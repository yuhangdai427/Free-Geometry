#!/usr/bin/env python3
"""Patch-token alpha interpolation -> 4-view depth metrics AND TSDF-fusion F1,
measured ON the 8->4 training pairs (not 32v).

Question: in the exact regime where the training signal lives (teacher 8v ->
shared 4v), does substituting student patch tokens with teacher's improve
(a) per-view depth metrics (AbsRel, d1.25; pair-wide scale), and
(b) the 4-view TSDF-fused reconstruction F1 vs GT?

patch_interp measured only per-patch log-res^2. Fusion F1 additionally
captures cross-view consistency - where an 8v-context teacher should help
most. GT extrinsics + student-predicted intrinsics are held fixed across
variants, so ONLY the depth maps differ.

Variants per pair: a0/a25/a50/a75/a100 (real interpolation, all 4 tap
layers), a100sham (teacher feats from pair+3), teacher (direct teacher 8->4
depth; parity check, should equal a100).

Depth scale: ONE pair-wide log-scale per (pair, variant) fitted vs GT depth
(offline diagnostic only; mirrors perpatch_logres2 centering), then fused
with GT extrinsics through the repo ScanNetPP recon_unposed TSDF pipeline
(voxel 0.02, trunc 0.15, max_depth 5m, threshold 5cm, AABB crop, down 0.02).

Outputs: depths/*.npz, depth_rows.jsonl, fusion_rows.jsonl, pair_metrics.csv,
summary.md. model_results trees are deleted inline.
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common import (  # noqa: E402
    STUDENT_INDICES, TAP_LAYERS, get_scene_data, load_manifest, stable_seed,
)
import modeling as M  # noqa: E402
from train_arms import perpatch_logres2  # noqa: E402
from depth_metrics import frame_metrics  # noqa: E402

OUT = "artifacts/diagnostics/patch_interp_f1"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
VARNAMES = ["a0", "a25", "a50", "a75", "a100", "a100sham", "teacher"]


# ---------------------------------------------------------------------------
# GPU phase
# ---------------------------------------------------------------------------

@torch.no_grad()
def gpu_phase(args, scenes):
    manifest = load_manifest(args.manifest)
    teacher = M.load_teacher(args.device)
    student = M.load_student(args.device)
    base = M.get_base_vggt(student)
    depths_dir = os.path.join(OUT, "depths")
    meta_dir = os.path.join(OUT, "pairs_meta")
    os.makedirs(depths_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    rows_path = os.path.join(OUT, "depth_rows.jsonl")
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    with open(rows_path, "w") as rows_f:
        for scene in scenes:
            sc = manifest["scenes"][scene]
            scene_data = get_scene_data(scene)
            M.reset_lora_(student)
            student.eval()
            pairs = sc["train_pairs"] + sc["probe_pairs"]
            all_pairs = list(pairs)
            if args.max_pairs:
                pairs = pairs[: args.max_pairs]
            for pi, pair in enumerate(pairs):
                images8, images4 = M.load_pair_images(scene_data, pair["teacher_frames"])
                H, W = images8.shape[-2:]
                ph, pw = H // 14, W // 14
                tfeats, tpsi = M.aggregator_all(teacher, images8)
                sfeats, spsi = student._get_aggregator()(images4)
                sfeats = [f.float() for f in sfeats]
                pair_sham = all_pairs[(pi + 3) % len(all_pairs)]
                images8_sh, _ = M.load_pair_images(scene_data, pair_sham["teacher_frames"])
                tfeats_sh, _ = M.aggregator_all(teacher, images8_sh)
                gt4 = M.load_probe_gt(scene_data, pair["student_frames"], (H, W))
                gt_ext44 = np.asarray(scene_data.extrinsics)[pair["student_frames"]]
                gt_ixt_raw = np.asarray(scene_data.aux.ixt_raw_list)[pair["student_frames"]]
                image_files = [scene_data.image_files[i] for i in pair["student_frames"]]

                def depth_with(t_src, alpha):
                    slots = [None] * 24
                    for l in TAP_LAYERS:
                        f = sfeats[l].clone()
                        f[:, :, M.PATCH_START_IDX:, :] = (
                            (1 - alpha) * f[:, :, M.PATCH_START_IDX:, :]
                            + alpha * t_src[l][:, STUDENT_INDICES, M.PATCH_START_IDX:, :].float())
                        slots[l] = f.contiguous()
                    d, _ = M.replay_depth_nograd(base, slots, images4, spsi)
                    return d.squeeze(0).squeeze(-1).float().cpu().numpy(), slots

                # student predicted intrinsics (alpha=0 camera replay), fixed for all variants
                _, slots0 = depth_with(tfeats, 0.0)
                pose_enc = M.replay_camera_nograd(base, slots0)
                _, ixt_pred = pose_encoding_to_extri_intri(
                    pose_enc.float(), image_size_hw=(H, W), pose_encoding_type="absT_quaR_FoV")
                ixt_pred = ixt_pred.squeeze(0).float().cpu().numpy()

                np.savez_compressed(
                    os.path.join(meta_dir, f"{scene}__p{pi:02d}.npz"),
                    gt_ext44=gt_ext44.astype(np.float32),
                    gt_ixt_raw=gt_ixt_raw.astype(np.float32),
                    ixt_pred=ixt_pred.astype(np.float32),
                    image_files=np.asarray(image_files))

                variant_depths = {}
                for name, alpha in zip(VARNAMES[:5], ALPHAS):
                    variant_depths[name] = depth_with(tfeats, alpha)[0]
                variant_depths["a100sham"] = depth_with(tfeats_sh, 1.0)[0]
                # teacher direct: teacher feats on shared slots (parity with a100)
                slots_t = [None] * 24
                for l in TAP_LAYERS:
                    slots_t[l] = tfeats[l][:, STUDENT_INDICES].float().contiguous()
                dt, _ = M.replay_depth_nograd(base, slots_t, images4, tpsi)
                variant_depths["teacher"] = dt.squeeze(0).squeeze(-1).float().cpu().numpy()

                valid = np.isfinite(gt4) & (gt4 > 0)
                for name, d in variant_depths.items():
                    r = np.log(np.clip(d, 1e-6, None)) - np.log(np.clip(gt4, 1e-6, None))
                    c = float(r[valid].mean())
                    ds = (d * np.exp(-c)).astype(np.float32)
                    absrels, d125s = [], []
                    for k in range(4):
                        fm = frame_metrics(ds[k], gt4[k])
                        if fm:
                            absrels.append(fm["absrel"])
                            d125s.append(fm["d125"])
                    pp = perpatch_logres2(d, gt4, (ph, pw))
                    np.savez_compressed(
                        os.path.join(depths_dir, f"{scene}__p{pi:02d}__{name}.npz"),
                        depth=ds.astype(np.float16))
                    rows_f.write(json.dumps({
                        "scene": scene, "pair": pi, "variant": name,
                        "absrel": float(np.mean(absrels)), "d125": float(np.mean(d125s)),
                        "logres2": float(np.nanmean(pp)), "scale": float(np.exp(-c))}) + "\n")
                    rows_f.flush()
                print(f"[{scene} p{pi}] 7 variants done, "
                      f"a0 absrel={json.loads(open(rows_path).readlines()[-7])['absrel']:.4f}",
                      flush=True)
                del tfeats, sfeats, tfeats_sh, variant_depths
                torch.cuda.empty_cache()
            del scene_data
    print("[gpu] done", flush=True)


# ---------------------------------------------------------------------------
# Fusion phase (CPU multiprocessing)
# ---------------------------------------------------------------------------

_DS = None
_GT_CACHE = {}


def _fusion_init():
    global _DS
    from depth_anything_3.bench.datasets.scannetpp import ScanNetPP
    _DS = ScanNetPP()


def _gt_pcd(scene):
    import open3d as o3d
    from depth_anything_3.bench.utils import sample_points_from_mesh
    if scene not in _GT_CACHE:
        gt_data = _DS.get_data(scene)
        o3d.utility.random.seed(stable_seed("gtpcd", scene) % (2**31))
        gt_mesh = o3d.io.read_triangle_mesh(gt_data.aux.gt_mesh_path)
        gt_pcd = sample_points_from_mesh(gt_mesh, _DS.sampling_number)
        aabb = gt_pcd.get_axis_aligned_bounding_box()
        gt_ds = gt_pcd.voxel_down_sample(_DS.down_sample)
        _GT_CACHE[scene] = (np.asarray(gt_ds.points), aabb)
    return _GT_CACHE[scene]


def _fusion_one(task):
    scene, pi, name = task
    import open3d as o3d
    from depth_anything_3.bench.utils import nn_correspondance
    work = os.path.join(OUT, "fusion_work", f"{scene}__p{pi:02d}__{name}")
    export_dir = os.path.join(work, "model_results", "scannetpp", scene, "unposed")
    os.makedirs(os.path.join(export_dir, "exports", "mini_npz"), exist_ok=True)
    try:
        z = np.load(os.path.join(OUT, "depths", f"{scene}__p{pi:02d}__{name}.npz"))
        meta = np.load(os.path.join(OUT, "pairs_meta", f"{scene}__p{pi:02d}.npz"),
                       allow_pickle=True)
        depth = z["depth"].astype(np.float32)
        gt_ext44 = meta["gt_ext44"]
        np.savez_compressed(
            os.path.join(export_dir, "exports", "mini_npz", "results.npz"),
            depth=depth, extrinsics=gt_ext44[:, :3, :].astype(np.float32),
            intrinsics=meta["ixt_pred"].astype(np.float32))
        np.savez_compressed(
            os.path.join(export_dir, "exports", "gt_meta.npz"),
            extrinsics=gt_ext44, intrinsics=meta["gt_ixt_raw"],
            image_files=meta["image_files"])
        o3d.utility.random.seed(stable_seed("fuse", scene, pi, name) % (2**31))
        fuse_path = os.path.join(export_dir, "exports", "fuse", "pcd.ply")
        _DS.fuse3d(scene, os.path.join(export_dir, "exports", "mini_npz", "results.npz"),
                   fuse_path, "recon_unposed")
        pred = o3d.io.read_point_cloud(fuse_path)
        gt_pts, aabb = _gt_pcd(scene)
        pts = np.asarray(pred.points)
        inside = (
            (pts[:, 0] >= aabb.min_bound[0] - 0.1) & (pts[:, 0] <= aabb.max_bound[0] + 0.1) &
            (pts[:, 1] >= aabb.min_bound[1] - 0.1) & (pts[:, 1] <= aabb.max_bound[1] + 0.1) &
            (pts[:, 2] >= aabb.min_bound[2] - 0.1) & (pts[:, 2] <= aabb.max_bound[2] + 0.1))
        pred = pred.select_by_index(inside.nonzero()[0])
        pred = pred.voxel_down_sample(_DS.down_sample)
        vp = np.asarray(pred.points)
        if len(vp) == 0 or len(gt_pts) == 0:
            m = {"acc": float("inf"), "comp": float("inf"), "overall": float("inf"),
                 "precision": 0.0, "recall": 0.0, "fscore": 0.0}
        else:
            dpg = nn_correspondance(gt_pts, vp)
            dgp = nn_correspondance(vp, gt_pts)
            acc, comp = float(np.mean(dpg)), float(np.mean(dgp))
            prec = float(np.mean(dpg < _DS.eval_threshold))
            rec = float(np.mean(dgp < _DS.eval_threshold))
            m = {"acc": acc, "comp": comp, "overall": (acc + comp) / 2,
                 "precision": prec, "recall": rec,
                 "fscore": 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0}
        return {"scene": scene, "pair": pi, "variant": name, **m}
    finally:
        shutil.rmtree(os.path.join(work, "model_results"), ignore_errors=True)


def fusion_phase(args, scenes):
    from multiprocessing import Pool
    manifest = load_manifest(args.manifest)
    tasks = []
    for scene in scenes:
        n = len(manifest["scenes"][scene]["train_pairs"] +
                manifest["scenes"][scene]["probe_pairs"])
        if args.max_pairs:
            n = min(n, args.max_pairs)
        for pi in range(n):
            for name in VARNAMES:
                if os.path.exists(os.path.join(OUT, "depths", f"{scene}__p{pi:02d}__{name}.npz")):
                    tasks.append((scene, pi, name))
    print(f"[fusion] {len(tasks)} tasks on {args.workers} workers", flush=True)
    t0 = time.time()
    with open(os.path.join(OUT, "fusion_rows.jsonl"), "w") as f:
        with Pool(args.workers, initializer=_fusion_init) as pool:
            for i, r in enumerate(pool.imap_unordered(_fusion_one, tasks)):
                f.write(json.dumps(r) + "\n")
                f.flush()
                if (i + 1) % 21 == 0:
                    dt = time.time() - t0
                    print(f"[fusion] {i + 1}/{len(tasks)} in {dt:.0f}s", flush=True)
    shutil.rmtree(os.path.join(OUT, "fusion_work"), ignore_errors=True)
    print("[fusion] done; fusion_work deleted", flush=True)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(args, scenes):
    import collections
    import csv
    import statistics as st
    drows = [json.loads(l) for l in open(os.path.join(OUT, "depth_rows.jsonl"))]
    frows = [json.loads(l) for l in open(os.path.join(OUT, "fusion_rows.jsonl"))]
    fmap = {(r["scene"], r["pair"], r["variant"]): r for r in frows}
    merged = []
    for r in drows:
        fr = fmap.get((r["scene"], r["pair"], r["variant"]))
        merged.append({**r, **({k: fr[k] for k in ("acc", "comp", "overall",
                                                  "precision", "recall", "fscore")} if fr else {})})
    with open(os.path.join(OUT, "pair_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "pair", "variant", "absrel", "d125",
                                          "logres2", "scale", "acc", "comp", "overall",
                                          "precision", "recall", "fscore"])
        w.writeheader()
        for r in merged:
            w.writerow(r)

    by = collections.defaultdict(list)
    for r in merged:
        by[r["variant"]].append(r)

    def agg(rows, k):
        v = [r[k] for r in rows if k in r and np.isfinite(r[k])]
        return float(np.mean(v)) if v else float("nan")

    lines = ["# patch interpolation on 8->4 pairs: depth metrics + 4-view fusion F1", "",
             f"scenes x pairs = {len(scenes)} x {len(merged) // (len(scenes) * len(VARNAMES))}; "
             "GT poses + student-pred intrinsics fixed; pair-wide depth scale fitted offline; "
             "repo TSDF (voxel 0.02, trunc 0.15, max 5m, thr 5cm).", "",
             "| variant | AbsRel | d1.25 | logres2 | F1 | CD | acc | comp |",
             "|---|---|---|---|---|---|---|---|"]
    for name in VARNAMES:
        rows = by[name]
        lines.append(f"| {name} | {agg(rows, 'absrel'):.4f} | {agg(rows, 'd125'):.4f} | "
                     f"{agg(rows, 'logres2'):.4f} | {agg(rows, 'fscore'):.4f} | "
                     f"{agg(rows, 'overall'):.4f} | {agg(rows, 'acc'):.4f} | {agg(rows, 'comp'):.4f} |")

    lines += ["", "## paired deltas vs a0 (per pair, then averaged)", "",
              "| variant | dF1 mean | F1 wins | dAbsRel mean | AbsRel wins |",
              "|---|---|---|---|---|"]
    base = {(r["scene"], r["pair"]): r for r in by["a0"]}
    for name in VARNAMES[1:]:
        df, dar = [], []
        for r in by[name]:
            b = base.get((r["scene"], r["pair"]))
            if b and np.isfinite(r.get("fscore", np.nan)) and np.isfinite(b.get("fscore", np.nan)):
                df.append(r["fscore"] - b["fscore"])
                dar.append(r["absrel"] - b["absrel"])
        wf = sum(1 for x in df if x > 0)
        wa = sum(1 for x in dar if x < 0)
        lines.append(f"| {name} | {st.mean(df):+.4f} | {wf}/{len(df)} | "
                     f"{st.mean(dar):+.5f} | {wa}/{len(dar)} |")

    lines += ["", "## per-scene F1 for a0 / a100 / teacher", "",
              "| scene | a0 | a25 | a50 | a75 | a100 | teacher |", "|---|---|---|---|---|---|---|"]
    for scene in scenes:
        vals = []
        for name in ["a0", "a25", "a50", "a75", "a100", "teacher"]:
            rows = [r for r in by[name] if r["scene"] == scene]
            vals.append(f"{agg(rows, 'fscore'):.4f}")
        lines.append(f"| {scene} | " + " | ".join(vals) + " |")

    # per-pair F1 gain of teacher over a0, correlation with logres2 gain
    gains = []
    for r in by["teacher"]:
        b = base.get((r["scene"], r["pair"]))
        if b and np.isfinite(r.get("fscore", np.nan)):
            gains.append((r["fscore"] - b["fscore"], b["logres2"] - r["logres2"]))
    if gains:
        try:
            import scipy.stats as sps
            rho = float(sps.spearmanr([g[0] for g in gains], [g[1] for g in gains]).statistic)
        except Exception:
            rho = float("nan")
        lines += ["", f"- pairs where teacher F1 > a0 F1: {sum(1 for g in gains if g[0] > 0)}/{len(gains)}",
                  f"- spearman(dF1, dLogres2 improvement) across pairs: {rho:.3f}"]
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write(txt)
    print(txt, flush=True)
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="artifacts/diagnostics/bakeoff_v2_transductive/scene_manifest.json")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--max_pairs", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fusion_only", action="store_true")
    ap.add_argument("--summarize_only", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    manifest = load_manifest(args.manifest)
    scenes = args.scenes or sorted(manifest["scenes"])
    if args.summarize_only:
        summarize(args, scenes)
        return
    if args.fusion_only:
        fusion_phase(args, scenes)
        return
    gpu_phase(args, scenes)
    fusion_phase(args, scenes)
    summarize(args, scenes)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
