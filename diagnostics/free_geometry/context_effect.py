#!/usr/bin/env python3
"""Stage 1 (context effect): does a longer aggregator context improve the
frozen VGGT teacher's 32-frame coverage reconstruction?

Per scene x ctx in {4,8,16,32}: the 32 eval frames (manifest eval32_frames,
eval order) are split into 32/ctx consecutive chunks; each chunk gets ONE
frozen forward (aggregator + depth head + camera head, no_grad, autocast off,
mirroring train_arms.infer_eval32). Products:

1. Per-frame per-pixel depth: depths/<scene>.npz (fp16, keys d_ctx{c}_f{pos:03d},
   pos = position in the eval32 list) + conf (c_ctx...) + per-frame AbsRel /
   delta1.25 / RMSE-log (scene-shared log scale, depth_metrics.py convention 1)
   + per-patch centered squared log-residual maps e_ctx{c} [32,P]
   (train_arms.perpatch_logres2 over all 32 views jointly = scene-shared scale).
2. Two TSDF reconstructions per ctx (repo ScanNetPP recon_unposed pipeline via
   VGGTEvaluator, crafted model_results trees):
   (a) gtpose:  GT extrinsics + predicted depth (per-chunk Umeyama scale s_c
       applied to depth so it is metric; predicted intrinsics kept identical to
       variant (b) so only the poses differ) -- isolates depth quality.
   (b) predpose: per-chunk Sim(3) Umeyama alignment of each chunk's predicted
       camera trajectory to the GT cameras (>=3 cams; deterministic, no ransac),
       aligned extrinsics + s_c-scaled depth. GT is used offline only.
   The evaluator's internal align_poses_umeyama then finds a residual ~identity
   (gtpose: exact identity), keeping the pipeline faithful. F1 = fscore,
   CD = overall. model_results trees are deleted right after metric extraction.
3. Pose AUC@3 per ctx: chunk-internal frame-pair AUC (modeling.pose_auc /
   compute_pose aligns to the first camera of the chunk), averaged over chunks.

Outputs: depths/<scene>.npz, metrics.csv, frame_metrics.csv, fusion_metrics.csv,
summary.md. GPU phase is the default; fusion runs as per-(ctx,variant)
subprocesses (--fusion_only mode) so no CUDA context is alive when
multiprocessing TSDF workers fork.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common import get_scene_data, load_gt_depth, load_image_model, load_manifest  # noqa: E402
import modeling as M  # noqa: E402
from train_arms import perpatch_logres2  # noqa: E402
from depth_metrics import frame_metrics  # noqa: E402

MANIFEST_DEFAULT = "artifacts/diagnostics/bakeoff_v2_transductive/scene_manifest.json"
OUT_DIR_DEFAULT = "artifacts/diagnostics/context_effect"
PROGRESS_MD = os.path.join("artifacts", "diagnostics", "PROGRESS.md")
CTXS_DEFAULT = [4, 8, 16, 32]

METRIC_FIELDS = ["scene", "ctx", "n_chunks", "absrel", "d125", "rmselog",
                 "auc03_mean", "auc30_mean", "peak_mem_gb"]
FRAME_FIELDS = ["scene", "ctx", "pos", "frame_idx", "absrel", "d125", "rmselog", "n_valid"]
FUSION_FIELDS = ["ctx", "variant", "scene", "acc", "comp", "overall",
                 "precision", "recall", "fscore"]


def progress(msg):
    line = f"- [{time.strftime('%H:%M')}] {msg}\n"
    try:
        with open(PROGRESS_MD, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass
    print(f"[progress] {msg}", flush=True)


def append_csv(path, fieldnames, rows, header_once=True):
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists or not header_once:
            w.writeheader()
        for r in rows:
            w.writerow(r)
        f.flush()


# --------------------------------------------------------------------------
# GPU phase
# --------------------------------------------------------------------------

@torch.no_grad()
def forward_chunk(vggt, images_chunk):
    """One frozen chunk forward. Returns depth [1,c,H,W,1], conf [1,c,H,W],
    pose_enc [1,c,9], feats24 (caller frees)."""
    feats24, psi = M.aggregator_all(vggt, images_chunk)
    depth, conf = M.replay_depth_nograd(vggt, feats24, images_chunk, psi)
    pose_enc = M.replay_camera_nograd(vggt, feats24)
    return depth, conf, pose_enc, feats24


def pose_to_ext_ixt(pose_enc, hw):
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    ext, ixt = pose_encoding_to_extri_intri(
        pose_enc.float(), image_size_hw=hw, pose_encoding_type="absT_quaR_FoV")
    return ext.squeeze(0).float().cpu().numpy(), ixt.squeeze(0).float().cpu().numpy()


def umeyama_to_gt(gt_ext44, pred_ext):
    """Chunk predicted w2c [c,3,4] -> Sim(3) aligned to GT w2c [c,4,4].
    Returns (s, ext_aligned [c,4,4]). Deterministic (no ransac; c>=3)."""
    from depth_anything_3.utils.pose_align import align_poses_umeyama
    _r, _t, s, ext_aligned = align_poses_umeyama(
        np.asarray(gt_ext44, dtype=np.float64), np.asarray(pred_ext, dtype=np.float64),
        return_aligned=True, ransac=False)
    return float(s), ext_aligned.astype(np.float32)


def scene_forwards(vggt, scene_data, ev, ctxs, device="cuda"):
    """All chunk forwards for one scene. Returns (per-ctx dict, images_hw, patch_hw)."""
    imgs = [load_image_model(scene_data.image_files[i]) for i in ev]
    H, W = imgs[0].shape[:2]
    patch_hw = (H // 14, W // 14)
    arr = np.stack(imgs, 0)
    images = torch.from_numpy(arr).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)

    out = {}
    for ctx in ctxs:
        n_chunks = 32 // ctx
        d_all = np.zeros((32, H, W), np.float32)
        c_all = np.zeros((32, H, W), np.float32)
        p_all = np.zeros((32, 9), np.float32)
        ext_all = np.zeros((32, 3, 4), np.float32)
        ixt_all = np.zeros((32, 3, 3), np.float32)
        extaln_all = np.zeros((32, 4, 4), np.float32)
        scales = np.zeros(n_chunks, np.float32)
        auc03 = np.zeros(n_chunks, np.float32)
        auc30 = np.zeros(n_chunks, np.float32)
        gt_ext_all = np.asarray(scene_data.extrinsics)[ev]  # [32,4,4] w2c
        for k in range(n_chunks):
            sl = slice(k * ctx, (k + 1) * ctx)
            images_c = images[:, sl].contiguous()
            depth, conf, pose_enc, feats24 = forward_chunk(vggt, images_c)
            d_all[sl] = depth.squeeze(0).squeeze(-1).float().cpu().numpy()
            c_all[sl] = conf.squeeze(0).float().cpu().numpy()
            p_all[sl] = pose_enc.squeeze(0).float().cpu().numpy()
            ext_c, ixt_c = pose_to_ext_ixt(pose_enc, (H, W))
            ext_all[sl] = ext_c[:, :3, :]
            ixt_all[sl] = ixt_c
            gt_ext_c = gt_ext_all[sl]
            s, ext_aln = umeyama_to_gt(gt_ext_c, ext_c)
            scales[k] = s
            extaln_all[sl] = ext_aln
            auc = M.pose_auc(pose_enc.float(), gt_ext_c)
            auc03[k], auc30[k] = auc["auc03"], auc["auc30"]
            del feats24, depth, conf, pose_enc, images_c
        out[ctx] = {"depth": d_all, "conf": c_all, "pose_enc": p_all,
                    "ext": ext_all, "ixt": ixt_all, "ext_aligned": extaln_all,
                    "scales": scales, "auc03": auc03, "auc30": auc30}
        print(f"  ctx={ctx:>2}: scales={np.round(scales, 4).tolist()} "
              f"auc03_mean={auc03.mean():.4f}", flush=True)
        torch.cuda.empty_cache()
    return out, (H, W), patch_hw, images


def depth_stats(depth32, gt32, patch_hw):
    """Scene-shared log-scale correction; per-frame metrics + per-patch errors."""
    preds = depth32.astype(np.float64)
    gts = gt32.astype(np.float64)
    m = np.isfinite(gts) & (gts > 0)
    c = float((np.log(np.clip(preds, 1e-6, None))[m] - np.log(gts[m])).mean())
    p1 = preds * np.exp(-c)
    frows, absrel, d125, rmselog = [], [], [], []
    for k in range(32):
        fm = frame_metrics(p1[k], gts[k])
        if fm is None:
            frows.append({"absrel": float("nan"), "d125": float("nan"),
                          "rmselog": float("nan"), "n_valid": 0})
            continue
        frows.append({"absrel": fm["absrel"], "d125": fm["d125"],
                      "rmselog": fm["rmselog"], "n_valid": fm["n"]})
        absrel.append(fm["absrel"]); d125.append(fm["d125"]); rmselog.append(fm["rmselog"])
    pp = perpatch_logres2(depth32, gt32, patch_hw)  # [32,P]
    return (frows, float(np.mean(absrel)), float(np.mean(d125)),
            float(np.mean(rmselog)), pp.astype(np.float32))


def run_gpu_phase(args, scenes):
    vggt = M.load_teacher(args.device)
    manifest = load_manifest(args.manifest)
    depths_dir = os.path.join(args.out_dir, "depths")
    os.makedirs(depths_dir, exist_ok=True)
    metric_rows, frame_rows = [], []
    for scene in scenes:
        ev = manifest["scenes"][scene]["eval32_frames"]
        assert len(ev) == 32
        for attempt in range(args.oom_retries + 1):
            try:
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                scene_data = get_scene_data(scene)
                res, (H, W), patch_hw, _images = scene_forwards(
                    vggt, scene_data, ev, args.ctxs, args.device)
                gt32 = np.stack([load_gt_depth(scene_data.aux.gt_depth_files[i], (H, W))
                                 for i in ev], 0)  # [32,H,W] NaN invalid
                save = {"frames": np.asarray(ev, np.int32),
                        "patch_hw": np.asarray(patch_hw, np.int32),
                        "gt_ext": np.asarray(scene_data.extrinsics)[ev].astype(np.float32),
                        "gt_ixt_raw": np.asarray(scene_data.aux.ixt_raw_list)[ev].astype(np.float32),
                        "image_files": np.asarray([scene_data.image_files[i] for i in ev])}
                for ctx in args.ctxs:
                    r = res[ctx]
                    frows, absrel, d125, rmselog, pp = depth_stats(r["depth"], gt32, patch_hw)
                    for pos in range(32):
                        save[f"d_ctx{ctx}_f{pos:03d}"] = r["depth"][pos].astype(np.float16)
                        save[f"c_ctx{ctx}_f{pos:03d}"] = r["conf"][pos].astype(np.float16)
                        frame_rows.append({"scene": scene, "ctx": ctx, "pos": pos,
                                           "frame_idx": ev[pos], **frows[pos]})
                    save[f"e_ctx{ctx}"] = pp
                    save[f"pose_ctx{ctx}"] = r["pose_enc"]
                    save[f"ext_ctx{ctx}"] = r["ext"]
                    save[f"extaln_ctx{ctx}"] = r["ext_aligned"][:, :3, :]
                    save[f"ixt_ctx{ctx}"] = r["ixt"]
                    save[f"scale_ctx{ctx}"] = r["scales"]
                    save[f"auc03_ctx{ctx}"] = r["auc03"]
                    save[f"auc30_ctx{ctx}"] = r["auc30"]
                    metric_rows.append({
                        "scene": scene, "ctx": ctx, "n_chunks": 32 // ctx,
                        "absrel": absrel, "d125": d125, "rmselog": rmselog,
                        "auc03_mean": float(r["auc03"].mean()),
                        "auc30_mean": float(r["auc30"].mean()),
                        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)})
                np.savez_compressed(os.path.join(depths_dir, f"{scene}.npz"), **save)
                dt = time.time() - t0
                print(f"[scene {scene}] done in {dt:.0f}s, "
                      f"peak={torch.cuda.max_memory_allocated() / 1024 ** 3:.1f}GB",
                      flush=True)
                del res, scene_data, gt32
                torch.cuda.empty_cache()
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if attempt == args.oom_retries:
                    raise
                print(f"[scene {scene}] OOM, retry in 60s "
                      f"({attempt + 1}/{args.oom_retries})", flush=True)
                time.sleep(60)
    append_csv(os.path.join(args.out_dir, "metrics.csv"), METRIC_FIELDS, metric_rows)
    append_csv(os.path.join(args.out_dir, "frame_metrics.csv"), FRAME_FIELDS, frame_rows)
    return metric_rows


# --------------------------------------------------------------------------
# Fusion phase (CPU, subprocess per ctx x variant)
# --------------------------------------------------------------------------

def fusion_run(args):
    """Build a crafted model_results tree for one (ctx, variant), evaluate with
    the repo VGGTEvaluator recon_unposed pipeline, append fusion_metrics.csv,
    delete model_results."""
    import open3d as o3d  # noqa: F401  (seed below)
    from vggt.vggt.bench.evaluator import VGGTEvaluator

    ctx, variant = args.ctx, args.variant
    manifest = load_manifest(args.manifest)
    scenes = args.scenes or sorted(manifest["scenes"])
    work_dir = os.path.join(args.out_dir, "fusion_work", f"ctx{ctx}_{variant}")
    depths_dir = os.path.join(args.out_dir, "depths")

    free_gb = shutil.disk_usage(args.out_dir).free / 1024 ** 3
    assert free_gb > 15.0, f"disk too low for fusion: {free_gb:.1f}G"

    for scene in scenes:
        z = np.load(os.path.join(depths_dir, f"{scene}.npz"), allow_pickle=True)
        n_chunks = 32 // ctx
        scales = z[f"scale_ctx{ctx}"]
        depth = np.stack([z[f"d_ctx{ctx}_f{pos:03d}"].astype(np.float32)
                          for pos in range(32)], 0)
        for k in range(n_chunks):
            depth[k * ctx:(k + 1) * ctx] *= scales[k]  # metric scale per chunk
        if variant == "gtpose":
            ext = z["gt_ext"][:, :3, :].astype(np.float32)
        else:
            ext = z[f"extaln_ctx{ctx}"].astype(np.float32)
        ixt = z[f"ixt_ctx{ctx}"].astype(np.float32)
        export_dir = os.path.join(work_dir, "model_results", "scannetpp", scene, "unposed")
        os.makedirs(os.path.join(export_dir, "exports", "mini_npz"), exist_ok=True)
        np.savez_compressed(os.path.join(export_dir, "exports", "mini_npz", "results.npz"),
                            depth=depth, extrinsics=ext, intrinsics=ixt)
        np.savez_compressed(os.path.join(export_dir, "exports", "gt_meta.npz"),
                            extrinsics=z["gt_ext"], intrinsics=z["gt_ixt_raw"],
                            image_files=z["image_files"])

    o3d.utility.random.seed(42)
    evaluator = VGGTEvaluator(work_dir=work_dir, datas=["scannetpp"],
                              modes=["recon_unposed"], scenes=scenes,
                              max_frames=-1, num_fusion_workers=4)
    metrics = evaluator.eval()
    per_scene = metrics["scannetpp_recon_unposed"]
    rows = []
    for scene in scenes:
        m = per_scene[scene]
        rows.append({"ctx": ctx, "variant": variant, "scene": scene,
                     **{k: m[k] for k in ("acc", "comp", "overall",
                                          "precision", "recall", "fscore")}})
    append_csv(os.path.join(args.out_dir, "fusion_metrics.csv"), FUSION_FIELDS, rows)
    shutil.rmtree(os.path.join(work_dir, "model_results"), ignore_errors=True)
    print(f"[fusion ctx{ctx} {variant}] done; model_results deleted", flush=True)
    print(json.dumps({f"ctx{ctx}_{variant}": per_scene.get("mean", {})}, indent=2),
          flush=True)


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

def _fmt(x, nd=4):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def write_summary(args, scenes):
    import collections
    met = list(csv.DictReader(open(os.path.join(args.out_dir, "metrics.csv"))))
    fus_path = os.path.join(args.out_dir, "fusion_metrics.csv")
    fus = list(csv.DictReader(open(fus_path))) if os.path.exists(fus_path) else []
    ctxs = sorted({int(r["ctx"]) for r in met})

    def agg(rows, key):
        vals = [float(r[key]) for r in rows if r[key] not in ("", "nan") and np.isfinite(float(r[key]))]
        return float(np.mean(vals)) if vals else None

    by_ctx = collections.defaultdict(list)
    for r in met:
        by_ctx[int(r["ctx"])].append(r)
    fus_by = collections.defaultdict(list)
    for r in fus:
        fus_by[(int(r["ctx"]), r["variant"])].append(r)

    lines = ["# Context effect: longer vs lower context on 32-frame coverage (frozen VGGT)",
             "",
             f"manifest: `{args.manifest}`; scenes: {', '.join(scenes)} (eval32_frames each).",
             "Depth metrics: scene-shared log scale (depth_metrics.py convention 1).",
             "AUC@3: chunk-internal frame-pair AUC (compute_pose aligns to chunk's first "
             "camera), averaged over the 32/ctx chunks; cross-chunk poses are not comparable.",
             "Fusion: repo ScanNetPP `recon_unposed` TSDF pipeline (voxel 0.02, trunc 0.15, "
             "max_depth 5m, threshold 5cm). Both variants pre-scale depth per chunk by the "
             "Sim(3) Umeyama scale s_c of that chunk's predicted trajectory vs GT cameras "
             "(deterministic, no ransac); gtpose then fuses with GT extrinsics (repo-internal "
             "residual alignment = identity), predpose with the chunk-aligned predicted "
             "extrinsics. Both keep predicted intrinsics, so variants differ ONLY in poses. "
             "F1 = fscore, CD = overall (acc+comp)/2, in metres.",
             "",
             "## Main table (6-scene means)",
             "",
             "| ctx | F1 (gtpose) | F1 (predpose) | CD (gtpose) | CD (predpose) | "
             "AUC@3 | AbsRel | delta1.25 |",
             "|---|---|---|---|---|---|---|---|"]
    for c in ctxs:
        fg = agg(fus_by.get((c, "gtpose"), []), "fscore")
        fp = agg(fus_by.get((c, "predpose"), []), "fscore")
        cg = agg(fus_by.get((c, "gtpose"), []), "overall")
        cp = agg(fus_by.get((c, "predpose"), []), "overall")
        a3 = agg(by_ctx[c], "auc03_mean")
        ar = agg(by_ctx[c], "absrel")
        d1 = agg(by_ctx[c], "d125")
        lines.append(f"| {c} | {_fmt(fg)} | {_fmt(fp)} | {_fmt(cg)} | {_fmt(cp)} | "
                     f"{_fmt(a3)} | {_fmt(ar)} | {_fmt(d1)} |")

    lines += ["", "## Per-scene detail", "",
              "| scene | ctx | F1 gt | F1 pred | CD gt | CD pred | AUC@3 | AbsRel | d1.25 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for scene in scenes:
        for c in ctxs:
            mr = [r for r in by_ctx[c] if r["scene"] == scene]
            fg = agg([r for r in fus_by.get((c, "gtpose"), []) if r["scene"] == scene], "fscore")
            fp = agg([r for r in fus_by.get((c, "predpose"), []) if r["scene"] == scene], "fscore")
            cg = agg([r for r in fus_by.get((c, "gtpose"), []) if r["scene"] == scene], "overall")
            cp = agg([r for r in fus_by.get((c, "predpose"), []) if r["scene"] == scene], "overall")
            lines.append(f"| {scene} | {c} | {_fmt(fg)} | {_fmt(fp)} | {_fmt(cg)} | {_fmt(cp)} | "
                         f"{_fmt(agg(mr, 'auc03_mean'))} | {_fmt(agg(mr, 'absrel'))} | "
                         f"{_fmt(agg(mr, 'd125'))} |")

    def delta(key_fn, c_lo=4, c_hi=32):
        lo, hi = key_fn(c_lo), key_fn(c_hi)
        if lo is None or hi is None:
            return None
        return hi - lo

    lines += ["", "## Reading", ""]
    d_fg = delta(lambda c: agg(fus_by.get((c, "gtpose"), []), "fscore"))
    d_fp = delta(lambda c: agg(fus_by.get((c, "predpose"), []), "fscore"))
    d_ar = delta(lambda c: agg(by_ctx[c], "absrel"))
    d_a3 = delta(lambda c: agg(by_ctx[c], "auc03_mean"))
    lines.append(f"- ctx 4 -> 32: F1(gtpose) {_fmt(d_fg, 4)}, F1(predpose) {_fmt(d_fp, 4)}, "
                 f"AbsRel {_fmt(d_ar, 4)}, AUC@3 {_fmt(d_a3, 4)}.")
    lines.append("- gtpose isolates depth quality (poses fixed to GT); predpose adds the "
                 "chunk-alignment/pose-consistency effect. If F1(gtpose) grows with ctx, "
                 "longer context improves per-view depth; the gtpose-vs-predpose gap is the "
                 "pose-consistency cost of short contexts.")
    lines.append("- depth npz keys: d_ctx{c}_f{pos:03d} / c_ctx... (fp16, pos = position in "
                 "eval32 order, `frames` maps pos -> scene frame index), e_ctx{c} [32,P] "
                 "per-patch centered squared log residuals (scene-shared scale), "
                 "scale/ext/extaln/ixt/pose_ctx{c}, auc03/30_ctx{c} per chunk.")
    out = os.path.join(args.out_dir, "summary.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST_DEFAULT)
    ap.add_argument("--out_dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--ctxs", nargs="*", type=int, default=CTXS_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--oom_retries", type=int, default=3)
    ap.add_argument("--skip_fusion", action="store_true")
    ap.add_argument("--fusion_only", action="store_true",
                    help="CPU mode: build+eval+delete one (ctx, variant) fusion")
    ap.add_argument("--ctx", type=int, default=None)
    ap.add_argument("--variant", default=None, choices=["gtpose", "predpose"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = load_manifest(args.manifest)
    scenes = args.scenes or sorted(manifest["scenes"])

    if args.fusion_only:
        assert args.ctx and args.variant
        fusion_run(args)
        return

    progress(f"context_effect Stage1 GPU 启动 scenes={scenes} ctxs={args.ctxs}")
    run_gpu_phase(args, scenes)
    progress("context_effect Stage1 GPU 完成（depths npz + depth/pose 指标）")

    if not args.skip_fusion:
        for ctx in args.ctxs:
            for variant in ("gtpose", "predpose"):
                for attempt in range(3):
                    r = subprocess.run(
                        [sys.executable, os.path.abspath(__file__),
                         "--fusion_only", "--manifest", args.manifest,
                         "--out_dir", args.out_dir, "--scenes", *scenes,
                         "--ctx", str(ctx), "--variant", variant])
                    if r.returncode == 0:
                        break
                    print(f"[fusion ctx{ctx} {variant}] rc={r.returncode}, "
                          f"retry in 60s ({attempt + 1}/3)", flush=True)
                    time.sleep(60)
                else:
                    raise RuntimeError(f"fusion ctx{ctx} {variant} failed 3x")
        progress("context_effect 融合评测完成（model_results 已清）")

    print(write_summary(args, scenes), flush=True)
    progress("context_effect Stage1 全部完成，summary.md 已写")
    print("DONE")


if __name__ == "__main__":
    main()
