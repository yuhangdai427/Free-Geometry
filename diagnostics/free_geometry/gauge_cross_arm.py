#!/usr/bin/env python3
"""Gauge-recoupling study for an ARBITRARY arm on a dataset (extension slot of
the gauge_cross_* pipeline, which hardcodes the per-dataset champion arm).

Use case driving this file: 7scenes C2M_maskrel@100v into a SEPARATE scratch
run_root (<ds>/gauge_cross_c2m) so the champion-arm artifacts under
<ds>/gauge_cross stay untouched. A0_baseline npz can be hardlink-reused from
an existing gauge_cross run (--reuse-baseline-from).

Stages (--stage): infer (GPU, waits >=20GB free per scene-arm), measure (CPU),
gtfree (CPU), fuse (CPU; set CUDA_VISIBLE_DEVICES="" in-process), report (CPU).
All stages resumable. New file; does not modify any existing module.
Run from repo root, e.g.:

  python3 diagnostics/free_geometry/gauge_cross_arm.py --dataset 7scenes \
      --arm C2M_maskrel --tag 100v --metric-file eval32_metrics_C2M.json \
      --out .../gauge_cross_c2m --reuse-baseline-from .../gauge_cross \
      --stage infer
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.join(_REPO, "src", "vggt"))

FP = os.path.join(_REPO, "artifacts", "diagnostics", "final_protocol")
STEP = 100
MIN_FREE_MIB = 20 * 1024
GRID = np.arange(0.90, 1.10 + 1e-9, 0.005)
PTS_PER_FRAME = 700
K_NN = 12
VARIANTS = ["orig", "fixgt", "fixfree"]


def as44(ext):
    ext = np.asarray(ext)
    if ext.shape[-2:] == (4, 4):
        return ext
    n = ext.shape[0]
    out = np.tile(np.eye(4, dtype=ext.dtype), (n, 1, 1))
    out[:, :3, :] = ext
    return out


def gpu_free_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
    return min(int(x) for x in out.decode().split())


def wait_gpu():
    while True:
        free = gpu_free_mib()
        if free >= MIN_FREE_MIB:
            return
        print(f"[wait] gpu free {free} MiB < {MIN_FREE_MIB}; sleeping 60s", flush=True)
        time.sleep(60)


def npz_dir(out, ds, arm, tag, scene):
    return os.path.join(out, "eval32", f"{arm}@{tag}", "model_results", ds,
                        scene, "unposed", "exports")


def npz_path(out, ds, arm, tag, scene):
    return os.path.join(npz_dir(out, ds, arm, tag, scene),
                        "mini_npz", "results.npz")


def load_npz(out, ds, arm, tag, scene):
    p = npz_dir(out, ds, arm, tag, scene)
    r = np.load(os.path.join(p, "mini_npz", "results.npz"))
    g = np.load(os.path.join(p, "gt_meta.npz"), allow_pickle=True)
    conf = r["conf"] if "conf" in r.files else None
    return (r["depth"], as44(r["extrinsics"]), r["intrinsics"], conf,
            as44(g["extrinsics"]), g["intrinsics"])


# --------------------------------------------------------------------------
def stage_infer(args, scenes):
    import common
    from common import get_scene_data, gt_ixt_raw
    import modeling as M
    from train_arms import infer_eval32, save_eval_npz

    RR = os.path.join(FP, args.dataset)
    # hardlink-reuse baseline npz
    if args.reuse_baseline_from:
        for scene in scenes:
            dst = npz_dir(args.out, args.dataset, "A0_baseline", args.tag, scene)
            if os.path.isfile(os.path.join(dst, "mini_npz", "results.npz")):
                continue
            src = npz_dir(args.reuse_baseline_from, args.dataset,
                          "A0_baseline", args.tag, scene)
            if not os.path.isfile(os.path.join(src, "mini_npz", "results.npz")):
                continue
            os.makedirs(os.path.join(dst, "mini_npz"), exist_ok=True)
            os.link(os.path.join(src, "mini_npz", "results.npz"),
                    os.path.join(dst, "mini_npz", "results.npz"))
            os.link(os.path.join(src, "gt_meta.npz"),
                    os.path.join(dst, "gt_meta.npz"))
            print(f"[link] A0_baseline {scene}", flush=True)

    manifest = args.manifest
    wait_gpu()
    for arm in ["A0_baseline", args.arm]:
        todo = [s for s in scenes
                if not os.path.isfile(npz_path(args.out, args.dataset, arm,
                                               args.tag, s))]
        if not todo:
            print(f"[skip] {arm}: all scenes present", flush=True)
            continue
        student = M.load_student()
        student.eval()
        if arm == "A0_baseline":
            M.reset_lora_(student)  # LoRA B=0 -> exact base model
        for scene in todo:
            wait_gpu()
            if arm != "A0_baseline":
                peft_dir = os.path.join(RR, "ckpts", scene, arm,
                                        f"step{STEP}_lora_peft")
                adapter = scene.replace("/", "__")
                student.vggt.load_adapter(peft_dir, adapter_name=adapter)
                student.vggt.set_adapter(adapter)
            scene_data = get_scene_data(scene)
            frames = manifest["scenes"][scene]["eval32_frames"]
            t0 = time.time()
            depth, ext, intr, _, conf = infer_eval32(student, scene_data, frames)
            save_eval_npz(args.out, f"{arm}@{args.tag}", scene, depth, ext, intr,
                          np.asarray(scene_data.extrinsics)[frames],
                          gt_ixt_raw(scene_data, frames),
                          [scene_data.image_files[i] for i in frames], frames,
                          conf=conf)
            print(f"[done] {arm} {scene} ({time.time()-t0:.0f}s)", flush=True)
        del student
        import torch
        torch.cuda.empty_cache()
    print("INFER DONE", flush=True)


# --------------------------------------------------------------------------
def stage_measure(args, scenes):
    import common
    from common import get_scene_data, load_gt_depth
    from depth_anything_3.utils.pose_align import align_poses_umeyama

    ds = args.dataset
    deltas = {}
    m = json.load(open(os.path.join(FP, ds, args.metric_file)))
    base = m[f"A0_baseline@{args.tag}"]
    tta = m[f"{args.arm}@{args.tag}"]
    for s in scenes:
        db = base[f"{ds}_recon_unposed"][s]
        dt = tta[f"{ds}_recon_unposed"][s]
        pb = base[f"{ds}_pose"][s]
        pt = tta[f"{ds}_pose"][s]
        deltas.setdefault(s, {})[args.arm] = {
            "f1_base_pub": db["fscore"], "f1_tta_pub": dt["fscore"],
            "dF1": dt["fscore"] - db["fscore"],
            "cd_base_pub": db["overall"], "cd_tta_pub": dt["overall"],
            "dCD": db["overall"] - dt["overall"],
            "auc_base_pub": pb["auc03"], "auc_tta_pub": pt["auc03"],
            "dAUC": pt["auc03"] - pb["auc03"],
        }

    rows = []
    for scene in scenes:
        sd = get_scene_data(scene)
        frames = args.manifest["scenes"][scene]["eval32_frames"]
        gt_depths = None
        for arm in ["A0_baseline", args.arm]:
            try:
                depth, pred_ext, intr, conf, gt_ext, gt_intr = load_npz(
                    args.out, ds, arm, args.tag, scene)
            except FileNotFoundError:
                print(f"[skip] {arm} {scene}: npz missing", flush=True)
                continue
            _, _, sp, _ = align_poses_umeyama(
                gt_ext.copy(), pred_ext.copy(), return_aligned=True,
                ransac=True, random_state=42)
            if gt_depths is None:
                hw = depth[0].shape[-2:]
                gt_depths = [load_gt_depth(sd.aux.gt_depth_files[i], hw)
                             for i in frames]
            logs = []
            for i in range(len(depth)):
                g = gt_depths[i]
                om = np.isfinite(g) & (g > 0) & np.isfinite(depth[i]) & (depth[i] > 0)
                logs.append(np.log(g[om]) - np.log(depth[i][om]))
            sdep = float(np.exp(np.median(np.concatenate(logs))))
            row = {"scene": scene, "arm": arm,
                   "s_pose": float(sp), "s_depth": sdep,
                   "mismatch": float(sp) / sdep - 1.0}
            row.update(deltas.get(scene, {}).get(arm, {}))
            rows.append(row)
            print(f"[gauge] {arm} {scene}: s_pose={sp:.4f} s_depth={sdep:.4f} "
                  f"mismatch={row['mismatch']*100:+.2f}%", flush=True)

    os.makedirs(args.out, exist_ok=True)
    keys = ["scene", "arm", "s_pose", "s_depth", "mismatch",
            "f1_base_pub", "f1_tta_pub", "dF1", "cd_base_pub", "cd_tta_pub",
            "dCD", "auc_base_pub", "auc_tta_pub", "dAUC"]
    with open(os.path.join(args.out, "gauge_mismatch.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(args.out, "gauge_mismatch.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("MEASURE DONE", flush=True)


# --------------------------------------------------------------------------
def sample_frame_points(depth, intr, conf, n_pts, rng):
    ys, xs = np.nonzero(np.isfinite(depth) & (depth > 0))
    if len(ys) == 0:
        return None
    sel = rng.choice(len(ys), size=min(n_pts, len(ys)), replace=False)
    ys, xs = ys[sel], xs[sel]
    z = depth[ys, xs].astype(np.float64)
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    rays = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(z)], 1)
    w = conf[ys, xs].astype(np.float64) if conf is not None else np.ones_like(z)
    w = np.clip(w, 1e-3, None)
    return rays, z, w


def stage_gtfree(args, scenes):
    from scipy.spatial import cKDTree

    f_gt = {}
    gm_path = os.path.join(args.out, "gauge_mismatch.csv")
    if os.path.isfile(gm_path):
        with open(gm_path) as f:
            for row in csv.DictReader(f):
                f_gt[(row["arm"], row["scene"])] = \
                    float(row["s_depth"]) / float(row["s_pose"])

    results = {}
    rj_path = os.path.join(args.out, "gauge_gtfree.json")
    if os.path.isfile(rj_path):
        results = json.load(open(rj_path))
    for arm in ["A0_baseline", args.arm]:
        for scene in scenes:
            key = f"{arm}/{scene}"
            if key in results:
                print(f"[skip] {key}", flush=True)
                continue
            try:
                depth, ext, intr, conf, _, _ = load_npz(
                    args.out, args.dataset, arm, args.tag, scene)
            except FileNotFoundError:
                continue
            c2w = np.linalg.inv(ext)
            rng = np.random.default_rng(11)
            frames = []
            for i in range(len(depth)):
                sp = sample_frame_points(depth[i], intr[i],
                                         conf[i] if conf is not None else None,
                                         PTS_PER_FRAME, rng)
                if sp is not None:
                    frames.append((i, *sp))
            if len(frames) < 2:
                continue
            med_d = float(np.median(np.concatenate([fr[2] for fr in frames])))
            thr = 0.02 * med_d
            objs_a, objs_b = [], []
            for f in GRID:
                pts_all, fid_all, w_all = [], [], []
                for i, rays, z, w in frames:
                    pw = (rays * (z * f)[:, None]) @ c2w[i, :3, :3].T \
                        + c2w[i, :3, 3]
                    pts_all.append(pw)
                    fid_all.append(np.full(len(pw), i))
                    w_all.append(w)
                pts = np.concatenate(pts_all)
                fid = np.concatenate(fid_all)
                w = np.concatenate(w_all)
                tree = cKDTree(pts)
                dd, ii = tree.query(pts, k=K_NN + 1, workers=4)
                d_cross = np.full(len(pts), np.nan)
                for k in range(1, K_NN + 1):
                    diff = fid[ii[:, k]] != fid
                    take = np.isnan(d_cross) & diff
                    d_cross[take] = dd[take, k]
                d_cross = np.where(np.isnan(d_cross), dd[:, -1], d_cross)
                objs_a.append(float(np.sum(w * d_cross) / np.sum(w)))
                objs_b.append(float(np.sum(w * (d_cross < thr)) / np.sum(w)))
            objs_a = np.array(objs_a)
            objs_b = np.array(objs_b)
            f_a = float(GRID[int(np.argmin(objs_a))])
            f_b = float(GRID[int(np.argmax(objs_b))])
            results[key] = {
                "arm": arm, "scene": scene,
                "f_free_dist": f_a, "f_free_inlier": f_b,
                "f_gt": f_gt.get((arm, scene)),
                "obj_dist": objs_a.tolist(), "obj_inlier": objs_b.tolist(),
                "grid": GRID.tolist(), "thr": thr,
            }
            fg = f_gt.get((arm, scene))
            dev = "" if fg is None else f" dev={(f_a-fg)*100:+.2f}pt"
            print(f"[gtfree] {key}: f_dist={f_a:.3f} f_inlier={f_b:.3f} "
                  f"f_gt={fg if fg is None else round(fg,4)}{dev}", flush=True)
            with open(rj_path, "w") as f:
                json.dump(results, f, indent=1)
    print("GTFREE DONE", flush=True)


# --------------------------------------------------------------------------
def stage_fuse(args, scenes):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import shutil

    ds = args.dataset
    f_gt, f_free = {}, {}
    gm = os.path.join(args.out, "gauge_mismatch.csv")
    if os.path.isfile(gm):
        with open(gm) as f:
            for row in csv.DictReader(f):
                f_gt[(row["arm"], row["scene"])] = \
                    float(row["s_depth"]) / float(row["s_pose"])
    gf = os.path.join(args.out, "gauge_gtfree.json")
    if os.path.isfile(gf):
        for v in json.load(open(gf)).values():
            f_free[(v["arm"], v["scene"])] = v["f_free_dist"]

    def make_variant_npz(arm, scene, factor, variant):
        src = npz_dir(args.out, ds, arm, args.tag, scene)
        dst = npz_dir(args.out, ds, f"{arm}_{variant}", args.tag, scene)
        r = np.load(os.path.join(src, "mini_npz", "results.npz"))
        os.makedirs(os.path.join(dst, "mini_npz"), exist_ok=True)
        kw = {k: r[k] for k in r.files}
        kw["depth"] = np.round(kw["depth"] * factor, 8)
        np.savez_compressed(os.path.join(dst, "mini_npz", "results.npz"), **kw)
        shutil.copyfile(os.path.join(src, "gt_meta.npz"),
                        os.path.join(dst, "gt_meta.npz"))

    metrics_path = os.path.join(args.out, "gauge_fix_metrics.json")
    all_metrics = {}
    if os.path.isfile(metrics_path):
        all_metrics = json.load(open(metrics_path))

    import open3d as o3d
    from vggt.bench.evaluator import VGGTEvaluator

    for arm in ["A0_baseline", args.arm]:
        for variant in args.variants:
            todo = []
            for scene in scenes:
                mkey = f"{arm}_{variant}@{args.tag}::{ds}_recon_unposed"
                if mkey in all_metrics and scene in all_metrics[mkey]:
                    continue
                if variant == "orig":
                    f = 1.0
                elif variant == "fixgt":
                    f = f_gt.get((arm, scene))
                else:
                    f = f_free.get((arm, scene))
                if f is None:
                    print(f"[skip] {arm} {scene} {variant}: no factor",
                          flush=True)
                    continue
                make_variant_npz(arm, scene, f, variant)
                todo.append(scene)
            if not todo:
                continue
            work_dir = os.path.join(args.out, "eval32",
                                    f"{arm}_{variant}@{args.tag}")
            o3d.utility.random.seed(42)
            ev = VGGTEvaluator(work_dir=work_dir, datas=[ds],
                               modes=["recon_unposed"], scenes=todo,
                               max_frames=0, num_fusion_workers=args.workers)
            m = ev.eval()
            for k, v in m.items():
                mkey = f"{arm}_{variant}@{args.tag}::{k}"
                all_metrics.setdefault(mkey, {}).update(v)
                for s in todo:
                    r = v[s]
                    print(f"RESULT {arm} {variant} {s}: F1={r['fscore']:.4f} "
                          f"CD={r['overall']:.4f} acc={r['acc']:.4f} "
                          f"comp={r['comp']:.4f}", flush=True)
            with open(metrics_path, "w") as fp:
                json.dump(all_metrics, fp, indent=1, default=str)
    print("FUSE DONE", flush=True)


# --------------------------------------------------------------------------
def mean(xs):
    xs = list(xs)
    return float(np.mean(xs)) if xs else float("nan")


def stage_report(args, scenes):
    ds = args.dataset
    out = args.out
    mm = {}
    with open(os.path.join(out, "gauge_mismatch.csv")) as f:
        for row in csv.DictReader(f):
            mm[(row["arm"], row["scene"])] = row
    gf = json.load(open(os.path.join(out, "gauge_gtfree.json")))
    f_free = {(v["arm"], v["scene"]): v["f_free_dist"] for v in gf.values()}
    fm = json.load(open(os.path.join(out, "gauge_fix_metrics.json")))
    pub = json.load(open(os.path.join(FP, ds, args.metric_file)))
    ARMS = ["A0_baseline", args.arm]

    def fused(arm, variant, scene):
        key = f"{arm}_{variant}@{args.tag}::{ds}_recon_unposed"
        v = fm.get(key, {}).get(scene)
        return v if isinstance(v, dict) else None

    def pubm(arm, scene):
        key = f"{'A0_baseline' if arm == 'A0_baseline' else args.arm}@{args.tag}"
        return pub[key][f"{ds}_recon_unposed"][scene]

    # AUC invariance: variant npz ext/intr bitwise == source npz (all scenes)
    auc_max_dev = 0.0
    n_pairs = 0
    for scene in scenes:
        for arm in ARMS:
            src = os.path.join(npz_dir(out, ds, arm, args.tag, scene),
                               "mini_npz", "results.npz")
            for variant in ("fixgt", "fixfree"):
                dst = os.path.join(npz_dir(out, ds, f"{arm}_{variant}",
                                           args.tag, scene),
                                   "mini_npz", "results.npz")
                if not (os.path.isfile(src) and os.path.isfile(dst)):
                    continue
                a = np.load(src)
                b = np.load(dst)
                de = max(float(np.abs(a[k] - b[k]).max())
                         for k in ("extrinsics", "intrinsics"))
                auc_max_dev = max(auc_max_dev, de)
                n_pairs += 1

    rows = []
    for scene in scenes:
        for arm in ARMS:
            r = mm.get((arm, scene))
            fo = fused(arm, "orig", scene)
            if r is None or fo is None:
                continue
            fg = fused(arm, "fixgt", scene)
            ff = fused(arm, "fixfree", scene)
            rows.append({
                "scene": scene, "arm": arm,
                "mismatch": float(r["mismatch"]),
                "f_gt": float(r["s_depth"]) / float(r["s_pose"]),
                "f_free": f_free.get((arm, scene)),
                "F1_orig": fo["fscore"],
                "F1_fixgt": fg["fscore"] if fg else None,
                "F1_fixfree": ff["fscore"] if ff else None,
                "CD_orig": fo["overall"],
                "CD_fixgt": fg["overall"] if fg else None,
                "CD_fixfree": ff["overall"] if ff else None,
                "F1_pub": pubm(arm, scene)["fscore"],
                "CD_pub": pubm(arm, scene)["overall"],
            })

    summary = {"dataset": ds, "arm": args.arm, "tag": args.tag,
               "auc_invariance_max_dev": auc_max_dev,
               "auc_invariance_pairs": n_pairs, "arms": {}}
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm
               and all(r[f"{m}_{v}"] is not None
                       for m in ("F1", "CD") for v in VARIANTS)]
        e = {"n": len(sub)}
        for m in ("F1", "CD"):
            for v in VARIANTS:
                e[f"{m}_{v}"] = mean(r[f"{m}_{v}"] for r in sub)
        e["F1_pub"] = mean(r["F1_pub"] for r in sub)
        e["CD_pub"] = mean(r["CD_pub"] for r in sub)
        e["mismatch_mean"] = mean(r["mismatch"] for r in sub)
        e["mismatch_meanabs"] = mean(abs(r["mismatch"]) for r in sub)
        devs = [r["f_free"] - r["f_gt"] for r in sub if r["f_free"] is not None]
        e["ffree_dev_mean_pt"] = mean(devs) * 100
        e["ffree_dev_meanabs_pt"] = mean(abs(d) for d in devs) * 100
        e["ffree_dev_maxabs_pt"] = max((abs(d) for d in devs),
                                       default=float("nan")) * 100
        summary["arms"][arm] = e

    paired = {}
    bmap = {r["scene"]: r for r in rows if r["arm"] == "A0_baseline"}
    tmap = {r["scene"]: r for r in rows if r["arm"] == args.arm}
    common_scenes = [s for s in scenes if s in bmap and s in tmap
                     and all(bmap[s][f"{m}_{v}"] is not None
                             and tmap[s][f"{m}_{v}"] is not None
                             for m in ("F1", "CD") for v in VARIANTS)]
    for v in VARIANTS:
        paired[f"dF1_{v}"] = mean(tmap[s][f"F1_{v}"] - bmap[s][f"F1_{v}"]
                                  for s in common_scenes)
        paired[f"dCD_{v}"] = mean(bmap[s][f"CD_{v}"] - tmap[s][f"CD_{v}"]
                                  for s in common_scenes)
    summary["paired"] = paired
    summary["paired_n"] = len(common_scenes)
    summary["orig_vs_pub"] = {
        arm: {"F1_meanabsdev": mean(abs(r["F1_orig"] - r["F1_pub"])
                                    for r in rows if r["arm"] == arm),
              "CD_meanabsdev": mean(abs(r["CD_orig"] - r["CD_pub"])
                                    for r in rows if r["arm"] == arm)}
        for arm in ARMS}

    with open(os.path.join(out, "gauge_cross_summary.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)

    print(f"\n### {ds} ({args.arm}@{args.tag}, n={len(common_scenes)} paired)\n")
    print("| 臂 | 失配 mean% | mean|失配|% | F1 orig | F1 fixgt | F1 fixfree |"
          " CD orig | CD fixgt | CD fixfree |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        e = summary["arms"][arm]
        print(f"| {arm} | {e['mismatch_mean']*100:+.2f} | "
              f"{e['mismatch_meanabs']*100:.2f} | "
              f"{e['F1_orig']:.4f} | {e['F1_fixgt']:.4f} | {e['F1_fixfree']:.4f} | "
              f"{e['CD_orig']:.4f} | {e['CD_fixgt']:.4f} | {e['CD_fixfree']:.4f} |")
    print(f"\npaired dF1: orig {paired['dF1_orig']:+.4f} -> "
          f"fixgt {paired['dF1_fixgt']:+.4f} -> fixfree {paired['dF1_fixfree']:+.4f}")
    print(f"paired dCD(+为好): orig {paired['dCD_orig']:+.4f} -> "
          f"fixgt {paired['dCD_fixgt']:+.4f} -> fixfree {paired['dCD_fixfree']:+.4f}")
    for arm in ARMS:
        e = summary["arms"][arm]
        print(f"f_free vs f_gt [{arm}]: dev mean {e['ffree_dev_mean_pt']:+.2f}pt "
              f"mean|dev| {e['ffree_dev_meanabs_pt']:.2f}pt "
              f"max|dev| {e['ffree_dev_maxabs_pt']:.2f}pt")
    print(f"orig vs published mean|dev|: "
          + "; ".join(f"{a} F1 {summary['orig_vs_pub'][a]['F1_meanabsdev']:.4f} "
                      f"CD {summary['orig_vs_pub'][a]['CD_meanabsdev']:.4f}"
                      for a in ARMS))
    print(f"AUC invariance: {n_pairs} pairs max|Delta ext/intr| = {auc_max_dev:.2e}")
    print("\nREPORT DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--metric-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reuse-baseline-from", default=None)
    ap.add_argument("--stage", required=True,
                    choices=["infer", "measure", "gtfree", "fuse", "report"])
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    import common
    from common import load_manifest
    args.manifest = load_manifest(
        os.path.join(FP, args.dataset, "scene_manifest.json"))
    common.set_dataset(args.manifest.get("dataset", args.dataset))
    scenes = sorted(args.manifest["scenes"])
    if args.scenes:
        scenes = [s for s in scenes if s in set(args.scenes)]

    {"infer": stage_infer, "measure": stage_measure, "gtfree": stage_gtfree,
     "fuse": stage_fuse, "report": stage_report}[args.stage](args, scenes)


if __name__ == "__main__":
    main()
