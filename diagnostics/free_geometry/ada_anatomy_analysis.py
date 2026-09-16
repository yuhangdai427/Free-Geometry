#!/usr/bin/env python3
"""Per-frame anatomy of scannetpp 1ada7a0617 (baseline vs TTA arms), CPU-only.

Inputs: eval npz exports under ada_anatomy/eval32 (A0_baseline, B5_maskdistill)
and single_scene_eval/eval32 (C2M_MC, C2M_SCL, ...). Analyses:
1. Per-frame pose error after the exact eval alignment (RANSAC Umeyama Sim3,
   random_state=42): rotation geodesic (deg) + camera-center error (m).
2. Pairwise relative R/t errors (the compute_pose AUC feedstock), first-camera
   aligned, to localise which pairs drive AUC.
3. Per-frame depth AbsRel (GT-valid Omega at model resolution), plus per-frame
   optimal log-scale dispersion (cross-view depth-scale drift detector).
4. Per-frame point placement: unproject pred depth with pred intrinsics,
   transform by (a) GT c2w, (b) Sim3-aligned pred c2w; mean NN distance to GT
   mesh sample (no TSDF). Separates depth-path vs pose-path damage per frame.
Writes ada_anatomy/per_frame_stats.json + per_frame_pose.csv + per_frame_depth.csv.
Run from repo root. New file; does not modify any existing module.
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.join(_REPO, "src", "vggt"))

import common  # noqa: E402
from common import get_scene_data, load_gt_depth  # noqa: E402
from depth_anything_3.utils.pose_align import align_poses_umeyama  # noqa: E402

RR = os.path.join(_REPO, "artifacts", "diagnostics", "final_protocol", "scannetpp")
SCENE = "1ada7a0617"
OUT = os.path.join(RR, "ada_anatomy")

EXPS = {  # exp name -> npz root
    "A0_baseline": os.path.join(OUT, "eval32", "A0_baseline@100v"),
    "B5_maskdistill": os.path.join(OUT, "eval32", "B5_maskdistill@100v"),
    "C2M_MC": os.path.join(RR, "single_scene_eval", "eval32", "C2M_MC@100v"),
    "C2M_SCL": os.path.join(RR, "single_scene_eval", "eval32", "C2M_SCL@100v"),
}


def as44(ext):
    ext = np.asarray(ext)
    if ext.shape[-2:] == (4, 4):
        return ext
    n = ext.shape[0]
    out = np.tile(np.eye(4, dtype=ext.dtype), (n, 1, 1))
    out[:, :3, :] = ext
    return out


def load_exp(root):
    p = os.path.join(root, "model_results", "scannetpp", SCENE, "unposed", "exports")
    r = np.load(os.path.join(p, "mini_npz", "results.npz"))
    g = np.load(os.path.join(p, "gt_meta.npz"), allow_pickle=True)
    return (r["depth"], as44(r["extrinsics"]), r["intrinsics"],
            as44(g["extrinsics"]), g["intrinsics"])


def rot_geodesic_deg(Ra, Rb):
    Rd = Ra @ Rb.T
    c = (np.trace(Rd) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def cam_centers(ext_w2c):
    R = ext_w2c[:, :3, :3]
    t = ext_w2c[:, :3, 3]
    return -np.einsum("nij,nj->ni", R.transpose(0, 2, 1), t)


def per_frame_pose(pred_ext, gt_ext):
    """Exact eval alignment (RANSAC Umeyama Sim3, rs=42), then per-frame errors."""
    _, _, scale, pred_al = align_poses_umeyama(
        gt_ext.copy(), pred_ext.copy(), return_aligned=True, ransac=True,
        random_state=42)
    n = len(pred_al)
    rerr = np.array([rot_geodesic_deg(pred_al[i, :3, :3], gt_ext[i, :3, :3])
                     for i in range(n)])
    cerr = np.linalg.norm(cam_centers(pred_al) - cam_centers(gt_ext), axis=1)
    return rerr, cerr, float(scale)


def pairwise_rel_errors(pred_ext, gt_ext):
    """compute_pose feedstock: first-camera alignment, all-pairs relative
    rotation (deg) and translation-direction (deg) errors. Returns [N,N] each."""
    sys.path.insert(0, os.path.join(_REPO, "src", "depth_anything_3"))
    from depth_anything_3.bench.utils import (
        align_to_first_camera, se3_to_relative_pose_error)
    import torch
    p = torch.from_numpy(pred_ext).float()
    g = torch.from_numpy(gt_ext).float()
    p = align_to_first_camera(p)
    g = align_to_first_camera(g)
    rr, tt = se3_to_relative_pose_error(p, g, len(p))
    rr = rr.cpu().numpy().reshape(-1)
    tt = tt.cpu().numpy().reshape(-1)
    i1, i2 = np.triu_indices(len(p), 1)
    n = len(p)
    RR_ = np.zeros((n, n))
    TT_ = np.zeros((n, n))
    RR_[i1, i2] = rr
    TT_[i1, i2] = tt
    return RR_, TT_


def per_frame_depth(depth, gt_depths):
    """AbsRel per frame with one shared scale (median ratio over all frames,
    mirroring absrel_shared); per-frame optimal log-scale for drift detection."""
    n = len(depth)
    logs = []
    omegas = []
    for i in range(n):
        g = gt_depths[i]
        om = np.isfinite(g) & (g > 0) & np.isfinite(depth[i]) & (depth[i] > 0)
        omegas.append(om)
        logs.append(np.log(g[om]) - np.log(depth[i][om]))
    all_log = np.concatenate(logs)
    shared_c = np.median(all_log)  # one shared multiplicative scale correction
    absrel = np.full(n, np.nan)
    frame_scale = np.full(n, np.nan)
    for i in range(n):
        r = logs[i] - shared_c
        absrel[i] = np.mean(np.abs(r)) if len(r) else np.nan
        frame_scale[i] = np.median(logs[i]) if len(logs[i]) else np.nan
    return absrel, frame_scale


def unproject(depth, intr, max_pts=3000, rng=None):
    h, w = depth.shape
    rng = rng or np.random.default_rng(0)
    ys, xs = np.nonzero(np.isfinite(depth) & (depth > 0))
    if len(ys) == 0:
        return np.zeros((0, 3))
    sel = rng.choice(len(ys), size=min(max_pts, len(ys)), replace=False)
    ys, xs = ys[sel], xs[sel]
    z = depth[ys, xs]
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z
    return np.stack([x, y, z], 1)


def main():
    common.set_dataset("scannetpp")
    sd = get_scene_data(SCENE)
    frames = list(range(100))
    gt_depths = [load_gt_depth(sd.aux.gt_depth_files[i], (378, 504)) for i in frames]

    data = {}
    for name, root in EXPS.items():
        if not os.path.exists(os.path.join(
                root, "model_results", "scannetpp", SCENE, "unposed", "exports",
                "mini_npz", "results.npz")):
            print(f"skip {name}: npz missing")
            continue
        data[name] = load_exp(root)
        print(f"loaded {name}", flush=True)

    stats = {}
    gt_ext = data["A0_baseline"][3]

    # 1+2: pose
    import csv
    pose_rows = []
    pair_err = {}
    for name, (depth, pred_ext, intr, gext, gintr) in data.items():
        rerr, cerr, scale = per_frame_pose(pred_ext, gext)
        rr, tt = pairwise_rel_errors(pred_ext, gext)
        pair_err[name] = (rr, tt)
        stats[name] = {
            "umeyama_scale": scale,
            "rot_deg": {"mean": float(rerr.mean()), "median": float(np.median(rerr)),
                        "p90": float(np.percentile(rerr, 90)), "max": float(rerr.max()),
                        "n>3deg": int((rerr > 3).sum()), "n>5deg": int((rerr > 5).sum())},
            "center_m": {"mean": float(cerr.mean()), "median": float(np.median(cerr)),
                         "p90": float(np.percentile(cerr, 90)), "max": float(cerr.max()),
                         "n>0.3m": int((cerr > 0.3).sum())},
        }
        pose_rows.append((name, rerr, cerr))
        print(f"[pose] {name}: rot mean {rerr.mean():.3f} p90 {np.percentile(rerr,90):.3f} "
              f"max {rerr.max():.2f} | center mean {cerr.mean():.4f} max {cerr.max():.3f} "
              f"| scale {scale:.4f}", flush=True)

    # which pairs drive AUC: B5 minus baseline max-error per pair
    if "B5_maskdistill" in pair_err and "A0_baseline" in pair_err:
        rb, tb = pair_err["A0_baseline"]
        rt, tt = pair_err["B5_maskdistill"]
        mb = np.maximum(rb, tb)
        mt = np.maximum(rt, tt)
        diff = mt - mb
        iu = np.triu_indices(len(mb), 1)
        stats["pair_auc_feedstock"] = {
            "pairs_improved_frac": float(np.mean(diff[iu] < 0)),
            "mean_diff_deg": float(np.mean(diff[iu])),
            "worst_regressions": sorted(
                [(int(i), int(j), float(diff[i, j]), float(mb[i, j]), float(mt[i, j]))
                 for i, j in zip(*iu)], key=lambda x: -x[2])[:15],
        }

    # 3: depth
    depth_rows = []
    for name, (depth, pred_ext, intr, gext, gintr) in data.items():
        absrel, fscale = per_frame_depth(depth, gt_depths)
        drift = float(np.nanstd(fscale))
        stats[name]["depth"] = {
            "absrel_mean": float(np.nanmean(absrel)),
            "frame_logscale_std": drift,
            "frame_logscale_range": float(np.nanmax(fscale) - np.nanmin(fscale)),
        }
        depth_rows.append((name, absrel, fscale))
        print(f"[depth] {name}: AbsRel {np.nanmean(absrel):.4f} "
              f"logscale std {drift:.4f} range {np.nanmax(fscale)-np.nanmin(fscale):.4f}",
              flush=True)

    # 4: point placement vs GT mesh (no TSDF)
    import open3d as o3d
    from scipy.spatial import cKDTree
    o3d.utility.random.seed(42)
    gt_mesh = o3d.io.read_triangle_mesh(sd.aux.gt_mesh_path)
    gt_pcd = gt_mesh.sample_points_uniformly(1_000_000)
    gt_pts = np.asarray(gt_pcd.points)
    tree = cKDTree(gt_pts)
    place_rows = []
    rng = np.random.default_rng(7)
    for name, (depth, pred_ext, intr, gext, gintr) in data.items():
        _, _, scale, pred_al = align_poses_umeyama(
            gext.copy(), pred_ext.copy(), return_aligned=True, ransac=True,
            random_state=42)
        pred_c2w_al = np.linalg.inv(pred_al)         # Sim3-aligned pred c2w
        gt_c2w = np.linalg.inv(gext)
        d_gtpose = np.full(100, np.nan)
        d_predpose = np.full(100, np.nan)
        for i in range(100):
            pts = unproject(depth[i], intr[i], rng=rng)
            if len(pts) == 0:
                continue
            for arr, c2w in ((d_gtpose, gt_c2w[i]), (d_predpose, pred_c2w_al[i])):
                pw = (c2w[:3, :3] @ pts.T).T + c2w[:3, 3]
                dd, _ = tree.query(pw, workers=4)
                arr[i] = float(np.mean(dd))
        stats[name]["placement"] = {
            "gtpose_mean_nn_m": float(np.nanmean(d_gtpose)),
            "predpose_mean_nn_m": float(np.nanmean(d_predpose)),
            "gtpose_p90_nn_m": float(np.nanpercentile(d_gtpose, 90)),
            "predpose_p90_nn_m": float(np.nanpercentile(d_predpose, 90)),
        }
        place_rows.append((name, d_gtpose, d_predpose))
        print(f"[place] {name}: GT-pose NN {np.nanmean(d_gtpose):.4f} m | "
              f"pred-pose NN {np.nanmean(d_predpose):.4f} m "
              f"(p90 {np.nanpercentile(d_predpose,90):.4f})", flush=True)

    with open(os.path.join(OUT, "per_frame_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)

    # CSVs: per-frame rows for baseline/B5 (+ others)
    names = list(data.keys())
    with open(os.path.join(OUT, "per_frame_pose.csv"), "w", newline="") as f:
        w = csv.writer(f)
        hdr = ["frame"]
        for n in names:
            hdr += [f"{n}_rot_deg", f"{n}_center_m"]
        w.writerow(hdr)
        pr = {n: (r, c) for n, r, c in pose_rows}
        for i in range(100):
            row = [i]
            for n in names:
                r, c = pr[n]
                row += [f"{r[i]:.4f}", f"{c[i]:.4f}"]
            w.writerow(row)
    with open(os.path.join(OUT, "per_frame_depth.csv"), "w", newline="") as f:
        w = csv.writer(f)
        hdr = ["frame"]
        for n in names:
            hdr += [f"{n}_absrel", f"{n}_logscale"]
        w.writerow(hdr)
        dr = {n: (a, s) for n, a, s in depth_rows}
        for i in range(100):
            row = [i]
            for n in names:
                a, s = dr[n]
                row += [f"{a[i]:.5f}", f"{s[i]:.5f}"]
            w.writerow(row)
    if place_rows:
        with open(os.path.join(OUT, "per_frame_placement.csv"), "w", newline="") as f:
            w = csv.writer(f)
            hdr = ["frame"]
            for n, _, _ in place_rows:
                hdr += [f"{n}_nn_gtpose", f"{n}_nn_predpose"]
            w.writerow(hdr)
            pr = {n: (a, b) for n, a, b in place_rows}
            for i in range(100):
                row = [i]
                for n, _, _ in place_rows:
                    a, b = pr[n]
                    row += [f"{a[i]:.5f}", f"{b[i]:.5f}"]
                w.writerow(row)
    print("ANALYSIS DONE", flush=True)


if __name__ == "__main__":
    main()
