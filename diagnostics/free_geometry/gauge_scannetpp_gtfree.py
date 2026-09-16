#!/usr/bin/env python3
"""GT-free gauge-repair factor search for scannetpp scenes x arms @100v.

For each scene-arm npz (gauge_scan/eval32/<arm>@100v), searches the depth
premultiply factor f on a grid over [0.90, 1.10] WITHOUT any GT, using
multi-view cross-projection consistency in the PREDICTED frame (gauge-free):
points unprojected from f*depth with predicted c2w poses must agree across
views. Objective A: conf-weighted mean cross-frame nearest-neighbour distance
(argmin). Objective B: conf-weighted inlier ratio at an adaptive threshold
(argmax). Reports f_star vs the GT-fitted factor f_gt = s_depth/s_pose from
gauge_scan/gauge_mismatch.csv. Writes gauge_scan/gauge_gtfree.json +
gauge_gtfree_curves.csv. Pure CPU. New file; no existing module modified."""

import csv
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
from common import load_manifest  # noqa: E402

RR = os.path.join(_REPO, "artifacts", "diagnostics", "final_protocol", "scannetpp")
OUT = os.path.join(RR, "gauge_scan")
ARMS = ["A0_baseline", "C2M_maskrel", "B5_maskdistill", "C2M_MC"]
GRID = np.arange(0.90, 1.10 + 1e-9, 0.005)
PTS_PER_FRAME = 700
K_NN = 12


def as44(ext):
    ext = np.asarray(ext)
    if ext.shape[-2:] == (4, 4):
        return ext
    n = ext.shape[0]
    out = np.tile(np.eye(4, dtype=ext.dtype), (n, 1, 1))
    out[:, :3, :] = ext
    return out


def load_npz(arm, scene):
    p = os.path.join(OUT, "eval32", f"{arm}@100v", "model_results", "scannetpp",
                     scene, "unposed", "exports")
    r = np.load(os.path.join(p, "mini_npz", "results.npz"))
    conf = r["conf"] if "conf" in r.files else None
    return r["depth"], as44(r["extrinsics"]), r["intrinsics"], conf


def sample_frame_points(depth, intr, conf, n_pts, rng):
    """Sample valid pixels; return cam-frame unit rays, depths, conf weights."""
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


def main():
    from scipy.spatial import cKDTree
    manifest = load_manifest(os.path.join(RR, "scene_manifest.json"))
    common.set_dataset(manifest.get("dataset", "scannetpp"))
    scenes = sorted(manifest["scenes"])

    # GT-fitted factors (if measurement already ran)
    f_gt = {}
    gm_path = os.path.join(OUT, "gauge_mismatch.csv")
    if os.path.isfile(gm_path):
        with open(gm_path) as f:
            for row in csv.DictReader(f):
                f_gt[(row["arm"], row["scene"])] = \
                    float(row["s_depth"]) / float(row["s_pose"])

    results = {}
    curve_rows = []
    for arm in ARMS:
        for scene in scenes:
            try:
                depth, ext, intr, conf = load_npz(arm, scene)
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
            thr = 0.02 * med_d  # adaptive inlier threshold (pred gauge)
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
                # first neighbour from a different frame
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
            key = f"{arm}/{scene}"
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
            for gi, f in enumerate(GRID):
                curve_rows.append({"arm": arm, "scene": scene, "f": float(f),
                                   "obj_dist": objs_a[gi], "obj_inlier": objs_b[gi]})

    with open(os.path.join(OUT, "gauge_gtfree.json"), "w") as f:
        json.dump(results, f, indent=1)
    with open(os.path.join(OUT, "gauge_gtfree_curves.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "scene", "f", "obj_dist", "obj_inlier"])
        w.writeheader()
        for r in curve_rows:
            w.writerow(r)
    print("GTFREE DONE", flush=True)


if __name__ == "__main__":
    main()
