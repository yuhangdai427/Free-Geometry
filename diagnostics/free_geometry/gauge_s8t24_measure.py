#!/usr/bin/env python3
"""Gauge-mismatch measurement over 20 scannetpp scenes x 2 arms @100v for the
24:8 (s8t24) B5_maskdistill run.

Per scene-arm, from the regenerated npz under gauge_scan/eval32/<arm>@100v:
  s_pose  : Sim3 scale from align_poses_umeyama(gt, pred, ransac=True, rs=42)
            (exactly the eval's _prep_unposed alignment; this scale is what the
            evaluator multiplies onto depth).
  s_depth : shared depth gauge = exp(median(log GT - log pred)) pooled over all
            valid pixels of all eval frames (mirrors absrel_shared).
  mismatch = s_pose / s_depth - 1   (>0: depth inflated vs trajectory gauge
            after eval scaling -> radial expansion, the 1ada7a0617 mechanism).
Joined with per-scene eval deltas (dF1, dCD, dAUC) from the run root's
eval32_metrics_B5_maskdistill.json (paired: the file carries its own
A0_baseline@100v). Writes gauge_scan/gauge_mismatch.csv + gauge_mismatch.json +
gauge_correlation.json. Pure CPU. New file; no existing module modified."""

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
from common import get_scene_data, load_gt_depth, load_manifest  # noqa: E402
from depth_anything_3.utils.pose_align import align_poses_umeyama  # noqa: E402

RR = os.path.join(_REPO, "artifacts", "diagnostics", "final_protocol_s8",
                  "scannetpp_s8t24")
OUT = os.path.join(RR, "gauge_scan")
ARMS = ["A0_baseline", "B5_maskdistill"]
METRIC_FILE = "eval32_metrics_B5_maskdistill.json"


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
    g = np.load(os.path.join(p, "gt_meta.npz"), allow_pickle=True)
    conf = r["conf"] if "conf" in r.files else None
    return (r["depth"], as44(r["extrinsics"]), r["intrinsics"], conf,
            as44(g["extrinsics"]), g["intrinsics"])


def s_pose_of(pred_ext, gt_ext):
    _, _, scale, _ = align_poses_umeyama(
        gt_ext.copy(), pred_ext.copy(), return_aligned=True, ransac=True,
        random_state=42)
    return float(scale)


def s_depth_of(depth, gt_depths):
    logs = []
    for i in range(len(depth)):
        g = gt_depths[i]
        om = np.isfinite(g) & (g > 0) & np.isfinite(depth[i]) & (depth[i] > 0)
        logs.append(np.log(g[om]) - np.log(depth[i][om]))
    return float(np.exp(np.median(np.concatenate(logs))))


def main():
    manifest = load_manifest(os.path.join(RR, "scene_manifest.json"))
    common.set_dataset(manifest.get("dataset", "scannetpp"))
    scenes = sorted(manifest["scenes"])

    # per-scene eval deltas from the run's own published metric file
    m = json.load(open(os.path.join(RR, METRIC_FILE)))
    base = m["A0_baseline@100v"]
    tta = m["B5_maskdistill@100v"]
    deltas = {}
    for s in scenes:
        db = base["scannetpp_recon_unposed"][s]
        dt = tta["scannetpp_recon_unposed"][s]
        pb = base["scannetpp_pose"][s]
        pt = tta["scannetpp_pose"][s]
        deltas[s] = {
            "f1_base": db["fscore"], "f1_tta": dt["fscore"],
            "dF1": dt["fscore"] - db["fscore"],
            "cd_base": db["overall"], "cd_tta": dt["overall"],
            "dCD": db["overall"] - dt["overall"],  # positive = improved
            "auc_base": pb["auc03"], "auc_tta": pt["auc03"],
            "dAUC": pt["auc03"] - pb["auc03"],
        }

    rows = []
    for scene in scenes:
        sd = get_scene_data(scene)
        frames = manifest["scenes"][scene]["eval32_frames"]
        gt_depths = [load_gt_depth(sd.aux.gt_depth_files[i], (378, 504))
                     for i in frames]
        for arm in ARMS:
            try:
                depth, pred_ext, intr, conf, gt_ext, gt_intr = load_npz(arm, scene)
            except FileNotFoundError:
                print(f"[skip] {arm} {scene}: npz missing", flush=True)
                continue
            sp = s_pose_of(pred_ext, gt_ext)
            sd_ = s_depth_of(depth, gt_depths)
            row = {
                "scene": scene, "arm": arm,
                "s_pose": sp, "s_depth": sd_,
                "mismatch": sp / sd_ - 1.0,
            }
            if arm != "A0_baseline":
                row.update(deltas.get(scene, {}))
            rows.append(row)
            print(f"[gauge] {arm} {scene}: s_pose={sp:.4f} s_depth={sd_:.4f} "
                  f"mismatch={row['mismatch']*100:+.2f}%", flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "gauge_mismatch.csv"), "w", newline="") as f:
        keys = ["scene", "arm", "s_pose", "s_depth", "mismatch",
                "f1_base", "f1_tta", "dF1", "cd_base", "cd_tta", "dCD",
                "auc_base", "auc_tta", "dAUC"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(OUT, "gauge_mismatch.json"), "w") as f:
        json.dump(rows, f, indent=1)

    # Spearman rho: B5 mismatch vs dF1 (absolute and delta-vs-baseline)
    from scipy.stats import spearmanr
    corr = {}
    sub = [r for r in rows if r["arm"] == "B5_maskdistill" and "dF1" in r]
    if len(sub) >= 3:
        x = np.array([r["mismatch"] for r in sub])
        y = np.array([r["dF1"] for r in sub])
        rho, p = spearmanr(x, y)
        rhoa, pa = spearmanr(np.abs(x), y)
        bmm = {r["scene"]: r["mismatch"] for r in rows
               if r["arm"] == "A0_baseline"}
        dm = np.array([r["mismatch"] - bmm[r["scene"]] for r in sub
                       if r["scene"] in bmm])
        rhod, pd_ = spearmanr(dm, y[:len(dm)])
        corr["B5_maskdistill"] = {
            "n": len(sub), "rho_mismatch_dF1": float(rho), "p": float(p),
            "rho_absmismatch_dF1": float(rhoa), "p_abs": float(pa),
            "rho_deltamismatch_dF1": float(rhod), "p_delta": float(pd_),
            "scatter": [{"scene": r["scene"], "mismatch": r["mismatch"],
                         "dF1": r["dF1"]} for r in sub]}
        print(f"[corr] B5_maskdistill: n={len(sub)} rho={rho:+.3f} (p={p:.2e}) "
              f"|mismatch| rho={rhoa:+.3f} (p={pa:.2e}) "
              f"delta-mismatch rho={rhod:+.3f} (p={pd_:.2e})", flush=True)
    with open(os.path.join(OUT, "gauge_correlation.json"), "w") as f:
        json.dump(corr, f, indent=1)
    print("MEASURE DONE", flush=True)


if __name__ == "__main__":
    main()
