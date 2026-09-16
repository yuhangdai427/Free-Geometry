#!/usr/bin/env python3
"""Cross-dataset gauge-mismatch measurement (7scenes / hiroom / eth3d).

Per scene-arm, from the regenerated npz under gauge_cross/eval32/<arm>@<tag>:
  s_pose  : Sim3 scale from align_poses_umeyama(gt, pred, ransac=True, rs=42)
            (exactly the eval's _prep_unposed alignment; this scale is what the
            evaluator multiplies onto depth).
  s_depth : shared depth gauge = exp(median(log GT - log pred)) pooled over all
            valid pixels of all eval frames (mirrors absrel_shared).
  mismatch = s_pose / s_depth - 1   (>0: depth inflated vs trajectory gauge
            after eval scaling -> radial expansion, the 1ada7a0617 mechanism).
Joined with per-scene eval deltas (dF1, dCD, dAUC) from the dataset's
published champion metric file (arm paired with that file's own A0_baseline).
Writes gauge_cross/gauge_mismatch.csv + .json + gauge_correlation.json.
Pure CPU. New file; no existing module modified. Run from repo root:

  python3 diagnostics/free_geometry/gauge_cross_measure.py --dataset 7scenes
"""

import argparse
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

FP = os.path.join(_REPO, "artifacts", "diagnostics", "final_protocol")
CHAMPION = {
    "7scenes": ("C2M_MC", "100v", "eval32_metrics_C2M_MC.json"),
    "hiroom": ("C2M_maskrel", "allv", "eval32_metrics_C2M.json"),
    "eth3d": ("C2M_SCL", "allv", "eval32_metrics_C2M_SCL.json"),
}


def as44(ext):
    ext = np.asarray(ext)
    if ext.shape[-2:] == (4, 4):
        return ext
    n = ext.shape[0]
    out = np.tile(np.eye(4, dtype=ext.dtype), (n, 1, 1))
    out[:, :3, :] = ext
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(CHAMPION))
    args = ap.parse_args()
    ds = args.dataset
    arm_tta, tag, metric_file = CHAMPION[ds]
    RR = os.path.join(FP, ds)
    OUT = os.path.join(RR, "gauge_cross")
    ARMS = ["A0_baseline", arm_tta]

    manifest = load_manifest(os.path.join(RR, "scene_manifest.json"))
    common.set_dataset(manifest.get("dataset", ds))
    scenes = sorted(manifest["scenes"])

    def load_npz(arm, scene):
        p = os.path.join(OUT, "eval32", f"{arm}@{tag}", "model_results", ds,
                         scene, "unposed", "exports")
        r = np.load(os.path.join(p, "mini_npz", "results.npz"))
        g = np.load(os.path.join(p, "gt_meta.npz"), allow_pickle=True)
        conf = r["conf"] if "conf" in r.files else None
        return (r["depth"], as44(r["extrinsics"]), r["intrinsics"], conf,
                as44(g["extrinsics"]), g["intrinsics"])

    # per-scene eval deltas from the published champion metric file
    deltas = {}
    m = json.load(open(os.path.join(RR, metric_file)))
    base = m[f"A0_baseline@{tag}"]
    tta = m[f"{arm_tta}@{tag}"]
    for s in scenes:
        db = base[f"{ds}_recon_unposed"][s]
        dt = tta[f"{ds}_recon_unposed"][s]
        pb = base[f"{ds}_pose"][s]
        pt = tta[f"{ds}_pose"][s]
        deltas.setdefault(s, {})[arm_tta] = {
            "f1_base_pub": db["fscore"], "f1_tta_pub": dt["fscore"],
            "dF1": dt["fscore"] - db["fscore"],
            "cd_base_pub": db["overall"], "cd_tta_pub": dt["overall"],
            "dCD": db["overall"] - dt["overall"],  # positive = improved
            "auc_base_pub": pb["auc03"], "auc_tta_pub": pt["auc03"],
            "dAUC": pt["auc03"] - pb["auc03"],
        }

    rows = []
    for scene in scenes:
        sd = get_scene_data(scene)
        frames = manifest["scenes"][scene]["eval32_frames"]
        gt_depths = None
        for arm in ARMS:
            try:
                depth, pred_ext, intr, conf, gt_ext, gt_intr = load_npz(arm, scene)
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

    os.makedirs(OUT, exist_ok=True)
    keys = ["scene", "arm", "s_pose", "s_depth", "mismatch",
            "f1_base_pub", "f1_tta_pub", "dF1", "cd_base_pub", "cd_tta_pub",
            "dCD", "auc_base_pub", "auc_tta_pub", "dAUC"]
    with open(os.path.join(OUT, "gauge_mismatch.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(OUT, "gauge_mismatch.json"), "w") as f:
        json.dump(rows, f, indent=1)

    from scipy.stats import spearmanr
    corr = {}
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm and "dF1" in r]
        if len(sub) < 3:
            continue
        x = np.array([r["mismatch"] for r in sub])
        y = np.array([r["dF1"] for r in sub])
        rho, p = spearmanr(x, y)
        rhoa, pa = spearmanr(np.abs(x), y)
        # delta-mismatch vs dF1 (controls scene-intrinsic mismatch)
        bm = {r["scene"]: r["mismatch"] for r in rows if r["arm"] == "A0_baseline"}
        dx = np.array([r["mismatch"] - bm[r["scene"]] for r in sub])
        rhod, pd_ = spearmanr(dx, y)
        corr[arm] = {"n": len(sub), "rho_mismatch_dF1": float(rho),
                     "p": float(p), "rho_absmismatch_dF1": float(rhoa),
                     "p_abs": float(pa),
                     "rho_deltamismatch_dF1": float(rhod), "p_delta": float(pd_),
                     "scatter": [{"scene": r["scene"], "mismatch": r["mismatch"],
                                  "dF1": r["dF1"]} for r in sub]}
        print(f"[corr] {arm}: n={len(sub)} rho={rho:+.3f} (p={p:.2e}) "
              f"|mismatch| rho={rhoa:+.3f} (p={pa:.2e}) "
              f"delta-mismatch rho={rhod:+.3f} (p={pd_:.2e})", flush=True)
    with open(os.path.join(OUT, "gauge_correlation.json"), "w") as f:
        json.dump(corr, f, indent=1)
    print(f"MEASURE {ds} DONE", flush=True)


if __name__ == "__main__":
    main()
