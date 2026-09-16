#!/usr/bin/env python3
"""Depth metrics (AbsRel/d1.25, shared + median conventions) for arms whose npz
live under gauge_scan/eval32/<arm>@100v (the main eval32 model_results were
cleaned). Mirrors depth_metrics.py frame logic exactly. Writes
<RR>/depth_metrics_C2M_ABS_from_gauge.csv. Pure CPU. New file; no existing
module modified."""

import csv
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
import common  # noqa: E402
from common import get_scene_data, load_gt_depth, load_manifest  # noqa: E402
from depth_metrics import frame_metrics  # noqa: E402

RR = os.path.join(_REPO, "artifacts", "diagnostics", "final_protocol", "scannetpp")
NPZ_ROOT = os.path.join(RR, "gauge_scan", "eval32")
ARMS = ["A0_baseline", "C2M_ABS"]
OUT_CSV = os.path.join(RR, "depth_metrics_C2M_ABS_from_gauge.csv")


def eval_arm(arm, scenes, manifest):
    out = {}
    for scene in scenes:
        npz_path = os.path.join(NPZ_ROOT, f"{arm}@100v", "model_results",
                                "scannetpp", scene, "unposed", "exports",
                                "mini_npz", "results.npz")
        meta_path = os.path.join(NPZ_ROOT, f"{arm}@100v", "model_results",
                                 "scannetpp", scene, "unposed", "exports",
                                 "gt_meta.npz")
        if not os.path.exists(npz_path):
            print(f"[skip] {arm} {scene}: npz missing", flush=True)
            continue
        pred = np.load(npz_path)["depth"]
        meta = np.load(meta_path, allow_pickle=True)
        frame_ids = meta["sampled_indices"].tolist()
        scene_data = get_scene_data(scene)
        preds, gts = [], []
        for k, fi in enumerate(frame_ids):
            gt = load_gt_depth(scene_data.aux.gt_depth_files[fi], pred[k].shape[-2:])
            preds.append(pred[k].astype(np.float64))
            gts.append(gt)
        preds = np.stack(preds)
        gts = np.stack(gts)

        m = np.isfinite(gts) & (gts > 0)
        c = (np.log(np.clip(preds, 1e-6, None))[m] - np.log(gts[m])).mean()
        p1 = preds * np.exp(-c)
        p2 = np.empty_like(preds)
        for k in range(len(preds)):
            mk = np.isfinite(gts[k]) & (gts[k] > 0)
            s = np.median(gts[k][mk]) / max(np.median(preds[k][mk]), 1e-9)
            p2[k] = preds[k] * s

        for tag, pp in (("shared", p1), ("median", p2)):
            rows = [frame_metrics(pp[k], gts[k]) for k in range(len(pp))]
            rows = [r for r in rows if r is not None]
            for key in ("absrel", "d125", "rmselog"):
                out.setdefault(scene, {})[f"{key}_{tag}"] = float(np.mean([r[key] for r in rows]))
    return out


def main():
    manifest = load_manifest(os.path.join(RR, "scene_manifest.json"))
    common.set_dataset(manifest.get("dataset", "scannetpp"))
    scenes = sorted(manifest["scenes"])

    rows = []
    for arm in ARMS:
        res = eval_arm(arm, scenes, manifest)
        for scene, mets in res.items():
            rows.append({"experiment": f"{arm}@100v", "scene": scene, **mets})
        print(f"[depth] {arm}: {len(res)} scenes", flush=True)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
