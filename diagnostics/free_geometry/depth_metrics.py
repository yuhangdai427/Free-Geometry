#!/usr/bin/env python3
"""Depth metrics (AbsRel, delta<1.25, RMSE-log) for every experiment in a
merged eval tree - computed from cached results.npz + GT depth, no GPU.

Two scale conventions per frame set:
  - scene-shared log-scale (one scalar per scene, as in E_depth)
  - per-frame median scaling (standard monocular convention)
GT: uint16 mm PNG, resize-only to pred grid, valid = 0<d<=5m.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import common  # noqa: E402
from common import get_scene_data, load_gt_depth, load_manifest  # noqa: E402


def frame_metrics(pred, gt):
    """pred, gt: [H,W] floats (gt NaN=invalid). Returns dict of metrics."""
    m = np.isfinite(gt) & (gt > 0) & np.isfinite(pred) & (pred > 0)
    if m.sum() < 100:
        return None
    p, g = pred[m], gt[m]
    absrel = np.mean(np.abs(p - g) / g)
    ratio = np.maximum(p / g, g / p)
    d125 = np.mean(ratio < 1.25)
    rmselog = np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))
    return {"absrel": absrel, "d125": d125, "rmselog": rmselog, "n": int(m.sum())}


def eval_experiment(npz_root, manifest_scenes, manifest):
    """Returns {scene: {metric: value}} for both scale conventions."""
    ds_name = manifest.get("dataset", "scannetpp")
    out = {}
    for scene in manifest_scenes:
        ev = manifest["scenes"][scene]["eval32_frames"]
        npz_path = os.path.join(npz_root, "model_results", ds_name, scene,
                                "unposed", "exports", "mini_npz", "results.npz")
        meta_path = os.path.join(npz_root, "model_results", ds_name, scene,
                                 "unposed", "exports", "gt_meta.npz")
        if not os.path.exists(npz_path):
            continue
        pred = np.load(npz_path)["depth"]           # [S,H,W]
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

        # convention 1: scene-shared log scale
        m = np.isfinite(gts) & (gts > 0)
        c = (np.log(np.clip(preds, 1e-6, None))[m] - np.log(gts[m])).mean()
        p1 = preds * np.exp(-c)
        # convention 2: per-frame median scale
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", default="artifacts/diagnostics/bakeoff_merged")
    ap.add_argument("--manifest", default="artifacts/diagnostics/bakeoff_v2_transductive/scene_manifest.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    common.set_dataset(manifest.get("dataset", "scannetpp"))
    scenes = sorted(manifest["scenes"])
    eval_root = os.path.join(args.run_root, "eval32")
    exps = sorted(os.listdir(eval_root))

    rows = []
    for exp in exps:
        npz_root = os.path.join(eval_root, exp)
        res = eval_experiment(npz_root, scenes, manifest)
        for scene, mets in res.items():
            rows.append({"experiment": exp, "scene": scene, **mets})
        print(f"[depth_metrics] {exp}: {len(res)} scenes", flush=True)

    out = args.out or os.path.join(args.run_root, "depth_metrics.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
