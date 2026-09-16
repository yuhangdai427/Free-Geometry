#!/usr/bin/env python3
"""Counterfactual gauge-repair fusion for the C2M_SCL arm on scannetpp @100v.
Copy of gauge_scannetpp_fuse.py restricted to arms=["C2M_SCL"], with factors
read from the SCL-specific files:
  fixgt   : f_gt = s_depth/s_pose from gauge_scan/gauge_mismatch_C2M_SCL.csv
  fixfree : f_free_dist from gauge_scan/gauge_gtfree_C2M_SCL.json
Variant npz go to gauge_scan/eval32/C2M_SCL_{fixgt,fixfree}@100v; metrics to
gauge_scan/gauge_fix_metrics_C2M_SCL.json.

MUST be launched with CUDA_VISIBLE_DEVICES="" (also set in-process). New file;
does not modify any existing module. Run from repo root:
  CUDA_VISIBLE_DEVICES="" python3 diagnostics/free_geometry/gauge_scannetpp_fuse_scl.py \
      --scenes <20 scenes> --variants fixgt fixfree
"""

import argparse
import csv
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import shutil
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.join(_REPO, "src", "vggt"))

RR = os.path.join(_REPO, "artifacts", "diagnostics", "final_protocol", "scannetpp")
OUT = os.path.join(RR, "gauge_scan")
ARMS = ["C2M_SCL"]


def load_factors():
    f_gt, f_free = {}, {}
    gm = os.path.join(OUT, "gauge_mismatch_C2M_SCL.csv")
    if os.path.isfile(gm):
        with open(gm) as f:
            for row in csv.DictReader(f):
                f_gt[(row["arm"], row["scene"])] = \
                    float(row["s_depth"]) / float(row["s_pose"])
    gf = os.path.join(OUT, "gauge_gtfree_C2M_SCL.json")
    if os.path.isfile(gf):
        for v in json.load(open(gf)).values():
            f_free[(v["arm"], v["scene"])] = v["f_free_dist"]
    return f_gt, f_free


def make_variant_npz(arm, scene, factor, variant):
    src = os.path.join(OUT, "eval32", f"{arm}@100v", "model_results",
                       "scannetpp", scene, "unposed", "exports")
    dst = os.path.join(OUT, "eval32", f"{arm}_{variant}@100v", "model_results",
                       "scannetpp", scene, "unposed", "exports")
    r = np.load(os.path.join(src, "mini_npz", "results.npz"))
    os.makedirs(os.path.join(dst, "mini_npz"), exist_ok=True)
    kw = {k: r[k] for k in r.files}
    kw["depth"] = np.round(kw["depth"] * factor, 8)
    np.savez_compressed(os.path.join(dst, "mini_npz", "results.npz"), **kw)
    shutil.copyfile(os.path.join(src, "gt_meta.npz"),
                    os.path.join(dst, "gt_meta.npz"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--variants", nargs="*", default=["fixgt", "fixfree"])
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    f_gt, f_free = load_factors()

    import open3d as o3d
    from vggt.bench.evaluator import VGGTEvaluator

    all_metrics = {}
    for arm in args.arms:
        for variant in args.variants:
            todo = []
            for scene in args.scenes:
                if variant == "orig":
                    f = 1.0
                elif variant == "fixgt":
                    f = f_gt.get((arm, scene))
                else:
                    f = f_free.get((arm, scene))
                if f is None:
                    print(f"[skip] {arm} {scene} {variant}: no factor", flush=True)
                    continue
                make_variant_npz(arm, scene, f, variant)
                todo.append(scene)
            if not todo:
                continue
            work_dir = os.path.join(OUT, "eval32", f"{arm}_{variant}@100v")
            o3d.utility.random.seed(42)
            ev = VGGTEvaluator(work_dir=work_dir, datas=["scannetpp"],
                               modes=["recon_unposed"], scenes=todo,
                               max_frames=0, num_fusion_workers=args.workers)
            m = ev.eval()
            for k, v in m.items():
                all_metrics[f"{arm}_{variant}@100v::{k}"] = v
                for s in todo:
                    r = v[s]
                    print(f"RESULT {arm} {variant} {s}: F1={r['fscore']:.4f} "
                          f"CD={r['overall']:.4f} acc={r['acc']:.4f} "
                          f"comp={r['comp']:.4f}", flush=True)
            with open(os.path.join(OUT, "gauge_fix_metrics_C2M_SCL.json"), "w") as fp:
                json.dump(all_metrics, fp, indent=1, default=str)
    print("FUSE DONE", flush=True)


if __name__ == "__main__":
    main()
