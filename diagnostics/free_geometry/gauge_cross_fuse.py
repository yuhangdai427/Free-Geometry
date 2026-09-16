#!/usr/bin/env python3
"""Cross-dataset counterfactual gauge-repair fusion (eval-side recoupling).

Variants per scene-arm (npz under gauge_cross/eval32/<arm>@<tag>):
  orig    : depth as-is (f=1.0)
  fixgt   : depth premultiplied by f_gt = s_depth/s_pose (GT-fitted oracle
            recoupling factor from gauge_cross/gauge_mismatch.csv)
  fixfree : depth premultiplied by the GT-free searched factor
            (gauge_cross/gauge_gtfree.json, objective = cross-view consistency)
The evaluator still applies its own Sim3 trajectory scale (s_pose) onto depth,
so premultiplying by f leaves the fused cloud at gauge f*s_pose; fixgt lands
exactly on the GT depth gauge (f_gt*s_pose = s_depth).

MUST be launched with CUDA_VISIBLE_DEVICES="" (also set in-process before any
CUDA-capable import) so fusion workers never touch the GPU - same safety
property as run_eval.py. New file; does not modify any existing module.
Run from repo root:

  CUDA_VISIBLE_DEVICES="" python3 diagnostics/free_geometry/gauge_cross_fuse.py \
      --dataset 7scenes --variants orig fixgt fixfree
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

FP = os.path.join(_REPO, "artifacts", "diagnostics", "final_protocol")
CHAMPION = {
    "7scenes": ("C2M_MC", "100v", "eval32_metrics_C2M_MC.json"),
    "hiroom": ("C2M_maskrel", "allv", "eval32_metrics_C2M.json"),
    "eth3d": ("C2M_SCL", "allv", "eval32_metrics_C2M_SCL.json"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(CHAMPION))
    ap.add_argument("--scenes", nargs="+", default=None)
    ap.add_argument("--variants", nargs="*", default=["orig", "fixgt", "fixfree"])
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    ds = args.dataset
    arm_tta, tag, _ = CHAMPION[ds]
    RR = os.path.join(FP, ds)
    OUT = os.path.join(RR, "gauge_cross")
    ARMS = ["A0_baseline", arm_tta]

    sys.path.insert(0, _HERE)
    from common import load_manifest
    manifest = load_manifest(os.path.join(RR, "scene_manifest.json"))
    scenes = sorted(manifest["scenes"])
    if args.scenes:
        scenes = [s for s in scenes if s in set(args.scenes)]

    f_gt, f_free = {}, {}
    gm = os.path.join(OUT, "gauge_mismatch.csv")
    if os.path.isfile(gm):
        with open(gm) as f:
            for row in csv.DictReader(f):
                f_gt[(row["arm"], row["scene"])] = \
                    float(row["s_depth"]) / float(row["s_pose"])
    gf = os.path.join(OUT, "gauge_gtfree.json")
    if os.path.isfile(gf):
        for v in json.load(open(gf)).values():
            f_free[(v["arm"], v["scene"])] = v["f_free_dist"]

    def make_variant_npz(arm, scene, factor, variant):
        src = os.path.join(OUT, "eval32", f"{arm}@{tag}", "model_results", ds,
                           scene, "unposed", "exports")
        dst = os.path.join(OUT, "eval32", f"{arm}_{variant}@{tag}",
                           "model_results", ds, scene, "unposed", "exports")
        r = np.load(os.path.join(src, "mini_npz", "results.npz"))
        os.makedirs(os.path.join(dst, "mini_npz"), exist_ok=True)
        kw = {k: r[k] for k in r.files}
        kw["depth"] = np.round(kw["depth"] * factor, 8)
        np.savez_compressed(os.path.join(dst, "mini_npz", "results.npz"), **kw)
        shutil.copyfile(os.path.join(src, "gt_meta.npz"),
                        os.path.join(dst, "gt_meta.npz"))

    metrics_path = os.path.join(OUT, "gauge_fix_metrics.json")
    all_metrics = {}
    if os.path.isfile(metrics_path):
        all_metrics = json.load(open(metrics_path))

    import open3d as o3d
    from vggt.bench.evaluator import VGGTEvaluator

    for arm in ARMS:
        for variant in args.variants:
            todo = []
            for scene in scenes:
                mkey = f"{arm}_{variant}@{tag}::{ds}_recon_unposed"
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
            work_dir = os.path.join(OUT, "eval32", f"{arm}_{variant}@{tag}")
            o3d.utility.random.seed(42)
            ev = VGGTEvaluator(work_dir=work_dir, datas=[ds],
                               modes=["recon_unposed"], scenes=todo,
                               max_frames=0, num_fusion_workers=args.workers)
            m = ev.eval()
            for k, v in m.items():
                mkey = f"{arm}_{variant}@{tag}::{k}"
                all_metrics.setdefault(mkey, {}).update(v)
                for s in todo:
                    r = v[s]
                    print(f"RESULT {arm} {variant} {s}: F1={r['fscore']:.4f} "
                          f"CD={r['overall']:.4f} acc={r['acc']:.4f} "
                          f"comp={r['comp']:.4f}", flush=True)
            with open(metrics_path, "w") as fp:
                json.dump(all_metrics, fp, indent=1, default=str)
    print(f"FUSE {ds} DONE", flush=True)


if __name__ == "__main__":
    main()
