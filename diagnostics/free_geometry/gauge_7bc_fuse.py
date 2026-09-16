#!/usr/bin/env python3
"""orig/fixgt/fixfree counterfactual fusion for the 7bc286c1b6 random_forced run
(arms A0_baseline, C2M_maskrel @100v, npz under gauge_7bc/eval32).

  orig    : factor 1.0 (npz re-fused through this same CPU evaluator pass, so
            all three variants are strictly paired)
  fixgt   : depth premultiplied by f_gt = s_depth/s_pose (gauge_mismatch.csv)
  fixfree : depth premultiplied by f_free_dist (gauge_gtfree.json)
Also verifies the variant npz extrinsics/intrinsics are bit-identical to the
orig npz (AUC structural invariance).

MUST be launched with CUDA_VISIBLE_DEVICES="" (also set in-process). New file;
does not modify any existing module. Run from repo root:
  CUDA_VISIBLE_DEVICES="" python3 diagnostics/free_geometry/gauge_7bc_fuse.py
"""

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

RR = os.path.join(_REPO, "artifacts", "diagnostics", "final_protocol",
                  "scannetpp_7bc_random")
OUT = os.path.join(RR, "gauge_7bc")
ARMS = ["A0_baseline", "C2M_maskrel"]
VARIANTS = ["orig", "fixgt", "fixfree"]


def load_factors():
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
    return f_gt, f_free


def make_variant_npz(arm, scene, factor, variant):
    src = os.path.join(OUT, "eval32", f"{arm}@100v", "model_results",
                       "scannetpp", scene, "unposed", "exports")
    dst = os.path.join(OUT, "eval32", f"{arm}_{variant}@100v", "model_results",
                       "scannetpp", scene, "unposed", "exports")
    r = np.load(os.path.join(src, "mini_npz", "results.npz"))
    os.makedirs(os.path.join(dst, "mini_npz"), exist_ok=True)
    kw = {k: r[k] for k in r.files}
    if variant != "orig":
        kw["depth"] = np.round(kw["depth"] * factor, 8)
    np.savez_compressed(os.path.join(dst, "mini_npz", "results.npz"), **kw)
    shutil.copyfile(os.path.join(src, "gt_meta.npz"),
                    os.path.join(dst, "gt_meta.npz"))
    # AUC invariance: extrinsics/intrinsics bit-identical to source npz
    v = np.load(os.path.join(dst, "mini_npz", "results.npz"))
    de = np.abs(v["extrinsics"] - r["extrinsics"]).max()
    di = np.abs(v["intrinsics"] - r["intrinsics"]).max()
    assert de == 0 and di == 0, (arm, scene, variant, de, di)


def main():
    f_gt, f_free = load_factors()

    import open3d as o3d
    from vggt.bench.evaluator import VGGTEvaluator

    manifest_scenes = ["7bc286c1b6"]
    all_metrics = {}
    for arm in ARMS:
        for variant in VARIANTS:
            todo = []
            for scene in manifest_scenes:
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
                               max_frames=0, num_fusion_workers=4)
            m = ev.eval()
            for k, v in m.items():
                all_metrics[f"{arm}_{variant}@100v::{k}"] = v
                for s in todo:
                    r = v[s]
                    print(f"RESULT {arm} {variant} {s}: F1={r['fscore']:.4f} "
                          f"CD={r['overall']:.4f} acc={r['acc']:.4f} "
                          f"comp={r['comp']:.4f}", flush=True)
            with open(os.path.join(OUT, "gauge_fix_metrics.json"), "w") as fp:
                json.dump(all_metrics, fp, indent=1, default=str)
    print("FUSE DONE (npz extr/intr bit-identical checks passed)", flush=True)


if __name__ == "__main__":
    main()
