#!/usr/bin/env python3
"""Counterfactual gauge-repair fusion for the 24:8 (s8t24) scannetpp run
(eval-side recoupling): 20 scenes x 2 arms (A0_baseline, B5_maskdistill) @100v.

Variants per scene-arm (npz under gauge_scan/eval32/<arm>@100v):
  orig    : evaluated in place on the original npz (bitwise identical to a
            depth x1.0 copy: depth is already round(.., 8) and x*1.0 == x)
  fixgt   : depth premultiplied by f_gt = s_depth/s_pose (GT-fitted oracle
            recoupling factor from gauge_scan/gauge_mismatch.csv)
  fixfree : depth premultiplied by the GT-free searched factor
            (gauge_scan/gauge_gtfree.json, objective = cross-view consistency)
The evaluator still applies its own Sim3 trajectory scale (s_pose) onto depth,
so premultiplying by f leaves the fused cloud at gauge f*s_pose; fixgt lands
exactly on the GT depth gauge (f_gt*s_pose = s_depth).

Also runs (CPU, no fusion):
  - pose-mode eval on the two ORIG arm dirs -> AUC@3 reproduction sanity vs
    the run's published eval32_metrics_B5_maskdistill.json;
  - bitwise extr/intr comparison of every variant npz vs its orig npz
    (AUC-invariance certificate: pose eval reads only extrinsics).

MUST be launched with CUDA_VISIBLE_DEVICES="" (also set in-process before any
CUDA-capable import) so fusion workers never touch the GPU - same safety
property as run_eval.py. New file; does not modify any existing module.
Run from repo root:

  CUDA_VISIBLE_DEVICES="" python3 diagnostics/free_geometry/gauge_s8t24_fuse.py
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

RR = os.path.join(_REPO, "artifacts", "diagnostics", "final_protocol_s8",
                  "scannetpp_s8t24")
OUT = os.path.join(RR, "gauge_scan")
ARMS = ["A0_baseline", "B5_maskdistill"]


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


def mini_npz_dir(work_dir, scene):
    return os.path.join(work_dir, "model_results", "scannetpp", scene,
                        "unposed", "exports")


def make_variant_npz(arm, scene, factor, variant):
    src = mini_npz_dir(os.path.join(OUT, "eval32", f"{arm}@100v"), scene)
    dst = mini_npz_dir(os.path.join(OUT, "eval32", f"{arm}_{variant}@100v"),
                       scene)
    r = np.load(os.path.join(src, "mini_npz", "results.npz"))
    os.makedirs(os.path.join(dst, "mini_npz"), exist_ok=True)
    kw = {k: r[k] for k in r.files}
    kw["depth"] = np.round(kw["depth"] * factor, 8)
    np.savez_compressed(os.path.join(dst, "mini_npz", "results.npz"), **kw)
    shutil.copyfile(os.path.join(src, "gt_meta.npz"),
                    os.path.join(dst, "gt_meta.npz"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--variants", nargs="*",
                    default=["orig", "fixgt", "fixfree"])
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    sys.path.insert(0, _HERE)
    import common
    from common import load_manifest
    manifest = load_manifest(os.path.join(RR, "scene_manifest.json"))
    common.set_dataset(manifest.get("dataset", "scannetpp"))
    scenes = args.scenes or sorted(manifest["scenes"])

    f_gt, f_free = load_factors()

    import open3d as o3d
    from vggt.bench.evaluator import VGGTEvaluator

    all_metrics = {}
    for arm in args.arms:
        for variant in args.variants:
            todo = []
            for scene in scenes:
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
                if variant != "orig":
                    make_variant_npz(arm, scene, f, variant)
                todo.append(scene)
            if not todo:
                continue
            work_dir = os.path.join(OUT, "eval32", f"{arm}@100v") \
                if variant == "orig" else \
                os.path.join(OUT, "eval32", f"{arm}_{variant}@100v")
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
            with open(os.path.join(OUT, "gauge_fix_metrics.json"), "w") as fp:
                json.dump(all_metrics, fp, indent=1, default=str)

    # AUC reproduction sanity: pose-mode eval on the ORIG arm dirs (CPU).
    for arm in args.arms:
        work_dir = os.path.join(OUT, "eval32", f"{arm}@100v")
        todo = [s for s in scenes
                if os.path.isfile(os.path.join(
                    mini_npz_dir(work_dir, s), "mini_npz", "results.npz"))]
        if not todo:
            continue
        ev = VGGTEvaluator(work_dir=work_dir, datas=["scannetpp"],
                           modes=["pose"], scenes=todo, max_frames=0)
        m = ev.eval()
        for k, v in m.items():
            all_metrics[f"{arm}@100v::{k}"] = v
        au = [v[s]["auc03"] for s in todo]
        print(f"POSE-CHECK {arm}: AUC@3 mean over {len(todo)} scenes = "
              f"{float(np.mean(au)):.4f}", flush=True)

    # AUC-invariance certificate: variant npz extr/intr vs orig, bitwise.
    max_diff = 0.0
    n_checked = 0
    for arm in args.arms:
        for s in scenes:
            p0 = os.path.join(mini_npz_dir(
                os.path.join(OUT, "eval32", f"{arm}@100v"), s),
                "mini_npz", "results.npz")
            for variant in ("fixgt", "fixfree"):
                p1 = os.path.join(mini_npz_dir(
                    os.path.join(OUT, "eval32", f"{arm}_{variant}@100v"), s),
                    "mini_npz", "results.npz")
                if not (os.path.isfile(p0) and os.path.isfile(p1)):
                    continue
                e0, e1 = np.load(p0)["extrinsics"], np.load(p1)["extrinsics"]
                i0, i1 = np.load(p0)["intrinsics"], np.load(p1)["intrinsics"]
                max_diff = max(max_diff, float(np.abs(e0 - e1).max()),
                               float(np.abs(i0 - i1).max()))
                n_checked += 1
    all_metrics["pose_invariance"] = {"max_abs_diff": max_diff,
                                      "n_pairs": n_checked}
    print(f"POSE-INVARIANCE: {n_checked} pairs, max|Delta| = {max_diff:.3e}",
          flush=True)
    with open(os.path.join(OUT, "gauge_fix_metrics.json"), "w") as fp:
        json.dump(all_metrics, fp, indent=1, default=str)
    print("FUSE DONE", flush=True)


if __name__ == "__main__":
    main()
