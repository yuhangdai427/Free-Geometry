#!/usr/bin/env python3
"""Evaluate all eval32 experiment exports (pose + recon_unposed) with the repo's
VGGTEvaluator. Open3D point sampling is seeded per experiment for determinism
(same scene order + same seed => identical GT/pred samples across arms)."""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from common import load_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", default="artifacts/diagnostics/bakeoff_v1")
    ap.add_argument("--experiments", nargs="*", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--datas", nargs="*", default=["scannetpp"])
    args = ap.parse_args()

    import open3d as o3d
    sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
    sys.path.insert(0, os.path.join(_HERE, "..", "..", "src", "vggt"))
    from vggt.vggt.bench.evaluator import VGGTEvaluator

    manifest = load_manifest(args.manifest or os.path.join(args.run_root, "scene_manifest.json"))
    scenes = sorted(manifest["scenes"])

    eval_root = os.path.join(args.run_root, "eval32")
    exps = args.experiments or sorted(os.listdir(eval_root))

    import re

    def exp_modes(exp):
        m = re.search(r"@(\d+)v$", exp)
        if m and int(m.group(1)) < 3:
            return ["recon_unposed"]  # <3 cams: trajectory Umeyama degenerate
        return ["pose", "recon_unposed"]

    summary = {}
    for exp in exps:
        work_dir = os.path.join(eval_root, exp)
        if not os.path.isdir(os.path.join(work_dir, "model_results")):
            continue
        exp_scenes = scenes
        for data_name in args.datas:
            present = os.path.join(work_dir, "model_results", data_name)
            exp_scenes = [s for s in exp_scenes if os.path.isfile(
                os.path.join(present, s, "unposed", "exports", "mini_npz", "results.npz"))]
        if not exp_scenes:
            continue
        o3d.utility.random.seed(42)  # deterministic point sampling for this exp
        evaluator = VGGTEvaluator(
            work_dir=work_dir, datas=args.datas,
            modes=exp_modes(exp), scenes=exp_scenes,
            max_frames=-1, num_fusion_workers=4)
        metrics = evaluator.eval()
        summary[exp] = metrics
        print(f"[eval] {exp} done", flush=True)

    out = os.path.join(args.run_root, "eval32_metrics.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
