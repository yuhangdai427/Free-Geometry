#!/usr/bin/env python3
"""Counterfactual fusion for the 1ada7a0617 anatomy: run the repo's
VGGTEvaluator (same engine as run_eval.py / run_posed_attr.py) over the
ada_anatomy scratch exps in recon_unposed AND recon_posed modes.
MUST be launched with CUDA_VISIBLE_DEVICES="" so the process (and its fusion
worker forks) never initialises CUDA - the same safety property run_eval.py
relies on. New file; does not modify any existing module. Run from repo root."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.join(_REPO, "src", "vggt"))

import open3d as o3d  # noqa: E402
from vggt.bench.evaluator import VGGTEvaluator  # noqa: E402

RR = os.path.join(_REPO, "artifacts", "diagnostics", "final_protocol",
                  "scannetpp", "ada_anatomy")
EXPS = ["A0_baseline@100v", "B5_maskdistill@100v",
        "B5_oraclescale@100v", "A0_badscale@100v"]

for exp in EXPS:
    work_dir = os.path.join(RR, "eval32", exp)
    if not os.path.isdir(os.path.join(work_dir, "model_results")):
        continue
    o3d.utility.random.seed(42)
    ev = VGGTEvaluator(work_dir=work_dir, datas=["scannetpp"],
                       modes=["recon_unposed", "recon_posed"],
                       scenes=["1ada7a0617"], max_frames=0, num_fusion_workers=1)
    m = ev.eval()
    for k, v in m.items():
        r = v["1ada7a0617"]
        print(f"VARIANT {exp} {k}: F1={r['fscore']:.4f} acc={r['acc']:.4f} "
              f"comp={r['comp']:.4f} prec={r['precision']:.4f} rec={r['recall']:.4f}",
              flush=True)
print("VARIANTS DONE")
