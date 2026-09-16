#!/usr/bin/env python3
"""Attribution fusion for 21d970d8de: run the official VGGTEvaluator (same
engine/TSDF constants as run_eval.py) over the attr_21d97 scratch exps in BOTH
recon_unposed and recon_posed modes. Exps:
  A0_baseline@100v, A0_baseline_gtdepth@100v,
  C2M_RKDC1@100v,  C2M_RKDC1_gtdepth@100v
-> {pred,GT} pose x {pred,GT} depth for each arm. Writes
attr_21d97/attr_metrics.json and deletes fused ply trees afterwards (disk
rule). MUST be launched with CUDA_VISIBLE_DEVICES="" (also set in-process).
New file; does not modify any existing module. Run from repo root."""

import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.join(_REPO, "src", "vggt"))

import open3d as o3d  # noqa: E402
from vggt.bench.evaluator import VGGTEvaluator  # noqa: E402

RR = os.path.join(_REPO, "artifacts", "diagnostics", "final_protocol",
                  "scannetpp_dev3b")
OUT = os.path.join(RR, "attr_21d97")
SCENE = "21d970d8de"
EXPS = ["A0_baseline@100v", "A0_baseline_gtdepth@100v",
        "C2M_RKDC1@100v", "C2M_RKDC1_gtdepth@100v"]


def main():
    all_metrics = {}
    for exp in EXPS:
        work_dir = os.path.join(OUT, "eval32", exp)
        if not os.path.isdir(os.path.join(work_dir, "model_results")):
            print(f"[skip] {exp}", flush=True)
            continue
        o3d.utility.random.seed(42)
        ev = VGGTEvaluator(work_dir=work_dir, datas=["scannetpp"],
                           modes=["recon_unposed", "recon_posed"],
                           scenes=[SCENE], max_frames=0, num_fusion_workers=4)
        m = ev.eval()
        for k, v in m.items():
            all_metrics[f"{exp}::{k}"] = v
            r = v[SCENE]
            print(f"RESULT {exp} {k}: F1={r['fscore']:.4f} "
                  f"CD={r['overall']:.4f} acc={r['acc']:.4f} "
                  f"comp={r['comp']:.4f}", flush=True)
    with open(os.path.join(OUT, "attr_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=1, default=str)

    # disk rule: remove fused ply trees (npz retained)
    for exp in EXPS:
        for posed in ("unposed", "posed"):
            fuse_dir = os.path.join(OUT, "eval32", exp, "model_results",
                                    "scannetpp", SCENE, posed, "exports", "fuse")
            if os.path.isdir(fuse_dir):
                shutil.rmtree(fuse_dir)
    print("FUSE DONE", flush=True)


if __name__ == "__main__":
    main()
