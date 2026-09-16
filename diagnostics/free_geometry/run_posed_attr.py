#!/usr/bin/env python3
"""ScanNet++ F1 attribution: same predictions, fuse with predicted poses
(recon_unposed) vs GT poses (recon_posed). If GT poses + TTA depth recovers
F1, the wound is the pose path; if not, it is cross-view depth-scale drift.
Run from repo root."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.join(_REPO, "src", "vggt"))

from vggt.bench.evaluator import VGGTEvaluator  # noqa: E402

RR = os.path.join(_REPO, "artifacts", "diagnostics", "final_protocol", "scannetpp")
SCENES = sorted(__import__("json").load(open(os.path.join(RR, "scene_manifest.json")))["scenes"])

for exp in ("A0_baseline@100v", "C2M_maskrel@100v"):
    ev = VGGTEvaluator(work_dir=os.path.join(RR, "eval32", exp),
                       datas=["scannetpp"], modes=["recon_unposed", "recon_posed"],
                       scenes=SCENES, max_frames=0)
    m = ev.eval()
    for k, v in m.items():
        f1 = [x["fscore"] for s, x in v.items() if s != "mean"]
        cd = [x["overall"] for s, x in v.items() if s != "mean"]
        print(f"ATTR {exp} {k}: F1={sum(f1)/len(f1):.4f} CD={sum(cd)/len(cd):.4f} n={len(f1)}",
              flush=True)
print("ATTR DONE")
