#!/usr/bin/env python3
"""Phase-4 sub8 manifests: copy the 8 sub8 scenes from final_protocol manifests
into phase4 run_roots (same pairs, same eval frames; arms differ at train time).
Also emits the run list. Run from repo root."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

from common import load_manifest, save_manifest  # noqa: E402

SUB8 = {
    "scannetpp": ["7831862f02", "bde1e479ad"],
    "7scenes": ["chess", "office"],
    "hiroom": ["20241230/828738/cam_sampled_08", "20241230/828749/cam_sampled_12"],
    "eth3d": ["courtyard", "office"],
}

for ds, scenes in SUB8.items():
    src = load_manifest(os.path.join(
        _REPO, "artifacts", "diagnostics", "final_protocol", ds, "scene_manifest.json"))
    out = {"dataset": ds, "seed_scene_select": 43,
           "note": "Phase-4 sub8 gating: pairs/eval frames identical to final_protocol",
           "run_root": os.path.join("artifacts", "diagnostics", "final_protocol_phase4", ds),
           "scenes": {s: src["scenes"][s] for s in scenes}}
    dst = os.path.join(_REPO, "artifacts", "diagnostics", "final_protocol_phase4", ds)
    save_manifest(out, os.path.join(dst, "scene_manifest.json"))
    print(f"[{ds}] {len(out['scenes'])} scenes -> {dst}")
