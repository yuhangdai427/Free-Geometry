#!/usr/bin/env python3
"""Build the eth3d scene manifest (fresh) with the SAME frame-selection strategy.
Scene list = the 11 ndispatch eth3d scenes. NOTE: most eth3d scenes have N<64
so the auto rule dispatches teacher 8 frames (8:4), unlike scannetpp's 16:4.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "src"))
sys.path.insert(0, os.path.join(ROOT, "..", "diagnostics", "free_geometry"))

from depth_anything_3.test_time_adaption import protocol_v1 as P
import common as fg

OUT = "workspace/ndispatch/eth3d"
SCENES = ("courtyard delivery_area electro facade kicker office pipes "
          "playground relief relief_2 terrains").split()

def main():
    fg.set_dataset("eth3d")
    out = {"dataset": "eth3d",
           "note": "eth3d manifest (2026-09-23) for the TTA ablation rerun; "
                   "protocol_v1 auto rule (N>=64 -> 16:4 else 8:4)",
           "run_root": OUT, "scenes": {}}
    for scene in SCENES:
        try:
            sd = fg.get_scene_data(scene)
            files = list(sd.image_files)
        except Exception as exc:
            print(f"eth3d/{scene}: SKIP ({exc})")
            continue
        N = len(files)
        proto = P.build_scene_protocol(files, scene, dataset="eth3d",
                                       n_train=10, n_shared=4, teacher_N=None)
        out["scenes"][scene] = {
            "num_frames_total": N,
            "num_frames_with_gt_depth": N,
            "eval32_frames": list(range(N)),
            "adaptation_pool_size": 0,
            "train_pairs": proto["train_pairs"],
            "probe_pairs": proto["probe_pairs"],
            "teacher_N": proto["teacher_N"],
            "tau": proto.get("tau"),
            "strategy": proto.get("strategy"),
        }
        print(f"eth3d/{scene}: N={N} tN={proto['teacher_N']} tau={proto.get('tau')}")
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "scene_manifest.json"), "w"), indent=1)
    print(f"-> {OUT}/scene_manifest.json ({len(out['scenes'])} scenes)")

if __name__ == "__main__":
    main()
