#!/usr/bin/env python3
"""Build the DTU-64 scene manifest for the raw+rel TTA campaign (2026-09-22).

Same construction as scripts/build_dtu_manifest_fg.py but for the pose-only
DTU-64 subset (13 scenes, 64 frames each). eval32_frames = ALL frames.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "src"))
sys.path.insert(0, os.path.join(ROOT, "..", "diagnostics", "free_geometry"))

from depth_anything_3.test_time_adaption import protocol_v1 as P
from depth_anything_3.utils.constants import DTU64_SCENES
import common as fg

OUT = "workspace/ndispatch/dtu64"


def main():
    fg.set_dataset("dtu64")
    out = {"dataset": "dtu64",
           "note": "DTU-64 manifest for raw+rel campaign (2026-09-22): fresh build, "
                   "eval = all frames (pose-only; eval3d raises NotImplementedError "
                   "and is skipped); pairs via protocol_v1 auto rule",
           "run_root": OUT,
           "scenes": {}}
    for scene in DTU64_SCENES:
        sd = fg.get_scene_data(scene)
        files = list(sd.image_files)
        N = len(files)
        proto = P.build_scene_protocol(files, scene, dataset="dtu64",
                                       n_train=10, n_shared=4, teacher_N=None)
        ev = list(range(N))
        out["scenes"][scene] = {
            "num_frames_total": N,
            "num_frames_with_gt_depth": N,
            "eval32_frames": ev,
            "adaptation_pool_size": N - len(set(ev)),
            "train_pairs": proto["train_pairs"],
            "probe_pairs": proto["probe_pairs"],
            "teacher_N": proto["teacher_N"],
            "tau": proto.get("tau"),
            "strategy": proto.get("strategy"),
        }
        print(f"dtu64/{scene}: N={N} tN={proto['teacher_N']} "
              f"eval={len(ev)} pairs={len(proto['train_pairs'])}")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "scene_manifest.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> {OUT}/scene_manifest.json ({len(out['scenes'])} scenes)")


if __name__ == "__main__":
    main()
