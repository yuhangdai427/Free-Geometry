#!/usr/bin/env python3
"""Build the DTU scene manifest for the raw+rel TTA campaign (2026-09-21).

Mirrors scripts/build_ndispatch_manifests.py's dtu branch, minus the
workspace/protocol_v2/baselines.json dependency (scene list comes from
depth_anything_3.utils.constants.DTU_SCENES instead), so it can run on a fresh
clone. eval32_frames = ALL frames (DTU is evaluated full-scene; pose + chamfer).
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "src"))
sys.path.insert(0, os.path.join(ROOT, "..", "diagnostics", "free_geometry"))

from depth_anything_3.test_time_adaption import protocol_v1 as P
from depth_anything_3.utils.constants import DTU_SCENES
import common as fg

OUT = "workspace/ndispatch/dtu"


def main():
    fg.set_dataset("dtu")
    out = {"dataset": "dtu",
           "note": "DTU manifest for raw+rel campaign (2026-09-21): fresh build, "
                   "eval = all frames (pose AUC + chamfer via fuse3d/eval3d); "
                   "pairs via protocol_v1 auto rule (N>=64 -> 16:4 else 8:4); "
                   "DA3/VGGT share the same seeded pair draws",
           "run_root": OUT,
           "scenes": {}}
    for scene in DTU_SCENES:
        sd = fg.get_scene_data(scene)
        files = list(sd.image_files)
        N = len(files)
        proto = P.build_scene_protocol(files, scene, dataset="dtu",
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
        print(f"dtu/{scene}: N={N} tN={proto['teacher_N']} "
              f"eval={len(ev)} pairs={len(proto['train_pairs'])}")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "scene_manifest.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> {OUT}/scene_manifest.json ({len(out['scenes'])} scenes)")


if __name__ == "__main__":
    main()
