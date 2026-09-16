#!/usr/bin/env python3
"""Measure the 32-view teacher upper bound: frozen VGGT on 64 views
(32 eval frames + 32 extra context frames from the same scene), outputs
sliced back to the 32 eval frames -> saved as a pseudo-experiment
'vT_teacher64to32' in the merged eval tree. This is the N->M analog of the
paper's 8->4 teacher endpoint, at the actual evaluation view count.
"""

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common import frames_with_gt_depth, get_scene_data, load_manifest, stable_seed
import modeling as M
from train_arms import infer_eval32, save_eval_npz
import random


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", default="artifacts/diagnostics/bakeoff_merged")
    ap.add_argument("--manifest", default="artifacts/diagnostics/bakeoff_v2_transductive/scene_manifest.json")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    student = M.load_student()  # fresh (LoRA B=0) == frozen baseline; heads intact
    M.reset_lora_(student)

    for scene in sorted(manifest["scenes"]):
        sc = manifest["scenes"][scene]
        scene_data = get_scene_data(scene)
        eval_frames = sc["eval32_frames"]
        ok, _ = frames_with_gt_depth(scene)
        pool = [i for i in ok if i not in set(eval_frames)]
        rng = random.Random(stable_seed("teacher64", scene))
        extra = sorted(rng.sample(pool, 32))
        all64 = sorted(eval_frames + extra)
        pos_of_eval = [all64.index(f) for f in eval_frames]

        # 64-view forward with the frozen model, then slice outputs to eval32
        depth64, ext64, intr64, _, _conf = infer_eval32(student, scene_data, all64)
        depth32 = depth64[pos_of_eval]
        ext32 = ext64[pos_of_eval]
        intr32 = intr64[pos_of_eval]

        save_eval_npz(args.run_root, "vT_teacher64to32", scene, depth32, ext32, intr32,
                      np.asarray(scene_data.extrinsics)[eval_frames],
                      np.asarray(scene_data.aux.ixt_raw_list)[eval_frames],
                      [scene_data.image_files[i] for i in eval_frames], eval_frames)
        print(f"[{scene}] teacher64to32 saved", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
