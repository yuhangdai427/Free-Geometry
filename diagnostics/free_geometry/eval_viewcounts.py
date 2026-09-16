#!/usr/bin/env python3
"""Evaluate saved per-scene LoRA checkpoints at LOW view counts (4v/8v nested
subsets of the fixed eval32 frames). Generic over ckpt root and arms."""

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import common
from common import get_scene_data, gt_ixt_raw, load_manifest
import modeling as M
from train_arms import infer_eval32, save_eval_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True, help="output tree (npz go to <run_root>/eval32/)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt_root", default=None, help="ckpts/<scene>/<arm>/step{N}_lora_peft")
    ap.add_argument("--step", type=int, default=30, help="checkpoint step (30 or 100)")
    ap.add_argument("--arms", nargs="*", default=["A1_deployed", "B5_conf", "C2_b5_rel"])
    ap.add_argument("--view_subsets", nargs="*", default=["8v", "4v"])
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    common.set_dataset(manifest.get("dataset", "scannetpp"))
    scenes = sorted(manifest["scenes"])
    student = M.load_student()

    arms = [("A0_baseline", None)] + [(a, a) for a in args.arms]
    if os.environ.get("SKIP_BASELINE", "0") == "1":
        arms = [(a, a) for a in args.arms]
    for arm, ckpt_arm in arms:
        for scene in scenes:
            sc = manifest["scenes"][scene]
            scene_data = get_scene_data(scene)
            ev = sc["eval32_frames"]
            views = {}
            for v in args.view_subsets:
                if v == "allv":
                    from common import frames_with_gt_depth
                    ok, _ = frames_with_gt_depth(scene)
                    views[v] = ok
                    continue
                n = int(v.rstrip("v"))
                views[v] = ev[::max(1, len(ev) // n)][:n]
            M.reset_lora_(student)
            if ckpt_arm is not None:
                student.load_lora_weights(
                    os.path.join(args.ckpt_root, scene, ckpt_arm, f"step{args.step}_lora.pt"))
            student.eval()
            for vc, frames in views.items():
                depth, ext, intr, _, _conf = infer_eval32(student, scene_data, frames)
                save_eval_npz(args.run_root, f"{arm}@{vc}", scene, depth, ext, intr,
                              np.asarray(scene_data.extrinsics)[frames],
                              gt_ixt_raw(scene_data, frames),
                              [scene_data.image_files[i] for i in frames], frames)
            print(f"[{arm} {scene}] done", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
