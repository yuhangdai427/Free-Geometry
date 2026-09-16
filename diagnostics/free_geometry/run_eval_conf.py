#!/usr/bin/env python3
"""Phase E: confidence-filtered TSDF re-evaluation.

Re-infers 32v for the given arms (from their per-scene step30 ckpts), saves
depth_conf, then writes shadow experiment trees with low-confidence depths
zeroed out (per-frame bottom-q quantile filtered). Baseline included - the
filter is applied to EVERY arm identically (fair comparison).

Usage: produces <run_root>/eval32/<arm>@confq<q> trees; evaluate with
run_eval.py afterwards.
"""

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common import get_scene_data, load_manifest
import modeling as M
from train_arms import infer_eval32, save_eval_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--arms", nargs="*", default=["A1_deployed", "B5_conf", "C2_b5_rel", "G4_conf_l23only"])
    ap.add_argument("--quantiles", nargs="*", type=float, default=[0.3, 0.5])
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    scenes = sorted(manifest["scenes"])
    student = M.load_student()

    arms = [("A0_baseline", None)] + [(a, a) for a in args.arms]
    for arm, ckpt_arm in arms:
        for scene in scenes:
            sc = manifest["scenes"][scene]
            scene_data = get_scene_data(scene)
            ev = sc["eval32_frames"]
            M.reset_lora_(student)
            if ckpt_arm is not None:
                student.load_lora_weights(
                    os.path.join(args.ckpt_root, scene, ckpt_arm, "step30_lora.pt"))
            student.eval()
            depth, ext, intr, _, conf = infer_eval32(student, scene_data, ev)
            assert conf is not None, "depth_conf missing"
            for q in args.quantiles:
                d = depth.copy()
                for k in range(len(d)):
                    thr = np.quantile(conf[k], q)
                    d[k][conf[k] < thr] = 0.0  # invalid for TSDF (depth<=0 skipped)
                save_eval_npz(args.run_root, f"{arm}@confq{int(q*100)}", scene, d, ext, intr,
                              np.asarray(scene_data.extrinsics)[ev],
                              np.asarray(scene_data.aux.ixt_raw_list)[ev],
                              [scene_data.image_files[i] for i in ev], ev)
            print(f"[{arm} {scene}] conf-filtered saved", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
