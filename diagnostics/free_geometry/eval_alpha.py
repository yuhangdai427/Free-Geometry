#!/usr/bin/env python3
"""Alpha-interpolation eval: scale trained LoRA deltas by alpha (wise-ft style
shrinkage toward baseline) and re-evaluate. Zero training.
Usage: python eval_alpha.py --run_root <rr> --arm C2M_maskrel --alphas 0.5 0.75 --views 100v
Run from repo root."""

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


def scale_lora_(student, alpha):
    n = 0
    for name, p in student.vggt.named_parameters():
        if "lora_B" in name:
            p.data.mul_(alpha)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--arm", default="C2M_maskrel")
    ap.add_argument("--alphas", nargs="*", type=float, default=[0.5, 0.75])
    ap.add_argument("--views", default="100v")
    ap.add_argument("--step", type=int, default=100)
    args = ap.parse_args()

    manifest = load_manifest(os.path.join(args.run_root, "scene_manifest.json"))
    common.set_dataset(manifest.get("dataset", "scannetpp"))
    scenes = sorted(manifest["scenes"])
    student = M.load_student()
    student.eval()
    for scene in scenes:
        sc = manifest["scenes"][scene]
        scene_data = get_scene_data(scene)
        ev = sc["eval32_frames"]
        v = args.views
        frames = ev if v == "allv" else ev[:: max(1, len(ev) // int(v.rstrip("v")))][: int(v.rstrip("v"))]
        ckpt = os.path.join(args.run_root, "ckpts", scene, args.arm, f"step{args.step}_lora.pt")
        for alpha in args.alphas:
            M.reset_lora_(student)
            student.load_lora_weights(ckpt)
            n = scale_lora_(student, alpha)
            depth, ext, intr, _, conf = infer_eval32(student, scene_data, frames)
            tag = f"{args.arm}_a{int(round(alpha * 100)):03d}@{v}"
            save_eval_npz(args.run_root, tag, scene, depth, ext, intr,
                          np.asarray(scene_data.extrinsics)[frames],
                          gt_ixt_raw(scene_data, frames),
                          [scene_data.image_files[i] for i in frames], frames)
            print(f"[{scene} a={alpha} n_loraB={n}] done", flush=True)
        del scene_data
        torch.cuda.empty_cache()
    print("ALPHA DONE")


if __name__ == "__main__":
    main()
