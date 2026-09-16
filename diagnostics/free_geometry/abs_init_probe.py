"""One-shot measurement: initial teacher(16v) vs student(4v) camera poses.

Answers, with concrete numbers on one training pair:
  1. what the raw initial camera values actually look like (centers, FoV, trajectory extent)
  2. how big the raw 4v-vs-16v gauge difference is (the "~2x" artifact decomposition)
  3. translation L1 error under three comparisons: raw-raw / teacher-normalized-vs-student-raw
     (ABS v1 target) / both-self-normalized (ABS-lite residual)
"""
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import common
from common import STUDENT_INDICES, get_scene_data, load_manifest
import modeling as M
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

IMAGE_HW = (378, 504)
SCENES = ["09c1414f1b", "1ada7a0617"]
ROOT = "artifacts/diagnostics/final_protocol/scannetpp"


def centers_and_fov(pose_enc):
    E, K = pose_encoding_to_extri_intri(pose_enc.float(), IMAGE_HW)  # w2c
    R, t = E[0, :, :3, :3], E[0, :, :3, 3]
    c = -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)  # camera centers (c2w translation)
    fov = pose_enc[0, :, 7:9]
    return c, fov, E, K


def stats(name, c, fov):
    cen = c.mean(0)
    d_cen = (c - cen).norm(dim=-1)
    bbox = (c.max(0).values - c.min(0).values)
    print(f"  {name}: bbox(xyz)={bbox[0]:.3f}/{bbox[1]:.3f}/{bbox[2]:.3f}  "
          f"mean|c-cen|={d_cen.mean():.3f}  max|c-cen|={d_cen.max():.3f}  "
          f"FoV_h={fov[:, 0].mean():.1f}deg 首帧中心={c[0].cpu().numpy().round(3)}")


def main():
    manifest = load_manifest(os.path.join(ROOT, "scene_manifest.json"))
    vggt = M.load_teacher("cuda")
    for scene in SCENES:
        entry = manifest["scenes"][scene] if "scenes" in manifest else manifest[scene]
        pair = entry["train_pairs"][0]
        scene_data = get_scene_data(scene)
        images16, images4 = M.load_pair_images(scene_data, pair["teacher_frames"], "cuda")

        with torch.no_grad(), torch.autocast("cuda", enabled=True):
            f16, psi = M.aggregator_all(vggt, images16)
            pose16 = M.replay_camera_nograd(vggt, f16)
            f4, _ = M.aggregator_all(vggt, images4)
            pose4 = M.replay_camera_nograd(vggt, f4)

        c16, fov16, E16, K16 = centers_and_fov(pose16)
        c4, fov4, E4, K4 = centers_and_fov(pose4)
        print(f"\n===== {scene} (pair0) =====")
        stats("teacher 16v (全部16帧)", c16.cpu(), fov16.cpu())
        stats("student 4v (共享4帧) ", c4.cpu(), fov4.cpu())

        # matched shared frames: teacher poses at STUDENT_INDICES
        ct = c16[STUDENT_INDICES]  # [4,3]
        cs = c4
        # raw-vs-raw (both already first-frame anchored by architecture)
        raw_l1 = (ct - cs).abs().mean().item()
        # scale ratio between the two trajectories (RMS distance from centroid)
        st_ = (ct - ct.mean(0)).norm(dim=-1).mean().item()
        ss_ = (cs - cs.mean(0)).norm(dim=-1).mean().item()
        # both self-normalized by own RMS scale -> real residual
        resid = ((ct - ct.mean(0)) / st_ - (cs - cs.mean(0)) / ss_).abs().mean().item()
        # ABS v1 setting: teacher normalized by point-scale (approx by RMS here) vs student raw
        abs_v1_like = (ct / st_ - cs).abs().mean().item()
        print(f"  尺度比 teacher/student = {st_/ss_:.3f}")
        print(f"  平移 L1：raw vs raw = {raw_l1:.4f}")
        print(f"  平移 L1：teacher归一化 vs student原始（≈ABS v1 的量）= {abs_v1_like:.4f}")
        print(f"  平移 L1：双方各自归一化后的真实残余 = {resid:.4f}")
        print(f"  首帧中心差 |c0_t - c0_s| = {(c16[0]-c4[0]).norm().item():.4f}")


if __name__ == "__main__":
    main()
