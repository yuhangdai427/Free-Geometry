"""Drift measurement: view-count sensitivity of the pose head.

For a scene's eval frames: run the model at view counts [4, 8, 16, 100v], decode
poses, and measure per-frame pose disagreement between each low-count context and
the 100v context (rotation angle deg + translation direction deg + scale ratio).
Compare A0_baseline vs a TTA arm (RKDC1) on 21d970d8de (fusion-drift) + 7831862f02 (control).
"""
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import common
from common import get_scene_data, load_manifest
import modeling as M
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

IMAGE_HW = (378, 504)
ROOT = "artifacts/diagnostics/final_protocol/scannetpp_dev3b"
SCENES = ["21d970d8de", "7831862f02"]
ARMS = ["A0_baseline", "C2M_RKDC1"]


def centers_and_rots(pose_enc):
    E, _ = pose_encoding_to_extri_intri(pose_enc.float(), IMAGE_HW)
    R, t = E[0, :, :3, :3], E[0, :, :3, 3]
    c = -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)
    return c, R


def main():
    manifest = load_manifest(os.path.join(ROOT, "scene_manifest.json"))
    base = M.load_teacher("cuda")
    for scene in SCENES:
        entry = manifest["scenes"][scene]
        eval_frames = entry["eval32_frames"]
        scene_data = get_scene_data(scene)
        imgs_all = [common.load_image_model(scene_data.image_files[i]) for i in eval_frames]
        arr = torch.from_numpy(np.stack(imgs_all)).permute(0, 3, 1, 2).float().unsqueeze(0).cuda()

        for arm in ARMS:
            model = base
            model_v = base
            if arm != "A0_baseline":
                model = M.load_student("cuda")
                M.reset_lora_(model)
                ckpt = os.path.join(ROOT, "ckpts", scene, arm, "step100_lora_peft")
                model.load_lora_weights(os.path.join(ROOT, "ckpts", scene, arm, "step100_lora.pt"))
                model_v = M.get_base_vggt(model)
            # reference: full 100v forward
            with torch.no_grad(), torch.autocast("cuda", enabled=True):
                f_full, psi = M.aggregator_all(model_v, arr)
                pose_full = M.replay_camera_nograd(model_v, f_full)
            c_full, R_full = centers_and_rots(pose_full)
            spread_full = (c_full - c_full.mean(0)).norm(dim=-1).pow(2).mean().sqrt().item()

            for k in [4, 8, 16]:
                idx = np.linspace(0, len(eval_frames) - 1, k).astype(int)
                sub = arr[:, torch.tensor(idx, device="cuda")]
                with torch.no_grad(), torch.autocast("cuda", enabled=True):
                    fs, _ = M.aggregator_all(model_v, sub)
                    pose_sub = M.replay_camera_nograd(model_v, fs)
                c_sub, R_sub = centers_and_rots(pose_sub)
                # compare on the SAME k frames against the full-context prediction at those indices
                c_ref = c_full[torch.tensor(idx, device="cuda")]
                R_ref = R_full[torch.tensor(idx, device="cuda")]
                # align scale: centers spread ratio
                s_sub = (c_sub - c_sub.mean(0)).norm(dim=-1).pow(2).mean().sqrt().item()
                s_ref = (c_ref - c_ref.mean(0)).norm(dim=-1).pow(2).mean().sqrt().item()
                # rotation error after first-frame anchoring (both are already first-frame anchored)
                dR = (R_sub @ R_ref.transpose(-1, -2))
                trace = dR.diagonal(dim1=-2, dim2=-1).sum(-1)
                ang = torch.rad2deg(torch.arccos(((trace - 1) / 2).clamp(-1, 1)))
                # translation direction disagreement (first-frame centered)
                tc_s = c_sub - c_sub[0]
                tc_r = c_ref - c_ref[0]
                tn_s = torch.nn.functional.normalize(tc_s, dim=-1, eps=1e-8)
                tn_r = torch.nn.functional.normalize(tc_r, dim=-1, eps=1e-8)
                dir_deg = torch.rad2deg(torch.arccos((tn_s * tn_r).sum(-1).clamp(-1, 1)))
                print(f"{scene} | {arm} | {k}v vs 100v: "
                      f"rot {ang.mean():.3f}°/max {ang.max():.3f}°  "
                      f"tdir {dir_deg.mean():.3f}°/max {dir_deg.max():.3f}°  "
                      f"scale_ratio {s_sub/s_ref:.4f}")


if __name__ == "__main__":
    main()
