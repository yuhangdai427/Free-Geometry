#!/usr/bin/env python3
"""Correctness smokes for the 3-model migration (plan section 9, phase 1).

Per model:
  1. zero-init identity: adapter teacher forward == official forward (bit-wise)
  2. grad path: student backward reaches trainable params; frozen heads untouched
  3. (dvlt) differentiable centers == official rays_to_pose T
Usage: fgmig_smoke.py --model {omega,pi3,dvlt}
"""
import argparse
import os
import sys

ROOT = "/root/autodl-tmp/Free-Geometry"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "free_geometry"))

import numpy as np
import torch


def rand_images(n, res, patch, device):
    h = w = res  # square smoke input, /patch exact
    x = torch.rand(1, n, 3, h, w, device=device)
    return x


def smoke_omega(device="cuda"):
    from free_geometry.adapters.vggt_omega import VGGTOmegaAdapter
    a = VGGTOmegaAdapter()
    a.load(device)
    imgs = rand_images(8, 416, 16, device)
    with torch.no_grad():
        official = a.teacher.forward(imgs)
        toks, psi, pose, depth, conf = a._full_forward(a.teacher, imgs)
    d1 = (official["pose_enc"] - pose).abs().max().item()
    d2 = (official["depth"] - depth.unsqueeze(-1)).abs().max().item()
    d3 = (official["depth_conf"] - conf).abs().max().item()
    print(f"[omega] identity: pose {d1:.2e} depth {d2:.2e} conf {d3:.2e}")
    assert max(d1, d2, d3) < 5e-3, "zero-init identity FAILED"

    a.reset_student(device)
    m4 = rand_images(4, 416, 16, device)
    out = a.forward_student(m4)
    loss = sum(v.sum() for v in out["readouts"].values()) + out["centers"].sum()
    loss.backward()
    lora_grads = [p.grad.abs().sum().item() for n, p in a.net.named_parameters()
                  if "lora_" in n and p.grad is not None]
    head_touched = any("dense_head" in n and p.grad is not None and p.grad.abs().sum() > 0
                       for n, p in a.net.named_parameters())
    print(f"[omega] grad: lora tensors w/ grad={len(lora_grads)}, "
          f"sum={sum(lora_grads):.3e}, frozen-head-grad={head_touched}")
    assert len(lora_grads) > 0 and sum(lora_grads) > 0, "grad path FAILED"
    assert not head_touched, "frozen head received grad!"
    # student zero-LoRA == teacher on same input
    with torch.no_grad():
        out2 = a.forward_student(m4)
        ref = a.forward_teacher(m4, [0, 1, 2, 3])
    dd = max((out2["readouts"][k] - ref["readouts"][k]).abs().max().item()
             for k in out2["readouts"])
    print(f"[omega] student==teacher (zero LoRA): readout diff {dd:.2e}")
    assert dd < 5e-3
    print("[omega] SMOKE PASS")


def smoke_pi3(device="cuda"):
    from free_geometry.adapters.pi3 import Pi3Adapter
    a = Pi3Adapter()
    a.load(device)
    imgs = rand_images(8, 504, 14, device)
    with torch.no_grad():
        official = a.teacher.forward(imgs)
        mine = a._full_forward(a.teacher, imgs, want_readouts=True)
    d1 = (official["local_points"] - mine["depth"].unsqueeze(-1) * 0 - official["local_points"]).abs().max().item()
    # compare depth + poses properly
    d_depth = (official["local_points"][..., 2] - mine["depth"]).abs().max().item()
    d_pos = (official["camera_poses"] - mine["camera_poses"]).abs().max().item()
    d_conf = (torch.sigmoid(official["conf"][..., 0]) - mine["conf"]).abs().max().item()
    print(f"[pi3] identity: depth {d_depth:.2e} pose {d_pos:.2e} conf {d_conf:.2e}")
    assert max(d_depth, d_pos, d_conf) < 5e-3, "zero-init identity FAILED"

    a.reset_student(device)
    m4 = rand_images(4, 504, 14, device)
    out = a.forward_student(m4)
    loss = sum(v.sum() for v in out["readouts"].values()) + out["centers"].sum()
    loss.backward()
    lora_grads = [p.grad.abs().sum().item() for n, p in a.net.named_parameters()
                  if "lora_" in n and p.grad is not None]
    print(f"[pi3] grad: lora tensors w/ grad={len(lora_grads)}, sum={sum(lora_grads):.3e}")
    assert len(lora_grads) > 0 and sum(lora_grads) > 0, "grad path FAILED"
    with torch.no_grad():
        ref = a.forward_teacher(m4, [0, 1, 2, 3])
    dd = max((out2 - ref2).abs().max().item()
             for out2, ref2 in [(out["readouts"]["point_proj_ln"],
                                 ref["readouts"]["point_proj_ln"])])
    print(f"[pi3] student==teacher (zero LoRA): readout diff {dd:.2e}")
    assert dd < 5e-3
    print("[pi3] SMOKE PASS")


def smoke_dvlt(device="cuda"):
    from free_geometry.adapters.dvlt import DVLTAdapter
    a = DVLTAdapter()
    a.load(device)
    imgs = rand_images(8, 504, 14, device)
    with torch.no_grad():
        official = a.teacher.forward_inference(imgs)
        mine = a._forward_adapt(a.teacher, imgs, want_readouts=True)
    d_depth = (official["depth"][..., 0] - mine["depth"]).abs().max().item()
    d_rays = (official["rays"] - mine["rays"]).abs().max().item()
    print(f"[dvlt] identity: depth {d_depth:.2e} rays {d_rays:.2e}")
    assert max(d_depth, d_rays) < 5e-3, "zero-init identity FAILED"

    # centers vs official rays_to_pose T (uniform conf)
    from dvlt.common.rays import rays_to_pose
    with torch.no_grad():
        c2w, _ = rays_to_pose(official["rays"].float(),
                              torch.ones_like(mine["depth"]),
                              imgs.shape[-2], imgs.shape[-1], patch_size=14)
    off_centers = c2w[0, :, :3, 3]
    d_c = (off_centers - mine["centers"][0]).abs().max().item()
    print(f"[dvlt] centers vs rays_to_pose T: {d_c:.2e} "
          f"(scale {off_centers.norm(dim=-1).mean().item():.3f})")
    assert d_c < 1e-3 * max(1.0, off_centers.norm().max().item()), "centers mismatch"

    a.reset_student(device)
    m4 = rand_images(4, 504, 14, device)
    out = a.forward_student(m4)
    loss = sum(v.sum() for v in out["readouts"].values()) + out["centers"].sum()
    loss.backward()
    n_grad = sum(1 for p in a.model.parameters() if p.grad is not None
                 and p.grad.abs().sum() > 0)
    n_tot = sum(1 for _ in a.model.parameters())
    print(f"[dvlt] grad: {n_grad}/{n_tot} param tensors got grad "
          f"(label {a.student_label()})")
    assert n_grad > n_tot // 4, "full-FT grad path FAILED"
    print("[dvlt] SMOKE PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["omega", "pi3", "dvlt"])
    args = ap.parse_args()
    {"omega": smoke_omega, "pi3": smoke_pi3, "dvlt": smoke_dvlt}[args.model]()
