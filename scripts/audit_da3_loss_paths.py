#!/usr/bin/env python3
"""DA3 loss-path numerical audit (read-only, no training). Checks:
A/B/C pose-path consistency:
  A = teacher cache path (forward_head_only extrinsics @ shared slots, 16-view ctx)
  B = student training path (decode_pose_w2c on layer-39 cam token)
      B16 = same 16-view forward (isolates decode formula from context)
      B4  = 4-view forward (what the rel loss actually supervises)
  C = eval path (api.inference extrinsics, 16-view ctx)
Plus identity tests: loss_pose_rel(X,X)==0, maskdistill(X,X)==0.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "diagnostics", "free_geometry"))

import common as fg_common
from depth_anything_3.test_time_adaption import protocol_v1 as P

MODEL = "model_weights/DA3-GIANT-1.1"
SLOTS = P.STUDENT_SLOTS


def rot_deg(Ra, Rb):
    Ra = Ra / np.linalg.det(Ra) ** (1 / 3)
    Rb = Rb / np.linalg.det(Rb) ** (1 / 3)
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def to_rt(ext):
    ext = np.asarray(ext, dtype=np.float64)
    return ext[..., :3, :3], ext[..., :3, 3]


def centers(R, t):
    return -(R.transpose(0, 2, 1) @ t[..., None]).squeeze(-1)


def cmp(name, X, Y):
    Rx, tx = to_rt(X)
    Ry, ty = to_rt(Y)
    angs = [rot_deg(Rx[i], Ry[i]) for i in range(len(Rx))]
    cx, cy = centers(Rx, tx), centers(Ry, ty)
    rel_c = np.linalg.norm(cx - cy, axis=-1) / np.maximum(np.linalg.norm(cy, axis=-1), 1e-9)
    print(f"  {name}: rot diff mean={np.mean(angs):.4f}° max={np.max(angs):.4f}° | "
          f"center rel-err mean={np.mean(rel_c)*100:.3f}% max={np.max(rel_c)*100:.3f}%")


def main():
    fg_common.set_dataset("7scenes")
    man = json.load(open("artifacts/diagnostics/final_protocol/7scenes/scene_manifest.json"))
    pair = man["scenes"]["chess"]["train_pairs"][0]
    data = fg_common.get_scene_data("chess")
    files16 = [data.image_files[i] for i in pair["teacher_frames"]]
    files4 = [data.image_files[i] for i in pair["student_frames"]]
    print(f"chess pair1: teacher={pair['teacher_frames']}")
    print(f"             student={pair['student_frames']}")

    teacher = P.create_teacher(MODEL, device="cuda")

    imgs16 = P.load_images_da3(files16).unsqueeze(0).cuda()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        feats, H, W = P.backbone_tapped_forward(teacher, imgs16, P.TAP_UNION)
    hf = [(f.float(), c.float()) for f, c in P.head_feats(feats, P.TAP_UNION)]
    with torch.autocast(device_type="cuda", enabled=False):
        preds = teacher.model.forward_head_only(hf, H=H, W=W, process_camera=True, process_sky=False)
    A = preds["extrinsics"][0, SLOTS].float().cpu().numpy()  # [4,3,4]

    pos39 = P.TAP_UNION.index(39)
    cam_tok39 = feats[pos39][1].float()  # [1,16,3072]
    with torch.autocast(device_type="cuda", enabled=False):
        B16 = P.decode_pose_w2c(teacher.model.cam_dec, cam_tok39[:, SLOTS], H, W)[0].cpu().numpy()

    imgs4 = P.load_images_da3(files4).unsqueeze(0).cuda()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        feats4, H4, W4 = P.backbone_tapped_forward(teacher, imgs4, P.TAP_LAYERS)
    cam4 = feats4[-1][1].float()
    with torch.autocast(device_type="cuda", enabled=False):
        B4 = P.decode_pose_w2c(teacher.model.cam_dec, cam4, H4, W4)[0].cpu().numpy()

    pred = teacher.inference(image=files16, process_res=P.PROCESS_RES,
                             process_res_method="upper_bound_resize", ref_view_strategy="first")
    C = np.asarray(pred.extrinsics, dtype=np.float32)[SLOTS]

    print("\n== pose-path consistency (teacher == base model) ==")
    cmp("A(cache) vs B16(decode, same ctx)", A, B16)
    cmp("A(cache) vs C(api.inference)   ", A, C)
    cmp("A(cache) vs B4 (decode, 4-view)", A, B4)
    cmp("B16       vs B4  (ctx 16 vs 4) ", B16, B4)

    print("\n== identity / sanity ==")
    for name, X in [("A", A), ("B16", B16), ("C", C)]:
        l, _ = P.loss_pose_rel(torch.from_numpy(X).float()[None], torch.from_numpy(X).float()[None])
        print(f"  loss_pose_rel({name},{name}) = {float(l):.2e} (expect ~0)")
    l_ab, ex = P.loss_pose_rel(torch.from_numpy(A).float()[None], torch.from_numpy(B4).float()[None])
    print(f"  loss_pose_rel(A,B4) = {float(l_ab):.4f}  {ex} (the actual supervision gap)")

    tap = P.split_tap_feats(feats, P.TAP_LAYERS)
    feats_shared = {l: t[:, SLOTS].detach() for l, t in tap.items()}
    conf4 = preds["depth_conf"][:, SLOTS].float()
    cache = {"feats": feats_shared, "conf4": conf4, "patch_hw": (H // P.PATCH_SIZE, W // P.PATCH_SIZE)}
    head_norm = teacher.model.head.norm
    pmask = torch.ones(1, 4, (H // P.PATCH_SIZE) * (W // P.PATCH_SIZE), device="cuda")
    l0, _ = P.loss_maskdistill(head_norm, cache, {l: t.clone() for l, t in feats_shared.items()}, pmask)
    print(f"  maskdistill(X,X) = {float(l0):.2e} (expect ~0)")
    tap4 = P.split_tap_feats(feats4, P.TAP_LAYERS)
    l1, ex1 = P.loss_maskdistill(head_norm, cache, tap4, pmask)
    print(f"  maskdistill(teacher16ctx, teacher4ctx) = {float(l1):.4f} (real distill gap, {ex1})")


if __name__ == "__main__":
    main()
