#!/usr/bin/env python3
"""Per-pair camera quality diagnostic: teacher vs GT rot error, masked
zero-LoRA student vs GT, and initial student-teacher rel-rot residual, for
the spike pairs of eth3d courtyard / delivery_area."""
import sys, os, json
sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "diagnostics/free_geometry")
import torch, numpy as np
import common as fg_common
from train_da3_protocol import make_dataset, get_scene_data
from depth_anything_3.test_time_adaption import protocol_v1 as P

def rot_angle_deg(R):
    tr = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))

def rel_err(w2c_pred, gt_f):
    """rotation angle of inv(gt) @ pred (w2c), deg"""
    E = (np.linalg.inv(gt_f) @ w2c_pred)[:3, :3]
    return rot_angle_deg(E)

fg_common.set_dataset("eth3d")
make_dataset("eth3d")
man = json.load(open("artifacts/diagnostics/final_protocol/eth3d/scene_manifest.json"))

for scene in ["courtyard", "delivery_area"]:
    sd = get_scene_data(scene)
    files = list(sd.image_files)
    gt = np.asarray(sd.extrinsics)  # w2c
    pairs = man["scenes"][scene]["train_pairs"]
    teacher = P.create_teacher("model_weights/DA3-GIANT-1.1").to("cuda")
    student = P.create_student("model_weights/DA3-GIANT-1.1", train_camera_token=False)
    P.reset_lora_(student); student.to("cuda").eval()
    print(f"===== {scene} (N={len(files)}) =====", flush=True)
    print(f"{'pair':>4}{'teachGT°':>10}{'studGT°':>10}{'Lrot0':>9}", flush=True)
    for pi, pr in enumerate(pairs):
        tf = pr["teacher_frames"]
        s4 = [tf[i] for i in [0, 2, 4, 6]]
        im8 = P.load_images_da3([files[i] for i in tf]).unsqueeze(0).to("cuda")
        ph, pw = im8.shape[-2] // 14, im8.shape[-1] // 14
        cache = P.cache_teacher_pair(teacher, im8, [0, 2, 4, 6], (ph, pw))
        ext_t = cache["ext4"].numpy()
        ims4 = im8[0, [0, 2, 4, 6]].unsqueeze(0)
        gen = torch.Generator(device="cuda").manual_seed(P.stable_seed("mask", scene, 0, pi, 0))
        ims4_in, _ = P.mask_image_blocks(ims4, 0.5, (ph, pw), gen)
        torch.manual_seed(P.stable_seed("cf_rng", scene, 0, pi, 0))
        with torch.no_grad():
            _, ext_s_t, _ = P.student_forward_c2m(student, ims4_in)
        ext_s = ext_s_t[0].detach().float().cpu().numpy()
        te = np.mean([rel_err(ext_t[k], gt[f]) for k, f in enumerate(s4)])
        se = np.mean([rel_err(ext_s[k], gt[f]) for k, f in enumerate(s4)])
        Rs, Rt = ext_s[:, :3, :3], ext_t[:, :3, :3]
        lr = np.mean([((Rs[i] @ Rs[j].T - Rt[i] @ Rt[j].T) ** 2).sum()
                      for i in range(4) for j in range(i + 1, 4)])
        print(f"{pi:>4}{te:>10.2f}{se:>10.2f}{lr:>9.4f}", flush=True)
        del im8, ims4, ims4_in, cache
    del teacher, student
    torch.cuda.empty_cache()
print("done", flush=True)
