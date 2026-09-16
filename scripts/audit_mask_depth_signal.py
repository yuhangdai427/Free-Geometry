#!/usr/bin/env python3
"""Verify: does the 50% input patch mask create a NON-saturated depth signal?
Compare base-model depth on masked input vs teacher(16v clean) depth at
MASKED vs UNMASKED patch positions (log-depth diff, chess pair 1)."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "diagnostics", "free_geometry"))

import common as fg
from depth_anything_3.test_time_adaption import protocol_v1 as P

fg.set_dataset("7scenes")
man = json.load(open("artifacts/diagnostics/final_protocol/7scenes/scene_manifest.json"))
pair = man["scenes"]["chess"]["train_pairs"][0]
data = fg.get_scene_data("chess")
teacher = P.create_teacher("model_weights/DA3-GIANT-1.1", device="cuda")

imgs16 = P.load_images_da3([data.image_files[i] for i in pair["teacher_frames"]]).unsqueeze(0).cuda()
ph, pw = imgs16.shape[-2] // 14, imgs16.shape[-1] // 14
cache = P.cache_teacher_pair(teacher, imgs16, P.STUDENT_SLOTS, (ph, pw))
imgs4 = imgs16[0, P.STUDENT_SLOTS].unsqueeze(0)  # [1,4,3,H,W]

images4_in, pmask = P.mask_image_blocks(imgs4, 0.5, (ph, pw),
                                        torch.Generator(device="cuda").manual_seed(0))

# base model forward on masked input (== student at step 0), WITH depth head
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    feats, H, W = P.backbone_tapped_forward(teacher, images4_in, P.TAP_LAYERS, "first")
hf = [(f.float(), c.float()) for f, c in P.head_feats(feats, P.TAP_LAYERS)]
with torch.autocast(device_type="cuda", enabled=False):
    preds = teacher.model.forward_head_only(hf, H=H, W=W, process_camera=False, process_sky=False)
depth_s = preds["depth"].float()  # [1,4,H,W]
depth_t = cache["depth4"].cuda().float()

# per-patch log-depth diff, split by masked/unmasked
K = H // ph
ds = torch.log(depth_s.clamp_min(1e-6))
dt = torch.log(depth_t.clamp_min(1e-6))
diff = (ds - dt).abs()  # [1,4,H,W]
patch_diff = torch.nn.functional.avg_pool2d(diff.reshape(4, 1, H, W), kernel_size=K).reshape(4, ph * pw)
m = pmask[0].bool()  # [4,P] True=masked
print(f"masked positions   : mean |dlog| = {patch_diff[:, m[0]].mean():.5f}  p90 = {patch_diff[:, m[0]].quantile(0.9):.5f}")
print(f"unmasked positions : mean |dlog| = {patch_diff[:, ~m[0]].mean():.5f}  p90 = {patch_diff[:, ~m[0]].quantile(0.9):.5f}")
print(f"overall mean |dlog| = {diff.mean():.5f}  (log-space; 0.01 ~= 1% depth error)")

# feature-space residual at masked positions for contrast (maskdistill on clean input)
tap = P.split_tap_feats(feats, P.TAP_LAYERS)
head_norm = teacher.model.head.norm
feats_c = {l: t.detach() for l, t in tap.items()}
l_clean, _ = P.loss_maskdistill(head_norm, cache, feats_c, pmask)
print(f"maskdistill(base masked-input feats vs teacher, masked pos) = {float(l_clean):.4f}")
