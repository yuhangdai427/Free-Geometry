#!/usr/bin/env python3
"""One manifest pair (eth3d courtyard pair0), SAME frames for both models.
Four camera groups, raw head outputs (9-dim pose enc: T3+quat4+fov2) plus
decoded w2c t:
  DA3-teacher (clean 16 views, student slots [0,2,4,6])
  DA3-student (same weights, zero-LoRA, MASKED 4 views, training mask seed)
  VGGT-teacher / VGGT-student (same recipe)
Full-precision dump -> workspace/camera_raw_compare.json"""
import sys, json
sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "diagnostics/free_geometry")
import torch, numpy as np
import common as fg_common
from train_da3_protocol import make_dataset, get_scene_data
from depth_anything_3.test_time_adaption import protocol_v1 as P
import modeling as M  # vggt diagnostics modeling

scene = "courtyard"
fg_common.set_dataset("eth3d"); make_dataset("eth3d")
sd = get_scene_data(scene); files = list(sd.image_files)
man = json.load(open("artifacts/diagnostics/final_protocol/eth3d/scene_manifest.json"))
pair = man["scenes"][scene]["train_pairs"][0]
tf, s4 = pair["teacher_frames"], pair["student_frames"]
print(f"scene={scene} teacher={tf} student={s4}", flush=True)
SLOTS = [tf.index(f) for f in s4]
out = {"scene": scene, "teacher_frames": tf, "student_frames": s4, "frames_used": s4, "groups": {}}

# ---------------- DA3 ----------------
from depth_anything_3.test_time_adaption.protocol_v1 import backbone_tapped_forward, TAP_LAYERS, TAP_UNION
da3 = P.create_student("model_weights/DA3-GIANT-1.1", train_camera_token=False)
P.reset_lora_(da3); da3.to("cuda").eval()
im16 = P.load_images_da3([files[i] for i in tf]).unsqueeze(0).to("cuda")
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    feats, H, W = backbone_tapped_forward(da3.da3, im16, TAP_UNION, "first")
cam_t16 = feats[-1][1]  # [1,16,3072] last-layer camera tokens
with torch.no_grad():
    enc_t = da3.da3.model.cam_dec(cam_t16.float())[0].cpu().numpy()      # [16,9] raw
# masked student 4
im4 = im16[0, SLOTS].unsqueeze(0)
ph, pw = im4.shape[-2]//14, im4.shape[-1]//14
gen = torch.Generator(device="cuda").manual_seed(P.stable_seed("mask", scene, 0, 0, 0))
im4_in, _ = P.mask_image_blocks(im4, 0.5, (ph, pw), gen)
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    feats_s, _, _ = backbone_tapped_forward(da3.da3, im4_in, TAP_UNION, "first")
with torch.no_grad():
    enc_s = da3.da3.model.cam_dec(feats_s[-1][1].float())[0].cpu().numpy()  # [4,9]
def dec_w2c_t(enc):
    from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
    from depth_anything_3.utils.geometry import affine_inverse
    c2w, _ = pose_encoding_to_extri_intri(torch.from_numpy(enc[None]), (H, W))
    return affine_inverse(c2w)[0].numpy()[:, :3, 3]
out["groups"]["da3_teacher"] = {"enc9": enc_t[SLOTS].tolist(), "t_w2c": dec_w2c_t(enc_t[SLOTS]).tolist()}
out["groups"]["da3_student"] = {"enc9": enc_s.tolist(), "t_w2c": dec_w2c_t(enc_s).tolist()}
del da3, im16, im4, im4_in, feats, feats_s; torch.cuda.empty_cache()

# ---------------- VGGT ----------------
vg = M.load_student(train_camera_token=False)
vg.to("cuda").eval()
from vggt.vggt.test_time_adaption.test3r_utils import load_images_for_vggt
v16 = load_images_for_vggt([files[i] for i in tf]).unsqueeze(0).to("cuda")
with torch.no_grad():
    out_list, psi = vg.vggt.aggregator(v16)
    enc_vt = vg.vggt.camera_head(out_list)[-1][0].float().cpu().numpy()   # [16,9]
v4 = v16[:, SLOTS]
vph, vpw = v4.shape[-2]//14, v4.shape[-1]//14
gen = torch.Generator(device="cuda").manual_seed(P.stable_seed("mask", scene, 0, 0, 0))
v4_in, _ = P.mask_image_blocks(v4, 0.5, (vph, vpw), gen)
with torch.no_grad():
    out_s, _ = vg.vggt.aggregator(v4_in)
    enc_vs = vg.vggt.camera_head(out_s)[-1][0].float().cpu().numpy()
def dec_w2c_v(enc, hh, ww):
    from vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri as pd
    ext, _ = pd(torch.from_numpy(enc[None]), (hh, ww))  # w2c [R|t]
    return ext[0].numpy()[:, :3, 3]
out["groups"]["vggt_teacher"] = {"enc9": enc_vt[SLOTS].tolist(),
                                 "t_w2c": dec_w2c_v(enc_vt[SLOTS], v16.shape[-2], v16.shape[-1]).tolist()}
out["groups"]["vggt_student"] = {"enc9": enc_vs.tolist(),
                                 "t_w2c": dec_w2c_v(enc_vs, v4.shape[-2], v4.shape[-1]).tolist()}
json.dump(out, open("workspace/camera_raw_compare.json", "w"), indent=1)

# console table
for g in ["da3_teacher", "da3_student", "vggt_teacher", "vggt_student"]:
    print(f"\n== {g} (4 views, frames {s4}) ==", flush=True)
    enc = np.array(out["groups"][g]["enc9"]); t = np.array(out["groups"][g]["t_w2c"])
    for k in range(4):
        print(f"  v{k}: enc9={np.round(enc[k],4).tolist()}  t_w2c={np.round(t[k],4).tolist()}", flush=True)
print("\nsaved -> workspace/camera_raw_compare.json", flush=True)
