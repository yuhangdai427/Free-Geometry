#!/usr/bin/env python3
"""Inference-only: for each scannetpp scene's 10 manifest pairs, run
teacher (16 clean frames) and student (4 masked frames) through both DA3
and VGGT, evaluate predicted cameras vs GT for the 4 shared frames.
Tests 'longer is better' (more context frames → better pose?)."""
import sys, os, json, time
sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "diagnostics/free_geometry")
import numpy as np, torch
import common as fg_common
from train_da3_protocol import make_dataset, get_scene_data
from depth_anything_3.test_time_adaption import protocol_v1 as P
from depth_anything_3.bench.utils import compute_pose
from depth_anything_3.utils.geometry import as_homogeneous

def rot_auc3(pred_w2c, gt_w2c):
    """AUC@3deg for the 4 shared frames"""
    p = compute_pose(as_homogeneous(torch.from_numpy(pred_w2c).float()),
                     as_homogeneous(torch.from_numpy(gt_w2c).float()))
    return float(p.auc03)

def run():
    fg_common.set_dataset("scannetpp")
    make_dataset("scannetpp")
    man = json.load(open("artifacts/diagnostics/final_protocol/scannetpp/scene_manifest.json"))
    out = {"scenes": {}}

    # load both models
    da3 = P.create_student("model_weights/DA3-GIANT-1.1", train_camera_token=False)
    P.reset_lora_(da3); da3.to("cuda").eval()
    import modeling as M
    vggt = M.load_student(train_camera_token=False)
    vggt.to("cuda").eval()

    for scene in sorted(man["scenes"].keys()):
        sd = get_scene_data(scene)
        files = list(sd.image_files)
        gt_ext = np.asarray(sd.extrinsics)  # [N,4,4] w2c
        pairs = man["scenes"][scene]["train_pairs"]
        scene_res = {"pairs": []}
        t0 = time.time()

        for pi, pair in enumerate(pairs):
            tf, sf = pair["teacher_frames"], pair["student_frames"]
            slots = [tf.index(f) for f in sf]  # student positions in teacher list
            gt4 = gt_ext[sf][:, :3, :].astype(np.float32)  # [4,3,4]

            # --- DA3 teacher (16 clean frames) ---
            im16 = P.load_images_da3([files[i] for i in tf]).unsqueeze(0).to("cuda")
            with torch.no_grad():
                tap, ext16, _ = P.student_forward_c2m(da3, im16)
            da3_teacher_auc = rot_auc3(ext16[0].cpu().numpy()[slots], gt4)

            # --- DA3 student (4 masked frames) ---
            im4 = im16[0, slots].unsqueeze(0)
            ph, pw = im4.shape[-2]//14, im4.shape[-1]//14
            gen = torch.Generator(device="cuda").manual_seed(P.stable_seed("mask", scene, 0, pi, 0))
            im4m, _ = P.mask_image_blocks(im4, 0.5, (ph,pw), gen)
            with torch.no_grad():
                _, ext4, _ = P.student_forward_c2m(da3, im4m)
            da3_student_auc = rot_auc3(ext4[0].cpu().numpy(), gt4)
            del im16, im4, im4m

            # --- VGGT teacher ---
            from vggt.vggt.test_time_adaption.test3r_utils import load_images_for_vggt
            v16 = load_images_for_vggt([files[i] for i in tf]).unsqueeze(0).to("cuda")
            from vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri as pd_v
            with torch.no_grad():
                out_list, psi = vggt.vggt.aggregator(v16)
                enc16 = vggt.vggt.camera_head(out_list)[-1]
            ext16v = pd_v(enc16.float(), (v16.shape[-2], v16.shape[-1]))[0][0].cpu().numpy()  # [16,3,4] w2c
            vg_teacher_auc = rot_auc3(ext16v[slots], gt4)

            # --- VGGT student (4 masked) ---
            v4 = v16[:, slots]
            vph, vpw = v4.shape[-2]//14, v4.shape[-1]//14
            gen2 = torch.Generator(device="cuda").manual_seed(P.stable_seed("mask", scene, 0, pi, 0))
            v4m, _ = P.mask_image_blocks(v4, 0.5, (vph,vpw), gen2)
            with torch.no_grad():
                out4, _ = vggt.vggt.aggregator(v4m)
                enc4 = vggt.vggt.camera_head(out4)[-1]
            ext4v = pd_v(enc4.float(), (v4.shape[-2], v4.shape[-1]))[0][0].cpu().numpy()
            vg_student_auc = rot_auc3(ext4v, gt4)
            del v16, v4, v4m

            scene_res["pairs"].append({
                "pi": pi, "student_frames": sf,
                "da3_teacher": da3_teacher_auc, "da3_student": da3_student_auc,
                "vggt_teacher": vg_teacher_auc, "vggt_student": vg_student_auc,
            })
            print(f"  [{scene[:20]}] p{pi}: DA3 t={da3_teacher_auc:.3f} s={da3_student_auc:.3f} "
                  f"| VGGT t={vg_teacher_auc:.3f} s={vg_student_auc:.3f}", flush=True)

        # summary
        for k in ["da3_teacher","da3_student","vggt_teacher","vggt_student"]:
            scene_res[k+"_mean"] = float(np.mean([p[k] for p in scene_res["pairs"]]))
        out["scenes"][scene] = scene_res
        print(f"[{scene}] done ({time.time()-t0:.0f}s) "
              f"DA3 t/s: {scene_res['da3_teacher_mean']:.3f}/{scene_res['da3_student_mean']:.3f} "
              f"VGGT t/s: {scene_res['vggt_teacher_mean']:.3f}/{scene_res['vggt_student_mean']:.3f}", flush=True)

    json.dump(out, open("workspace/teacher_vs_student.json", "w"), indent=1)
    # grand summary
    print("\n===== GRAND SUMMARY =====")
    for k in ["da3_teacher","da3_student","vggt_teacher","vggt_student"]:
        vals = [s[k+"_mean"] for s in out["scenes"].values()]
        print(f"  {k}: mean={np.mean(vals):.4f}")

if __name__ == "__main__":
    run()
