#!/usr/bin/env python3
"""Ablation: trained vs untrained student on TRAINING PAIRS.
For scenes [09c1414f1b, 578511c8a9], 100 pairs each:
  (a) trained + 4 masked    → AUC per pair
  (b) trained + 4 clean     → AUC
  (c) trained + 16 teacher  → AUC at student slots
  (d) untrained + 4 masked  → AUC
  (e) untrained + 4 clean   → AUC
  (f) untrained + 16 teacher → AUC at student slots"""
import sys, os, json, time
sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "diagnostics/free_geometry")
import numpy as np, torch
import common as fg_common
from train_da3_protocol import make_dataset, get_scene_data
from depth_anything_3.test_time_adaption import protocol_v1 as P
from depth_anything_3.bench.utils import compute_pose
from depth_anything_3.utils.geometry import as_homogeneous

def auc3(pred, gt):
    return float(compute_pose(
        as_homogeneous(torch.from_numpy(pred).float()),
        as_homogeneous(torch.from_numpy(gt).float())).auc03)

def run_scene(scene, ckpt_dir):
    fg_common.set_dataset("scannetpp"); make_dataset("scannetpp")
    sd = get_scene_data(scene)
    files = list(sd.image_files)
    gt_ext = np.asarray(sd.extrinsics)[:, :3, :].astype(np.float32)

    man = json.load(open("artifacts/diagnostics/final_protocol/scannetpp/scene_manifest.json"))
    # regenerate 100 pairs (same seed as training)
    proto = P.build_scene_protocol(files, scene, dataset="scannetpp",
                                     n_train=100, n_shared=4, teacher_N=16)
    pairs = proto["train_pairs"]

    # load trained student
    trained = P.create_student("model_weights/DA3-GIANT-1.1", train_camera_token=False)
    P.reset_lora_(trained)
    ckpt = os.path.join(ckpt_dir, scene, "c2m_final_lora.pt")
    ckpt_p = ckpt.replace(".pt", "_peft")
    if os.path.exists(ckpt_p):
        trained.load_lora_weights(ckpt)
    else:
        raise FileNotFoundError(f"no ckpt at {ckpt} or {ckpt_p}")
    trained.to("cuda").eval()

    # untrained student (zero LoRA)
    untrained = P.create_student("model_weights/DA3-GIANT-1.1", train_camera_token=False)
    P.reset_lora_(untrained)
    untrained.to("cuda").eval()

    results = {"pairs": []}
    for pi, pair in enumerate(pairs):
        tf, sf = pair["teacher_frames"], pair["student_frames"]
        slots = [tf.index(f) for f in sf]
        gt4 = gt_ext[sf]

        im16 = P.load_images_da3([files[i] for i in tf]).unsqueeze(0).to("cuda")
        im4 = im16[0, slots].unsqueeze(0)
        ph, pw = im4.shape[-2]//14, im4.shape[-1]//14
        gen = torch.Generator(device="cuda").manual_seed(P.stable_seed("mask", scene, 0, pi, 0))
        im4m, _ = P.mask_image_blocks(im4, 0.5, (ph,pw), gen)

        row = {"pi": pi}
        with torch.no_grad():
            # (a) trained + 4 masked
            _, ext, _ = P.student_forward_c2m(trained, im4m)
            row["tr_mask4"] = auc3(ext[0].cpu().numpy(), gt4)
            # (b) trained + 4 clean
            _, ext, _ = P.student_forward_c2m(trained, im4)
            row["tr_clean4"] = auc3(ext[0].cpu().numpy(), gt4)
            # (c) trained: 16帧 encode → 只取 student 4帧的 camera token decode
            from depth_anything_3.test_time_adaption.protocol_v1 import \
                backbone_tapped_forward, decode_pose_w2c
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                feats16, H16, W16 = backbone_tapped_forward(
                    trained.da3, im16, P.TAP_LAYERS, "first")
            cam_tok_all = feats16[-1][1]  # [1, 16, 3072] all frames' camera tokens
            cam_tok_4 = cam_tok_all[:, slots]  # ← 只取 student 4帧的 token
            with torch.autocast(device_type="cuda", enabled=False):
                ext4_from16 = decode_pose_w2c(
                    trained.da3.model.cam_dec, cam_tok_4.float(), H16, W16)
            row["tr_teach16"] = auc3(ext4_from16[0].cpu().numpy(), gt4)
            # (d) untrained + 4 masked
            _, ext, _ = P.student_forward_c2m(untrained, im4m)
            row["un_mask4"] = auc3(ext[0].cpu().numpy(), gt4)
            # (e) untrained + 4 clean
            _, ext, _ = P.student_forward_c2m(untrained, im4)
            row["un_clean4"] = auc3(ext[0].cpu().numpy(), gt4)
            # (f) untrained: 16帧 encode → 只取 student 4帧 decode
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                feats16u, H16u, W16u = backbone_tapped_forward(
                    untrained.da3, im16, P.TAP_LAYERS, "first")
            cam_tok_4u = feats16u[-1][1][:, slots]  # ← only student slots
            with torch.autocast(device_type="cuda", enabled=False):
                ext4u = decode_pose_w2c(
                    untrained.da3.model.cam_dec, cam_tok_4u.float(), H16u, W16u)
            row["un_teach16"] = auc3(ext4u[0].cpu().numpy(), gt4)

        results["pairs"].append(row)
        del im16, im4, im4m
        if pi % 20 == 0:
            print(f"  pair {pi}/100 done", flush=True)

    for k in ["tr_mask4","tr_clean4","tr_teach16","un_mask4","un_clean4","un_teach16"]:
        results[k+"_mean"] = float(np.mean([p[k] for p in results["pairs"]]))
    return results

if __name__ == "__main__":
    out = {}
    for scene in ["09c1414f1b", "578511c8a9"]:
        ckpt_dir = f"workspace/spp100smooth_ablation/{scene}"
        # retrain with ckpt saved
        print(f"[{scene}] retraining with ckpt saved...", flush=True)
        os.system(
            f"/root/miniconda3/envs/da3/bin/python scripts/train_pw0_accum.py "
            f"--dataset scannetpp --scene {scene} --vggt_sync --half_mode vggt --camrel "
            f"--n_train 100 --updates 100 --accum 1 "
            f"--manifest artifacts/diagnostics/final_protocol/scannetpp/scene_manifest.json "
            f"--out {ckpt_dir} > logs/ablation_train_{scene}.log 2>&1")
        print(f"[{scene}] running ablation...", flush=True)
        out[scene] = run_scene(scene, ckpt_dir)
        r = out[scene]
        print(f"[{scene}] RESULTS:", flush=True)
        for k in ["tr_mask4","tr_clean4","tr_teach16","un_mask4","un_clean4","un_teach16"]:
            print(f"  {k}: {r[k+'_mean']:.4f}", flush=True)
    json.dump(out, open("workspace/ablation_results.json", "w"), indent=1)
    print("saved -> workspace/ablation_results.json")
