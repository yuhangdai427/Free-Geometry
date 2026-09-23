#!/usr/bin/env python3
"""Ablation v2: trained vs untrained, with FULL decoder output (cameras + depth)
for F1 computation. 6 conditions × 100 pairs × 2 scenes.

For 16→4 conditions: encode 16 frames (full cross-attention), then slice BOTH
camera tokens AND patch features to the 4 student frames before decoding.
Depth head sees only 4 frames' features, but those features carry 16-frame context."""
import sys, os, json, time, shutil
sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "diagnostics/free_geometry")
import numpy as np, torch
import common as fg_common
from train_da3_protocol import make_dataset, get_scene_data, evaluate_scene
from depth_anything_3.test_time_adaption import protocol_v1 as P
from depth_anything_3.bench.utils import compute_pose
from depth_anything_3.utils.geometry import as_homogeneous

def auc3(pred, gt):
    return float(compute_pose(
        as_homogeneous(torch.from_numpy(pred).float()),
        as_homogeneous(torch.from_numpy(gt).float())).auc03)

def forward_16_decode_4(model, im16, slots):
    """Encode 16 frames, decode ONLY the 4 student frames (cam + depth)."""
    from depth_anything_3.test_time_adaption.protocol_v1 import \
        backbone_tapped_forward, decode_pose_w2c, split_tap_feats, \
        TAP_LAYERS, TAP_UNION, head_feats
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        feats, H, W = backbone_tapped_forward(model.da3, im16, TAP_UNION, "first")
    # camera: only student slots
    cam_tok = feats[-1][1][:, slots]  # [1, 4, 3072]
    with torch.autocast(device_type="cuda", enabled=False):
        ext = decode_pose_w2c(model.da3.model.cam_dec, cam_tok.float(), H, W)
    # depth: slice patch features to student frames, then run head
    feats_4 = [(f[:, slots], c[:, slots]) for f, c in
               [(feats[i][0][:, slots], feats[i][1][:, slots]) for i in range(len(feats))]]
    # rebuild head input from sliced feats
    hf = head_feats(feats, TAP_LAYERS)  # full 16-frame head feats
    hf_4 = [(f[:, slots], c[:, slots]) for f, c in hf]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        preds = model.da3.model.forward_head_only(hf_4, H=H, W=W,
                                                    process_camera=False, process_sky=False)
    depth = preds["depth"]  # [1, 4, H, W, 1] or [1, 4, H, W]
    return ext, depth

def forward_4(model, im4):
    """Standard 4-frame forward (cam + depth)."""
    from depth_anything_3.test_time_adaption.protocol_v1 import \
        backbone_tapped_forward, decode_pose_w2c, TAP_LAYERS, head_feats
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        feats, H, W = backbone_tapped_forward(model.da3, im4, TAP_UNION, "first")
    cam_tok = feats[-1][1]
    with torch.autocast(device_type="cuda", enabled=False):
        ext = decode_pose_w2c(model.da3.model.cam_dec, cam_tok.float(), H, W)
    hf = head_feats(feats, TAP_LAYERS)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        preds = model.da3.model.forward_head_only(hf, H=H, W=W,
                                                    process_camera=False, process_sky=False)
    return ext, preds["depth"]

def run_scene(scene, ckpt_dir, ds_obj, sd):
    files = list(sd.image_files)
    gt_ext = np.asarray(sd.extrinsics)[:, :3, :].astype(np.float32)
    proto = P.build_scene_protocol(files, scene, dataset="scannetpp",
                                     n_train=100, n_shared=4, teacher_N=16)
    pairs = proto["train_pairs"]

    trained = P.create_student("model_weights/DA3-GIANT-1.1", train_camera_token=False)
    P.reset_lora_(trained)
    ckpt = os.path.join(ckpt_dir, scene, "c2m_final_lora.pt")
    trained.load_lora_weights(ckpt)
    trained.to("cuda").eval()

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
            for tag, model in [("tr", trained), ("un", untrained)]:
                # (a/d) 4 masked
                ext, dep = forward_4(model, im4m)
                row[f"{tag}_mask4_auc"] = auc3(ext[0].cpu().numpy(), gt4)
                row[f"{tag}_mask4_dep"] = dep.squeeze().float().cpu().numpy()
                # (b/e) 4 clean
                ext, dep = forward_4(model, im4)
                row[f"{tag}_clean4_auc"] = auc3(ext[0].cpu().numpy(), gt4)
                row[f"{tag}_clean4_dep"] = dep.squeeze().float().cpu().numpy()
                # (c/f) 16 encode → 4 decode
                ext, dep = forward_16_decode_4(model, im16, slots)
                row[f"{tag}_teach16_auc"] = auc3(ext[0].cpu().numpy(), gt4)
                row[f"{tag}_teach16_dep"] = dep.squeeze().float().cpu().numpy()

        results["pairs"].append(row)
        del im16, im4, im4m
        if pi % 20 == 0:
            print(f"  pair {pi}/100", flush=True)

        # compute mini-F1 for this pair (TSDF from 4 depth maps)
        # Use the first pair's result as representative; computing F1 for all
        # 100 pairs × 6 conditions would be too slow
        if pi == 0:
            for cond in ["tr_mask4", "tr_clean4", "tr_teach16",
                          "un_mask4", "un_clean4", "un_teach16"]:
                dep = row[f"{cond}_dep"]  # [4, H, W]
                ext = None
                # get matching camera
                if "mask4" in cond:
                    m2 = trained if cond.startswith("tr") else untrained
                    if "mask" in cond:
                        e2, _ = forward_4(m2, im4m) if pi == 0 else (None, None)
                    else:
                        e2, _ = forward_4(m2, im4) if pi == 0 else (None, None)
                else:
                    m2 = trained if cond.startswith("tr") else untrained
                    e2, _ = forward_16_decode_4(m2, im16, slots) if pi == 0 else (None, None)
                # skip F1 for now (4-frame TSDF too sparse); AUC is the main signal
        # Clean up depth arrays from row to save memory
        for k in list(row.keys()):
            if k.endswith("_dep"):
                del row[k]
        results["pairs"][-1] = row

    for k in ["tr_mask4_auc","tr_clean4_auc","tr_teach16_auc",
              "un_mask4_auc","un_clean4_auc","un_teach16_auc"]:
        results[k.replace("_auc","_mean")] = float(np.mean([p[k] for p in results["pairs"]]))
    return results

if __name__ == "__main__":
    fg_common.set_dataset("scannetpp")
    ds_obj = make_dataset("scannetpp")
    out = {}
    for scene in ["09c1414f1b", "578511c8a9"]:
        ckpt_dir = f"workspace/spp100smooth_ablation"
        ckpt = os.path.join(ckpt_dir, scene, "c2m_final_lora.pt")
        if not os.path.exists(ckpt) and not os.path.exists(ckpt.replace(".pt","_peft")):
            print(f"[{scene}] retraining...", flush=True)
            os.system(
                f"/root/miniconda3/envs/da3/bin/python scripts/train_pw0_accum.py "
                f"--dataset scannetpp --scene {scene} --vggt_sync --half_mode vggt --camrel "
                f"--n_train 100 --updates 100 --accum 1 "
                f"--manifest artifacts/diagnostics/final_protocol/scannetpp/scene_manifest.json "
                f"--out {ckpt_dir}/{scene} > logs/ablation_train_{scene}.log 2>&1")
        sd = get_scene_data(scene)
        print(f"[{scene}] ablation...", flush=True)
        out[scene] = run_scene(scene, ckpt_dir, ds_obj, sd)
        r = out[scene]
        print(f"\n[{scene}] AUC RESULTS (100-pair mean):")
        print(f"  {'condition':<16} {'trained':>8} {'untrained':>10} {'Δ(tr-un)':>8}")
        for cond in ["mask4", "clean4", "teach16"]:
            tr = r[f"tr_{cond}_mean"]
            un = r[f"un_{cond}_mean"]
            print(f"  {cond:<16} {tr:>8.4f} {un:>10.4f} {tr-un:>+8.4f}")
        print(f"  {'tr: 16→4 vs 4m':<16} {r['tr_teach16_mean']:>8.4f} vs {r['tr_mask4_mean']:>8.4f} (Δ={r['tr_teach16_mean']-r['tr_mask4_mean']:+.4f})")
        print(f"  {'un: 16→4 vs 4m':<16} {r['un_teach16_mean']:>8.4f} vs {r['un_mask4_mean']:>8.4f} (Δ={r['un_teach16_mean']-r['un_mask4_mean']:+.4f})")
    json.dump(out, open("workspace/ablation_results.json", "w"), indent=1)
    print("\nsaved -> workspace/ablation_results.json")
