#!/usr/bin/env python3
"""Ablation v3: trained vs untrained, AUC + F1.
Conditions: 4-masked / 4-clean / 16-encode-then-4-decode.
F1 computed via TSDF fusion of the 4 student-frame depth maps (per pair).
10 pairs per condition (pairs 0,10,20,...,90) to keep runtime manageable."""
import sys, os, json, time, shutil
sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "diagnostics/free_geometry")
import numpy as np, torch
import common as fg_common
from train_da3_protocol import make_dataset, get_scene_data
from depth_anything_3.test_time_adaption import protocol_v1 as P
from depth_anything_3.bench.utils import compute_pose
from depth_anything_3.utils.geometry import as_homogeneous, affine_inverse
from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri

SCENES = ["09c1414f1b", "578511c8a9"]
PAIRS_TO_EVAL = list(range(0, 100, 10))  # 10 pairs: 0,10,20,...,90
CONDITIONS = ["mask4", "clean4", "teach16"]
MODELS = ["tr", "un"]

def decode_full(cam_dec, cam_tok, H, W):
    """Decode 9-dim encoding -> (w2c ext [S,3,4], intrinsics [S,3,3])."""
    enc = cam_dec(cam_tok.float())  # [B,S,9]
    c2w, ixt = pose_encoding_to_extri_intri(enc, (H, W))
    w2c = affine_inverse(c2w)
    return w2c[0].cpu().numpy()[:, :3, :], ixt[0].cpu().numpy()

def forward_condition(model, im, slots=None):
    """Returns (ext [S,3,4], intr [S,3,3], depth [S,H,W]).
    If slots is not None: im has 16 frames, decode only slots."""
    from depth_anything_3.test_time_adaption.protocol_v1 import \
        backbone_tapped_forward, head_feats, TAP_UNION, TAP_LAYERS
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        feats, H, W = backbone_tapped_forward(model.da3, im, TAP_UNION, "first")
    if slots is not None:
        cam_tok = feats[-1][1][:, slots]
        hf = head_feats(feats, TAP_LAYERS)
        hf = [(f[:, slots], c[:, slots]) for f, c in hf]
    else:
        cam_tok = feats[-1][1]
        hf = head_feats(feats, TAP_LAYERS)
    ext, intr = decode_full(model.da3.model.cam_dec, cam_tok, H, W)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        preds = model.da3.model.forward_head_only(
            hf, H=H, W=W, process_camera=False, process_sky=False)
    dep = preds["depth"]
    if dep.dim() == 5:
        dep = dep[..., 0]
    dep = dep.squeeze(0).float().cpu().numpy()  # [S,H,W]
    return ext, intr, dep

def eval_f1(scene, ds_obj, depth, ext, intr, gt_ext, gt_intr, img_files, tag, tmp_root):
    """Write npz, fuse, evaluate F1."""
    work = os.path.join(tmp_root, tag)
    npz_dir = os.path.join(work, "exports", "mini_npz")
    os.makedirs(npz_dir, exist_ok=True)
    np.savez_compressed(os.path.join(npz_dir, "results.npz"),
                        depth=np.round(depth, 8), extrinsics=ext, intrinsics=intr)
    np.savez_compressed(os.path.join(work, "exports", "gt_meta.npz"),
                        extrinsics=gt_ext, intrinsics=gt_intr,
                        image_files=np.array(img_files, dtype=object))
    fuse = os.path.join(work, "exports", "fuse", "pcd.ply")
    os.makedirs(os.path.dirname(fuse), exist_ok=True)
    try:
        ds_obj.fuse3d(scene, os.path.join(npz_dir, "results.npz"), fuse, "recon_unposed")
        r = ds_obj.eval3d(scene, fuse)
        return float(r.get("fscore", 0)), float(r.get("precision", 0)), float(r.get("recall", 0))
    except Exception as e:
        print(f"    F1 error ({tag}): {e}", flush=True)
        return 0.0, 0.0, 0.0
    finally:
        shutil.rmtree(work, ignore_errors=True)

def run():
    fg_common.set_dataset("scannetpp")
    ds_obj = make_dataset("scannetpp")
    tmp_root = "/tmp/ablation_f1"
    os.makedirs(tmp_root, exist_ok=True)
    all_results = {}

    for scene in SCENES:
        sd = get_scene_data(scene)
        files = list(sd.image_files)
        gt_ext_all = np.asarray(sd.extrinsics)
        gt_intr_all = np.asarray(sd.intrinsics)
        proto = P.build_scene_protocol(files, scene, dataset="scannetpp",
                                        n_train=100, n_shared=4, teacher_N=16)
        pairs = proto["train_pairs"]

        # load models
        ckpt_dir = f"workspace/spp100smooth_ablation/{scene}"
        ckpt = os.path.join(ckpt_dir, "ckpts", scene, "c2m_final_lora.pt")
        if not os.path.exists(ckpt.replace(".pt", "_peft")):
            print(f"[{scene}] retraining...", flush=True)
            os.system(
                f"/root/miniconda3/envs/da3/bin/python scripts/train_pw0_accum.py "
                f"--dataset scannetpp --scene {scene} --vggt_sync --half_mode vggt --camrel "
                f"--n_train 100 --updates 100 --accum 1 "
                f"--manifest artifacts/diagnostics/final_protocol/scannetpp/scene_manifest.json "
                f"--out {ckpt_dir} > logs/ablation_train_{scene}.log 2>&1")

        trained = P.create_student("model_weights/DA3-GIANT-1.1", train_camera_token=False)
        P.reset_lora_(trained)
        trained.load_lora_weights(ckpt)
        trained.to("cuda").eval()
        untrained = P.create_student("model_weights/DA3-GIANT-1.1", train_camera_token=False)
        P.reset_lora_(untrained)
        untrained.to("cuda").eval()

        sr = {}
        for pi in PAIRS_TO_EVAL:
            pair = pairs[pi]
            tf, sf = pair["teacher_frames"], pair["student_frames"]
            slots = [tf.index(f) for f in sf]
            gt4 = gt_ext_all[sf][:, :3, :].astype(np.float32)
            gt_intr4 = gt_intr_all[sf].astype(np.float32)
            img4_files = [files[i] for i in sf]

            im16 = P.load_images_da3([files[i] for i in tf]).unsqueeze(0).to("cuda")
            im4 = im16[0, slots].unsqueeze(0)
            ph, pw = im4.shape[-2]//14, im4.shape[-1]//14
            gen = torch.Generator(device="cuda").manual_seed(
                P.stable_seed("mask", scene, 0, pi, 0))
            im4m, _ = P.mask_image_blocks(im4, 0.5, (ph, pw), gen)

            for model_tag, model in [("tr", trained), ("un", untrained)]:
                for cond, inp, sl in [("mask4", im4m, None),
                                        ("clean4", im4, None),
                                        ("teach16", im16, slots)]:
                    key = f"{model_tag}_{cond}"
                    with torch.no_grad():
                        ext, intr, dep = forward_condition(model, inp, sl)
                    # AUC
                    a = float(compute_pose(
                        as_homogeneous(torch.from_numpy(ext).float()),
                        as_homogeneous(torch.from_numpy(gt4).float())).auc03)
                    # F1
                    f1, prec, rec = eval_f1(scene, ds_obj, dep, ext, intr,
                                             gt4, gt_intr4, img4_files,
                                             f"{key}_p{pi}", tmp_root)
                    sr.setdefault(key, []).append(
                        {"pi": pi, "auc": a, "f1": f1, "prec": prec, "recall": rec})
                    print(f"  [{scene[:12]}] p{pi} {key}: AUC={a:.4f} F1={f1:.4f}", flush=True)

            del im16, im4, im4m

        # summary
        print(f"\n===== [{scene}] SUMMARY (mean over {len(PAIRS_TO_EVAL)} pairs) =====")
        print(f"  {'condition':<16} {'AUC':>8} {'F1':>8} {'prec':>8} {'recall':>8}")
        for mt in MODELS:
            for c in CONDITIONS:
                key = f"{mt}_{c}"
                vals = sr[key]
                print(f"  {key:<16} {np.mean([v['auc'] for v in vals]):>8.4f} "
                      f"{np.mean([v['f1'] for v in vals]):>8.4f} "
                      f"{np.mean([v['prec'] for v in vals]):>8.4f} "
                      f"{np.mean([v['recall'] for v in vals]):>8.4f}")
        all_results[scene] = sr
        del trained, untrained
        torch.cuda.empty_cache()

    json.dump(all_results, open("workspace/ablation_f1_results.json", "w"), indent=1)
    print("\nsaved -> workspace/ablation_f1_results.json")

if __name__ == "__main__":
    run()
