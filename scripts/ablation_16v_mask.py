#!/usr/bin/env python3
"""追加消融：student 也看 16 帧（但 student 位置的 4 帧 masked），
对比训练前后。外加 100 帧全场景 benchmark。只跑 09c1414f1b 一个场景。"""
import sys, os, json, time, shutil
sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "diagnostics/free_geometry")
import numpy as np, torch
import common as fg_common
from train_da3_protocol import make_dataset, get_scene_data, evaluate_scene
from depth_anything_3.test_time_adaption import protocol_v1 as P
from depth_anything_3.bench.utils import compute_pose
from depth_anything_3.utils.geometry import as_homogeneous, affine_inverse
from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri

SCENE = "09c1414f1b"
PAIRS = list(range(0, 100, 10))

def decode_full(cam_dec, cam_tok, H, W):
    enc = cam_dec(cam_tok.float())
    c2w, ixt = pose_encoding_to_extri_intri(enc, (H, W))
    w2c = affine_inverse(c2w)
    return w2c[0].detach().cpu().numpy()[:, :3, :], ixt[0].detach().cpu().numpy()

def forward_16(model, im16, mask_slots=None, mask_ratio=0.5, patch_hw=None, gen=None):
    """16-frame forward. If mask_slots given, apply mask ONLY to those frames."""
    from depth_anything_3.test_time_adaption.protocol_v1 import \
        backbone_tapped_forward, head_feats, TAP_UNION, TAP_LAYERS
    im_in = im16.clone()
    if mask_slots is not None:
        # mask only the student frames within the 16-frame input
        for s in mask_slots:
            im_in[0, s] = P.mask_image_blocks(
                im16[0, s:s+1], mask_ratio, patch_hw, gen)[0][0]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        feats, H, W = backbone_tapped_forward(model.da3, im_in, TAP_UNION, "first")
    cam_tok = feats[-1][1]  # [1, 16, 3072]
    ext, intr = decode_full(model.da3.model.cam_dec, cam_tok, H, W)
    hf = head_feats(feats, TAP_LAYERS)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        preds = model.da3.model.forward_head_only(
            hf, H=H, W=W, process_camera=False, process_sky=False)
    dep = preds["depth"]
    if dep.dim() == 5: dep = dep[..., 0]
    dep = dep.squeeze(0).float().detach().cpu().numpy()
    return ext, intr, dep

def eval_f1(scene, ds_obj, depth, ext, intr, gt_ext, gt_intr, img_files, tag, tmp):
    work = os.path.join(tmp, tag)
    npz = os.path.join(work, "exports", "mini_npz")
    os.makedirs(npz, exist_ok=True)
    np.savez_compressed(os.path.join(npz, "results.npz"),
                        depth=np.round(depth, 8), extrinsics=ext, intrinsics=intr)
    np.savez_compressed(os.path.join(work, "exports", "gt_meta.npz"),
                        extrinsics=gt_ext, intrinsics=gt_intr,
                        image_files=np.array(img_files, dtype=object))
    fuse = os.path.join(work, "exports", "fuse", "pcd.ply")
    os.makedirs(os.path.dirname(fuse), exist_ok=True)
    try:
        ds_obj.fuse3d(scene, os.path.join(npz, "results.npz"), fuse, "recon_unposed")
        r = ds_obj.eval3d(scene, fuse)
        return float(r.get("fscore", 0)), float(r.get("precision", 0)), float(r.get("recall", 0))
    except Exception as e:
        print(f"    F1 err ({tag}): {e}", flush=True)
        return 0., 0., 0.
    finally:
        shutil.rmtree(work, ignore_errors=True)

def run():
    fg_common.set_dataset("scannetpp")
    ds_obj = make_dataset("scannetpp")
    sd = get_scene_data(SCENE)
    files = list(sd.image_files)
    N = len(files)
    gt_ext_all = np.asarray(sd.extrinsics)
    gt_intr_all = np.asarray(sd.intrinsics)
    proto = P.build_scene_protocol(files, SCENE, dataset="scannetpp",
                                    n_train=100, n_shared=4, teacher_N=16)
    pairs = proto["train_pairs"]
    tmp = "/tmp/abl16"
    os.makedirs(tmp, exist_ok=True)

    # load models
    ckpt_dir = f"workspace/spp100smooth_ablation/{SCENE}"
    trained = P.create_student("model_weights/DA3-GIANT-1.1", train_camera_token=False)
    P.reset_lora_(trained)
    trained.load_lora_weights(os.path.join(ckpt_dir, "ckpts", SCENE, "c2m_final_lora.pt"))
    trained.to("cuda").eval()
    untrained = P.create_student("model_weights/DA3-GIANT-1.1", train_camera_token=False)
    P.reset_lora_(untrained)
    untrained.to("cuda").eval()

    results = {}
    for pi in PAIRS:
        pair = pairs[pi]
        tf, sf = pair["teacher_frames"], pair["student_frames"]
        slots = [tf.index(f) for f in sf]
        gt16 = gt_ext_all[tf][:, :3, :].astype(np.float32)
        gt_intr16 = gt_intr_all[tf].astype(np.float32)
        img16_files = [files[i] for i in tf]
        im16 = P.load_images_da3([files[i] for i in tf]).unsqueeze(0).to("cuda")
        ph, pw = im16.shape[-2]//14, im16.shape[-1]//14

        for mtag, model in [("tr", trained), ("un", untrained)]:
            # condition A: 16 clean (all clean, like teacher)
            ext, intr, dep = forward_16(model, im16)
            a16 = float(compute_pose(
                as_homogeneous(torch.from_numpy(ext).float()),
                as_homogeneous(torch.from_numpy(gt16).float())).auc03)
            f16, p16, r16 = eval_f1(SCENE, ds_obj, dep, ext, intr, gt16, gt_intr16,
                                      img16_files, f"{mtag}_clean16_p{pi}", tmp)
            # condition B: 16 ALL frames masked (50% block mask on every frame)
            gen = torch.Generator(device="cuda").manual_seed(
                P.stable_seed("mask", SCENE, 0, pi, 0))
            im16m, _ = P.mask_image_blocks(im16, 0.5, (ph, pw), gen)
            ext2, intr2, dep2 = forward_16(model, im16m)
            # AUC on student slots only (masked frames)
            a16m = float(compute_pose(
                as_homogeneous(torch.from_numpy(ext2[slots]).float()),
                as_homogeneous(torch.from_numpy(gt16[slots]).float())).auc03)
            # F1 from all 16 frames' depth (all were masked, cross-attn recovers)
            f16m, p16m, r16m = eval_f1(SCENE, ds_obj, dep2, ext2, intr2, gt16,
                                         gt_intr16, img16_files, f"{mtag}_mask16_p{pi}", tmp)
            del im16m
            results.setdefault(f"{mtag}_clean16", []).append(
                {"pi": pi, "auc": a16, "f1": f16, "prec": p16, "rec": r16})
            results.setdefault(f"{mtag}_mask16", []).append(
                {"pi": pi, "auc_stu": a16m, "f1": f16m, "prec": p16m, "rec": r16m})
            print(f"  p{pi} {mtag} clean16: AUC={a16:.4f} F1={f16:.4f} | "
                  f"mask16(stu): AUC={a16m:.4f} F1={f16m:.4f}", flush=True)
        del im16

    # summary
    print(f"\n===== [{SCENE}] 16-frame conditions =====")
    for key in sorted(results):
        vals = results[key]
        auc_k = "auc_stu" if "mask16" in key else "auc"
        print(f"  {key:<16} AUC={np.mean([v[auc_k] for v in vals]):.4f}  "
              f"F1={np.mean([v['f1'] for v in vals]):.4f}  "
              f"prec={np.mean([v['prec'] for v in vals]):.4f}  "
              f"rec={np.mean([v['rec'] for v in vals]):.4f}")

    # benchmark: manifest's standard eval frames (100 random, seed-fixed)
    man = json.load(open("artifacts/diagnostics/final_protocol/scannetpp/scene_manifest.json"))
    bench_frames = man["scenes"][SCENE]["eval32_frames"]
    print(f"\n===== [{SCENE}] Benchmark ({len(bench_frames)} manifest eval frames) =====")
    for mtag, model in [("trained", trained), ("untrained", untrained)]:
        ev = evaluate_scene(model, sd, bench_frames, scene=SCENE,
                            dataset_obj=ds_obj,
                            export_dir=f"{tmp}/bench_{mtag}")
        print(f"  {mtag}: AUC={ev['auc03']:.4f}  F1={ev['recon_fscore']:.4f}  "
              f"abs_rel={ev.get('abs_rel', 0):.4f}", flush=True)

    json.dump(results, open("workspace/ablation_16v_results.json", "w"), indent=1)
    print("saved -> workspace/ablation_16v_results.json")

if __name__ == "__main__":
    run()
