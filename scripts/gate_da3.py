"""Probe-pair residual probe for the GT-free per-scene acceptance gate.

For one completed TTA run (workspace dir with ckpts/<scene>/c2m_final_lora.pt),
measure, on the scene's 2 held-out PROBE pairs (never trained on):
  - md:  feature residual (maskdistill form, all positions, vs frozen teacher)
  - rkd: pose-structure residual (camera-center RKD, vs frozen teacher)
  - cp:  gauge-coupling residual (depth/pose coupling, vs frozen teacher)
for BOTH the baseline student (fresh LoRA = baseline model) and the adapted
student (saved LoRA). The gate premise: a harmful adaptation moves the student
AWAY from its teacher on held-out pairs (residual increases vs baseline).

Usage:
  python3 scripts/gate_da3.py --run workspace/da3_eth3d_2stage --dataset eth3d
Emits <run>/gate_residuals.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "diagnostics", "free_geometry"))

import common as fg_common
from depth_anything_3.test_time_adaption import protocol_v1 as P


@torch.no_grad()
def allv_stats(student, image_files):
    """GT-free long-context drift signature over the FULL eval-frame inference:
    couple_stat = log(RMS of camera centers) - log(mean depth)  (gauge coupling),
    plus center RMS and mean depth separately (scale drift). The 4-frame probe
    residuals cannot see 100-view drift; this runs the deployment-shape
    inference itself. Returns dict(couple, center_rms, mean_depth)."""
    student.eval()
    pred = student.da3.inference(
        image=list(image_files),
        process_res=P.PROCESS_RES,
        process_res_method="upper_bound_resize",
        ref_view_strategy="first",
        export_dir=None,
        export_format="mini_npz",
    )
    depth = np.asarray(pred.depth, dtype=np.float32)          # [N,H,W]
    ext = np.asarray(pred.extrinsics, dtype=np.float32)       # [N,4,4] w2c
    R = ext[:, :3, :3]
    t = ext[:, :3, 3]
    centers = -np.einsum("nij,nj->ni", R.transpose(0, 2, 1), t)  # [N,3]
    center_rms = float(np.sqrt(((centers - centers.mean(0, keepdims=True)) ** 2).sum(-1).mean()))
    valid = np.isfinite(depth) & (depth > 0)
    mean_depth = float(depth[valid].mean())
    couple = float(np.log(max(center_rms, 1e-8)) - np.log(max(mean_depth, 1e-8)))
    return {"couple": couple, "center_rms": center_rms, "mean_depth": mean_depth}


def scene_residuals(student, caches, images, patch_hw, device):
    """Mean (md, rkd, cp) residuals of the student over the given probe pairs."""
    md_s, rkd_s, cp_s = [], [], []
    for cache, imgs in zip(caches, images):
        images4 = imgs.unsqueeze(0).to(device)
        tap, ext_s, depth_s = P.student_forward_c2m(student, images4, with_depth=True)
        c_gpu = {
            "feats": {l: t.to(device) for l, t in cache["feats"].items()},
            "conf4": cache["conf4"].to(device),
            "patch_hw": patch_hw,
        }
        pmask = torch.ones(1, images4.shape[1], patch_hw[0] * patch_hw[1], device=device)
        md, _ = P.loss_maskdistill(student.da3.model.head.norm, c_gpu, tap, pmask)
        ext_t = cache["ext4"].to(device)
        rkd, _ = P.loss_rkd_shared_pose_huber_w2c(ext_s, ext_t)
        cp, _ = P.loss_couple_w2c(ext_s, depth_s, ext_t,
                                  cache["depth4"].to(device), cache["conf4"].to(device))
        md_s.append(float(md))
        rkd_s.append(float(rkd))
        cp_s.append(float(cp))
    n = max(1, len(caches))
    return {"md": sum(md_s) / n, "rkd": sum(rkd_s) / n, "cp": sum(cp_s) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model_name", default="model_weights/DA3-GIANT-1.1")
    ap.add_argument("--scenes", nargs="*", default=None)
    args = ap.parse_args()

    device = "cuda"
    summary = json.load(open(os.path.join(args.run, "smoke_summary.json")))
    scenes = args.scenes or list(summary["scenes"].keys())

    fg_common.set_dataset(args.dataset)

    print("Loading teacher + student ...")
    teacher = P.create_teacher(args.model_name)
    student = P.create_student(args.model_name)

    out = {}
    for scene in scenes:
        proto = json.load(open(os.path.join(args.run, "ckpts", scene, "protocol.json")))
        probe_pairs = proto["probe_pairs"]
        image_files = list(fg_common.get_scene_data(scene).image_files)

        teacher.to(device)
        caches, imgs_cpu = [], []
        for pair in probe_pairs:
            imgs = P.load_images_da3([image_files[i] for i in pair["teacher_frames"]])
            imgs = imgs.unsqueeze(0).to(device)
            ph, pw = imgs.shape[-2] // P.PATCH_SIZE, imgs.shape[-1] // P.PATCH_SIZE
            slots = list(range(0, 2 * len(pair["student_frames"]), 2))
            caches.append(P.cache_teacher_pair(teacher, imgs, slots, (ph, pw)))
            imgs_cpu.append(imgs[0, slots].cpu().contiguous())
            del imgs
        teacher.to("cpu")
        torch.cuda.empty_cache()
        patch_hw = caches[0]["patch_hw"]

        student.to(device)
        student.eval()
        eval_files = [image_files[i] for i in proto["eval_frames"]]
        with torch.no_grad():
            P.reset_lora_(student)
            base = scene_residuals(student, caches, imgs_cpu, patch_hw, device)
            drift_base = allv_stats(student, eval_files)
            student.load_lora_weights(os.path.join(args.run, "ckpts", scene, "c2m_final_lora.pt"))
            adapt = scene_residuals(student, caches, imgs_cpu, patch_hw, device)
            drift_adapt = allv_stats(student, eval_files)
        student.to("cpu")
        out[scene] = {"baseline": base, "adapted": adapt,
                      "d_md": adapt["md"] - base["md"],
                      "d_rkd": adapt["rkd"] - base["rkd"],
                      "d_cp": adapt["cp"] - base["cp"],
                      "drift_base": drift_base, "drift_adapt": drift_adapt,
                      "d_couple_allv": drift_adapt["couple"] - drift_base["couple"],
                      "center_ratio": drift_adapt["center_rms"] / max(drift_base["center_rms"], 1e-8),
                      "depth_ratio": drift_adapt["mean_depth"] / max(drift_base["mean_depth"], 1e-8)}
        print(f"[{scene}] d_md={out[scene]['d_md']:+.4f} d_rkd={out[scene]['d_rkd']:+.4f} "
              f"d_cp={out[scene]['d_cp']:+.4f} | d_couple_allv={out[scene]['d_couple_allv']:+.4f} "
              f"cratio={out[scene]['center_ratio']:.4f} dratio={out[scene]['depth_ratio']:.4f}",
              flush=True)

    path = os.path.join(args.run, "gate_residuals.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
