#!/usr/bin/env python3
"""Evaluate the VGGT feature-loss-only (cosw1, n_train=5) ckpts AND the
zero-LoRA baseline on eth3d scenes, metric-identical to the DA3
evaluate_scene path: full-scene inference -> pose AUC@3 (compute_pose, w2c),
one-LS-scale abs_rel/delta1.25 (GT-valid px), TSDF recon F1/CD via
dataset_obj.fuse3d/eval3d from the mini_npz export.

Usage: python scripts/eval_vggt_cosw1.py --scenes courtyard facade
"""
import argparse, json, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "src"))
sys.path.insert(0, os.path.join(ROOT, "..", "src", "vggt"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "..", "diagnostics", "free_geometry"))

import common
from common import get_scene_data, gt_ixt_raw, load_gt_depth
import modeling as M
from train_arms import infer_eval32, save_eval_npz
from train_da3_protocol import make_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=["courtyard", "facade"])
    ap.add_argument("--run_root", default="workspace/vggtcosw1_eval")
    ap.add_argument("--dataset", default="eth3d")
    ap.add_argument("--arm", default=None,
                    help="evaluate train_arms ckpts at "
                         "{run_root}/ckpts/{scene}/{arm}/step{N}_lora.pt "
                         "instead of the cosw1 mirror layout")
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--skip_baseline", action="store_true",
                    help="evaluate ONLY the TTA arm (baseline evaluated once "
                         "per dataset elsewhere)")
    ap.add_argument("--manifest", default=None,
                    help="use the manifest's eval32_frames instead of all "
                         "scene frames (required for large scenes)")
    args = ap.parse_args()
    os.makedirs(args.run_root, exist_ok=True)

    common.set_dataset(args.dataset)
    ds_obj = make_dataset(args.dataset)
    from depth_anything_3.bench.utils import compute_pose
    from depth_anything_3.utils.geometry import as_homogeneous

    results = {}
    for scene in args.scenes:
        scene_data = get_scene_data(scene)
        files = list(scene_data.image_files)
        frames = list(range(len(files)))
        if args.manifest:
            import json as _j
            frames = list(_j.load(open(args.manifest))["scenes"][scene]["eval32_frames"])
        gt_ext_arr = np.asarray(scene_data.extrinsics)[frames]
        gt = None
        try:
            _gtf = scene_data.aux["gt_depth_files"]
            if _gtf and isinstance(_gtf[0], str):
                gt = None  # loaded after inference (needs depth hw)
        except Exception:
            gt = None
        arms = [("A0_baseline", None)] if not args.skip_baseline else []
        arms.append(("TTA",
                     (os.path.join(args.run_root, "ckpts", scene, args.arm,
                                   f"step{args.step}_lora.pt") if args.arm
                      else os.path.join("workspace", f"vggtcosw1_grad_{scene}",
                                        "ckpts", scene, "vggt_final_lora.pt"))))
        for exp, ckpt in arms:
            student = M.load_student()
            if ckpt:
                student.load_lora_weights(ckpt)
            depth, ext, intr, images, conf = infer_eval32(student, scene_data, frames)
            pose = compute_pose(
                as_homogeneous(torch.from_numpy(ext).float()),
                as_homogeneous(torch.from_numpy(gt_ext_arr).float()))
            out = {"n_eval_frames": len(frames), "auc03": float(pose.auc03),
                   "auc30": float(pose.auc30)}
            # depth metrics: one LS scale per scene, GT-valid px (DA3 recipe)
            try:
                _gtf = scene_data.aux["gt_depth_files"]
                if _gtf and isinstance(_gtf[0], str):
                    g = np.stack([load_gt_depth(_gtf[i], depth.shape[-2:]) for i in frames])
                    omega = np.isfinite(g) & (g > 0) & np.isfinite(depth) & (depth > 0)
                    if omega.sum() > 0:
                        p, gg = depth[omega].astype(np.float64), g[omega].astype(np.float64)
                        s = float((p * gg).sum() / max((p * p).sum(), 1e-12))
                        ps = p * s
                        out.update(abs_rel=float(np.mean(np.abs(ps - gg) / gg)),
                                   delta125=float(np.mean(np.maximum(ps / gg, gg / ps) < 1.25)),
                                   ls_scale=s)
            except Exception as e:
                out["depth_metric_error"] = str(e)
            # recon F1 on the same fuse3d/eval3d path as the DA3 numbers
            save_eval_npz(args.run_root, exp, scene, depth, ext, intr,
                          gt_ext_arr, gt_ixt_raw(scene_data, frames),
                          [files[i] for i in frames], frames,
                          conf=(conf if conf is not None else None),
                          dataset=args.dataset)
            # DTU fuse3d/eval3d read mask_files from gt_meta (same as the DA3
            # evaluate_scene recipe); save_eval_npz doesn't write them.
            aux = scene_data.aux
            if getattr(aux, "get", None) and aux.get("mask_files") is not None:
                meta_path = os.path.join(args.run_root, "eval32", exp, "model_results",
                                         args.dataset, scene, "unposed", "exports",
                                         "gt_meta.npz")
                _m = dict(np.load(meta_path, allow_pickle=True))
                _m["mask_files"] = np.array([aux["mask_files"][i] for i in frames],
                                            dtype=object)
                np.savez_compressed(meta_path, **_m)
            result_path = os.path.join(args.run_root, "eval32", exp, "model_results",
                                      args.dataset, scene, "unposed", "exports",
                                      "mini_npz", "results.npz")
            fuse_path = os.path.join(os.path.dirname(os.path.dirname(result_path)),
                                     "fuse", "pcd.ply")
            os.makedirs(os.path.dirname(fuse_path), exist_ok=True)
            try:
                ds_obj.fuse3d(scene, result_path, fuse_path, "recon_unposed")
                recon = ds_obj.eval3d(scene, fuse_path)
                out.update({f"recon_{k}": v for k, v in recon.items()
                            if isinstance(v, (int, float))})
            except Exception as e:
                import traceback
                out["recon_error"] = str(e) + " || " + "|".join(
                    traceback.format_exc().strip().splitlines()[-3:])
            print(f"[{scene}] {exp}: AUC@3={out['auc03']:.4f} "
                  f"F1={out.get('recon_fscore', float('nan')):.4f} "
                  f"abs_rel={out.get('abs_rel', float('nan')):.4f}", flush=True)
            results.setdefault(scene, {})[exp] = out
            del student
            torch.cuda.empty_cache()

    dst = os.path.join(args.run_root, "vggt_cosw1_eval.json")
    json.dump(results, open(dst, "w"), indent=1)
    print("->", dst)


if __name__ == "__main__":
    main()
