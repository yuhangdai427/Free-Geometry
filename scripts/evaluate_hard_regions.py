#!/usr/bin/env python3
"""Evaluate VGGT predictions only on textureless or normalized-occlusion pixels.

Both predicted and GT 3D points originate exclusively from the same per-frame
mask.  This deliberately differs from the usual full-scene TSDF metric: no
unmasked mesh/point-cloud surface can contribute to regional F1 or overall.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "src" / "vggt")]
from depth_anything_3.bench.utils import evaluate_3d_reconstruction
from depth_anything_3.utils.pose_align import align_poses_umeyama, apply_umeyama_alignment_to_ext
from scripts.benchmark_vggt_covisibility_010 import (  # noqa: E402
    infer_8v_extract_4v, infer_direct, load_images,
)
from vggt.models.vggt import VGGT
from vggt.vggt.test_time_adaption.covisibility_sequences import _image_from_depth, load_selected_scenes, sequence_calibration
from vggt.vggt.test_time_adaption.models import VGGTStudentModel

STUDENT = [0, 2, 4, 6]
ARMS = ("base_8v", "base_4v", "base_8v_extract_4v", "lora_4v", "lora_8v")
# Match each benchmark dataset's own reconstruction threshold.
THRESHOLDS = {"eth3d": 0.25, "scannetpp": 0.05, "7scenes": 0.05, "hiroom": 0.05}


def source_depths(selection):
    """Map sampled RGB path to its exact matrix depth/calibration record."""
    output = {}
    for line in Path(selection).read_text().splitlines():
        item = json.loads(line); matrix = np.load(item["matrix_path"], allow_pickle=False)
        for depth, intr, ext in zip(matrix["depth_paths"], matrix["intrinsics"], matrix["extrinsics_w2c"]):
            output[(item["dataset"], _image_from_depth(item["dataset"], str(depth)))] = (str(depth), intr.astype(np.float32), ext.astype(np.float32))
    return output


def read_depth(dataset, path):
    if dataset == "eth3d":
        image = cv2.imread(_image_from_depth(dataset, path), cv2.IMREAD_UNCHANGED)
        values = np.fromfile(path, dtype=np.float32)
        depth = values.reshape(image.shape[:2])
        # ETH3D encodes missing laser depth as infinity, not just zero.
        # Never let those samples enter masks, backprojection, or metrics.
        return depth, np.isfinite(depth) & (depth > 0)
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None: raise FileNotFoundError(path)
    if dataset in {"scannetpp", "7scenes"}:
        depth = raw.astype(np.float32) / 1000.0
        valid = np.isfinite(depth) & (depth > 0)
        if dataset == "7scenes":
            valid &= raw != 65535
    elif dataset == "hiroom":
        depth = raw.astype(np.float32) / 65535.0 * 100.0
        valid = np.isfinite(depth) & (depth > 0)
        alias = Path(path).parents[1] / "aliasing_mask" / Path(path).name
        if alias.exists():
            aliasing = cv2.imread(str(alias), cv2.IMREAD_UNCHANGED)
            valid &= aliasing == 0
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return depth, valid


def resize_depth(depth, valid, shape):
    h, w = shape
    return (cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST),
            cv2.resize(valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool))


def texture_score(image_path, shape):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(image.astype(np.float32) / 255.0)[None, None]
    sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
    mag = torch.sqrt(F.conv2d(x, sx, padding=1).square() + F.conv2d(x, sx.transpose(-1, -2), padding=1).square())
    return F.avg_pool2d(mag, 14, 14)[0, 0].numpy()


def texture_mask(image_path, shape, q30):
    patch = texture_score(image_path, shape) < q30
    patch = torch.from_numpy(patch.astype(np.float32))[None, None]
    return F.interpolate(patch, size=shape, mode="nearest")[0, 0].numpy().astype(bool)


@torch.inference_mode()
def occlusion_mask(depths, valids, intrinsics, w2cs, source):
    """True normalized-occlusion pixels for one source in an actual 8V window."""
    dev = depths.device; h, w = depths.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")
    pix = torch.stack((xx.flatten(), yy.flatten(), torch.ones(h*w, device=dev)))
    targets = [i for i in range(len(depths)) if i != source]
    z = depths[source].flatten(); xyz = torch.linalg.solve(intrinsics[source], pix) * z
    world = torch.linalg.inv(w2cs[source]) @ torch.cat((xyz, torch.ones(1, h*w, device=dev)))
    cam = w2cs[targets] @ world.unsqueeze(0); pz = cam[:, 2]
    uv = torch.bmm(intrinsics[targets], cam[:, :3]); u, v = uv[:, 0]/uv[:, 2], uv[:, 1]/uv[:, 2]
    inside = (pz > 0) & (u >= 0) & (u <= w-1) & (v >= 0) & (v <= h-1)
    grid = torch.stack((2*u/max(w-1,1)-1, 2*v/max(h-1,1)-1), -1).view(7, h, w, 2)
    td = F.grid_sample(depths[targets,None], grid, mode="nearest", padding_mode="zeros", align_corners=True)[:,0].reshape(7,-1)
    tv = F.grid_sample(valids[targets,None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=True)[:,0].reshape(7,-1)>0.5
    candidate = inside & tv; testable = valids[source].flatten() & (candidate.sum(0) >= 2)
    seen = candidate & ((td-pz).abs() <= .02*pz)
    return (testable & (seen.sum(0) < 2)).reshape(h,w).cpu().numpy()


def points(depth, intrinsic, w2c, mask, cap=100000):
    y, x = np.where(mask & np.isfinite(depth) & (depth > 0))
    if len(x) > cap:
        take = np.linspace(0, len(x)-1, cap, dtype=int); y, x = y[take], x[take]
    z = depth[y, x]; pix = np.stack((x, y, np.ones_like(x)), axis=0)
    cam = np.linalg.solve(intrinsic, pix) * z
    return (np.linalg.inv(w2c) @ np.vstack((cam, np.ones((1, len(x))))) )[:3].T


def run(args):
    device = torch.device(args.device)
    records = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()][:args.max_windows]
    scenes = {(x.dataset, x.scene): x for x in load_selected_scenes(args.selection, [args.dataset])}
    lookup = source_depths(args.selection)
    report = json.loads(Path(args.selection_report).read_text())
    q30 = report.get("patch_gradient_q30")
    # The earlier 7Scenes/HiRoom manifests recorded the Q4 frame criterion,
    # but not the underlying dataset-wide patch Q30. Reconstruct it from the
    # exact real candidate RGB frames used by this analysis.
    if q30 is None and args.criterion == "texture":
        q30 = float(np.percentile(np.concatenate([
            texture_score(path, (504, 504)).reshape(-1)
            for scene in scenes.values() for path in scene.image_files
        ]), 30))
        print(f"[derived patch Q30] {args.dataset}={q30:.8f}", flush=True)
    base = VGGT.from_pretrained(args.model_name).to(device).eval()
    lora = None
    if any(a.startswith("lora") for a in args.arms):
        student = VGGTStudentModel(model_name=args.model_name, lora_rank=32, lora_alpha=32, lora_layers=list(range(24)))
        student.load_lora_weights(args.lora); lora = student.to(device).eval()._get_vggt_model()
    results = defaultdict(list)
    for record in records:
        scene = scenes[(args.dataset, record["scene"])]
        # Native matrix grid supplies exact real GT geometry for masks and GT points.
        native = [lookup[(args.dataset, path)] for path in record["eight_image_files"]]
        raw = [read_depth(args.dataset, x[0]) for x in native]
        h0, w0 = raw[0][0].shape; scale = min(1.0, 224 / max(h0,w0)); h,w=round(h0*scale),round(w0*scale)
        gt_d, gt_v = zip(*(resize_depth(d,v,(h,w)) for d,v in raw))
        gt_d=np.stack(gt_d); gt_v=np.stack(gt_v)
        # Matrix intrinsics are already expressed in the matrix's max-side
        # grid (the same grid produced just above).  Do not rescale them a
        # second time from native ETH3D/ScanNet++ resolution.
        k=np.stack([x[1] for x in native]); k[:,2,2]=1
        ext=np.stack([x[2] for x in native])
        occ = None
        if args.criterion == "occlusion":
            occ_all = [occlusion_mask(torch.from_numpy(gt_d).to(device), torch.from_numpy(gt_v).to(device), torch.from_numpy(k).to(device), torch.from_numpy(ext).to(device), i) for i in range(8)]
        for arm in args.arms:
            model = lora if arm.startswith("lora") else base
            image_files, gt_ext, _ = sequence_calibration(scene, record, arm)
            paths = record["eight_image_files"] if arm == "base_8v_extract_4v" else image_files
            images = load_images(paths, args.image_size).to(device)
            pred = infer_8v_extract_4v(model, images) if arm == "base_8v_extract_4v" else infer_direct(model, images)
            # All methods are compared on the same four student cameras.  The
            # 8V arms merely condition those predictions on four extra views.
            is_eight_view_arm = arm in {"base_8v", "lora_8v"}
            # Direct 8V outputs retain their original positions.  Four-view
            # arms produce outputs in the locked 0,2,4,6 order instead.
            pred_positions = STUDENT if is_eight_view_arm else list(range(4))
            source_positions = STUDENT
            # Align predicted camera coordinate system and depth scale exactly once per arm/window.
            _, _, scale_pred, aligned_ext = align_poses_umeyama(gt_ext, pred["extrinsics"].cpu().numpy(), return_aligned=True, ransac=True, random_state=42)
            pred_d=pred["depth"].detach().float().cpu().numpy()*scale_pred
            pred_d = np.squeeze(pred_d)
            pred_i=pred["intrinsics"].detach().float().cpu().numpy()
            pred_pts=[]; gt_pts=[]; absrels=[]; mask_pixels=[]
            for pred_i_idx, seq_i in zip(pred_positions, source_positions):
                pd=pred_d[pred_i_idx]; gd,gv=resize_depth(gt_d[seq_i],gt_v[seq_i],pd.shape)
                if args.criterion == "texture": mask=texture_mask(record["eight_image_files"][seq_i], pd.shape, q30)
                else: mask=cv2.resize(occ_all[seq_i].astype(np.uint8),(pd.shape[1],pd.shape[0]),interpolation=cv2.INTER_NEAREST).astype(bool)
                valid=mask & gv & np.isfinite(gd) & (gd > 0) & np.isfinite(pd) & (pd>0); mask_pixels.append(int(valid.sum()))
                if valid.any(): absrels.append(float(np.mean(np.abs(pd[valid]-gd[valid])/np.maximum(gd[valid],1e-6))))
                gt_k = k[seq_i].copy()
                gt_k[0] *= pd.shape[1] / w
                gt_k[1] *= pd.shape[0] / h
                ext_i = seq_i if is_eight_view_arm else pred_i_idx
                gt_pts.append(points(gd, gt_k, gt_ext[ext_i], valid))
                pred_pts.append(points(pd, pred_i[pred_i_idx], aligned_ext[pred_i_idx], valid))
            if not absrels or not any(len(x) for x in pred_pts) or not any(len(x) for x in gt_pts):
                raise RuntimeError(f"{record['scene']}/{arm}: empty masked prediction or GT support")
            metrics=evaluate_3d_reconstruction(np.concatenate(pred_pts),np.concatenate(gt_pts),threshold=THRESHOLDS[args.dataset])
            if args.debug:
                pp, gp = np.concatenate(pred_pts), np.concatenate(gt_pts)
                print(f"[debug] {arm} scale={scale_pred:.6g} depth_pred=({np.nanmin(pred_d):.4g},{np.nanmax(pred_d):.4g}) "
                      f"depth_gt=({np.nanmin(gt_d):.4g},{np.nanmax(gt_d):.4g}) "
                      f"centers_gt={np.linalg.inv(gt_ext)[:,:3,3].mean(0)} centers_pred={np.linalg.inv(aligned_ext)[:,:3,3].mean(0)} "
                      f"points_gt=({gp.min(0)},{gp.max(0)}) points_pred=({pp.min(0)},{pp.max(0)})", flush=True)
            metrics.update({"masked_absrel":float(np.mean(absrels)),"masked_pixels":int(sum(mask_pixels)),"scene":record["scene"],"sample_idx":record["sample_idx"]})
            results[arm].append(metrics)
            print(f"[{record['scene']}] {arm}: pixels={metrics['masked_pixels']} F1={metrics['fscore']:.4f} overall={metrics['overall']:.4f}",flush=True)
    out={"dataset":args.dataset,"criterion":args.criterion,"windows":len(records),"per_window":results,"mean":{}}
    for arm, rows in results.items(): out["mean"][arm]={key:float(np.mean([x[key] for x in rows])) for key in ("masked_absrel","acc","comp","overall","precision","recall","fscore","masked_pixels")}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out,indent=2,sort_keys=True)); print(args.output)


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",choices=("eth3d","scannetpp","7scenes","hiroom"),required=True); ap.add_argument("--criterion",choices=("texture","occlusion"),required=True)
    ap.add_argument("--selection",required=True); ap.add_argument("--manifest",required=True); ap.add_argument("--selection-report",required=True); ap.add_argument("--lora",required=True); ap.add_argument("--model-name",default=str(ROOT/"model_weights/VGGT-1B")); ap.add_argument("--output",required=True); ap.add_argument("--arms",nargs="+",choices=ARMS,default=list(ARMS)); ap.add_argument("--max-windows",type=int,default=25); ap.add_argument("--image-size",type=int,default=504); ap.add_argument("--device",default="cuda:0"); ap.add_argument("--debug",action="store_true")
    run(ap.parse_args())
