#!/usr/bin/env python3
"""DA3-GIANT-1.1 baseline (NO adaptation) on the frozen protocol manifests.

For each scene in the manifest: inference on the protocol eval frames with the
base model, then pose AUC (w2c, compute_pose) and scale-fitted AbsRel / δ1.25
(GT depth via the shared 4-dataset convention in diagnostics/free_geometry/common.py).

Example:
    python scripts/eval_da3_baseline.py \
        --dataset 7scenes \
        --manifest artifacts/diagnostics/final_protocol/7scenes/scene_manifest.json \
        --out artifacts/diagnostics/final_protocol/da3_baseline/7scenes_baseline.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "diagnostics", "free_geometry"))

import common as fg_common
from depth_anything_3.test_time_adaption import protocol_v1 as P


@torch.no_grad()
def eval_scene(api, image_files, gt_ext_all, gt_depth_files, eval_frames) -> dict:
    from depth_anything_3.bench.utils import compute_pose
    from depth_anything_3.utils.geometry import as_homogeneous

    files = [image_files[i] for i in eval_frames]
    pred = api.inference(
        image=files,
        process_res=P.PROCESS_RES,
        process_res_method="upper_bound_resize",
        ref_view_strategy="first",
    )
    depth = np.asarray(pred.depth, dtype=np.float32)  # [N,H,W]
    ext = np.asarray(pred.extrinsics, dtype=np.float32)  # [N,4,4] w2c

    gt_ext = np.asarray(gt_ext_all)[eval_frames]
    pose = compute_pose(
        as_homogeneous(torch.from_numpy(ext).float()),
        as_homogeneous(torch.from_numpy(gt_ext).float()),
    )

    gt = np.stack([
        fg_common.load_gt_depth(gt_depth_files[i], depth.shape[-2:]) for i in eval_frames
    ])
    omega = np.isfinite(gt) & (gt > 0) & np.isfinite(depth) & (depth > 0)
    out = {
        "n_eval_frames": len(eval_frames),
        "auc03": float(pose.auc03),
        "auc05": float(pose.auc05),
        "auc15": float(pose.auc15),
        "auc30": float(pose.auc30),
        "n_valid_px": int(omega.sum()),
    }
    if int(omega.sum()) > 0:
        p, g = depth[omega].astype(np.float64), gt[omega].astype(np.float64)
        scale = float((p * g).sum() / max((p * p).sum(), 1e-12))
        ps = p * scale
        out.update({
            "abs_rel": float(np.mean(np.abs(ps - g) / g)),
            "delta125": float(np.mean(np.maximum(ps / g, g / ps) < 1.25)),
            "ls_scale": scale,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="DA3 baseline on frozen protocol manifests")
    ap.add_argument("--dataset", required=True,
                    choices=["scannetpp", "7scenes", "hiroom", "eth3d"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model_name", default="model_weights/DA3-GIANT-1.1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    scenes = sorted(man["scenes"])
    fg_common.set_dataset(args.dataset)

    print(f"Loading base model: {args.model_name}")
    teacher = P.create_teacher(args.model_name, device="cuda")

    results = {}
    for scene in scenes:
        t0 = time.time()
        sc = man["scenes"][scene]
        data = fg_common.get_scene_data(scene)
        gt_depth_files = list(data.aux.gt_depth_files)
        assert len(gt_depth_files) == len(data.image_files), (
            f"{scene}: gt_depth_files {len(gt_depth_files)} != image_files "
            f"{len(data.image_files)}")
        ev = eval_scene(teacher, list(data.image_files), data.extrinsics,
                        gt_depth_files, sc["eval32_frames"])
        ev["time_s"] = time.time() - t0
        results[scene] = ev
        print(f"[{args.dataset}/{scene}] auc03={ev['auc03']:.4f} auc30={ev['auc30']:.4f} "
              f"abs_rel={ev.get('abs_rel', float('nan')):.4f} "
              f"d1.25={ev.get('delta125', float('nan')):.4f} "
              f"({ev['n_eval_frames']}f, {ev['time_s']:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"dataset": args.dataset, "model": args.model_name,
                   "manifest": args.manifest, "scenes": results}, f, indent=2)
    print(f"Wrote {args.out}")
    print("DONE")


if __name__ == "__main__":
    main()
