#!/usr/bin/env python3
"""Metrics for migration exports (runs in the da3 env).

Reads <output_root>/<scene_tag>/exports/mini_npz/results.npz + the protocol
JSON, computes AUC@3 (all-pairs rel-pose, first-camera aligned), LS-scaled
AbsRel, and recon_unposed F1/CD via the SAME bench chain as all frozen
results (fuse3d TSDF + eval3d). Writes <output_root>/metrics.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = "/root/autodl-tmp/Free-Geometry"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "diagnostics", "free_geometry"))

import common as fg_common  # noqa: E402


def make_dataset(ds):
    import importlib
    table = {
        "scannetpp": ("depth_anything_3.bench.datasets.scannetpp", "ScanNetPP"),
        "7scenes": ("depth_anything_3.bench.datasets.sevenscenes", "SevenScenes"),
        "hiroom": ("depth_anything_3.bench.datasets.hiroom", "HiRoomDataset"),
        "eth3d": ("depth_anything_3.bench.datasets.eth3d", "ETH3D"),
    }
    mod = importlib.import_module(table[ds][0])
    return getattr(mod, table[ds][1])()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--skip_recon", action="store_true")
    args = ap.parse_args()

    from depth_anything_3.bench.utils import compute_pose
    from depth_anything_3.utils.geometry import as_homogeneous

    fg_common.set_dataset(args.dataset)
    dataset_obj = make_dataset(args.dataset)
    pdir = os.path.join(ROOT, "workspace/fgmig/protocols", args.dataset)
    out = {}
    for tag in sorted(os.listdir(args.output_root)):
        d = os.path.join(args.output_root, tag, "exports", "mini_npz", "results.npz")
        if not os.path.isfile(os.path.join(args.output_root, tag, "exports", "mini_npz", "results.npz")):
            continue
        scene = tag.replace("__", "/")
        proto = json.load(open(os.path.join(pdir, f"{tag}.json")))
        frames = proto["eval_frames"]
        data = fg_common.get_scene_data(scene)
        z = np.load(d)
        depth, ext, intr = z["depth"], z["extrinsics"], z["intrinsics"]
        gt_ext = np.asarray(data.extrinsics)[frames]

        pose = compute_pose(as_homogeneous(torch.from_numpy(ext).float()),
                            as_homogeneous(torch.from_numpy(gt_ext).float()))
        rec = {"n_eval_frames": len(frames), "auc03": float(pose.auc03),
               "auc05": float(pose.auc05), "auc30": float(pose.auc30)}

        gt = np.stack([fg_common.load_gt_depth(data.aux.gt_depth_files[i], depth.shape[-2:])
                       for i in frames])
        omega = np.isfinite(gt) & (gt > 0) & np.isfinite(depth) & (depth > 0)
        if omega.sum() > 0:
            p, g = depth[omega].astype(np.float64), gt[omega].astype(np.float64)
            scale = (p * g).sum() / max((p * p).sum(), 1e-12)
            rec["abs_rel"] = float(np.mean(np.abs(p * scale - g) / g))

        if not args.skip_recon:
            exp_dir = os.path.join(args.output_root, tag, "exports")
            np.savez_compressed(os.path.join(exp_dir, "gt_meta.npz"),
                                extrinsics=gt_ext,
                                intrinsics=np.asarray(data.intrinsics)[frames],
                                image_files=np.array(
                                    [data.image_files[i] for i in frames], dtype=object))
            fuse_path = os.path.join(exp_dir, "fuse", "pcd.ply")
            os.makedirs(os.path.dirname(fuse_path), exist_ok=True)
            dataset_obj.fuse3d(scene, d, fuse_path, "recon_unposed")
            r = dataset_obj.eval3d(scene, fuse_path)
            for k, v in dict(r).items():
                rec[f"recon_{k}"] = float(v)
        out[tag] = rec
        print(f"[{tag}] auc03={rec['auc03']:.4f} "
              f"fscore={rec.get('recon_fscore', float('nan')):.4f}", flush=True)

    with open(os.path.join(args.output_root, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
