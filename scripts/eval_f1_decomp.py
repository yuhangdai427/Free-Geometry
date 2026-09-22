#!/usr/bin/env python3
"""F1 decomposition: same TTA depth maps, fused under (a) PREDICTED cameras
(the reported F1) vs (b) GT cameras. If (b) recovers to baseline level, the
F1 drop is camera-depth MISALIGNMENT; if (b) also drops, the depth maps
themselves degraded (per-frame scale/quality beyond the global LS scale)."""
import sys, os, json, shutil
sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "diagnostics/free_geometry")
import numpy as np
import common as fg_common
from train_da3_protocol import make_dataset, get_scene_data

scene = sys.argv[1] if len(sys.argv) > 1 else "facade"
root = sys.argv[2] if len(sys.argv) > 2 else f"workspace/f1diag_{scene}"
fg_common.set_dataset("eth3d")
ds = make_dataset("eth3d")
sd = get_scene_data(scene)
npz = os.path.join(root, "recon", scene, "exports", "mini_npz", "results.npz")
d = np.load(npz, allow_pickle=True)
depth, ext, intr = d["depth"], d["extrinsics"], d["intrinsics"]
gt_ext = np.asarray(sd.extrinsics)[list(range(len(sd.image_files)))][:, :3, :]
gt_intr = np.asarray(sd.intrinsics)[list(range(len(sd.image_files)))]
print(f"{scene}: depth{depth.shape} ext{ext.shape} intr{intr.shape} gt_ext{gt_ext.shape}")

def fuse_and_eval(tag, E, K):
    work = os.path.join(root, "decomp", tag)
    os.makedirs(os.path.join(work, "exports", "mini_npz"), exist_ok=True)
    np.savez_compressed(os.path.join(work, "exports", "mini_npz", "results.npz"),
                        depth=np.round(depth, 8), extrinsics=E, intrinsics=K)
    fuse = os.path.join(work, "exports", "fuse", "pcd.ply")
    os.makedirs(os.path.dirname(fuse), exist_ok=True)
    ds.fuse3d(scene, os.path.join(work, "exports", "mini_npz", "results.npz"), fuse, "recon_unposed")
    r = ds.eval3d(scene, fuse)
    f1 = r.get("fscore", float("nan"))
    print(f"  {tag:<28} F1={f1:.4f}")
    return f1

f_pred = fuse_and_eval("pred_cam_pred_intr", ext, intr)
f_gt   = fuse_and_eval("gt_cam_pred_intr",   gt_ext, intr)
f_gtki = fuse_and_eval("gt_cam_gt_intr",     gt_ext, gt_intr)
print(f"\n结论: 深度不变换 GT 相机后 F1 {'恢复 → 相机-深度错配' if f_gt > f_pred + 0.02 else ('仍低 → 深度图本身退化' if f_gt < f_pred + 0.02 else '基本持平 → 两者都有贡献')}")
