#!/usr/bin/env python3
"""Attribution variants for 21d970d8de: gauge mismatch + GT-depth variant npz +
posed/ mirrors.

Per arm (A0_baseline, C2M_RKDC1):
  1. s_pose = Sim3 scale from align_poses_umeyama(gt, pred, ransac=True, rs=42)
     (same call the evaluator's _prep_unposed/_prep_posed makes on this npz);
     s_depth = exp(median(log GT - log pred)) pooled over valid pixels of the
     100 frames; mismatch = s_pose/s_depth - 1.
  2. <arm>_gtdepth@100v variant: depth = GT_depth(frame, resized to the npz
     grid) / s_pose (invalid GT pixels -> 0; the evaluator masks them via the
     GT mask anyway). The eval pipeline multiplies depth by s_pose again ->
     the fused cloud sees metric GT depth. Extrinsics/intrinsics/conf unchanged
     (bit-identical), so recon_unposed on the variant = {pred pose, GT depth}
     and recon_posed = {GT pose, GT depth}.
  3. Every exp's npz + gt_meta mirrored into posed/exports/ so the evaluator's
     recon_posed mode (posed=True export dir) finds them.

Writes attr_21d97/gauge.json. Pure CPU. New file; no existing module modified.
Run from repo root."""

import json
import os
import shutil
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.join(_REPO, "src", "vggt"))

import common  # noqa: E402
from common import get_scene_data, load_gt_depth, load_manifest  # noqa: E402
from depth_anything_3.utils.pose_align import align_poses_umeyama  # noqa: E402

RR = os.path.join(_REPO, "artifacts", "diagnostics", "final_protocol",
                  "scannetpp_dev3b")
OUT = os.path.join(RR, "attr_21d97")
SCENE = "21d970d8de"
ARMS = ["A0_baseline", "C2M_RKDC1"]


def as44(ext):
    ext = np.asarray(ext)
    if ext.shape[-2:] == (4, 4):
        return ext
    n = ext.shape[0]
    out = np.tile(np.eye(4, dtype=ext.dtype), (n, 1, 1))
    out[:, :3, :] = ext
    return out


def exp_exports(exp, posed=False):
    return os.path.join(OUT, "eval32", exp, "model_results", "scannetpp",
                        SCENE, "posed" if posed else "unposed", "exports")


def main():
    manifest = load_manifest(os.path.join(RR, "scene_manifest.json"))
    common.set_dataset(manifest.get("dataset", "scannetpp"))
    frames = manifest["scenes"][SCENE]["eval32_frames"]
    scene_data = get_scene_data(SCENE)

    report = {}
    for arm in ARMS:
        exp = f"{arm}@100v"
        src = exp_exports(exp)
        r = np.load(os.path.join(src, "mini_npz", "results.npz"))
        g = np.load(os.path.join(src, "gt_meta.npz"), allow_pickle=True)
        depth = r["depth"]
        pred_ext = as44(r["extrinsics"])
        gt_ext = as44(g["extrinsics"])
        H, W = depth.shape[-2:]

        _, _, s_pose, _ = align_poses_umeyama(
            gt_ext.copy(), pred_ext.copy(), return_aligned=True, ransac=True,
            random_state=42)
        s_pose = float(s_pose)

        gt_depths = [load_gt_depth(scene_data.aux.gt_depth_files[i], (H, W))
                     for i in frames]
        logs = []
        for i in range(len(depth)):
            gd = gt_depths[i]
            om = np.isfinite(gd) & (gd > 0) & np.isfinite(depth[i]) & (depth[i] > 0)
            logs.append(np.log(gd[om]) - np.log(depth[i][om]))
        s_depth = float(np.exp(np.median(np.concatenate(logs))))
        mismatch = s_pose / s_depth - 1.0
        report[arm] = {"s_pose": s_pose, "s_depth": s_depth,
                       "mismatch": mismatch}
        print(f"[gauge] {arm}: s_pose={s_pose:.4f} s_depth={s_depth:.4f} "
              f"mismatch={mismatch*100:+.2f}%", flush=True)

        # GT-depth variant (depth in pred gauge so eval rescale lands on metric)
        kw = {k: r[k] for k in r.files}
        vd = np.zeros_like(depth)
        for i in range(len(depth)):
            gd = gt_depths[i]
            valid = np.isfinite(gd) & (gd > 0)
            vd[i][valid] = gd[valid] / s_pose
        kw["depth"] = np.round(vd, 8)
        dst = exp_exports(f"{arm}_gtdepth@100v")
        os.makedirs(os.path.join(dst, "mini_npz"), exist_ok=True)
        np.savez_compressed(os.path.join(dst, "mini_npz", "results.npz"), **kw)
        shutil.copyfile(os.path.join(src, "gt_meta.npz"),
                        os.path.join(dst, "gt_meta.npz"))
        v = np.load(os.path.join(dst, "mini_npz", "results.npz"))
        assert np.abs(v["extrinsics"] - r["extrinsics"]).max() == 0
        assert np.abs(v["intrinsics"] - r["intrinsics"]).max() == 0

        # posed/ mirrors for every exp of this arm
        for e in (exp, f"{arm}_gtdepth@100v"):
            sdir = exp_exports(e)
            pdir = exp_exports(e, posed=True)
            os.makedirs(os.path.join(pdir, "mini_npz"), exist_ok=True)
            shutil.copyfile(os.path.join(sdir, "mini_npz", "results.npz"),
                            os.path.join(pdir, "mini_npz", "results.npz"))
            shutil.copyfile(os.path.join(sdir, "gt_meta.npz"),
                            os.path.join(pdir, "gt_meta.npz"))
    with open(os.path.join(OUT, "gauge.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("VARIANTS DONE", flush=True)


if __name__ == "__main__":
    main()
