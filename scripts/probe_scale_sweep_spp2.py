#!/usr/bin/env python3
"""Scale-pairing confirmation probe (2026-09-21): the 2x2 attribution shows the
TTA damage is a depth<->pose scale/consistency break, not component damage.
Decisive test: globally rescale the maskrel pp depth by s (pose fixed) and
re-run TSDF fusion. If F1 recovers at s != 1, the pairing break is confirmed
and the GT-free fix is a couple-style re-coupling.
Also computes the GT-free couple statistic per arm: log(RMS centers/mean depth).
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "diagnostics", "free_geometry"))
os.chdir(os.path.join(ROOT, ".."))

RUN = "workspace/probe_spp2"
SCENES = ["1ada7a0617", "21d970d8de"]
SCALES = [0.85, 0.925, 1.0, 1.075, 1.15]

import common
from train_arms import save_eval_npz
common.set_dataset("scannetpp")
manifest = json.load(open(f"{RUN}/scene_manifest.json"))

def couple_stat(ext, depth):
    R, t = ext[..., :3, :3], ext[..., :3, 3]
    c = -(R.transpose(0, 2, 1) @ t[..., None])[..., 0]
    rms = float(np.sqrt(((c - c.mean(0)) ** 2).sum(1).mean()))
    md = float(np.nanmean(depth))
    return rms / md

for scene in SCENES:
    sc = manifest["scenes"][scene]
    ev = sc["eval32_frames"]
    scene_data = common.get_scene_data(scene)
    gt_ext = np.asarray(scene_data.extrinsics)[ev]
    files = [scene_data.image_files[i] for i in ev]
    for arm in ("A0_baseline", "C2M_maskrel"):
        p = f"{RUN}/eval32/{arm}_pp/model_results/scannetpp/{scene}/unposed/exports/mini_npz/results.npz"
        z = np.load(p, allow_pickle=True)
        depth, ext, intr = z["depth"], z["extrinsics"], z["intrinsics"]
        conf = z["conf"] if "conf" in z else None
        cp = couple_stat(ext, depth)
        print(f"[{scene}/{arm}] couple stat RMS/meanD = {cp:.4f}", flush=True)
        if arm != "C2M_maskrel":
            continue
        # GT-free re-coupling scale: match maskrel's couple ratio to BASELINE's
        zb = np.load(f"{RUN}/eval32/A0_baseline_pp/model_results/scannetpp/{scene}/unposed/exports/mini_npz/results.npz", allow_pickle=True)
        s0 = couple_stat(zb["extrinsics"], zb["depth"]) / cp
        print(f"   GT-free re-couple scale s0 = {s0:.4f}", flush=True)
        for s in SCALES + [round(s0, 4)]:
            save_eval_npz(RUN, f"C2M_maskrel_s{str(s).replace('.','_')}", scene,
                          depth * s, ext, intr, gt_ext,
                          np.asarray(scene_data.intrinsics)[ev], files, ev, conf=conf)
        print(f"   wrote {len(SCALES)+1} scaled conditions", flush=True)
print("EXPORTS DONE — run: python diagnostics/free_geometry/run_eval.py --run_root workspace/probe_spp2 --datas scannetpp")
