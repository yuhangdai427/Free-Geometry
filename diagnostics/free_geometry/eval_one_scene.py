#!/usr/bin/env python3
"""Single-scene fast eval: baseline + one arm's ckpt on one scene, then
pose+recon metrics. Usage: python eval_one_scene.py --run_root <rr> --scene <s> --arm C2M_HARD
Run from repo root."""

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.join(_REPO, "src", "vggt"))

import common
from common import get_scene_data, gt_ixt_raw, load_manifest
import modeling as M
from train_arms import infer_eval32, save_eval_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--views", default="100v")
    ap.add_argument("--out_root", default=None)
    args = ap.parse_args()

    manifest = load_manifest(os.path.join(args.run_root, "scene_manifest.json"))
    ds = manifest.get("dataset", "scannetpp")
    common.set_dataset(ds)
    sc = manifest["scenes"][args.scene]
    scene_data = get_scene_data(args.scene)
    ev = sc["eval32_frames"]
    v = args.views
    frames = ev if v == "allv" else ev[:: max(1, len(ev) // int(v.rstrip("v")))][: int(v.rstrip("v"))]

    out_root = args.out_root or os.path.join(args.run_root, "single_scene_eval")
    student = M.load_student()
    student.eval()
    for exp, use_ckpt in (("A0_baseline", False), (args.arm, True)):
        M.reset_lora_(student)
        if use_ckpt:
            student.load_lora_weights(os.path.join(
                args.run_root, "ckpts", args.scene, args.arm, f"step{args.step}_lora.pt"))
        depth, ext, intr, _, conf = infer_eval32(student, scene_data, frames)
        save_eval_npz(out_root, f"{exp}@{v}", args.scene, depth, ext, intr,
                      np.asarray(scene_data.extrinsics)[frames],
                      gt_ixt_raw(scene_data, frames),
                      [scene_data.image_files[i] for i in frames], frames)
        print(f"[{exp} {args.scene}] inferred", flush=True)

    from vggt.bench.evaluator import VGGTEvaluator
    for exp in ("A0_baseline", args.arm):
        ev_ = VGGTEvaluator(work_dir=os.path.join(out_root, "eval32", f"{exp}@{v}"),
                            datas=[ds], modes=["pose", "recon_unposed"],
                            scenes=[args.scene], max_frames=0)
        m = ev_.eval()
        p = m[f"{ds}_pose"][args.scene]
        r = m[f"{ds}_recon_unposed"][args.scene]
        print(f"RESULT {exp}@{v} {args.scene}: AUC@3={p['auc03']:.4f} AUC@30={p['auc30']:.4f} "
              f"F1={r['fscore']:.4f} CD={r['overall']:.4f} "
              f"acc={r['acc']:.4f} comp={r['comp']:.4f} "
              f"prec={r['precision']:.4f} rec={r['recall']:.4f}", flush=True)
    print("SINGLE DONE")


if __name__ == "__main__":
    main()
