#!/usr/bin/env python3
"""One-off anatomy inference: B5_maskdistill step100 on scannetpp 1ada7a0617
eval frames (100v), saved into the ada_anatomy scratch run_root for paired
per-frame analysis. New file; does not modify any existing module.
Run from repo root."""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.join(_REPO, "src", "vggt"))

import common  # noqa: E402
from common import get_scene_data, gt_ixt_raw, load_manifest  # noqa: E402
import modeling as M  # noqa: E402
from train_arms import infer_eval32, save_eval_npz  # noqa: E402

RR = os.path.join(_REPO, "artifacts", "diagnostics", "final_protocol", "scannetpp")
SCENE = "1ada7a0617"
ARM = "B5_maskdistill"
STEP = 100
OUT = os.path.join(RR, "ada_anatomy")


def main():
    manifest = load_manifest(os.path.join(RR, "scene_manifest.json"))
    common.set_dataset(manifest.get("dataset", "scannetpp"))
    sc = manifest["scenes"][SCENE]
    scene_data = get_scene_data(SCENE)
    frames = sc["eval32_frames"]

    student = M.load_student()
    student.eval()
    M.reset_lora_(student)
    student.load_lora_weights(os.path.join(
        RR, "ckpts", SCENE, ARM, f"step{STEP}_lora.pt"))
    depth, ext, intr, _, conf = infer_eval32(student, scene_data, frames)
    save_eval_npz(OUT, f"{ARM}@100v", SCENE, depth, ext, intr,
                  np.asarray(scene_data.extrinsics)[frames],
                  gt_ixt_raw(scene_data, frames),
                  [scene_data.image_files[i] for i in frames], frames)
    print("INFER DONE", flush=True)


if __name__ == "__main__":
    main()
