#!/usr/bin/env python3
"""Attribution infer for scannetpp 21d970d8de (dev3b manifest): A0_baseline and
C2M_RKDC1 @100v WITH conf, npz into
artifacts/diagnostics/final_protocol/scannetpp_dev3b/attr_21d97/eval32/<arm>@100v.
GPU inference; waits until >=20 GB free. New file; does not modify any existing
module. Run from repo root."""

import os
import subprocess
import sys
import time

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

RR = os.path.join(_REPO, "artifacts", "diagnostics", "final_protocol",
                  "scannetpp_dev3b")
OUT = os.path.join(RR, "attr_21d97")
SCENE = "21d970d8de"
ARMS = ["A0_baseline", "C2M_RKDC1"]
STEP = 100
MIN_FREE_MIB = 20 * 1024


def gpu_free_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
    return min(int(x) for x in out.decode().split())


def wait_gpu():
    while True:
        free = gpu_free_mib()
        if free >= MIN_FREE_MIB:
            return
        print(f"[wait] gpu free {free} MiB < {MIN_FREE_MIB}; sleeping 60s", flush=True)
        time.sleep(60)


def npz_path(arm):
    return os.path.join(OUT, "eval32", f"{arm}@100v", "model_results",
                        "scannetpp", SCENE, "unposed", "exports", "mini_npz",
                        "results.npz")


def main():
    manifest = load_manifest(os.path.join(RR, "scene_manifest.json"))
    common.set_dataset(manifest.get("dataset", "scannetpp"))
    frames = manifest["scenes"][SCENE]["eval32_frames"]
    assert len(frames) == 100

    wait_gpu()
    for arm in ARMS:
        student = M.load_student()
        student.eval()
        if arm == "A0_baseline":
            M.reset_lora_(student)
        else:
            peft_dir = os.path.join(RR, "ckpts", SCENE, arm, f"step{STEP}_lora_peft")
            student.vggt.load_adapter(peft_dir, adapter_name=SCENE)
            student.vggt.set_adapter(SCENE)
        if os.path.isfile(npz_path(arm)):
            print(f"[skip] {arm}", flush=True)
            continue
        scene_data = get_scene_data(SCENE)
        t0 = time.time()
        depth, ext, intr, _, conf = infer_eval32(student, scene_data, frames)
        save_eval_npz(OUT, f"{arm}@100v", SCENE, depth, ext, intr,
                      np.asarray(scene_data.extrinsics)[frames],
                      gt_ixt_raw(scene_data, frames),
                      [scene_data.image_files[i] for i in frames], frames,
                      conf=conf)
        print(f"[done] {arm} depth={depth.shape} ({time.time()-t0:.0f}s)", flush=True)
        del student
        import torch
        torch.cuda.empty_cache()
    print("INFER ALL DONE", flush=True)


if __name__ == "__main__":
    main()
