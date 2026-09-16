#!/usr/bin/env python3
"""Cross-dataset gauge-recoupling study, inference stage: regenerate eval npz
(depth/extr/intr/conf) for the champion arm + A0_baseline on 7scenes / hiroom /
eth3d from final_protocol ckpts (step100). Mirrors gauge_scannetpp_infer.py but
parameterized by dataset. Output goes to a NEW scratch run_root
<ds>/gauge_cross (nothing under eval32/ is touched). Resumable: skips
scene-arm pairs whose results.npz already exists. GPU inference; waits until
>=20 GB free before each scene-arm (pure-inference rule). New file; does not
modify any existing module. Run from repo root:

  python3 diagnostics/free_geometry/gauge_cross_infer.py --dataset 7scenes
"""

import argparse
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

FP = os.path.join(_REPO, "artifacts", "diagnostics", "final_protocol")

# dataset -> (champion arm, eval tag, published metric file for pairing)
CHAMPION = {
    "7scenes": ("C2M_MC", "100v", "eval32_metrics_C2M_MC.json"),
    "hiroom": ("C2M_maskrel", "allv", "eval32_metrics_C2M.json"),
    "eth3d": ("C2M_SCL", "allv", "eval32_metrics_C2M_SCL.json"),
}
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(CHAMPION))
    ap.add_argument("--scenes", nargs="*", default=None)
    args = ap.parse_args()

    arm_tta, tag, _ = CHAMPION[args.dataset]
    RR = os.path.join(FP, args.dataset)
    OUT = os.path.join(RR, "gauge_cross")
    ARMS = ["A0_baseline", arm_tta]

    manifest = load_manifest(os.path.join(RR, "scene_manifest.json"))
    common.set_dataset(manifest.get("dataset", args.dataset))
    scenes = sorted(manifest["scenes"])
    if args.scenes:
        scenes = [s for s in scenes if s in set(args.scenes)]

    def npz_path(arm, scene):
        return os.path.join(OUT, "eval32", f"{arm}@{tag}", "model_results",
                            args.dataset, scene, "unposed", "exports",
                            "mini_npz", "results.npz")

    wait_gpu()
    for arm in ARMS:
        student = M.load_student()
        student.eval()
        if arm == "A0_baseline":
            M.reset_lora_(student)  # LoRA B=0 -> exact base model
        for scene in scenes:
            if os.path.isfile(npz_path(arm, scene)):
                print(f"[skip] {arm} {scene}", flush=True)
                continue
            wait_gpu()
            if arm != "A0_baseline":
                peft_dir = os.path.join(RR, "ckpts", scene, arm,
                                        f"step{STEP}_lora_peft")
                adapter = scene.replace("/", "__")
                student.vggt.load_adapter(peft_dir, adapter_name=adapter)
                student.vggt.set_adapter(adapter)
            scene_data = get_scene_data(scene)
            frames = manifest["scenes"][scene]["eval32_frames"]
            t0 = time.time()
            depth, ext, intr, _, conf = infer_eval32(student, scene_data, frames)
            save_eval_npz(OUT, f"{arm}@{tag}", scene, depth, ext, intr,
                          np.asarray(scene_data.extrinsics)[frames],
                          gt_ixt_raw(scene_data, frames),
                          [scene_data.image_files[i] for i in frames], frames,
                          conf=conf)
            print(f"[done] {arm} {scene} ({time.time()-t0:.0f}s)", flush=True)
        del student
        import torch
        torch.cuda.empty_cache()
    print(f"INFER {args.dataset} ALL DONE", flush=True)


if __name__ == "__main__":
    main()
