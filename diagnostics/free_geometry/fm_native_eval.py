#!/usr/bin/env python3
"""FM-native inference probe: the FM_zero student was TRAINED on 8-slot inputs
(real shared frames at slots [0,2,4,6], zeroed extra frames at [1,3,5,7]).
The vc battery evaluated it with plain real inputs (transfer). Here we test the
training-consistent protocol: real frames interleaved with zero slots.

Modes: (a) plain 4v/8v (real only), (b) native: 4 real -> 8 slots interleaved;
8 real -> 16 slots interleaved. Same repo recon_unposed + pose eval via
crafted model_results + run_eval.

Output: artifacts/diagnostics/fm_native/eval32_metrics.json (arms
FMplain@4v/8v, FMnative@4v/8v, A0_baseline@4v/8v) + summary printed.
"""

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common import get_scene_data, load_manifest  # noqa: E402
import modeling as M  # noqa: E402
from train_arms import save_eval_npz  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402

OUT = "artifacts/diagnostics/fm_native"
CK = "artifacts/diagnostics/bakeoff_v2_transductive/ckpts"
MANIFEST = "artifacts/diagnostics/bakeoff_v2_transductive/scene_manifest.json"


@torch.no_grad()
def infer_slots(student, scene_data, frames, interleave_zeros, device="cuda"):
    base = M.get_base_vggt(student)
    imgs = [M.load_image_model(scene_data.image_files[i]) for i in frames]
    arr = np.stack(imgs, 0)
    if interleave_zeros:
        n = arr.shape[0]
        z = np.zeros_like(arr)
        merged = []
        for i in range(n):
            merged += [arr[i], z[0]]
        arr = np.stack(merged, 0)
    images = torch.from_numpy(arr).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda", enabled=False):
        preds = base(images)
    depth = preds["depth"].squeeze(0).squeeze(-1).float().cpu().numpy()
    pose_enc = preds["pose_enc"]
    if pose_enc.dim() == 2:
        pose_enc = pose_enc.unsqueeze(0)
    H, W = images.shape[-2:]
    ext, intr = pose_encoding_to_extri_intri(
        pose_enc, image_size_hw=(H, W), pose_encoding_type="absT_quaR_FoV")
    ext = ext.squeeze(0).float().cpu().numpy()
    intr = intr.squeeze(0).float().cpu().numpy()
    if interleave_zeros:  # keep real slots [0,2,4,...]
        keep = list(range(0, arr.shape[0], 2))
        depth, ext, intr = depth[keep], ext[keep], intr[keep]
    return depth, ext, intr


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = load_manifest(MANIFEST)
    scenes = sorted(manifest["scenes"])
    student = M.load_student("cuda")
    for scene in scenes:
        sc = manifest["scenes"][scene]
        scene_data = get_scene_data(scene)
        ev = sc["eval32_frames"]
        views = {"4v": ev[::8][:4], "8v": ev[::4][:8]}
        for tag, ckpt in (("A0_baseline", None), ("FMplain", "FM_zero"), ("FMnative", "FM_zero")):
            M.reset_lora_(student)
            if ckpt:
                student.load_lora_weights(os.path.join(CK, scene, ckpt, "step100_lora.pt"))
            student.eval()
            for vc, frames in views.items():
                native = tag == "FMnative"
                depth, ext, intr = infer_slots(student, scene_data, frames, native)
                save_eval_npz(OUT, f"{tag}@{vc}", scene, depth, ext, intr,
                              np.asarray(scene_data.extrinsics)[frames],
                              np.asarray(scene_data.aux.ixt_raw_list)[frames],
                              [scene_data.image_files[i] for i in frames], list(frames))
                print(f"[{scene} {tag}@{vc}] saved", flush=True)
        del scene_data
        torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
