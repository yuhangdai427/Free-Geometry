#!/usr/bin/env python3
"""Consensus K-scaling probe: how does the teacher's local depth noise scale
with the number of ensembled contexts K? Answers "can the 39% bad-patch rate
go to 5%?" with a measured curve (mean and median consensus).

Per pair: sample 16 contexts (same shared 4 frames at slots [0,2,4,6], extra 4
resampled each k from the GT-frame pool). Log-depth consensus over the first
K contexts, K in {1,2,4,8,16}; median over K in {4,8,16}. Metrics vs student:
mean per-patch err and teacher-worse-than-student rate. GT offline only.

Output: artifacts/diagnostics/consensus_k/summary.md
"""

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common import STUDENT_INDICES, TAP_LAYERS, get_scene_data, load_manifest, frames_with_gt_depth, stable_seed  # noqa: E402
import modeling as M  # noqa: E402
from train_arms import perpatch_logres2  # noqa: E402
import random

OUT = "artifacts/diagnostics/consensus_k"
K_MAX = 16
SCENES = ["1ada7a0617", "38d58a7a31", "7831862f02"]
PAIRS_PER_SCENE = 4


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = load_manifest("artifacts/diagnostics/bakeoff_v2_transductive/scene_manifest.json")
    teacher = M.load_teacher("cuda")
    student = M.load_student("cuda")
    agg_mean = {k: [[], []] for k in (1, 2, 4, 8, 16)}   # K -> [err, bad]
    agg_med = {k: [[], []] for k in (4, 8, 16)}
    for scene in SCENES:
        sc = manifest["scenes"][scene]
        scene_data = get_scene_data(scene)
        ok, _ = frames_with_gt_depth(scene)
        M.reset_lora_(student); student.eval()
        for pi, pair in enumerate(sc["train_pairs"][:PAIRS_PER_SCENE]):
            shared = list(pair["student_frames"])
            pool = [f for f in ok if f not in set(shared)]
            rng = random.Random(stable_seed(53_000, scene, pi))
            images4 = torch.from_numpy(np.stack(
                [M.load_image_model(scene_data.image_files[i]) for i in shared], 0)
            ).permute(0, 3, 1, 2).float().unsqueeze(0).cuda()
            _, _, spreds = M.student_preds(student, images4)
            sdepth = spreds["depth"].squeeze(0).squeeze(-1).float().cpu().numpy()
            gt4 = M.load_probe_gt(scene_data, shared, images4.shape[-2:])
            ph, pw = images4.shape[-2] // 14, images4.shape[-1] // 14
            pp_s = perpatch_logres2(sdepth, gt4, (ph, pw))
            valid = np.isfinite(pp_s)
            logds = []
            for k in range(K_MAX):
                frames = [None] * 8
                for si, s in zip(STUDENT_INDICES, shared):
                    frames[si] = s
                extras = rng.sample(pool, 4)
                for ei, e in zip([1, 3, 5, 7], extras):
                    frames[ei] = e
                imgs = torch.from_numpy(np.stack(
                    [M.load_image_model(scene_data.image_files[i]) for i in frames], 0)
                ).permute(0, 3, 1, 2).float().unsqueeze(0).cuda()
                feats24, psi = M.aggregator_all(teacher, imgs)
                slots = [None] * 24
                for l in TAP_LAYERS:
                    slots[l] = feats24[l][:, STUDENT_INDICES].float().contiguous()
                d4, _ = M.replay_depth_nograd(teacher, slots, imgs[:, STUDENT_INDICES], psi)
                logds.append(torch.log(d4.squeeze(0).squeeze(-1).float().clamp_min(1e-6)).cpu().numpy())
                del feats24
            singles = np.stack(logds, 0)  # [16,4,H,W]
            for K in (1, 2, 4, 8, 16):
                cons = np.exp(singles[:K].mean(0))
                pp = perpatch_logres2(cons, gt4, (ph, pw))
                agg_mean[K][0].append(float(pp[valid].mean()))
                agg_mean[K][1].append(float(((pp > pp_s) & valid).mean()))
            for K in (4, 8, 16):
                cons = np.exp(np.median(singles[:K], axis=0))
                pp = perpatch_logres2(cons, gt4, (ph, pw))
                agg_med[K][0].append(float(pp[valid].mean()))
                agg_med[K][1].append(float(((pp > pp_s) & valid).mean()))
            print(f"[{scene} p{pi}] done", flush=True)
        torch.cuda.empty_cache()
    lines = ["# consensus K-scaling (3 scenes x 4 pairs)", "",
             "| K | mean-cons err | mean-cons bad-rate | median-cons err | median-cons bad-rate |",
             "|---|---|---|---|---|"]
    for K in (1, 2, 4, 8, 16):
        me, mb = np.mean(agg_mean[K][0]), np.mean(agg_mean[K][1])
        if K in agg_med:
            de, db = np.mean(agg_med[K][0]), np.mean(agg_med[K][1])
            lines.append(f"| {K} | {me:.5f} | {mb:.3f} | {de:.5f} | {db:.3f} |")
        else:
            lines.append(f"| {K} | {me:.5f} | {mb:.3f} | | |")
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[-6:]), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
