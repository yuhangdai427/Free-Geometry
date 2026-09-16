#!/usr/bin/env python3
"""Consensus-teacher probe: does ensembling the teacher over K different
8-view supersets (same shared 4 frames, different extra 4) reduce its local
depth noise (the ~41% teacher-worse-than-student patches)?

If the bad patches are subset-dependent (random), averaging K contexts cancels
them; if intrinsic, it won't. Decides whether a "consensus teacher" can provide
genuinely better depth targets. GT used offline only.

Output: artifacts/diagnostics/consensus_probe/summary.md
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

OUT = "artifacts/diagnostics/consensus_probe"
K = 4
SCENES = ["1ada7a0617", "38d58a7a31"]
PAIRS_PER_SCENE = 4


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = load_manifest("artifacts/diagnostics/bakeoff_v2_transductive/scene_manifest.json")
    teacher = M.load_teacher("cuda")
    student = M.load_student("cuda")
    lines = ["# consensus teacher probe", ""]
    tot = {"single_pp": [], "cons_pp": [], "single_bad": [], "cons_bad": []}
    for scene in SCENES:
        sc = manifest["scenes"][scene]
        scene_data = get_scene_data(scene)
        ok, _ = frames_with_gt_depth(scene)
        M.reset_lora_(student); student.eval()
        for pi, pair in enumerate(sc["train_pairs"][:PAIRS_PER_SCENE]):
            shared = list(pair["student_frames"])
            base_extras = [f for i, f in enumerate(pair["teacher_frames"]) if i not in STUDENT_INDICES]
            pool = [f for f in ok if f not in set(shared)]
            rng = random.Random(stable_seed(51_000, scene, pi))
            images4 = torch.from_numpy(np.stack(
                [M.load_image_model(scene_data.image_files[i]) for i in shared], 0)
            ).permute(0, 3, 1, 2).float().unsqueeze(0).cuda()
            _, _, spreds = M.student_preds(student, images4)
            sdepth = spreds["depth"].squeeze(0).squeeze(-1).float().cpu().numpy()
            gt4 = M.load_probe_gt(scene_data, shared, images4.shape[-2:])
            ph, pw = images4.shape[-2] // 14, images4.shape[-1] // 14
            pp_s = perpatch_logres2(sdepth, gt4, (ph, pw))
            logds = []
            for k in range(K):
                extras = base_extras if k == 0 else rng.sample(pool, 4)
                frames = [None] * 8
                for si, s in zip(STUDENT_INDICES, shared):
                    frames[si] = s
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
            singles = np.stack(logds, 0)                       # [K,4,H,W]
            cons = singles.mean(0)                             # log-space mean
            pp_single = np.stack([perpatch_logres2(np.exp(ld), gt4, (ph, pw)) for ld in singles], 0)  # [K,4,P]
            pp_cons = perpatch_logres2(np.exp(cons), gt4, (ph, pw))
            valid = np.isfinite(pp_s)
            bad_single = ((pp_single > pp_s[None]) & valid[None]).mean()
            bad_cons = ((pp_cons > pp_s) & valid).mean()
            tot["single_pp"].append(float(pp_single[:, valid].mean()))
            tot["cons_pp"].append(float(pp_cons[valid].mean()))
            tot["single_bad"].append(float(bad_single))
            tot["cons_bad"].append(float(bad_cons))
            lines.append(f"[{scene} p{pi}] mean_pp single {pp_single[:, valid].mean():.5f} -> cons {pp_cons[valid].mean():.5f} | "
                         f"teacher-worse-than-student {bad_single:.3f} -> {bad_cons:.3f}")
            print(lines[-1], flush=True)
    lines += ["", f"K={K}, scenes={SCENES}, pairs={PAIRS_PER_SCENE}/scene", "",
              f"mean per-patch err: single {np.mean(tot['single_pp']):.5f} -> consensus {np.mean(tot['cons_pp']):.5f}",
              f"teacher-worse-than-student rate: single {np.mean(tot['single_bad']):.3f} -> consensus {np.mean(tot['cons_bad']):.3f}"]
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
