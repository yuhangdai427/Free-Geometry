#!/usr/bin/env python3
"""Channel-subspace probe (user idea): is the teacher's distillable signal
concentrated in a subset of feature channels / a subspace?

Two GT-free-relevant statistics + one GT label (offline only):
1. Channel separation: per channel c, does |h_t - h_s|_c differ between
   patches where teacher is better vs worse (GT label)?
2. Channel self-consistency (GT-free): per channel variance of teacher
   features across K=4 resampled extra-view contexts (reliable subspace).
3. Cross-check: are reliable (low-variance) channels also the separating
   ones? And how concentrated is the separation (top-k channel share)?

Output: artifacts/diagnostics/channel_probe/summary.md
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

OUT = "artifacts/diagnostics/channel_probe"
K = 4
SCENES = ["1ada7a0617", "38d58a7a31", "7831862f02"]
PAIRS_PER_SCENE = 4
LAYER = 23


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = load_manifest("artifacts/diagnostics/bakeoff_v2_transductive/scene_manifest.json")
    teacher = M.load_teacher("cuda")
    student = M.load_student("cuda")
    dh = teacher.depth_head
    Var_all, SEP = [], []
    for scene in SCENES:
        sc = manifest["scenes"][scene]
        scene_data = get_scene_data(scene)
        ok, _ = frames_with_gt_depth(scene)
        M.reset_lora_(student); student.eval()
        for pi, pair in enumerate(sc["train_pairs"][:PAIRS_PER_SCENE]):
            shared = list(pair["student_frames"])
            pool = [f for f in ok if f not in set(shared)]
            rng = random.Random(stable_seed(55_000, scene, pi))
            images4 = torch.from_numpy(np.stack(
                [M.load_image_model(scene_data.image_files[i]) for i in shared], 0)
            ).permute(0, 3, 1, 2).float().unsqueeze(0).cuda()
            sfeats24, _, sp = M.student_preds(student, images4)
            sdepth = sp["depth"].squeeze(0).squeeze(-1).float().cpu().numpy()
            hs = M.to_norm(dh, M.to_patch(sfeats24[LAYER].float()))     # [1,4,P,C]
            gt4 = M.load_probe_gt(scene_data, shared, images4.shape[-2:])
            ph, pw = images4.shape[-2] // 14, images4.shape[-1] // 14
            pp_s = perpatch_logres2(sdepth, gt4, (ph, pw))
            hts, tds = [], []
            for k in range(K):
                frames = [None] * 8
                for si, s in zip(STUDENT_INDICES, shared):
                    frames[si] = s
                for ei, e in zip([1, 3, 5, 7], rng.sample(pool, 4)):
                    frames[ei] = e
                imgs = torch.from_numpy(np.stack(
                    [M.load_image_model(scene_data.image_files[i]) for i in frames], 0)
                ).permute(0, 3, 1, 2).float().unsqueeze(0).cuda()
                feats24, psi = M.aggregator_all(teacher, imgs)
                ht = M.to_norm(dh, M.to_patch(feats24[LAYER][:, STUDENT_INDICES].float()))
                hts.append(ht)
                slots = [None] * 24
                for l in TAP_LAYERS:
                    slots[l] = feats24[l][:, STUDENT_INDICES].float().contiguous()
                d4, _ = M.replay_depth_nograd(teacher, slots, imgs[:, STUDENT_INDICES], psi)
                tds.append(d4.squeeze(0).squeeze(-1).float().cpu().numpy())
                del feats24
            HT = torch.stack(hts, 0)                    # [K,1,4,P,C]
            var_c = HT.var(0).mean(dim=(0, 1, 2))       # [C] cross-context var
            dh_c = (hts[0] - hs).abs()                  # [1,4,P,C]
            pp_t = perpatch_logres2(tds[0], gt4, (ph, pw))
            better = (pp_t < pp_s).reshape(-1)
            valid = np.isfinite(pp_t.reshape(-1))
            d = dh_c[0].reshape(-1, dh_c.shape[-1]).cpu().numpy()   # [4P,C]
            d, b = d[valid], better[valid]
            Var_all.append(var_c.cpu().numpy())
            SEP.append((d, b))
            print(f"[{scene} p{pi}] done", flush=True)
        torch.cuda.empty_cache()
    # channel separation: |E[dh|better] - E[dh|worse]| / pooled std
    d_all = np.concatenate([x[0] for x in SEP], 0)
    b_all = np.concatenate([x[1] for x in SEP], 0)
    mu_b, mu_w = d_all[b_all].mean(0), d_all[~b_all].mean(0)
    sd = d_all.std(0) + 1e-8
    sep = np.abs(mu_b - mu_w) / sd                     # [C]
    var_c = np.mean(Var_all, 0)                        # [C]
    order = np.argsort(-sep)
    share = np.cumsum(sep[order]) / sep.sum()
    k90 = int(np.searchsorted(share, 0.90))
    rho = float(np.corrcoef(sep, var_c)[0, 1])
    top_share_var = 1 - var_c[order[:2048 // 10]].mean() / var_c.mean()
    lines = ["# channel subspace probe (L23, 3 scenes x 4 pairs x K=4)", "",
             f"channels={d_all.shape[1]}, patches={d_all.shape[0]}",
             f"separation (|mu_better-mu_worse|/std): max {sep.max():.3f} mean {sep.mean():.3f}",
             f"top-10% channels carry {sep[order[:204]].sum()/sep.sum():.1%} of separation; 90% sep needs {k90} channels",
             f"corr(separation, cross-context variance) = {rho:.3f}",
             f"top-sep-10% channels have {top_share_var:+.1%} lower cross-context variance than average",
             "", "interpretation: sep>>0 & k90 small & rho<0 -> a reliable distillable subspace exists"]
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    np.savez_compressed(os.path.join(OUT, "channels.npz"), sep=sep, var=var_c, order=order)
    print("\n".join(lines), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
