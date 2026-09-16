#!/usr/bin/env python3
"""Sim-probe: does cross-view patch similarity predict where the teacher is right?

User hypothesis: extra-view patches similar to shared-view patches (post-norm QK
cosine) are more likely correct. Test both directions with GT (offline only):
  SHARED side: label = teacher 8v locally better than student 4v (pp_t < pp_s).
    signals: sim to extra-view patches (max / top10 mean, L23+L17), teacher conf,
    |h_t - h_s| feature distance.
  EXTRA side: label = teacher depth error below pair median.
    signals: sim to shared-view patches (max / top10 mean), teacher conf.
Outputs: artifacts/diagnostics/sim_probe/{results.csv, summary.md}
"""

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common import STUDENT_INDICES, TAP_LAYERS, get_scene_data, load_manifest  # noqa: E402
import modeling as M  # noqa: E402
from train_arms import perpatch_logres2, teacher_patch_conf  # noqa: E402

EXTRA_INDICES = [i for i in range(8) if i not in STUDENT_INDICES]
MANIFEST = "artifacts/diagnostics/bakeoff_v2_transductive/scene_manifest.json"
OUT = "artifacts/diagnostics/sim_probe"


def cosmax(q, bank, topk=10):
    """q [N,C], bank [M,C] (both L2-normed) -> (max sim [N], mean-of-topk [N])."""
    s = q @ bank.T
    vmax = s.max(dim=1).values
    k = min(topk, s.shape[1])
    vtop = s.topk(k, dim=1).values.mean(dim=1)
    return vmax, vtop


def auc(scores, labels):
    """ROC-AUC of scores for binary labels (ties in label excluded by caller)."""
    m = np.isfinite(scores)
    s, y = scores[m], labels[m].astype(float)
    if y.sum() in (0, len(y)):
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for tied scores
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = load_manifest(MANIFEST)
    teacher = M.load_teacher("cuda")
    student = M.load_student("cuda")
    rows = []
    for scene in sorted(manifest["scenes"]):
        sc = manifest["scenes"][scene]
        scene_data = get_scene_data(scene)
        M.reset_lora_(student); student.eval()
        for pi, pair in enumerate(sc["train_pairs"] + sc["probe_pairs"]):
            images8, images4 = M.load_pair_images(scene_data, pair["teacher_frames"])
            ph, pw = images8.shape[-2] // 14, images8.shape[-1] // 14
            feats24, psi = M.aggregator_all(teacher, images8)
            slots8 = [None] * 24
            for l in TAP_LAYERS:
                slots8[l] = feats24[l].float().contiguous()
            depth8, conf8 = M.replay_depth_nograd(teacher, slots8, images8, psi)
            sfeats24, _, spreds = M.student_preds(student, images4)
            sdepth = spreds["depth"].squeeze(0).squeeze(-1).float()
            gt4 = M.load_probe_gt(scene_data, pair["student_frames"], images8.shape[-2:])
            extra_frames = [pair["teacher_frames"][i] for i in EXTRA_INDICES]
            gt8x = M.load_probe_gt(scene_data, extra_frames, images8.shape[-2:])
            depth8 = depth8.squeeze(0).squeeze(-1).float().cpu().numpy()
            pp_t4 = perpatch_logres2(depth8[STUDENT_INDICES], gt4, (ph, pw))
            pp_s = perpatch_logres2(sdepth.cpu().numpy(), gt4, (ph, pw))
            pp_t8x = perpatch_logres2(depth8[EXTRA_INDICES], gt8x, (ph, pw))
            conf_t4 = teacher_patch_conf({"conf4": conf8[:, STUDENT_INDICES]}, (ph, pw))
            conf_x4 = teacher_patch_conf({"conf4": conf8[:, EXTRA_INDICES]}, (ph, pw))
            feats = {l: feats24[l].float() for l in (17, 23)}
            for layer, lname in ((23, "23"), (17, "17")):
                ht8 = M.to_norm(teacher.depth_head, M.to_patch(feats[layer]))
                hs = M.to_norm(teacher.depth_head,
                               M.to_patch(sfeats24[layer].float()))
                C = ht8.shape[-1]
                ht_sh = ht8[:, STUDENT_INDICES].reshape(-1, C)
                ht_ex = ht8[:, EXTRA_INDICES].reshape(-1, C)
                hs_f = hs.reshape(-1, C)
                ht_sh_n = torch.nn.functional.normalize(ht_sh, dim=-1)
                ht_ex_n = torch.nn.functional.normalize(ht_ex, dim=-1)
                smax, stop = cosmax(ht_sh_n, ht_ex_n)
                xmax, xtop = cosmax(ht_ex_n, ht_sh_n)
                row = dict(
                    scene=scene, pair=pi, layer=lname,
                    pp_t=pp_t4.reshape(-1), pp_s=pp_s.reshape(-1),
                    conf_t=conf_t4.reshape(-1).cpu().numpy(),
                    sim_max=smax.cpu().numpy(), sim_top=stop.cpu().numpy(),
                    fdist=(ht_sh - hs_f).norm(dim=-1).cpu().numpy() if layer == 23 else np.zeros(4 * ph * pw),
                    pp_tx=pp_t8x.reshape(-1),
                    conf_x=conf_x4.reshape(-1).cpu().numpy(),
                    xsim_max=xmax.cpu().numpy(), xsim_top=xtop.cpu().numpy())
                rows.append(row)
            print(f"[{scene} p{pi}] done", flush=True)
        del feats24
        torch.cuda.empty_cache()

    import csv
    keys = ["scene", "pair", "layer"]
    with open(os.path.join(OUT, "results_raw.npz"), "wb") as f:
        np.savez_compressed(f, **{f"r{i}": rows[i] for i in range(len(rows))})

    # pooled AUC per layer
    lines = ["# sim_probe summary", "",
             "SHARED side: label = teacher 8v locally better than student 4v (pp_t<pp_s)", "",
             "| layer | signal | AUC(pooled) | AUC(per-scene mean) |", "|---|---|---|---|"]
    for lname in ("17", "23"):
        rl = [r for r in rows if r["layer"] == lname]
        pp_t = np.concatenate([r["pp_t"] for r in rl]); pp_s = np.concatenate([r["pp_s"] for r in rl])
        valid = np.isfinite(pp_t) & np.isfinite(pp_s)
        lab = (pp_t < pp_s).astype(float)
        for sig in ("sim_max", "sim_top", "conf_t", "fdist"):
            s = np.concatenate([r[sig] for r in rl])
            a_pool = auc(s[valid], lab[valid])
            per = [auc(np.concatenate([r[sig] for r in rl if r["scene"] == sc2]),
                       np.concatenate([(r["pp_t"] < r["pp_s"]).astype(float) for r in rl if r["scene"] == sc2]))
                   for sc2 in sorted({r["scene"] for r in rl})]
            lines.append(f"| L{lname} | {sig} | {a_pool:.4f} | {np.nanmean(per):.4f} |")
        base = float(lab[valid].mean())
        lines.append(f"| L{lname} | base rate | {base:.3f} | |")
    lines += ["", "EXTRA side: label = teacher depth error below pair median", "",
              "| layer | signal | AUC(pooled) |", "|---|---|---|"]
    for lname in ("17", "23"):
        rl = [r for r in rows if r["layer"] == lname]
        e = np.concatenate([r["pp_tx"] for r in rl])
        med = np.concatenate([r["pp_tx"] - np.nanmedian(r["pp_tx"]) for r in rl])
        valid = np.isfinite(e)
        lab = (med < 0).astype(float)
        for sig in ("xsim_max", "xsim_top", "conf_x"):
            s = np.concatenate([r[sig] for r in rl])
            lines.append(f"| L{lname} | {sig} | {auc(s[valid], lab[valid]):.4f} |")
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
