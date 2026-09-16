#!/usr/bin/env python3
"""Does the teacher-conf gate actually know WHICH patches the teacher improves?

Per patch on the 72 8->4 pairs: improvement = err(a0) - err(teacher=a100)
(per-patch centered log-res^2 from the saved scaled depths, pair-wide scale
already applied). Gate score = teacher conf avg-pooled to the patch grid,
mean-normalized (exactly teacher_patch_conf's convention in train_arms).

Reports: ROC-AUC of conf for the "teacher-better" label, Spearman(conf,
improvement), and the conf-gated vs uniform-gated EXPECTED per-patch
improvement capture: sum(w * imp) / sum(w) vs mean(imp) - i.e. how much more
of the true improvement the conf weighting harvests compared to uniform.
Positive label base rate also reported.
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common import get_scene_data, load_manifest  # noqa: E402
import modeling as M  # noqa: E402
from train_arms import perpatch_logres2  # noqa: E402

OUT = "artifacts/diagnostics/patch_interp_f1"


def roc_auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # tie handling: average ranks
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    n1 = labels.sum()
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    manifest = load_manifest("artifacts/diagnostics/bakeoff_v2_transductive/scene_manifest.json")
    scenes = sorted(manifest["scenes"])
    all_w, all_imp = [], []
    per_scene = {}
    for scene in scenes:
        scene_data = get_scene_data(scene)
        pairs = manifest["scenes"][scene]["train_pairs"] + manifest["scenes"][scene]["probe_pairs"]
        sw, si = [], []
        for pi, pair in enumerate(pairs):
            z0 = np.load(os.path.join(OUT, "depths", f"{scene}__p{pi:02d}__a0.npz"))
            zt = np.load(os.path.join(OUT, "depths", f"{scene}__p{pi:02d}__a100.npz"))
            zc = np.load(os.path.join(OUT, "conf", f"{scene}__p{pi:02d}.npz"))
            d0 = z0["depth"].astype(np.float32)
            dt = zt["depth"].astype(np.float32)
            conf = zc["conf"].astype(np.float32)
            H, W = d0.shape[-2:]
            ph, pw = H // 14, W // 14
            gt4 = M.load_probe_gt(scene_data, pair["student_frames"], (H, W))
            e0 = perpatch_logres2(d0, gt4, (ph, pw))   # [4,P]
            et = perpatch_logres2(dt, gt4, (ph, pw))
            imp = e0 - et                               # >0: teacher better
            # conf -> patch weight (teacher_patch_conf convention)
            cf = conf.reshape(4, H, W)
            kH, kW = H // ph, W // pw
            cfp = cf.reshape(4, ph, kH, pw, kW).mean(axis=(2, 4))
            w = (cfp / max(cfp.mean(), 1e-8)).reshape(4, ph * pw)
            m = np.isfinite(imp)
            sw.append(w[m]); si.append(imp[m])
            all_w.append(w[m]); all_imp.append(imp[m])
        per_scene[scene] = (np.concatenate(sw), np.concatenate(si))

    W = np.concatenate(all_w)
    I = np.concatenate(all_imp)
    label = (I > 0).astype(np.int64)

    lines = ["# conf-gate mechanism: does teacher conf know which patches improve?", "",
             f"pooled patches: {len(I)} (72 pairs, GT-valid only); base rate teacher-better: {label.mean():.3f}", ""]
    lines.append(f"- ROC-AUC(conf, teacher-better): {roc_auc(W, label):.4f}")
    rho = float(np.corrcoef(np.argsort(np.argsort(W)), np.argsort(np.argsort(I)))[0, 1])
    lines.append(f"- Spearman(conf weight, per-patch improvement): {rho:+.4f}")
    gated = float((W * I).sum() / W.sum())
    unif = float(I.mean())
    lines.append(f"- expected improvement under conf weighting: {gated:+.6f} vs uniform {unif:+.6f} "
                 f"(ratio {gated / unif if unif != 0 else float('nan'):.3f})")
    lines.append("")
    lines.append("| scene | AUC | Spearman | gated/unif | base rate |")
    lines.append("|---|---|---|---|---|")
    for scene, (w, i) in per_scene.items():
        lab = (i > 0).astype(np.int64)
        g = float((w * i).sum() / w.sum())
        u = float(i.mean())
        r = float(np.corrcoef(np.argsort(np.argsort(w)), np.argsort(np.argsort(i)))[0, 1])
        lines.append(f"| {scene} | {roc_auc(w, lab):.4f} | {r:+.4f} | {g / u if u != 0 else float('nan'):.3f} | {lab.mean():.3f} |")
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "conf_gate_analysis.md"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
