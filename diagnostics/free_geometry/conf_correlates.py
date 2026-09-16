#!/usr/bin/env python3
"""What does teacher confidence actually correlate with? (72 pairs, saved npz)

Checks per-pixel on GT-valid points (subsampled):
- conf vs GT depth value (far = low conf?)
- conf vs image gradient magnitude (textureless = low conf?)
- conf vs GT-depth gradient magnitude (occlusion/depth edges = low conf?)
And per-patch teacher improvement (err_a0 - err_aX) split by conf quartile
for X in {a100 (8v teacher), t16, t32}: where does the teacher's headroom live?
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common import get_scene_data, load_manifest, load_image_model  # noqa: E402
import modeling as M  # noqa: E402
from train_arms import perpatch_logres2  # noqa: E402

OUT = "artifacts/diagnostics/patch_interp_f1"
N_SUB = 4000


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    manifest = load_manifest("artifacts/diagnostics/bakeoff_v2_transductive/scene_manifest.json")
    scenes = sorted(manifest["scenes"])
    dep, tex, edge = [], [], []
    conf_all = []
    quart = {v: [[] for _ in range(4)] for v in ("a100", "t16", "t32")}
    rng = np.random.default_rng(0)
    for scene in scenes:
        scene_data = get_scene_data(scene)
        pairs = manifest["scenes"][scene]["train_pairs"] + manifest["scenes"][scene]["probe_pairs"]
        for pi, pair in enumerate(pairs):
            zc = np.load(os.path.join(OUT, "conf", f"{scene}__p{pi:02d}.npz"))
            conf = zc["conf"].astype(np.float32).reshape(4, *zc["conf"].shape[-2:])
            H, W = conf.shape[-2:]
            gt4 = M.load_probe_gt(scene_data, pair["student_frames"], (H, W))
            imgs = [load_image_model(scene_data.image_files[i]) for i in pair["student_frames"]]
            for k in range(4):
                g = gt4[k]
                m = np.isfinite(g) & (g > 0)
                if m.sum() < 100:
                    continue
                gray = imgs[k].mean(-1)
                gy, gx = np.gradient(gray)
                tex_mag = np.sqrt(gx * gx + gy * gy)
                dgy, dgx = np.gradient(np.where(m, g, np.nanmedian(g[m])))
                edge_mag = np.sqrt(dgx * dgx + dgy * dgy)
                idx = rng.choice(np.flatnonzero(m), size=min(N_SUB, m.sum()), replace=False)
                cf = conf[k].reshape(-1)[idx]
                conf_all.append(cf)
                dep.append(g.reshape(-1)[idx])
                tex.append(tex_mag.reshape(-1)[idx])
                edge.append(edge_mag.reshape(-1)[idx])
            # per-patch improvement by conf quartile
            ph, pw = H // 14, W // 14
            cfp = conf.reshape(4, ph, H // ph, pw, W // pw).mean(axis=(2, 4)).reshape(4, ph * pw)
            q = np.quantile(cfp, [0.25, 0.5, 0.75])
            binid = np.digitize(cfp, q)  # [4,P] in 0..3
            z0 = np.load(os.path.join(OUT, "depths", f"{scene}__p{pi:02d}__a0.npz"))
            d0 = z0["depth"].astype(np.float32)
            e0 = perpatch_logres2(d0, gt4, (ph, pw))
            for v in ("a100", "t16", "t32"):
                zv = np.load(os.path.join(OUT, "depths", f"{scene}__p{pi:02d}__{v}.npz"))
                ev = perpatch_logres2(zv["depth"].astype(np.float32), gt4, (ph, pw))
                imp = e0 - ev
                mm = np.isfinite(imp)
                for b in range(4):
                    sel = mm & (binid == b)
                    if sel.sum() > 10:
                        quart[v][b].append(float(np.mean(imp[sel])))
        print(f"[{scene}] done", flush=True)

    C = np.concatenate(conf_all)
    D = np.concatenate(dep)
    T = np.concatenate(tex)
    E = np.concatenate(edge)
    lines = ["# what teacher confidence correlates with (72 pairs, ~1.1M px)", "",
             f"- Spearman(conf, GT depth):        {spearman(C, D):+.3f}   (negative => far = low conf)",
             f"- Spearman(conf, image gradient):  {spearman(C, T):+.3f}   (positive => textureless = low conf)",
             f"- Spearman(conf, GT depth edges):  {spearman(C, E):+.3f}   (negative => depth edges/occlusion = low conf)",
             f"- conf percentiles: p10={np.percentile(C, 10):.3f} p50={np.percentile(C, 50):.3f} p90={np.percentile(C, 90):.3f}",
             "",
             "## per-patch teacher improvement (err_a0 - err_teacher) by conf quartile", "",
             "| teacher | Q1 (lowest conf) | Q2 | Q3 | Q4 (highest conf) |",
             "|---|---|---|---|---|"]
    for v in ("a100", "t16", "t32"):
        means = [np.mean(quart[v][b]) if quart[v][b] else float("nan") for b in range(4)]
        lines.append(f"| {v} | " + " | ".join(f"{m:+.5f}" for m in means) + " |")
    lines += ["", "(positive = teacher better than a0 student in that quartile)"]
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "conf_correlates.md"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
