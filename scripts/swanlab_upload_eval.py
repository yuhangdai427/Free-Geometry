#!/usr/bin/env python3
"""Upload 3-state eval results (AUC/F1 vs baseline) to swanlab.

One experiment per scene in project `free-geometry-tta-eval`:
  auc03/theta0, auc03/theta_base, auc03/theta_rel   (+auc05/15/30, fscore, abs_rel, delta125)
  dAUC3%_base, dAUC3%_rel, dF1%_base, dF1%_rel, dAUC3%_rel_vs_base ...
The experiment description carries a one-line verdict per scene.
"""
import json
import os
import sys

ROOT = "/root/autodl-tmp/Free-Geometry"
os.chdir(ROOT)

import swanlab  # noqa: E402

SRC = "workspace/fg_diag_v2_eval.json"

METRICS = ["auc03", "auc05", "auc15", "auc30", "recon_fscore", "abs_rel", "delta125"]


def pct(x):
    return (x - 1.0) * 100.0


def main():
    d = json.load(open(SRC))
    for scene, sc in d.items():
        st = sc["states"]
        if "theta_rel" not in st or "theta_base" not in st:
            continue
        run = swanlab.init(
            project="free-geometry-tta-eval",
            experiment_name=f"{scene}_eval",
            description=(
                f"{sc['dataset']} | DA3-GIANT 8:4 loss_all_pos 100步 | "
                f"AUC3: θ₀ {st['theta0']['auc03']:.4f} -> base {st['theta_base']['auc03']:.4f} "
                f"({pct(st['theta_base']['auc03']/st['theta0']['auc03']):+.1f}%) / "
                f"rel {st['theta_rel']['auc03']:.4f} "
                f"({pct(st['theta_rel']['auc03']/st['theta0']['auc03']):+.1f}%) | "
                f"F1: θ₀ {st['theta0']['recon_fscore']:.4f} -> base {st['theta_base']['recon_fscore']:.4f} "
                f"({pct(st['theta_base']['recon_fscore']/st['theta0']['recon_fscore']):+.1f}%) / "
                f"rel {st['theta_rel']['recon_fscore']:.4f} "
                f"({pct(st['theta_rel']['recon_fscore']/st['theta0']['recon_fscore']):+.1f}%)"
            ),
            config={"dataset": sc["dataset"], "scene": scene,
                    "states": "theta0=frozen | theta_base=rkdc1h(no rel) | theta_rel=rkdc1hr(corrected rel)",
                    "source": SRC},
        )
        m = {}
        for metric in METRICS:
            for state in ("theta0", "theta_base", "theta_rel"):
                if metric in st[state]:
                    m[f"{metric}/{state}"] = st[state][metric]
        # deltas vs theta0 (and rel vs base)
        for a, b, tag in [("theta_base", "theta0", "base"),
                          ("theta_rel", "theta0", "rel"),
                          ("theta_rel", "theta_base", "rel_vs_base")]:
            for metric in ("auc03", "auc30", "recon_fscore"):
                if metric in st[a] and st[b][metric]:
                    m[f"d{metric}%/{tag}"] = pct(st[a][metric] / st[b][metric])
        run.log(m)
        run.finish()
        print(f"[uploaded] {scene}: "
              f"dAUC3 base {pct(st['theta_base']['auc03']/st['theta0']['auc03']):+.2f}% "
              f"rel {pct(st['theta_rel']['auc03']/st['theta0']['auc03']):+.2f}% | "
              f"dF1 base {pct(st['theta_base']['recon_fscore']/st['theta0']['recon_fscore']):+.2f}% "
              f"rel {pct(st['theta_rel']['recon_fscore']/st['theta0']['recon_fscore']):+.2f}%",
              flush=True)


if __name__ == "__main__":
    main()
