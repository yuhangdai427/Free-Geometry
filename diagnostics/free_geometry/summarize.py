#!/usr/bin/env python3
"""Aggregate all bake-off results into paired_summary.csv + report.md.

Decision logic (pre-registered in the plan):
- Primary: paired per-scene probe E_depth deltas (deterministic) + 32-view
  AUC@3/AUC@30 deltas vs the A0 baseline and vs the A1 (current-loss) anchor.
- 32-view F1/CD descriptive only at n=6 (MDE 0.02-0.07 F1).
- Winner must beat BOTH A0 and A1 on primary metrics with bootstrap CI.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def bootstrap_ci(deltas, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas, dtype=float)
    if len(d) == 0:
        return float("nan"), float("nan")
    means = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_stats(values_by_scene, baseline_by_scene):
    scenes = sorted(set(values_by_scene) & set(baseline_by_scene))
    deltas = np.array([values_by_scene[s] - baseline_by_scene[s] for s in scenes])
    lo, hi = bootstrap_ci(deltas)
    return {
        "n": len(scenes),
        "mean_delta": float(deltas.mean()) if len(deltas) else float("nan"),
        "win_rate": float((deltas > 0).mean()) if len(deltas) else float("nan"),
        "ci_lo": lo, "ci_hi": hi,
        "per_scene": {s: float(d) for s, d in zip(scenes, deltas)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", default="artifacts/diagnostics/bakeoff_v1")
    args = ap.parse_args()
    R = args.run_root

    interp = load_csv(os.path.join(R, "feature_interpolation.csv"))
    direction = load_csv(os.path.join(R, "feature_direction.csv"))
    cameras = load_csv(os.path.join(R, "camera_endpoints.csv"))
    probes = load_csv(os.path.join(R, "probe_metrics.csv"))
    trace = load_csv(os.path.join(R, "training_trace.csv"))
    eval32_path = os.path.join(R, "eval32_metrics.json")
    eval32 = json.load(open(eval32_path)) if os.path.exists(eval32_path) else {}

    L = []
    w = L.append

    w("# Free-Geometry bake-off report (VGGT x ScanNet++, 6 dev scenes)\n")

    # ---------------- D1 ----------------
    w("\n## D1: teacher-target interpolation (frozen weights)\n")
    by_pair = defaultdict(list)
    for r in interp:
        by_pair[(r["scene"], r["pair_id"], r["sham"])].append(r)
    wins, tot, sham_ok = 0, 0, 0
    curves = []
    for (scene, pid, sham), rows in sorted(by_pair.items()):
        rows = sorted(rows, key=lambda r: float(r["alpha"]))
        e0, e1 = float(rows[0]["e_depth"]), float(rows[-1]["e_depth"])
        if sham == "0":
            tot += 1
            wins += e1 < e0
            curves.append(f"| {scene} p{pid} | {e0:.4f} | {e1:.4f} | {'better' if e1 < e0 else 'WORSE'} |")
        elif e1 > e0:
            sham_ok += 1
    w(f"- teacher endpoint better on depth: **{wins}/{tot}**")
    w(f"- sham control degraded (readout sensitive): **{sham_ok}/{tot}**")
    w("\n| pair | E_depth(α=0) | E_depth(α=1) | verdict |\n|---|---|---|---|")
    L.extend(curves)

    cam_wins = sum(1 for r in cameras if float(r["teacher_auc03"]) > float(r["student_auc03"]))
    cam_ties = sum(1 for r in cameras if float(r["teacher_auc03"]) == float(r["student_auc03"]))
    w(f"\n- camera endpoint (AUC@3): teacher better {cam_wins}/{len(cameras)}, ties {cam_ties}")

    # ---------------- D2 ----------------
    w("\n## D2: equal-norm direction test (ΔE_depth, negative = better; overshoot-excluded)\n")
    groups = defaultdict(list)
    for r in direction:
        if r["overshoot"] == "1":
            continue
        groups[(r["loss"], r["variant"], r["beta"])].append(
            float(r["e_depth_after"]) - float(r["e_depth_before"]))
    w("\n| loss | variant | beta | mean ΔE | median ΔE | win rate | n |\n|---|---|---|---|---|---|---|")
    for k in sorted(groups, key=lambda x: (x[0], x[1], float(x[2]))):
        v = np.array(groups[k])
        w(f"| {k[0]} | {k[1]} | {k[2]} | {v.mean():.5f} | {np.median(v):.5f} | {(v < 0).mean():.2f} | {len(v)} |")

    # ---------------- D3 probes ----------------
    w("\n## D3: probe metrics per arm (mean over 2 probe pairs, per scene)\n")
    probe_by = defaultdict(dict)
    for r in probes:
        probe_by[(r["arm"], int(r["step"]))][r["scene"]] = float(r["e_depth"])
    arms = sorted({a for a, _ in probe_by})
    base_e = probe_by.get(("A0_baseline", 0), {})
    w("\n### Probe E_depth paired deltas vs baseline (negative = better)\n")
    w("\n| arm | step | mean ΔE | win rate | 95% CI |\n|---|---|---|---|---|")
    probe_stats = {}
    for arm in arms:
        if arm == "A0_baseline":
            continue
        for step in sorted({s for a, s in probe_by if a == arm}):
            st = paired_stats(probe_by[(arm, step)], base_e)
            probe_stats[(arm, step)] = st
            w(f"| {arm} | {step} | {st['mean_delta']:+.5f} | {1 - st['win_rate']:.2f} | [{st['ci_lo']:+.5f}, {st['ci_hi']:+.5f}] |")
    # pose probe
    probe_pose = defaultdict(dict)
    for r in probes:
        if r.get("probe_auc03"):
            probe_pose[(r["arm"], int(r["step"]))][r["scene"]] = float(r["probe_auc03"])
    w("\n### Probe pose AUC@3 paired deltas vs baseline (positive = better)\n")
    w("\n| arm | step | mean Δ | win rate | 95% CI |\n|---|---|---|---|---|")
    base_p = probe_pose.get(("A0_baseline", 0), {})
    for arm in sorted({a for a, _ in probe_pose}):
        if arm == "A0_baseline":
            continue
        for step in sorted({s for a, s in probe_pose if a == arm}):
            st = paired_stats(probe_pose[(arm, step)], base_p)
            w(f"| {arm} | {step} | {st['mean_delta']:+.4f} | {st['win_rate']:.2f} | [{st['ci_lo']:+.4f}, {st['ci_hi']:+.4f}] |")

    # ---------------- 32-view eval ----------------
    w("\n## D3: fixed 32-view scene evaluation\n")
    if eval32:
        table = defaultdict(lambda: defaultdict(dict))  # metric -> exp -> scene -> value
        for exp, entry in eval32.items():
            for mode_key, scenemap in entry.items():
                if not isinstance(scenemap, dict):
                    continue
                mode = mode_key.split("_", 1)[-1] if "_" in mode_key else mode_key
                for scene, m in scenemap.items():
                    if scene == "mean" or not isinstance(m, dict):
                        continue
                    for k, v in m.items():
                        try:
                            table[k][exp][scene] = float(v)
                        except (TypeError, ValueError):
                            pass
        exps = sorted(eval32)
        base_exp = "A0_baseline" if "A0_baseline" in eval32 else None
        for metric in ("auc03", "auc30", "fscore", "overall"):
            if metric not in table:
                continue
            w(f"\n### 32-view {metric} (paired deltas vs baseline; for fscore/auc positive=better, for overall/CD negative=better)\n")
            w("\n| exp | mean | mean Δ | win rate | 95% CI |\n|---|---|---|---|---|")
            for exp in exps:
                vals = table[metric].get(exp, {})
                mean_v = float(np.mean(list(vals.values()))) if vals else float("nan")
                if exp == base_exp or base_exp is None:
                    w(f"| {exp} | {mean_v:.4f} | — | — | — |")
                    continue
                st = paired_stats(vals, table[metric].get(base_exp, {}))
                w(f"| {exp} | {mean_v:.4f} | {st['mean_delta']:+.4f} | {st['win_rate']:.2f} | [{st['ci_lo']:+.4f}, {st['ci_hi']:+.4f}] |")
    else:
        w("\n(eval32_metrics.json not found — run run_eval.py first)")

    # ---------------- training traces ----------------
    if trace:
        w("\n## Training loss trajectories (mean loss per 10-step bin)\n")
        bins = defaultdict(list)
        for r in trace:
            bins[(r["arm"], (int(r["step"]) - 1) // 10)].append(float(r["loss"]))
        w("\n| arm | steps 1-10 | 11-20 | ... | 91-100 |\n|---|---|---|---|---|")
        for arm in sorted({a for a, _ in bins}):
            vals = [np.mean(bins[(arm, b)]) for b in sorted({b for a, b in bins if a == arm})]
            w(f"| {arm} | " + " | ".join(f"{v:.4f}" for v in vals[:3]) + " ... | " + f"{vals[-1]:.4f} |")

    w("\n## Decision notes\n")
    w("- Primary readout: probe E_depth + probe/32-view AUC, paired per scene, bootstrap CI.")
    w("- 32-view F1/CD at n=6 is descriptive (MDE 0.02-0.07).")
    w("- RAW MSE decrease is NOT evidence of geometric improvement (see D1 sham/camera split).")

    out = os.path.join(R, "report.md")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
