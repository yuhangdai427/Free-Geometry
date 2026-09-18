#!/usr/bin/env python3
"""Compare mv13 (LoRA 13-39, multi-view only) vs old all40 (0-39) DA3 rkdc1h runs.

Old runs:  workspace/da3_protocol_{ds}_t8s4_lossall/smoke_summary.json  (arm rkdc1h,
           8:4, loss_all_pos, seed 0 — identical protocol except LoRA scope 0-39)
New runs:  workspace/mv13_{ds}/smoke_summary.json                        (same, 13-39)

Per dataset: mean AUC@3 / F1 for old vs new + per-scene delta, on the COMMON
scenes (should be identical lists).
"""
import json
import sys

ROOT = "/root/autodl-tmp/Free-Geometry"

DATASETS = ["7scenes", "eth3d", "hiroom", "scannetpp"]


def load(path):
    try:
        return json.load(open(f"{ROOT}/{path}"))["scenes"]
    except FileNotFoundError:
        return None


def mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    print(f"{'dataset':10s} {'n':>3s} | {'AUC3 old(0-39)':>14s} {'new(13-39)':>10s} {'Δabs':>8s} | "
          f"{'F1 old':>8s} {'new':>8s} {'Δabs':>8s} | AUC 逐场景变化")
    totals = {"a_old": [], "a_new": [], "f_old": [], "f_new": []}
    for ds in DATASETS:
        old = load(f"workspace/da3_protocol_{ds}_t8s4_lossall/smoke_summary.json")
        new = load(f"workspace/mv13_{ds}/smoke_summary.json")
        if not old or not new:
            print(f"{ds:10s}   - | (missing: {'old' if not old else ''}{' new' if not new else ''})")
            continue
        common = [s for s in old if s in new and "eval" in old[s] and "eval" in new[s]]
        a_old = [old[s]["eval"]["auc03"] for s in common]
        a_new = [new[s]["eval"]["auc03"] for s in common]
        f_old = [old[s]["eval"]["recon_fscore"] for s in common]
        f_new = [new[s]["eval"]["recon_fscore"] for s in common]
        per_scene = " ".join(
            f"{s.split('/')[-1][:6]}:{'+' if b - a >= 0 else ''}{(b - a) * 100:.2f}"
            for s, a, b in zip(common, a_old, a_new))
        print(f"{ds:10s} {len(common):3d} | {mean(a_old):14.4f} {mean(a_new):10.4f} "
              f"{(mean(a_new) - mean(a_old)) * 100:+8.2f} | {mean(f_old):8.4f} {mean(f_new):8.4f} "
              f"{(mean(f_new) - mean(f_old)) * 100:+8.2f}")
        print(f"{'':10s}     | AUC pp×100 per-scene (new−old): {per_scene}")
        totals["a_old"] += a_old; totals["a_new"] += a_new
        totals["f_old"] += f_old; totals["f_new"] += f_new
    if totals["a_old"]:
        print(f"{'ALL':10s} {len(totals['a_old']):3d} | {mean(totals['a_old']):14.4f} "
              f"{mean(totals['a_new']):10.4f} {(mean(totals['a_new']) - mean(totals['a_old'])) * 100:+8.2f} | "
              f"{mean(totals['f_old']):8.4f} {mean(totals['f_new']):8.4f} "
              f"{(mean(totals['f_new']) - mean(totals['f_old'])) * 100:+8.2f}")


if __name__ == "__main__":
    main()
