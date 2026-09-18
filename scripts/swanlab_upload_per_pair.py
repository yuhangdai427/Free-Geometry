#!/usr/bin/env python3
"""Re-upload completed training traces to swanlab as PER-PAIR curves.

The global loss-vs-step line mixes 10 interleaved training pairs (tasks) and
is unreadable. For each (scene, arm) this creates one experiment with:
  - global metrics logged with step = training step (1..100)
  - per-pair metrics "pair{pi}/{metric}" logged with step = visit index of
    that pair (1..10), so each pair shows its own clean descent trajectory.

Reads the training_trace.csv files written by train_da3_protocol.py — no GPU,
no retraining; exact same numbers as the runs that produced the checkpoints.
"""
import csv
import os
import sys

ROOT = "/root/autodl-tmp/Free-Geometry"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import swanlab  # noqa: E402

RUNS = [
    ("workspace/diag2_spp_rkdc1h/training_trace.csv", "scannetpp", "rkdc1h"),
    ("workspace/diag2_spp_rkdc1hr/training_trace.csv", "scannetpp", "rkdc1hr"),
    ("workspace/diag2_7s_rkdc1h/training_trace.csv", "7scenes", "rkdc1h"),
    ("workspace/diag2_7s_rkdc1hr/training_trace.csv", "7scenes", "rkdc1hr"),
]

GLOBAL_KEYS = ["loss", "lr", "grad_norm", "maskdistill", "rkd_sh_d",
               "rkd_sh_a", "couple", "rel_rot", "rel_tdir"]
PAIR_KEYS = ["loss", "maskdistill", "rkd_sh_d", "rkd_sh_a", "couple",
             "rel_rot", "rel_tdir", "grad_norm"]


def num(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def main():
    for path, dataset, arm in RUNS:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        scenes = list(dict.fromkeys(r["scene"] for r in rows))
        for scene in scenes:
            srows = sorted((r for r in rows if r["scene"] == scene),
                           key=lambda r: int(r["step"]))
            run = swanlab.init(
                project="free-geometry-tta-pairs",
                experiment_name=f"{scene}_{arm}",
                description=f"{dataset} | {arm} | per-pair curves "
                            f"(x = visit index of that pair) + global (x = step)",
                config={"dataset": dataset, "arm": arm, "scene": scene,
                        "source_csv": path, "n_steps": len(srows)},
            )
            visits = {}
            for r in srows:
                g = {k: num(r.get(k)) for k in GLOBAL_KEYS}
                g = {k: v for k, v in g.items() if v is not None}
                g["pair_idx"] = int(r["pair_idx"])
                run.log(g, step=int(r["step"]))
                pi = int(r["pair_idx"])
                visits[pi] = visits.get(pi, 0) + 1
                pm = {f"pair{pi}/{k}": v for k, v in
                      ((k, num(r.get(k))) for k in PAIR_KEYS) if v is not None}
                run.log(pm, step=visits[pi])
            run.finish()
            print(f"[uploaded] {scene}_{arm}: {len(srows)} steps, "
                  f"{len(visits)} pairs", flush=True)


if __name__ == "__main__":
    main()
