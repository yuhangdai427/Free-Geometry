#!/usr/bin/env python3
"""Phase-3 summary for the final_protocol run: baseline vs TTA arms tables,
acceptance rate, and sub-3%/decline flags.

Merges metric sources per dataset run_root:
  eval32_metrics.json      (latest run_eval pass, e.g. B5 diagnosis)
  eval32_metrics_C2M.json  (backup of the C2M pass)
  depth_metrics.csv / depth_metrics_C2M.csv (same split)
Experiments are keyed '<arm>@<view>'; baseline = A0_baseline@<main view>.

Run from repo root: python diagnostics/free_geometry/summarize_final.py
"""

import csv
import json
import os

ROOT = os.path.join("artifacts", "diagnostics", "final_protocol")
MAIN_VIEW = {"scannetpp": "100v", "7scenes": "100v", "hiroom": "allv", "eth3d": "allv"}
HIGHER_BETTER = {"AUC@3": True, "F1": True, "d1.25": True, "CD": False, "AbsRel": False}
METRICS = ["AUC@3", "F1", "CD", "AbsRel", "d1.25"]
ARMS = ["C2M_maskrel", "B5_maskdistill"]


def load_ds(ds):
    rr = os.path.join(ROOT, ds)
    table = {}  # (arm, view) -> scene -> metric -> value
    for suf in ("", "_C2M"):
        mp = os.path.join(rr, f"eval32_metrics{suf}.json")
        if not os.path.exists(mp):
            continue
        em = json.load(open(mp))
        for key, sects in em.items():
            if "@" not in key:
                continue
            arm, view = key.rsplit("@", 1)
            if (arm, view) in table:  # first source wins for duplicates
                continue
            rec = {}
            pose = sects.get(f"{ds}_pose", {})
            recon = sects.get(f"{ds}_recon_unposed", {})
            for scene, x in pose.items():
                if scene != "mean":
                    rec.setdefault(scene, {})["AUC@3"] = x["auc03"]
            for scene, x in recon.items():
                if scene != "mean":
                    rec.setdefault(scene, {}).update({"F1": x["fscore"], "CD": x["overall"]})
            table[(arm, view)] = rec
    for suf in ("", "_C2M"):
        dp = os.path.join(rr, f"depth_metrics{suf}.csv")
        if not os.path.exists(dp):
            continue
        for row in csv.DictReader(open(dp)):
            exp = row["experiment"]
            if "@" not in exp:
                continue
            arm, view = exp.rsplit("@", 1)
            rec = table.setdefault((arm, view), {})
            if rec and "AbsRel" in rec.get(row["scene"], {}):
                continue  # first source wins
            if row["scene"] == "mean":
                continue
            rec.setdefault(row["scene"], {}).update({
                "AbsRel": float(row["absrel_shared"]), "d1.25": float(row["d125_shared"])})
    return table


def rel_gain(base, tta, higher_better):
    if base == 0:
        return float("nan")
    d = (tta - base) / base
    return d if higher_better else -d


def main():
    lines = ["# Final-protocol results (baseline vs TTA, same frames/evaluator)", ""]
    accept = {}   # (arm, metric) -> [wins, total]
    cells = []
    for ds in ("scannetpp", "7scenes", "hiroom", "eth3d"):
        table = load_ds(ds)
        v = MAIN_VIEW[ds]
        base = table.get(("A0_baseline", v))
        if not base:
            lines.append(f"## {ds}: NOT READY")
            continue
        for arm in ARMS:
            tta = table.get((arm, v))
            if not tta:
                continue
            scenes = sorted(set(base) & set(tta))
            lines.append(f"## {ds} ({len(scenes)} scenes, eval={v}) — {arm}")
            lines.append("| metric | baseline | TTA | abs delta | relative | >=3%? |")
            lines.append("|---|---|---|---|---|---|")
            for m in METRICS:
                bs = [base[s][m] for s in scenes if m in base[s] and m in tta[s]]
                ts = [tta[s][m] for s in scenes if m in base[s] and m in tta[s]]
                if not bs:
                    continue
                mb, mt = sum(bs) / len(bs), sum(ts) / len(ts)
                rg = rel_gain(mb, mt, HIGHER_BETTER[m])
                ok = "YES" if rg >= 0.03 else ("decline" if rg < 0 else "weak")
                cells.append((ds, arm, m, mb, mt, rg))
                lines.append(f"| {m} | {mb:.4f} | {mt:.4f} | {mt - mb:+.4f} | {rg * 100:+.1f}% | {ok} |")
            lines.append("")
            lines.append("| scene | dAUC@3 | dF1 | dAbsRel |")
            lines.append("|---|---|---|---|")
            for s in scenes:
                da = tta[s]["AUC@3"] - base[s]["AUC@3"] if "AUC@3" in base[s] else float("nan")
                df = tta[s]["F1"] - base[s]["F1"] if "F1" in base[s] else float("nan")
                dr = tta[s]["AbsRel"] - base[s]["AbsRel"] if "AbsRel" in base[s] else float("nan")
                lines.append(f"| {s} | {da:+.4f} | {df:+.4f} | {dr:+.4f} |")
                for m, d in (("AUC@3", da), ("F1", df)):
                    if d == d:
                        a = accept.setdefault((arm, m), [0, 0])
                        a[1] += 1
                        a[0] += int(d > 0)
            lines.append("")
    lines.append("## Acceptance rate")
    for (arm, m), (w, n) in sorted(accept.items()):
        lines.append(f"- {arm}: scenes with positive {m} delta: {w}/{n}"
                     + (f" ({100.0 * w / n:.0f}%)" if n else ""))
    for arm in ARMS:
        ac = [c for c in cells if c[1] == arm]
        if not ac:
            continue
        ok = sum(1 for c in ac if c[5] >= 0.03)
        lines.append(f"- {arm}: metric cells with relative gain >=3%: {ok}/{len(ac)}")
        bad = [c for c in ac if c[5] < 0.03]
        if bad:
            lines.append(f"  - {arm} cells needing diagnosis: "
                         + ", ".join(f"{c[0]}/{c[2]} ({c[5] * 100:+.1f}%)" for c in bad))
    with open(os.path.join(ROOT, "FINAL_RESULTS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[-22:]))
    print(f"\nWrote {os.path.join(ROOT, 'FINAL_RESULTS.md')}")


if __name__ == "__main__":
    main()
