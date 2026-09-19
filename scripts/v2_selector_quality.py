#!/usr/bin/env python3
"""Selector quality from final_report_table.json (per-scene baseline/final/selected).

Reports per run: fallback-free comparison of final-step vs selector-selected
deltas, disaster rescues (final dF1<=-10% -> selected > -10%), harms, and mean
uplift. Also overall fallback accounting from analysis JSONs.
"""
import json

T = "workspace/protocol_v2/analysis/final_report_table.json"


def rel(base, x):
    return (x - base) / base if base else None


def main():
    d = json.load(open(T))
    print(f"{'run':18} {'n':>3} {'dAUC fin':>9} {'dAUC sel':>9} {'dF1 fin':>9} {'dF1 sel':>9} "
          f"{'upliftA':>8} {'upliftF':>8} {'resc':>4} {'harm':>4}")
    for run, blk in d.items():
        scenes = blk.get("scenes") or {}
        rows = []
        for sc, s in scenes.items():
            if s.get("sel_auc03") is None or s.get("final_auc03") is None:
                continue
            ba, bf = s["baseline_auc03"], s["baseline_f1"]
            rows.append({
                "dA_fin": rel(ba, s["final_auc03"]), "dA_sel": rel(ba, s["sel_auc03"]),
                "dF_fin": rel(bf, s["final_f1"]), "dF_sel": rel(bf, s["sel_f1"]),
            })
        if not rows:
            continue
        n = len(rows)
        m = lambda k: sum(r[k] for r in rows) / n
        resc = sum(1 for r in rows if r["dF_fin"] is not None and r["dF_fin"] <= -0.10
                   and r["dF_sel"] is not None and r["dF_sel"] > -0.10)
        harm = sum(1 for r in rows if r["dF_fin"] is not None and r["dF_fin"] > -0.02
                   and r["dF_sel"] is not None and r["dF_sel"] <= -0.10)
        print(f"{run:18} {n:>3} {100*m('dA_fin'):>+8.2f}% {100*m('dA_sel'):>+8.2f}% "
              f"{100*m('dF_fin'):>+8.2f}% {100*m('dF_sel'):>+8.2f}% "
              f"{100*(m('dA_sel')-m('dA_fin')):>+7.2f}pp {100*(m('dF_sel')-m('dF_fin')):>+7.2f}pp "
              f"{resc:>4} {harm:>4}")


if __name__ == "__main__":
    main()
