#!/usr/bin/env python3
"""Offline selector-variant study (GT used ONLY for this offline evaluation).

For every scene in completed runs we have: baseline (d=0), final-step GT delta,
selector-selected GT delta, and the probe trace. We compare strategies:
  always_baseline, always_final, selector_current, oracle_max(baseline, final)
and one GT-free variant: trust-gated selector (if step-0 probe rot_deg or
feature is an outlier -> distrust probe; accept final when its feature
component improved, else baseline).
Output: mean dAUC/dF1 per strategy per run + overall.
"""
import glob
import json
import os
import statistics as st

RUNS = {
    "da3/eth3d": ("full_da3_eth3d", "da3_eth3d"),
    "da3/7scenes": ("full_da3_7scenes", "da3_7scenes"),
    "da3/scannetpp": ("full_da3_scannetpp", "da3_scannetpp"),
    "da3/hiroom": ("full_da3_hiroom", "da3_hiroom"),
    "vggt/eth3d": ("full16_vggt_eth3d", "vggt_eth3d"),
    "vggt/7scenes": ("full16_vggt_7scenes", "vggt_7scenes"),
    "vggt/scannetpp": ("full16_vggt_scannetpp", "vggt_scannetpp"),
}
AN = "workspace/protocol_v2/analysis"


def scene_step0(trace_path):
    by_step = {}
    for line in open(trace_path):
        r = json.loads(line)
        recs = r.get("records") or ([r] if "components" in r else [])
        by_step.setdefault(int(r["step"]), []).extend(recs)
    if 0 not in by_step:
        return None
    comps = [x["components"] for x in by_step[0] if "components" in x]
    if not comps:
        return None
    out = {}
    for k in comps[0]:
        vals = [c[k] for c in comps if k in c and isinstance(c[k], (int, float))]
        if vals:
            out[k] = st.mean(vals)
    # also feature improvement at final step
    lasts = by_step[max(by_step)]
    lc = [x["components"] for x in lasts if "components" in x]
    if lc and "feature" in comps[0]:
        out["feature_final"] = st.mean(c["feature"] for c in lc if "feature" in c)
    return out


def main():
    strats = ["always_final", "selector", "oracle", "trustgate"]
    agg = {s: {"A": [], "F": []} for s in strats}
    for run, (rd, an) in RUNS.items():
        ap = os.path.join(AN, an + ".json")
        if not os.path.exists(ap):
            continue
        d = json.load(open(ap))
        sel = d.get("selector", {})
        rows = {}
        for r in d.get("scenes", []):
            if "d_f1_rel" in r:
                rows[r["scene"]] = (r["d_auc03_rel"], r["d_f1_rel"])
        # selected-step GT deltas from final_report_table
        frt = json.load(open(os.path.join(AN, "final_report_table.json")))
        sel_gt = {}
        if run in frt and frt[run].get("scenes"):
            for sc, s in frt[run]["scenes"].items():
                if s.get("sel_auc03") is not None and s.get("baseline_auc03"):
                    sel_gt[sc] = (s["sel_auc03"] / s["baseline_auc03"] - 1,
                                  s["sel_f1"] / s["baseline_f1"] - 1)
        per = {s: {"A": [], "F": []} for s in strats}
        for sc, (fa, ff) in rows.items():
            tp = os.path.join("workspace/protocol_v2", rd, "probe_trace", sc + ".jsonl")
            s0 = scene_step0(tp) if os.path.exists(tp) else None
            sa, sf = sel_gt.get(sc, (fa, ff))  # selected GT delta; default final
            decisions = {
                "always_final": (fa, ff),
                "selector": (sa, sf),
                "oracle": (max(0.0, fa), max(0.0, ff)),
            }
            # trust gate: big step-0 rot_deg => teacher unreliable on this scene
            if s0 is not None and "rot_deg" in s0 and "feature" in s0 and "feature_final" in s0:
                trust = s0["rot_deg"] < 0.6  # rad; ~34 deg
                feat_improved = s0["feature_final"] < s0["feature"]
                if trust:
                    tg = (sa, sf)  # probe trustworthy: follow selector
                else:
                    tg = (fa, ff) if feat_improved else (0.0, 0.0)
            else:
                tg = (sa, sf)
            decisions["trustgate"] = tg
            for k, (a, f) in decisions.items():
                per[k]["A"].append(a)
                per[k]["F"].append(f)
                agg[k]["A"].append(a)
                agg[k]["F"].append(f)
        n = len(rows)
        print(f"{run} (n={n})")
        for k in strats:
            print(f"  {k:14} dAUC {100*st.mean(per[k]['A']):+6.2f}%  dF1 {100*st.mean(per[k]['F']):+6.2f}%")
    print("== OVERALL ==")
    for k in strats:
        print(f"  {k:14} dAUC {100*st.mean(agg[k]['A']):+6.2f}%  dF1 {100*st.mean(agg[k]['F']):+6.2f}%  n={len(agg[k]['A'])}")


if __name__ == "__main__":
    main()
