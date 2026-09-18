#!/usr/bin/env python3
"""Protocol v2 morning-report consolidation.

For each (model, dataset) with a completed run, produce the three-way table:
  baseline vs final-step vs selector-selected (the protocol output).
Writes workspace/protocol_v2/analysis/final_report_table.json + .md.
Read-only on run dirs; never touches baselines except reading baselines.json.

Usage: python scripts/v2_final_report.py
"""
import json
import os
import re
import sys

BASELINES = "workspace/protocol_v2/baselines.json"
RUNS = [
    # (model, dataset, kind, path)
    ("da3", "eth3d", "full", "workspace/protocol_v2/full_da3_eth3d"),
    ("da3", "7scenes", "full", "workspace/protocol_v2/full_da3_7scenes"),
    ("da3", "scannetpp", "full", "workspace/protocol_v2/full_da3_scannetpp"),
    ("da3", "hiroom", "full", "workspace/protocol_v2/full_da3_hiroom"),
    ("da3", "dtu", "full", "workspace/protocol_v2/full_da3_dtu"),
    ("da3", "dtu64", "full", "workspace/protocol_v2/full_da3_dtu64"),
    ("vggt", "eth3d", "full16", "workspace/protocol_v2/full16_vggt_eth3d"),
    ("vggt", "7scenes", "full16", "workspace/protocol_v2/full16_vggt_7scenes"),
    ("vggt", "scannetpp", "full16", "workspace/protocol_v2/full16_vggt_scannetpp"),
    ("vggt", "hiroom", "full16", "workspace/protocol_v2/full16_vggt_hiroom"),
]


def find_metrics(node, out):
    if isinstance(node, dict):
        if "auc03" in node or "fscore" in node or "overall" in node:
            out.update(node)
            return
        for v in node.values():
            find_metrics(v, out)


def vggt_scenes(path):
    s = {}
    if not os.path.exists(path):
        return s
    d = json.load(open(path))

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, dict) and ("auc03" in v or "fscore" in v):
                    s.setdefault(k, {}).update(v)
                else:
                    walk(v)
    walk(d)
    return s


def da3_scenes(run_dir):
    p = os.path.join(run_dir, "smoke_summary.json")
    if not os.path.exists(p):
        return {}
    out = {}
    for sc, blk in json.load(open(p))["scenes"].items():
        ev, es = blk.get("eval"), blk.get("eval_selected")
        if ev:
            out[sc] = {
                "final": {"auc03": ev.get("auc03"), "fscore": ev.get("recon_fscore"),
                          "overall": ev.get("recon_overall")},
                "selected": ({"auc03": es.get("auc03"), "fscore": es.get("recon_fscore"),
                              "overall": es.get("recon_overall"),
                              "loaded_step": es.get("actual_loaded_step",
                                                    blk.get("actual_loaded_step"))}
                             if es else None),
            }
    return out


def rel(t, b):
    return None if (t is None or not b) else (t - b) / abs(b)


def main():
    base = json.load(open(BASELINES))
    table = {}
    for model, ds, kind, rd in RUNS:
        bb = base.get(model, {}).get(ds, {}).get("baseline", {})
        if model == "da3":
            scenes = da3_scenes(rd)
            sel_map = {}
            fin_map = {}
            for sc, d in scenes.items():
                fin_map[sc] = d["final"]
                sel_map[sc] = d["selected"] or d["final"]
        else:
            fin_map = vggt_scenes(os.path.join(rd, "eval32_metrics_V2final.json"))
            selx = vggt_scenes(os.path.join(rd, "eval32_metrics_V2selected.json"))
            sel_map = {sc: selx.get(sc) or fin_map.get(sc) for sc in fin_map}
            # rename keys for uniformity
            fin_map = {k: v for k, v in fin_map.items()}
        rows = {}
        for sc in fin_map:
            b = bb.get(sc)
            if not b:
                continue
            f_m, s_m = fin_map[sc], sel_map.get(sc)
            rows[sc] = {
                "baseline_auc03": b.get("auc03"), "baseline_f1": b.get("fscore"),
                "baseline_overall": b.get("overall"),
                "final_auc03": f_m.get("auc03"), "final_f1": f_m.get("fscore"),
                "final_overall": f_m.get("overall"),
                "sel_auc03": (s_m or {}).get("auc03"), "sel_f1": (s_m or {}).get("fscore"),
                "sel_overall": (s_m or {}).get("overall"),
                "sel_step": (s_m or {}).get("loaded_step"),
            }
        def mean(key):
            v = [r[key] for r in rows.values() if r.get(key) is not None]
            return sum(v) / len(v) if v else None
        def dmean(mkey, bkey):
            v = [rel(r[mkey], r[bkey]) for r in rows.values()
                 if r.get(mkey) is not None and r.get(bkey)]
            return sum(v) / len(v) if v else None
        table[f"{model}/{ds}"] = {
            "n_scenes": len(rows),
            "baseline": {"auc03": mean("baseline_auc03"), "f1": mean("baseline_f1"),
                         "overall": mean("baseline_overall")},
            "final": {"auc03": mean("final_auc03"), "f1": mean("final_f1"),
                      "overall": mean("final_overall")},
            "selected": {"auc03": mean("sel_auc03"), "f1": mean("sel_f1"),
                         "overall": mean("sel_overall")},
            "d_rel_final": {"auc03": dmean("final_auc03", "baseline_auc03"),
                            "f1": dmean("final_f1", "baseline_f1"),
                            "overall": dmean("final_overall", "baseline_overall")},
            "d_rel_selected": {"auc03": dmean("sel_auc03", "baseline_auc03"),
                               "f1": dmean("sel_f1", "baseline_f1"),
                               "overall": dmean("sel_overall", "baseline_overall")},
            "scenes": rows,
        }
        print(f"{model}/{ds}: n={len(rows)} "
              f"final dAUC={_fmt(table[f'{model}/{ds}']['d_rel_final']['auc03'])} "
              f"dF1={_fmt(table[f'{model}/{ds}']['d_rel_final']['f1'])} | "
              f"selected dAUC={_fmt(table[f'{model}/{ds}']['d_rel_selected']['auc03'])} "
              f"dF1={_fmt(table[f'{model}/{ds}']['d_rel_selected']['f1'])}")

    os.makedirs("workspace/protocol_v2/analysis", exist_ok=True)
    with open("workspace/protocol_v2/analysis/final_report_table.json", "w") as f:
        json.dump(table, f, indent=1)
    print("-> workspace/protocol_v2/analysis/final_report_table.json")


def _fmt(x):
    return f"{x * 100:+.2f}%" if isinstance(x, float) else "  n/a "


if __name__ == "__main__":
    main()
