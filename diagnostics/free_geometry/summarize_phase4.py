#!/usr/bin/env python3
"""Phase-4 gating summary: stacked arms + student control vs C2M/B5 on sub8.

For each dataset, reads:
  final_protocol_phase4/<ds>/eval32_metrics.json   (A0 + C2M_CamRel/CONFD_REL/C2M_CTM)
  final_protocol_studentctl/<ds>/eval32_metrics.json (C2M on clustered-student control)
  final_protocol/<ds>/eval32_metrics_C2M.json + eval32_metrics.json (C2M, B5 refs)
Prints per-arm means over the 2 sub8 scenes + paired delta vs C2M_maskrel.
Run from repo root: python diagnostics/free_geometry/summarize_phase4.py
"""

import json
import os

MAIN_VIEW = {"scannetpp": "100v", "7scenes": "100v", "hiroom": "allv", "eth3d": "allv"}
SUB8 = {
    "scannetpp": ["7831862f02", "bde1e479ad"],
    "7scenes": ["chess", "office"],
    "hiroom": ["20241230/828738/cam_sampled_08", "20241230/828749/cam_sampled_12"],
    "eth3d": ["courtyard", "office"],
}
ARMS = ["A0_baseline", "C2M_maskrel", "B5_maskdistill", "C2M_studentctl",
        "C2M_CamRel", "CONFD_REL", "C2M_CTM"]


def load(path):
    if not os.path.exists(path):
        return {}
    m = json.load(open(path))
    out = {}
    for k, v in m.items():
        if "@" not in k:
            continue
        exp, view = k.rsplit("@", 1)
        if (exp, view) in out:
            continue
        rec = {}
        for sect, sv in v.items():
            if sect.endswith("_pose"):
                for s, x in sv.items():
                    if s != "mean":
                        rec.setdefault(s, {})["AUC@3"] = x["auc03"]
            if sect.endswith("_recon_unposed"):
                for s, x in sv.items():
                    if s != "mean":
                        rec.setdefault(s, {}).update(F1=x["fscore"], CD=x["overall"])
        out[(exp, view)] = rec
    return out


def load_glob(pattern):
    """Merge all matching metric files; first file (sorted) wins per key."""
    import glob
    out = {}
    for p in sorted(glob.glob(pattern)):
        for k, v in load(p).items():
            out.setdefault(k, v)
    return out


def main():
    lines = ["# Phase-4 sub8 gating (2 scenes/dataset; main eval view)", ""]
    for ds in ("scannetpp", "7scenes", "hiroom", "eth3d"):
        v = MAIN_VIEW[ds]
        p4 = load_glob(f"artifacts/diagnostics/final_protocol_phase4/{ds}/eval32_metrics*.json")
        ctl = load(f"artifacts/diagnostics/final_protocol_studentctl/{ds}/eval32_metrics.json")
        ref_c2m = load(f"artifacts/diagnostics/final_protocol/{ds}/eval32_metrics_C2M.json")
        ref_b5 = load(f"artifacts/diagnostics/final_protocol/{ds}/eval32_metrics.json")
        data = {}
        for exp in ("A0_baseline", "C2M_CamRel", "CONFD_REL", "C2M_CTM"):
            data[exp] = p4.get((exp, v), {})
        data["C2M_studentctl"] = ctl.get(("C2M_maskrel", v), {})
        data["C2M_maskrel"] = ref_c2m.get(("C2M_maskrel", v), {})
        data["B5_maskdistill"] = ref_b5.get(("B5_maskdistill", v), {})
        scenes = SUB8[ds]
        lines.append(f"## {ds} ({v})")
        lines.append("| arm | AUC@3 | d vs C2M | F1 | d vs C2M | CD |")
        lines.append("|---|---|---|---|---|---|")
        ref = data["C2M_maskrel"]
        for arm in ARMS:
            rec = data[arm]
            if not all(s in rec for s in scenes):
                lines.append(f"| {arm} | MISSING | | | | |")
                continue
            auc = sum(rec[s]["AUC@3"] for s in scenes) / 2
            f1 = sum(rec[s]["F1"] for s in scenes) / 2
            cd = sum(rec[s]["CD"] for s in scenes) / 2
            if all(s in ref for s in scenes) and arm not in ("C2M_maskrel",):
                da = auc - sum(ref[s]["AUC@3"] for s in scenes) / 2
                df = f1 - sum(ref[s]["F1"] for s in scenes) / 2
                lines.append(f"| {arm} | {auc:.4f} | {da:+.4f} | {f1:.4f} | {df:+.4f} | {cd:.4f} |")
            else:
                lines.append(f"| {arm} | {auc:.4f} | - | {f1:.4f} | - | {cd:.4f} |")
        lines.append("")
    out = "\n".join(lines) + "\n"
    with open("artifacts/diagnostics/final_protocol/PHASE4_GATING.md", "w") as f:
        f.write(out)
    print(out)


if __name__ == "__main__":
    main()
