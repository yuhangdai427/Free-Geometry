#!/usr/bin/env python3
"""Morning report generator (2026-09-17): both result tables.

Table 1: loss-position ablation (VGGT + DA3, full-patch vs masked-only loss)
  sources: final_protocol_lossall/*/eval32_metrics_*_lossall.json (VGGT),
           workspace/da3_protocol_*_t8s4_lossall/smoke_summary.json (DA3),
           references: frozen champions (masked-only).
Table 2: 3-model migration (omega/pi3/dvlt): a0 baseline + rkdc_allpos paired
  deltas per dataset, from workspace/fgmig/runs/*/metrics.json.
Writes workspace/fgmig/MORNING_REPORT.md
"""
import glob
import json
import os

import numpy as np

ROOT = "/root/autodl-tmp/Free-Geometry"
OUT = os.path.join(ROOT, "workspace/fgmig/MORNING_REPORT.md")


def rel(entries):
    return float(np.mean(entries)) if entries else float("nan")


def table1():
    rows = []
    base_da3 = lambda ds: json.load(open(
        f"{ROOT}/artifacts/diagnostics/final_protocol/da3_baseline/{ds}_baseline.json"))["scenes"]
    rec_da3 = lambda ds: json.load(open(
        f"{ROOT}/artifacts/diagnostics/final_protocol/da3_baseline/{ds}_recon_baseline.json"))["scenes"]
    da3_ref = {"7scenes": (+5.22, +2.56), "eth3d": (+2.95, -1.21),
               "hiroom": (+3.74, +1.57), "scannetpp": (-0.17, +0.36)}
    for ds in ("7scenes", "eth3d", "hiroom", "scannetpp"):
        p = f"{ROOT}/workspace/da3_protocol_{ds}_t8s4_lossall/metrics.json"
        p2 = f"{ROOT}/workspace/da3_protocol_{ds}_t8s4_lossall/smoke_summary.json"
        if os.path.exists(p2):
            s = json.load(open(p2))["scenes"]
            ba, br = base_da3(ds), rec_da3(ds)
            da = [(r['eval']['auc03'] - ba[sc]['auc03']) / ba[sc]['auc03'] * 100
                  for sc, r in s.items() if ba[sc]['auc03'] > 0]
            df = [(r['eval']['recon_fscore'] - br[sc]['fscore']) / br[sc]['fscore'] * 100
                  for sc, r in s.items() if br[sc]['fscore'] > 0]
            rows.append(f"| DA3 × {ds} | {da3_ref[ds][0]:+.2f} / {da3_ref[ds][1]:+.2f} "
                        f"| **{rel(da):+.2f} / {rel(df):+.2f}** (n={len(da)}) |")
        else:
            rows.append(f"| DA3 × {ds} | {da3_ref[ds][0]:+.2f} / {da3_ref[ds][1]:+.2f} | PENDING |")
    vggt_ref = {"7scenes": (+0.62, +14.10), "eth3d": (+32.24, +25.46),
                "hiroom": (+18.54, +18.52), "scannetpp": (+5.50, +2.22)}
    for ds in ("7scenes", "eth3d", "hiroom", "scannetpp"):
        fs = glob.glob(f"{ROOT}/artifacts/diagnostics/final_protocol_lossall/{ds}/"
                       "eval32_metrics_*lossall.json")
        if not fs:
            rows.append(f"| VGGT × {ds} | {vggt_ref[ds][0]:+.2f} / {vggt_ref[ds][1]:+.2f} | PENDING |")
            continue
        m = json.load(open(fs[0]))
        key = [k for k in m if k.startswith("A0_baseline@")][0].split("@")[1]
        pt, pb = m[f"C2M_{'RKDC1H' if 'RKDC1H' in fs[0] else 'maskrel'}@{key}"][f"{ds}_pose"], \
                 m[f"A0_baseline@{key}"][f"{ds}_pose"]
        rt, rb = m[f"C2M_{'RKDC1H' if 'RKDC1H' in fs[0] else 'maskrel'}@{key}"][f"{ds}_recon_unposed"], \
                 m[f"A0_baseline@{key}"][f"{ds}_recon_unposed"]
        da = [(pt[s]['auc03'] - pb[s]['auc03']) / pb[s]['auc03'] * 100
              for s in pt if s != 'mean' and pb[s]['auc03'] > 0]
        df = [(rt[s]['fscore'] - rb[s]['fscore']) / rb[s]['fscore'] * 100
              for s in rt if s != 'mean' and rb[s]['fscore'] > 0]
        rows.append(f"| VGGT × {ds} | {vggt_ref[ds][0]:+.2f} / {vggt_ref[ds][1]:+.2f} "
                    f"| **{rel(da):+.2f} / {rel(df):+.2f}** (n={len(da)}) |")
    return rows


def table2():
    rows, notes = [], []
    for model in ("omega", "pi3", "dvlt"):
        for ds in ("7scenes", "eth3d", "hiroom", "scannetpp"):
            p0 = f"{ROOT}/workspace/fgmig/runs/{model}_{ds}_a0/metrics.json"
            p1 = f"{ROOT}/workspace/fgmig/runs/{model}_{ds}_rkdc_allpos/metrics.json"
            if not (os.path.exists(p0) and os.path.exists(p1)):
                rows.append(f"| {model} × {ds} | PENDING | | |")
                continue
            m0, m1 = json.load(open(p0)), json.load(open(p1))
            common = [s for s in m0 if s in m1]
            if not common:
                rows.append(f"| {model} × {ds} | no common scenes | | |")
                continue
            b_auc = np.mean([m0[s]['auc03'] for s in common])
            da = [(m1[s]['auc03'] - m0[s]['auc03']) / m0[s]['auc03'] * 100
                  for s in common if m0[s]['auc03'] > 0]
            if model == "pi3":
                # F1 protocol-limited: scale-invariant depth x GT-K (no scale fit)
                b_f1, df_s = float('nan'), "n/a*"
            else:
                b_f1 = np.mean([m0[s].get('recon_fscore', np.nan) for s in common])
                dfv = [(m1[s]['recon_fscore'] - m0[s]['recon_fscore']) / m0[s]['recon_fscore'] * 100
                       for s in common if m0[s].get('recon_fscore', 0) > 0]
                df_s = f"**{rel(dfv):+.2f}**"
            rows.append(f"| {model} × {ds} | AUC {b_auc:.3f} / F1 {b_f1:.3f} "
                        f"| **{rel(da):+.2f}** | {df_s} (n={len(common)}) |")
    return rows, notes


def main():
    t1 = table1()
    t2, _ = table2()
    lines = ["# Morning Report 2026-09-17", "",
             "## Table 1 — loss-position ablation (full-patch vs masked-only, dAUC/dF1 %)",
             "", "| cell | masked-only (ref) | full-patch loss |", "|---|---|---|", *t1, "",
             "## Table 2 — 3-model migration (baseline level; paired dAUC/dF1 % vs own a0)",
             "", "| cell | a0 baseline | dAUC@3 | dF1 |", "|---|---|---|---|", *t2, "",
             "Pi3 F1 uses GT intrinsics (GT-K, eval-only, protocol deviation labeled).",
             "DVLT is full fine-tune (no LoRA), lr 1e-5."]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
