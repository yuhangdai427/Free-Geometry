#!/usr/bin/env python3
"""Summarize DTU/DTU-64 raw+rel campaign results vs baselines.

Reads:
  workspace/overnight/da3_dtu_u100/<scan>/eval.json    (DA3 TTA, 22 scenes)
  workspace/overnight/da3_dtu_u0/<scan>/eval.json      (DA3 zero-update baseline)
  workspace/overnight/vggt_dtu_u100/vggt_cosw1_eval.json  (VGGT baseline+TTA)
  .../da3_dtu64_u100, da3_dtu64_u0, vggt_dtu64_u100    (13 scenes, pose-only)

Reports per model x dataset: scene-macro means for AUC@3/AUC@30 and DTU chamfer
(acc/comp/overall, mm), plus relative improvement (TTA vs baseline).
"""
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path("/localhdd02/yuhang/code/free_geometry_latest/workspace/overnight")


def load_da3(sub):
    out = {}
    d = ROOT / sub
    if not d.exists():
        return out
    for p in sorted(d.glob("*/eval.json")):
        try:
            out[p.parent.name] = json.loads(p.read_text())
        except Exception:
            pass
    return out


def mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(vals) / len(vals) if vals else None


def fmt(v, pct=False):
    if v is None:
        return "—"
    return f"{v * 100:+.2f}%" if pct else f"{v:.4f}"


def rel_gain(base, tta, lower_better=False):
    """Relative improvement of TTA over baseline, in percent."""
    if base is None or tta is None or base == 0:
        return None
    d = (base - tta) / base if lower_better else (tta - base) / base
    return d * 100


def collect(dataset):
    res = {}
    da3_tta = load_da3(f"da3_{dataset}_u100")
    da3_base = load_da3(f"da3_{dataset}_u0")
    res["DA3"] = (da3_base, da3_tta)
    vj = ROOT / f"vggt_{dataset}_u100" / "vggt_cosw1_eval.json"
    if vj.exists():
        v = json.loads(vj.read_text())
        base = {s: r.get("A0_baseline", {}) for s, r in v.items() if "A0_baseline" in r}
        tta = {s: r.get("TTA", {}) for s, r in v.items() if "TTA" in r}
        res["VGGT"] = (base, tta)
    return res


def report(dataset):
    print(f"\n===== {dataset} =====")
    data = collect(dataset)
    for model, (base, tta) in data.items():
        scenes_b = set(base) or set(tta)
        common = sorted(set(base) & set(tta))
        print(f"\n-- {model} (n={len(common)} paired scenes) --")
        if not base:
            print("   baseline 缺失（需 --updates 0 补评）")
        for key, lb in [("auc03", False), ("auc30", False),
                        ("recon_acc", True), ("recon_comp", True),
                        ("recon_overall", True)]:
            bv = [base[s].get(key) for s in common]
            tv = [tta[s].get(key) for s in common]
            bm, tm = mean(bv), mean(tv)
            g = rel_gain(bm, tm, lb)
            if bm is None and tm is None:
                continue
            print(f"   {key:<14} base={fmt(bm):>8}  TTA={fmt(tm):>8}  "
                  f"gain={fmt(g, pct=True) if g is not None else '—'}")
    return data


if __name__ == "__main__":
    for ds in sys.argv[1:] or ["dtu", "dtu64"]:
        report(ds)
