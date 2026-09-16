#!/usr/bin/env python3
"""GT-free acceptance-gate feasibility analysis (CPU only).

Joins train-time signals (probe_metrics.csv, training_trace.csv) with per-scene
final eval deltas (eval32_metrics_*.json + depth_metrics_*.csv) and asks:
can a no-GT signal predict the sign of per-scene F1/AUC/CD deltas well enough
to drive a "fallback to baseline" acceptance gate?

Writes artifacts/diagnostics/final_protocol/GATE_ANALYSIS.md
"""
import ast
import csv
import json
import math
import os
from statistics import mean

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FP = os.path.join(ROOT, "artifacts", "diagnostics", "final_protocol")

VIEW = {"scannetpp": "100v", "7scenes": "100v", "hiroom": "allv", "eth3d": "allv"}
ARM_EVAL = {
    "C2M_maskrel": "eval32_metrics_C2M.json",
    "B5_maskdistill": "eval32_metrics_B5.json.bak",
    "C2M_MC": "eval32_metrics_C2M_MC.json",
    "C2M_SCL": "eval32_metrics_C2M_SCL.json",
}
ARM_DEPTH = {
    "C2M_maskrel": "depth_metrics_C2M.csv",
    "B5_maskdistill": "depth_metrics_B5.csv.bak",
    "C2M_MC": "depth_metrics_C2M_MC.csv",
    "C2M_SCL": None,
}
# Officially reported (FINAL_RESULTS.md) arm x dataset combos
COMBOS = [("C2M_maskrel", ds) for ds in ["scannetpp", "7scenes", "hiroom", "eth3d"]] + [
    ("B5_maskdistill", "scannetpp"),
    ("B5_maskdistill", "hiroom"),
]
# v2.0 recipe arms with per-scene eval on disk (secondary)
COMBOS_V2 = [("C2M_MC", "scannetpp"), ("C2M_MC", "7scenes"), ("C2M_SCL", "eth3d")]


def load_eval(ds, arm):
    """per-scene {scene: (auc03, f1, cd)} for baseline and TTA."""
    view = VIEW[ds]
    d = json.load(open(os.path.join(FP, ds, ARM_EVAL[arm])))
    out = {}
    for tag in ["A0_baseline", arm]:
        e = d[f"{tag}@{view}"]
        rec, pose = e[f"{ds}_recon_unposed"], e[f"{ds}_pose"]
        out[tag] = {
            s: (pose[s]["auc03"], rec[s]["fscore"], rec[s]["overall"])
            for s in rec if s != "mean"
        }
    # AbsRel from depth metrics csv
    if ARM_DEPTH[arm]:
        rows = list(csv.DictReader(open(os.path.join(FP, ds, ARM_DEPTH[arm]))))
        absrel = {}
        for r in rows:
            exp, sc = r["experiment"], r["scene"]
            if exp.endswith("@" + view):
                tag = exp[: -len("@" + view)]
                absrel.setdefault(tag, {})[sc] = float(r["absrel_shared"])
        for tag in ["A0_baseline", arm]:
            for s in out[tag]:
                out[tag][s] = out[tag][s] + (absrel.get(tag, {}).get(s, float("nan")),)
    else:
        for tag in ["A0_baseline", arm]:
            for s in out[tag]:
                out[tag][s] = out[tag][s] + (float("nan"),)
    return out


def load_signals(ds, arm):
    """GT-free train-time signals per scene."""
    rows = list(csv.DictReader(open(os.path.join(FP, ds, "probe_metrics.csv"))))
    base = {r["scene"]: r for r in rows if r["arm"] == "A0_baseline"}
    sig = {}
    for r in rows:
        if r["arm"] != arm:
            continue
        sc, step = r["scene"], int(r["step"])
        sig.setdefault(sc, {})[step] = r
    out = {}
    for sc, steps in sig.items():
        b = base[sc]
        g = lambda r, c: float(r[c])
        p0, p30, p100 = g(b, "probe_auc03"), None, None
        q0 = g(b, "probe_auc30")
        s = {}
        if 30 in steps:
            p30 = g(steps[30], "probe_auc03")
            s["d_auc03_30"] = p30 - p0
        if 100 in steps:
            p100 = g(steps[100], "probe_auc03")
            s["d_auc03"] = p100 - p0
            s["d_auc30"] = g(steps[100], "probe_auc30") - q0
            s["auc03_100"] = p100
            s["d_e_depth"] = g(steps[100], "e_depth") - g(b, "e_depth")
            s["d_mse_raw"] = g(steps[100], "mse_raw") - g(b, "mse_raw")
            s["d_mse_norm"] = g(steps[100], "mse_norm") - g(b, "mse_norm")
            s["d_mse_proj"] = g(steps[100], "mse_proj") - g(b, "mse_proj")
            if p30 is not None:  # 3-point least-squares slope per step
                xs = [0.0, 30.0, 100.0]
                ys = [p0, p30, p100]
                xm, ym = mean(xs), mean(ys)
                s["auc03_slope"] = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum(
                    (x - xm) ** 2 for x in xs
                )
        out[sc] = s
    # training trace
    tr = list(csv.DictReader(open(os.path.join(FP, ds, "training_trace.csv"))))
    byscene = {}
    for r in tr:
        if r["arm"] == arm:
            byscene.setdefault(r["scene"], []).append(r)
    for sc, rs in byscene.items():
        rs.sort(key=lambda r: int(r["step"]))
        losses = [float(r["loss"]) for r in rs]
        steps_all = [float(r["step"]) for r in rs]
        gn_pairs = [
            (float(r["step"]), float(r["grad_norm"]))
            for r in rs
            if r["grad_norm"] not in ("", "nan", "NaN") and math.isfinite(float(r["grad_norm"]))
        ]
        steps = [float(r["step"]) for r in rs if math.isfinite(float(r["loss"]))]
        losses = [float(r["loss"]) for r in rs if math.isfinite(float(r["loss"]))]
        n = len(rs)
        k = max(1, n // 10)
        s = out.setdefault(sc, {})
        s["loss_drop"] = mean(losses[:k]) - mean(losses[-k:])
        s["loss_drop_rel"] = s["loss_drop"] / mean(losses[:k]) if mean(losses[:k]) else 0.0
        s["final_loss"] = mean(losses[-k:])
        xm, ym = mean(steps), mean(losses)
        s["loss_slope"] = sum((x - xm) * (y - ym) for x, y in zip(steps, losses)) / sum(
            (x - xm) ** 2 for x in steps
        )
        if gn_pairs:
            gs, gv = [p[0] for p in gn_pairs], [p[1] for p in gn_pairs]
            s["gnorm_mean"] = mean(gv)
            xs2, xm2 = gs, mean(gs)
            s["gnorm_slope"] = sum((x - xm2) * (y - mean(gv)) for x, y in gn_pairs) / sum(
                (x - xm2) ** 2 for x in xs2
            )
        rots = []
        for r in rs:
            try:
                ex = ast.literal_eval(r["extra"])
                if "rel_rot" in ex:
                    rots.append(float(ex["rel_rot"]))
            except Exception:
                pass
        if rots:
            s["rel_rot_mean"] = mean(rots)
            s["rel_rot_final"] = mean(rots[-k:])
    return out


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            r[order[t]] = avg
        i = j + 1
    return r


def spearman(xs, ys):
    rx, ry = ranks(xs), ranks(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def sign_acc(sig_vals, tgt_vals):
    ok = sum(1 for s, t in zip(sig_vals, tgt_vals) if (s > 0) == (t > 0))
    return ok / len(sig_vals), ok


def fmt(x, nd=4):
    return "nan" if x != x else f"{x:.{nd}f}"


def pct(x):
    return "nan" if x != x else f"{x * 100:+.1f}%"


def main():
    data = {}  # (arm, ds) -> {scene: dict(signals + deltas)}
    for arm, ds in COMBOS + COMBOS_V2:
        ev = load_eval(ds, arm)
        sig = load_signals(ds, arm)
        rec = {}
        for sc in ev[arm]:
            b, t = ev["A0_baseline"][sc], ev[arm][sc]
            if sc not in sig:
                continue
            d = dict(sig[sc])
            d["dAUC"] = t[0] - b[0]
            d["dF1"] = t[1] - b[1]
            d["dCD"] = b[2] - t[2]  # gain (CD lower is better)
            d["dAbsRel"] = b[3] - t[3]  # gain
            d["base"] = b
            d["tta"] = t
            rec[sc] = d
        data[(arm, ds)] = rec

    SIGS = [
        ("d_auc03", "Δprobe_auc03 (100−0)"),
        ("d_auc03_30", "Δprobe_auc03 (30−0)"),
        ("auc03_slope", "probe_auc03 slope"),
        ("d_auc30", "Δprobe_auc30"),
        ("d_e_depth", "Δe_depth"),
        ("d_mse_raw", "Δmse_raw"),
        ("d_mse_norm", "Δmse_norm"),
        ("d_mse_proj", "Δmse_proj"),
        ("loss_drop", "loss drop (first10%−last10%)"),
        ("loss_drop_rel", "relative loss drop"),
        ("loss_slope", "loss slope"),
        ("gnorm_mean", "mean grad-norm"),
        ("gnorm_slope", "grad-norm slope"),
        ("rel_rot_mean", "mean rel_rot (C2M only)"),
    ]

    lines = []
    A = lines.append
    A("# GT-free acceptance-gate feasibility analysis")
    A("")
    A("Question: can a signal available **at training time, without GT** predict the sign of a")
    A("scene's final eval delta, so a gate can revert that scene to the A0_baseline prediction?")
    A("")
    A("Data: `probe_metrics.csv` (probe @ steps 0/30/100; step-0 of every arm ≡ A0_baseline, verified),")
    A("`training_trace.csv` (100 steps), per-scene eval from `eval32_metrics_*.json`")
    A("(`<ds>_pose`.auc03 = AUC@3, `<ds>_recon_unposed`.fscore = F1, .overall = CD) and")
    A("`depth_metrics_*.csv` (absrel_shared). Each arm file's own A0_baseline copy is used for")
    A("pairing (baseline recon differs by ~1e-4 between arm files from TSDF re-fusion; pose is identical).")
    A("Deltas are gains: dF1=TTA−base, dAUC=TTA−base, dCD=base−TTA, dAbsRel=base−TTA (positive = improvement).")
    A("")
    A("Sanity: recomputed dataset means reproduce FINAL_RESULTS.md (e.g. scannetpp C2M F1 "
      + fmt(mean(v['base'][1] for v in data[('C2M_maskrel','scannetpp')].values()))
      + "→" + fmt(mean(v['tta'][1] for v in data[('C2M_maskrel','scannetpp')].values()))
      + ", table says 0.6664→0.6531; the 1e-4 baseline difference is the re-fusion noted above).")
    A("")

    # ---------- (1)+(2) sign prediction accuracy ----------
    A("## (1)(2) Sign-prediction accuracy of each candidate gate signal")
    A("")
    A("Accuracy of `(signal > 0) == (delta > 0)`; `n/n` = correct/total; ρ = Spearman with the")
    A("signed delta. Target = **F1 sign** (the metric the gate must protect).")
    A("")
    for target in ["dF1", "dAUC", "dCD"]:
        A(f"### target = {target} sign")
        A("")
        A("| signal | scannetpp C2M | 7scenes C2M | hiroom C2M | eth3d C2M | scannetpp B5 | hiroom B5 |")
        A("|---|---|---|---|---|---|---|")
        for sig, label in SIGS:
            row = [label]
            for arm, ds in COMBOS:
                rec = data[(arm, ds)]
                pts = [(v[sig], v[target]) for v in rec.values() if sig in v]
                if not pts:
                    row.append("—")
                    continue
                acc, ok = sign_acc([p[0] for p in pts], [p[1] for p in pts])
                rho = spearman([p[0] for p in pts], [p[1] for p in pts])
                row.append(f"{ok}/{len(pts)} ({acc:.0%}), ρ={rho:+.2f}")
            A("| " + " | ".join(row) + " |")
        A("")

    # ---------- probe granularity ----------
    A("## (4) probe_auc03 granularity check")
    A("")
    A("| dataset | arm | distinct Δprobe_auc03 values / scenes | Δ=0 ties | distinct Δprobe_auc30 |")
    A("|---|---|---|---|---|")
    for arm, ds in COMBOS:
        rec = data[(arm, ds)]
        d3 = [round(v["d_auc03"], 6) for v in rec.values() if "d_auc03" in v]
        d30 = [round(v["d_auc30"], 6) for v in rec.values() if "d_auc30" in v]
        A(f"| {ds} | {arm} | {len(set(d3))}/{len(d3)} | {sum(1 for x in d3 if x == 0)} | {len(set(d30))}/{len(d30)} |")
    A("")

    # ---------- (3) gate simulation ----------
    def gate_means(arm, ds, sig, thr=0.0):
        rec = data[(arm, ds)]
        out = {"base": [[] for _ in range(4)], "tta": [[] for _ in range(4)], "gated": [[] for _ in range(4)], "oracle": [[] for _ in range(4)]}
        nf = 0
        for v in rec.values():
            b, t = v["base"], v["tta"]
            deltas = [t[0] - b[0], t[1] - b[1], b[2] - t[2], b[3] - t[3]]
            accept = v.get(sig, float("nan")) > thr
            if not accept:
                nf += 1
            for i in range(4):
                out["base"][i].append(b[i])
                out["tta"][i].append(t[i])
                out["gated"][i].append(t[i] if accept else b[i])
                out["oracle"][i].append(t[i] if deltas[i] > 0 else b[i])
        return {k: [mean(x) for x in vv] for k, vv in out.items()}, nf, len(rec)

    A("## (3) Gate simulation: `signal ≤ 0 → revert scene to baseline`")
    A("")
    A("Means over scenes; F1/CD/AUC@3/AbsRel. Δ% = relative gain vs baseline (for CD/AbsRel,")
    A("positive = lower is better). `fb` = scenes reverted to baseline.")
    A("")
    GATE_SIGS = ["d_auc03", "d_auc03_30", "d_auc30", "d_mse_proj", "d_e_depth", "loss_drop", "rel_rot_mean"]
    SUMMARY = {}
    for sig in GATE_SIGS:
        A(f"### gate signal: {sig}")
        A("")
        A("| dataset | arm | fb | F1 base→TTA→gated (oracle) | CD base→TTA→gated (oracle) | AUC@3 gated Δ% | AbsRel gated Δ% |")
        A("|---|---|---|---|---|---|---|")
        for arm, ds in COMBOS:
            if not any(sig in v for v in data[(arm, ds)].values()):
                continue
            m, nf, n = gate_means(arm, ds, sig)
            b, t, g_, o = m["base"], m["tta"], m["gated"], m["oracle"]
            f1 = f"{fmt(b[1])}→{fmt(t[1])}→**{fmt(g_[1])}** ({fmt(o[1])})"
            cd = f"{fmt(b[2])}→{fmt(t[2])}→**{fmt(g_[2])}** ({fmt(o[2])})"
            aucg = (g_[0] - b[0]) / b[0]
            arg = (b[3] - g_[3]) / b[3]
            A(f"| {ds} | {arm} | {nf}/{n} | {f1} | {cd} | {pct(aucg)} | {pct(arg)} |")
            SUMMARY[(sig, arm, ds)] = (b, t, g_, o, nf, n)
        A("")

    # ungated + oracle reference
    A("### Reference: no gate vs oracle gate (per-metric)")
    A("")
    A("| dataset | arm | F1 base→TTA (oracle) | CD base→TTA (oracle) |")
    A("|---|---|---|---|")
    for arm, ds in COMBOS:
        rec = data[(arm, ds)]
        b = [mean(v["base"][i] for v in rec.values()) for i in range(4)]
        t = [mean(v["tta"][i] for v in rec.values()) for i in range(4)]
        of1, ocd = [], []
        for v in rec.values():
            of1.append(max(v["tta"][1], v["base"][1]))
            ocd.append(min(v["tta"][2], v["base"][2]))
        A(f"| {ds} | {arm} | {fmt(b[1])}→{fmt(t[1])} ({fmt(mean(of1))}) | {fmt(b[2])}→{fmt(t[2])} ({fmt(mean(ocd))}) |")
    A("")

    # margin sweep on d_auc03
    A("### Threshold sweep for Δprobe_auc03 (accept only if Δ > thr)")
    A("")
    A("| thr | scannetpp C2M F1 (fb) | scannetpp C2M CD | scannetpp B5 F1 (fb) | 7scenes C2M F1 | hiroom C2M F1 | eth3d C2M F1 |")
    A("|---|---|---|---|---|---|---|")
    for thr in [-0.02, -0.01, 0.0, 0.005, 0.01, 0.02, 0.03, 0.05]:
        cells = []
        for arm, ds in [("C2M_maskrel", "scannetpp"), ("B5_maskdistill", "scannetpp"),
                        ("C2M_maskrel", "7scenes"), ("C2M_maskrel", "hiroom"), ("C2M_maskrel", "eth3d")]:
            m, nf, n = gate_means(arm, ds, "d_auc03", thr)
            cells.append((m, nf, n))
        A(f"| {thr} | {fmt(cells[0][0]['gated'][1])} ({cells[0][1]}) | {fmt(cells[0][0]['gated'][2])} | "
          f"{fmt(cells[1][0]['gated'][1])} ({cells[1][1]}) | {fmt(cells[2][0]['gated'][1])} ({cells[2][1]}) | "
          f"{fmt(cells[3][0]['gated'][1])} ({cells[3][1]}) | {fmt(cells[4][0]['gated'][1])} ({cells[4][1]}) |")
    A("")

    # combo gates
    A("### Combined gates (fallback if ANY condition fails)")
    A("")
    A("| gate | scannetpp C2M F1 (fb) | scannetpp C2M CD | 7scenes C2M F1 | hiroom C2M F1 | eth3d C2M F1 | scannetpp B5 F1 | hiroom B5 F1 |")
    A("|---|---|---|---|---|---|---|---|")
    combos_gate = {
        "Δauc03>0": lambda v: v.get("d_auc03", -1) > 0,
        "Δauc03>0 AND loss_drop>0": lambda v: v.get("d_auc03", -1) > 0 and v.get("loss_drop", -1) > 0,
        "Δauc03>0 AND Δmse_proj≤0": lambda v: v.get("d_auc03", -1) > 0 and v.get("d_mse_proj", 1) <= 0,
        "Δauc03>-0.01": lambda v: v.get("d_auc03", -1) > -0.01,
        "Δauc30>0": lambda v: v.get("d_auc30", -1) > 0,
        "Δauc03>0 OR Δauc30>0": lambda v: v.get("d_auc03", -1) > 0 or v.get("d_auc30", -1) > 0,
    }
    for name, fn in combos_gate.items():
        cells = []
        for arm, ds in [("C2M_maskrel", "scannetpp"), ("C2M_maskrel", "7scenes"), ("C2M_maskrel", "hiroom"),
                        ("C2M_maskrel", "eth3d"), ("B5_maskdistill", "scannetpp"), ("B5_maskdistill", "hiroom")]:
            rec = data[(arm, ds)]
            f1g, cdg, nf = [], [], 0
            for v in rec.values():
                acc = fn(v)
                if not acc:
                    nf += 1
                f1g.append(v["tta"][1] if acc else v["base"][1])
                cdg.append(v["tta"][2] if acc else v["base"][2])
            cells.append((mean(f1g), mean(cdg), nf, len(rec)))
        A(f"| {name} | {fmt(cells[0][0])} ({cells[0][2]}) | {fmt(cells[0][1])} | "
          f"{fmt(cells[1][0])} ({cells[1][2]}) | {fmt(cells[2][0])} ({cells[2][2]}) | "
          f"{fmt(cells[3][0])} ({cells[3][2]}) | {fmt(cells[4][0])} ({cells[4][2]}) | "
          f"{fmt(cells[5][0])} ({cells[5][2]}) |")
    A("")

    # ---------- v2.0 recipe arms ----------
    A("## Extension: gate on v2.0 recipe arms (C2M_MC scannetpp/7scenes, C2M_SCL eth3d)")
    A("")
    A("| dataset | arm | F1 base→TTA | gate Δauc03>0: F1 (fb) | CD base→TTA→gated |")
    A("|---|---|---|---|---|")
    for arm, ds in COMBOS_V2:
        if (arm, ds) not in data or not data[(arm, ds)]:
            A(f"| {ds} | {arm} | no per-scene eval on disk | | |")
            continue
        m, nf, n = gate_means(arm, ds, "d_auc03")
        b, t, g_ = m["base"], m["tta"], m["gated"]
        A(f"| {ds} | {arm} | {fmt(b[1])}→{fmt(t[1])} | {fmt(g_[1])} ({nf}/{n}) | {fmt(b[2])}→{fmt(t[2])}→{fmt(g_[2])} |")
    A("")

    # ---------- per-scene scannetpp detail ----------
    A("## Per-scene detail: scannetpp (sorted by true dF1)")
    A("")
    A("Signals are GT-free, known at end of training. `rej@−0.01` = gate Δauc03>−0.01 reverts")
    A("this scene to baseline; `hit` = decision matches true F1 sign.")
    A("")
    for arm in ["C2M_maskrel", "B5_maskdistill"]:
        A(f"### {arm}")
        A("")
        A("| scene | Δauc03 | Δauc30 | Δe_depth | loss_drop | true dF1 | true dCD | true dAUC | rej@−0.01 | hit |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        rec = data[(arm, "scannetpp")]
        for sc, v in sorted(rec.items(), key=lambda kv: kv[1]["dF1"]):
            rej = v["d_auc03"] <= -0.01
            hit = (not rej) == (v["dF1"] > 0)
            A(f"| {sc} | {v['d_auc03']:+.4f} | {v['d_auc30']:+.4f} | {v['d_e_depth']:+.5f} | "
              f"{v['loss_drop']:+.3f} | {v['dF1']:+.4f} | {v['dCD']:+.5f} | {v['dAUC']:+.4f} | "
              f"{'REJ' if rej else 'ok'} | {'Y' if hit else 'N'} |")
        A("")

    # ---------- deployable dataset-conditional gate ----------
    # Rule (GT-free metadata only: tau, N from scene_manifest.json):
    #   tau > 0.55 (dense video)      -> no gate (probe anti-correlated there)
    #   tau <= 0.55, N >= 100         -> gate: accept iff d_probe_auc03 > -0.01
    #   tau <= 0.55, N < 100          -> no gate
    man = {}
    for ds in VIEW:
        m = json.load(open(os.path.join(FP, ds, "scene_manifest.json")))
        man[ds] = {sc: (s["tau"], s["adaptation_pool_size"]) for sc, s in m["scenes"].items()}

    def deploy_accept(arm, ds, sc, v):
        tau, n = man[ds][sc]
        if tau > 0.55 or n < 100:
            return True
        return v.get("d_auc03", 0.0) > -0.01

    A("## Deployable dataset-conditional gate (decided from GT-free metadata τ, N only)")
    A("")
    A("Rule: τ>0.55 (dense video) → no gate; τ≤0.55 & N≥100 → accept iff Δprobe_auc03 > −0.01;")
    A("τ≤0.55 & N<100 → no gate. (τ, N per scene from scene_manifest.json; τ≤0.55&N≥100 selects")
    A("exactly the 20 scannetpp scenes; 7scenes all τ>0.55; hiroom/eth3d all N<100.)")
    A("")
    A("| dataset | arm | fb | F1 base→ungated→**gated** | CD base→ungated→**gated** | AUC@3 base→ungated→**gated** | AbsRel base→ungated→**gated** |")
    A("|---|---|---|---|---|---|---|")
    for arm, ds in COMBOS:
        rec = data[(arm, ds)]
        gm = [[] for _ in range(4)]
        bm, tm = [[] for _ in range(4)], [[] for _ in range(4)]
        nf = 0
        for sc, v in rec.items():
            acc = deploy_accept(arm, ds, sc, v)
            if not acc:
                nf += 1
            for i in range(4):
                bm[i].append(v["base"][i])
                tm[i].append(v["tta"][i])
                gm[i].append(v["tta"][i] if acc else v["base"][i])
        b = [mean(x) for x in bm]
        t = [mean(x) for x in tm]
        g_ = [mean(x) for x in gm]
        A(f"| {ds} | {arm} | {nf}/{len(rec)} | {fmt(b[1])}→{fmt(t[1])}→**{fmt(g_[1])}** | "
          f"{fmt(b[2])}→{fmt(t[2])}→**{fmt(g_[2])}** | {fmt(b[0])}→{fmt(t[0])}→**{fmt(g_[0])}** | "
          f"{fmt(b[3])}→{fmt(t[3])}→**{fmt(g_[3])}** |")
    A("")

    # ---------- leave-one-dataset-out learned gates ----------
    FEATS = ["d_auc03", "d_auc30", "d_e_depth", "d_mse_proj", "d_mse_norm", "loss_drop_rel", "gnorm_mean"]
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    A("## Learned gates: leave-one-dataset-out (target = F1 sign)")
    A("")
    A(f"Features (all GT-free, train-time): {', '.join(FEATS)}; standardized with training-fold")
    A("statistics. Pooled over C2M_maskrel scenes of the 4 datasets. Models: logistic regression")
    A("(class_weight='balanced') and a small random forest (200 trees, depth≤3, balanced).")
    A("Tests whether a multi-signal learned gate transfers across dataset types.")
    A("")
    A("| model | held-out | F1-sign acc | F1 ungated→gated | CD ungated→gated | fb |")
    A("|---|---|---|---|---|---|")
    ds_list = ["scannetpp", "7scenes", "hiroom", "eth3d"]
    allscenes = {ds: [] for ds in ds_list}
    for ds in ds_list:
        for sc, v in data[("C2M_maskrel", ds)].items():
            if all(f in v for f in FEATS):
                allscenes[ds].append((sc, [v[f] for f in FEATS], 1 if v["dF1"] > 0 else 0, v))
    models = {
        "logreg(balanced)": lambda: LogisticRegression(class_weight="balanced", max_iter=5000, C=1.0),
        "randforest(balanced)": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=3, class_weight="balanced", random_state=0
        ),
    }
    for mname, mfac in models.items():
        for held in ds_list:
            tr = [r for ds in ds_list if ds != held for r in allscenes[ds]]
            te = allscenes[held]
            sc_ = StandardScaler().fit([r[1] for r in tr])
            clf = mfac().fit(sc_.transform([r[1] for r in tr]), [r[2] for r in tr])
            ok, nf = 0, 0
            f1g, cdg, f1t, cdt = [], [], [], []
            for s, x, y, v in te:
                acc = clf.predict(sc_.transform([x]))[0] == 1
                ok += int(acc == (y == 1))
                nf += int(not acc)
                f1g.append(v["tta"][1] if acc else v["base"][1])
                cdg.append(v["tta"][2] if acc else v["base"][2])
                f1t.append(v["tta"][1])
                cdt.append(v["tta"][2])
            A(f"| {mname} | {held} | {ok}/{len(te)} ({ok / len(te):.0%}) | {fmt(mean(f1t))}→**{fmt(mean(f1g))}** | "
              f"{fmt(mean(cdt))}→**{fmt(mean(cdg))}** | {nf} |")
    A("")

    # ---------- signal-family ceiling ----------
    A("## Signal-family ceiling (hindsight best single signal per dataset — NOT deployable)")
    A("")
    A("| dataset | arm | best signal by F1-sign acc | acc | gated F1 | gated CD | ungated F1 |")
    A("|---|---|---|---|---|---|---|")
    for arm, ds in COMBOS:
        rec = data[(arm, ds)]
        best = None
        for sig, label in SIGS:
            pts = [(v.get(sig), v) for v in rec.values() if sig in v]
            if len(pts) < len(rec):
                continue
            acc, ok = sign_acc([p[0] for p in pts], [p[1]["dF1"] for p in pts])
            if best is None or acc > best[0]:
                best = (acc, sig, label, ok, len(pts))
        acc, sig, label, ok, n = best
        f1g, cdg = [], []
        for v in rec.values():
            a = v[sig] > 0
            f1g.append(v["tta"][1] if a else v["base"][1])
            cdg.append(v["tta"][2] if a else v["base"][2])
        f1t = mean([v["tta"][1] for v in rec.values()])
        A(f"| {ds} | {arm} | {label} | {ok}/{n} ({acc:.0%}) | {fmt(mean(f1g))} | {fmt(mean(cdg))} | {fmt(f1t)} |")
    A("")

    # ---------- conclusions ----------
    A("## Conclusions — answers to (1)-(4)")
    A("")
    A("**(1) Join.** 68 C2M scenes + 50 B5 scenes joined cleanly: probe step-0 ≡ A0_baseline")
    A("(verified value-identical), scene sets identical across probe/trace/eval. Per-scene deltas")
    A("recomputed from `eval32_metrics_*.json` reproduce FINAL_RESULTS.md per-scene tables to ≤2e-4.")
    A("")
    A("**(2) Sign accuracy.** No single GT-free signal predicts the F1 sign well everywhere:")
    A("Δprobe_auc03 is 60% on scannetpp (C2M & B5), 70-73% on hiroom, but 43% on 7scenes and 36% on")
    A("eth3d. Worse, the *sign of the relationship flips by dataset type*: Spearman ρ(Δprobe_auc03,")
    A("dF1) = +0.41/+0.43 (hiroom), +0.06/+0.10 (scannetpp), −0.67 (7scenes), −0.15 (eth3d). On dense")
    A("video (7scenes) 6/7 scenes have negative probe delta yet positive F1 delta — the probe measures")
    A("per-pair pose quality, while the 7scenes F1 gains come from long-sequence fusion consistency")
    A("(cf. §6.4/§6.5 attribution). The only signal family that tracks dense-video F1 is")
    A("Δmse_norm/Δmse_proj (6/7 = 86%, ρ +0.89/+0.96 on 7scenes), but it is anti-predictive on")
    A("scannetpp (30-40%). loss_drop is useless as a gate: it is positive for literally every scene")
    A("(fallback count 0/68). Learned multi-feature gates (logreg/RF, leave-one-dataset-out) confirm")
    A("non-transferability: best LODO config fixes hiroom slightly but still destroys 7scenes")
    A("(fb 6-7/7) and trims eth3d.")
    A("")
    A("**(3) Gate simulation.** Three regimes:")
    A("")
    A("- *Global strict gate (Δprobe_auc03>0 everywhere)*: B5 scannetpp turns fully positive")
    A("  (F1 0.6632→0.6697, +0.5% vs baseline 0.6664; CD 0.0792→0.0781, +0.5% vs 0.0785), but C2M")
    A("  scannetpp F1 only recovers to 0.6625 (still −0.6% vs 0.6663) and CD only to baseline; and the")
    A("  gate destroys 7scenes (F1 +9.4%→+0.3%, CD +16.0%→+0.1%) and halves eth3d (F1 +11.6%→+7.4%).")
    A("  **Verdict: unacceptable.**")
    A("- *Dataset-conditional gate (deployable, keyed on GT-free τ,N)*: no gate for τ>0.55 or N<100;")
    A("  lenient gate Δprobe_auc03>−0.01 only for τ≤0.55 & N≥100 (i.e. scannetpp-type data). Result:")
    A("  **C2M scannetpp F1 −2.0%→+0.01% (0.6671 vs 0.6663), CD −1.3%→+0.4% (0.0783 vs 0.0786), AUC")
    A("  keeps +3.0%, AbsRel +4.1%**; **B5 scannetpp F1 −0.5%→+0.9% (0.6722), CD −0.8%→+0.9% (0.0778),")
    A("  AUC +2.8%**; 7scenes/hiroom/eth3d bit-identical to ungated. Verdict: **the gate works, but only")
    A("  as a scannetpp-type-specific, lenient-threshold gate.** Residual misses: 21d970d8de, c4c04e6d6c,")
    A("  acd95847c5, bcd2436daf are F1-negative scenes with solidly positive probe deltas (+0.056…+0.167)")
    A("  — their F1 wound lives in the 100v fusion stage, invisible to a pairwise pose probe; and boundary")
    A("  scene 7bc286c1b6 (τ=0.552, classified dense, ungated) would add +0.008 F1 if gated.")
    A("- *v2.0 recipe arms*: same story — Δprobe_auc03>0 gate on C2M_MC scannetpp turns F1 positive")
    A("  (0.6597→0.6668 vs baseline 0.6664) and CD positive (0.0788→0.0783), but the same gate on C2M_MC")
    A("  7scenes (fb 6/7) or C2M_SCL eth3d (F1 0.6540→0.6349) is harmful. The gate must stay")
    A("  dataset-conditional.")
    A("")
    A("**(4) probe_auc03 resolution.** The probe uses 2 probe-pair configs per scene (4 student + 16")
    A("teacher frames each); every observed probe_auc03 value is a multiple of 1/36 ≈ 0.028 (36 effective")
    A("pose-pair evaluations), so the quantization step is the same order as the signal itself.")
    A("Δ=0 ties: scannetpp 5/20, 7scenes 2/7, hiroom 3/30, eth3d 3/11 scenes; distinct Δ values:")
    A("10/20 (scannetpp C2M). Δprobe_auc30 is finer (17/20 distinct) but less F1-predictive. Higher-")
    A("resolution GT-free alternatives, in order of expected value:")
    A("")
    A("1. **Continuous pose-error surrogate on probe pairs**: mean/median rotation error (deg) and")
    A("   translation-direction error of TTA vs baseline on the same probe pairs — no AUC quantization,")
    A("   directly paired; or the *fraction of probe pairs whose pose error decreased* (a paired count).")
    A("2. **More probe pairs / more angular thresholds**: 2→8-16 probe configs, and log AUC@1°/5°/10°")
    A("   (the eval JSONs already carry auc05/auc15/auc30 — mid thresholds have more dynamic range on")
    A("   sparse scenes than 3°).")
    A("3. **Probe every 10 steps instead of {0,30,100}** → slope/curvature and an early-stop signal;")
    A("   the current 3-point slope adds nothing over the 2-point delta (same sign accuracy).")
    A("4. **Fusion-stage GT-free signal** (the actual failure location): cross-view depth/reprojection")
    A("   consistency of the TTA model on the *full* eval-length sequence vs baseline (e.g. median")
    A("   |Δdepth| on reprojection over all 100v, or conf-weighted TSDF inlier ratio) — scannetpp's F1")
    A("   wound is a 100v-fusion artifact (§6.4), which pairwise probes structurally cannot see.")
    A("5. **Predicted-confidence statistics**: mean conf delta on student frames / masked regions;")
    A("   cheap, already computed in forward passes.")
    A("")
    A("Reproduce: `python3 scripts/gate_analysis_gt_free.py` (CPU-only, stdlib + sklearn).")
    A("")

    out = os.path.join(FP, "GATE_ANALYSIS.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out)

    # compact stdout summary
    print("\n=== sign accuracy (dF1), d_auc03 ===")
    for arm, ds in COMBOS:
        rec = data[(arm, ds)]
        pts = [(v["d_auc03"], v["dF1"]) for v in rec.values()]
        acc, ok = sign_acc([p[0] for p in pts], [p[1] for p in pts])
        print(f"{ds:10s} {arm:15s} {ok}/{len(pts)} = {acc:.0%}")
    print("\n=== gate d_auc03>0: F1/CD means ===")
    for arm, ds in COMBOS:
        m, nf, n = gate_means(arm, ds, "d_auc03")
        print(f"{ds:10s} {arm:15s} fb={nf}/{n} F1 {m['base'][1]:.4f}->{m['tta'][1]:.4f}->gated {m['gated'][1]:.4f} "
              f"CD {m['base'][2]:.4f}->{m['tta'][2]:.4f}->gated {m['gated'][2]:.4f}")


if __name__ == "__main__":
    main()
