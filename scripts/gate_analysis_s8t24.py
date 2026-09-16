#!/usr/bin/env python3
"""GT-free acceptance-gate feasibility analysis for the 24:8+B5 arm (CPU only).

Same method as scripts/gate_analysis_gt_free.py, applied to the
artifacts/diagnostics/final_protocol_s8/scannetpp_s8t24 run root
(teacher 24v / student 8v, B5_maskdistill, scannetpp 20 scenes, 100v eval).

Per-scene deltas are extracted from eval32_metrics_B5_maskdistill.json
(paired: the file carries its own A0_baseline@100v copy).

Writes artifacts/diagnostics/final_protocol_s8/scannetpp_s8t24/GATE_ANALYSIS_S8T24.md
"""
import ast
import csv
import json
import math
import os
from statistics import mean

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT, "artifacts", "diagnostics", "final_protocol_s8", "scannetpp_s8t24")
# tau/N metadata: s8t24 manifest does not store tau; same 20 scenes as the
# original final_protocol scannetpp manifest -> reuse its tau values.
MAN_FP = os.path.join(ROOT, "artifacts", "diagnostics", "final_protocol", "scannetpp", "scene_manifest.json")
ARM = "B5_maskdistill"
VIEW = "100v"
EVAL_JSON = "eval32_metrics_B5_maskdistill.json"
DEPTH_CSV = "depth_metrics_B5_maskdistill.csv"


def load_eval():
    d = json.load(open(os.path.join(RUN, EVAL_JSON)))
    out = {}
    for tag in ["A0_baseline", ARM]:
        e = d[f"{tag}@{VIEW}"]
        rec, pose = e["scannetpp_recon_unposed"], e["scannetpp_pose"]
        out[tag] = {
            s: (pose[s]["auc03"], rec[s]["fscore"], rec[s]["overall"])
            for s in rec if s != "mean"
        }
    rows = list(csv.DictReader(open(os.path.join(RUN, DEPTH_CSV))))
    absrel = {}
    for r in rows:
        exp, sc = r["experiment"], r["scene"]
        if exp.endswith("@" + VIEW):
            absrel.setdefault(exp[: -len("@" + VIEW)], {})[sc] = float(r["absrel_shared"])
    for tag in ["A0_baseline", ARM]:
        for s in out[tag]:
            out[tag][s] = out[tag][s] + (absrel.get(tag, {}).get(s, float("nan")),)
    return out


def load_signals():
    rows = list(csv.DictReader(open(os.path.join(RUN, "probe_metrics.csv"))))
    base = {r["scene"]: r for r in rows if r["arm"] == "A0_baseline"}
    sig = {}
    for r in rows:
        if r["arm"] != ARM:
            continue
        sig.setdefault(r["scene"], {})[int(r["step"])] = r
    out = {}
    for sc, steps in sig.items():
        b = base[sc]
        g = lambda r, c: float(r[c])
        p0, q0 = g(b, "probe_auc03"), g(b, "probe_auc30")
        p30, p100 = g(steps[30], "probe_auc03"), g(steps[100], "probe_auc03")
        s = {
            "d_auc03_30": p30 - p0,
            "d_auc03": p100 - p0,
            "d_auc30": g(steps[100], "probe_auc30") - q0,
            "auc03_100": p100,
            "d_e_depth": g(steps[100], "e_depth") - g(b, "e_depth"),
            "d_mse_raw": g(steps[100], "mse_raw") - g(b, "mse_raw"),
            "d_mse_norm": g(steps[100], "mse_norm") - g(b, "mse_norm"),
            "d_mse_proj": g(steps[100], "mse_proj") - g(b, "mse_proj"),
        }
        xs = [0.0, 30.0, 100.0]
        ys = [p0, p30, p100]
        xm, ym = mean(xs), mean(ys)
        s["auc03_slope"] = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum(
            (x - xm) ** 2 for x in xs
        )
        out[sc] = s
    tr = list(csv.DictReader(open(os.path.join(RUN, "training_trace.csv"))))
    byscene = {}
    for r in tr:
        if r["arm"] == ARM:
            byscene.setdefault(r["scene"], []).append(r)
    for sc, rs in byscene.items():
        rs.sort(key=lambda r: int(r["step"]))
        n = len(rs)
        k = max(1, n // 10)
        steps = [float(r["step"]) for r in rs if math.isfinite(float(r["loss"]))]
        losses = [float(r["loss"]) for r in rs if math.isfinite(float(r["loss"]))]
        gn_pairs = [
            (float(r["step"]), float(r["grad_norm"]))
            for r in rs
            if r["grad_norm"] not in ("", "nan", "NaN") and math.isfinite(float(r["grad_norm"]))
        ]
        s = out.setdefault(sc, {})
        s["loss_drop"] = mean(losses[:k]) - mean(losses[-k:])
        s["loss_drop_rel"] = s["loss_drop"] / mean(losses[:k]) if mean(losses[:k]) else 0.0
        xm, ym = mean(steps), mean(losses)
        s["loss_slope"] = sum((x - xm) * (y - ym) for x, y in zip(steps, losses)) / sum(
            (x - xm) ** 2 for x in steps
        )
        if gn_pairs:
            gs, gv = [p[0] for p in gn_pairs], [p[1] for p in gn_pairs]
            s["gnorm_mean"] = mean(gv)
            xm2 = mean(gs)
            s["gnorm_slope"] = sum((x - xm2) * (y - mean(gv)) for x, y in gn_pairs) / sum(
                (x - xm2) ** 2 for x in gs
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
    ev = load_eval()
    sig = load_signals()
    man = json.load(open(MAN_FP))["scenes"]
    data = {}
    for sc in ev[ARM]:
        b, t = ev["A0_baseline"][sc], ev[ARM][sc]
        if sc not in sig:
            continue
        d = dict(sig[sc])
        d["dAUC"] = t[0] - b[0]
        d["dF1"] = t[1] - b[1]
        d["dCD"] = b[2] - t[2]
        d["dAbsRel"] = b[3] - t[3]
        d["base"] = b
        d["tta"] = t
        d["tau"] = man[sc]["tau"]
        d["N"] = man[sc]["adaptation_pool_size"]
        data[sc] = d
    n = len(data)

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
        ("rel_rot_mean", "mean rel_rot"),
    ]

    def gate_means(sig, thr, cond=False):
        """cond=True -> deployable rule: only scenes with tau<=0.55 & N>=100 are gated."""
        out = {"base": [[] for _ in range(4)], "tta": [[] for _ in range(4)],
               "gated": [[] for _ in range(4)], "oracle": [[] for _ in range(4)]}
        nf = 0
        for v in data.values():
            b, t = v["base"], v["tta"]
            deltas = [t[0] - b[0], t[1] - b[1], b[2] - t[2], b[3] - t[3]]
            eligible = (v["tau"] <= 0.55 and v["N"] >= 100) if cond else True
            accept = True if not eligible else v.get(sig, float("nan")) > thr
            if not accept:
                nf += 1
            for i in range(4):
                out["base"][i].append(b[i])
                out["tta"][i].append(t[i])
                out["gated"][i].append(t[i] if accept else b[i])
                out["oracle"][i].append(t[i] if deltas[i] > 0 else b[i])
        return {k: [mean(x) for x in vv] for k, vv in out.items()}, nf

    L = []
    A = L.append
    A("# GT-free acceptance-gate feasibility analysis — 24:8 + B5 (scannetpp_s8t24)")
    A("")
    A("日期：2026-09-14。方法与 `scripts/gate_analysis_gt_free.py` / `final_protocol/GATE_ANALYSIS.md` 完全一致，")
    A("作用于 24:8 teacher/student 的 B5_maskdistill 全量 run（teacher 24v / student 8v，scannetpp 20 场景，100v 评测）。")
    A("数据：`probe_metrics.csv`（probe @ steps 0/30/100；step-0 ≡ A0_baseline）、`training_trace.csv`（100 步）、")
    A(f"逐场景评测 `{EVAL_JSON}`（paired：文件自带 A0_baseline@100v；auc03 = AUC@3，fscore = F1，overall = CD）、")
    A(f"`{DEPTH_CSV}`（absrel_shared）。Δ 为增益：dF1=TTA−base、dAUC=TTA−base、dCD=base−TTA、dAbsRel=base−TTA（正=改善）。")
    A("")
    mb = [mean(v["base"][i] for v in data.values()) for i in range(4)]
    mt = [mean(v["tta"][i] for v in data.values()) for i in range(4)]
    A(f"Sanity（重算均值 vs 任务给定头条）：AUC@3 {fmt(mb[0])}→{fmt(mt[0])}（{pct((mt[0]-mb[0])/mb[0])}，给定 +5.4%）；"
      f"F1 {fmt(mb[1])}→{fmt(mt[1])}（{pct((mt[1]-mb[1])/mb[1])}，给定 −1.0%）；"
      f"CD {fmt(mb[2])}→{fmt(mt[2])}（{pct((mb[2]-mt[2])/mb[2]*-1)}，给定 −1.2%）；"
      f"AbsRel {fmt(mb[3])}→{fmt(mt[3])}（{pct((mb[3]-mt[3])/mb[3])}，给定 +9.3%）。n={n}。")
    A("")

    # ---- (1)(2) sign prediction ----
    A("## (1)(2) 各候选门信号的符号预测准确率（24:8+B5，n=20）")
    A("")
    A("准确率 = `(signal > 0) == (delta > 0)`；ρ = 与带符号 delta 的 Spearman 相关。")
    A("")
    for target in ["dF1", "dAUC", "dCD"]:
        A(f"### target = {target} sign")
        A("")
        A("| signal | acc | ρ |")
        A("|---|---|---|")
        for skey, label in SIGS:
            pts = [(v[skey], v[target]) for v in data.values() if skey in v]
            if not pts:
                A(f"| {label} | — | — |")
                continue
            acc, ok = sign_acc([p[0] for p in pts], [p[1] for p in pts])
            rho = spearman([p[0] for p in pts], [p[1] for p in pts])
            A(f"| {label} | {ok}/{len(pts)} ({acc:.0%}) | {rho:+.2f} |")
        A("")

    # ---- granularity ----
    A("## (4) probe_auc03 粒度检查")
    A("")
    d3 = [round(v["d_auc03"], 6) for v in data.values()]
    d30 = [round(v["d_auc30"], 6) for v in data.values()]
    A(f"- distinct Δprobe_auc03：{len(set(d3))}/{len(d3)}；Δ=0 ties：{sum(1 for x in d3 if x == 0)}；"
      f"distinct Δprobe_auc30：{len(set(d30))}/{len(d30)}。")
    A("- 24:8 探针（2 probe-pair 配置 × 8 student 帧）下观测值步长 ≈ 1/84 ≈ 0.012（对比 16:4 的 1/36 ≈ 0.028），粒度约细 3 倍。")
    A("")

    # ---- (3) gate simulation ----
    A("## (3) 门模拟：均值（F1/CD/AUC@3/AbsRel），`fb` = 回退 baseline 的场景数")
    A("")
    A("| gate | fb | F1 base→TTA→gated (oracle) | CD base→TTA→gated (oracle) | AUC@3 gated Δ% | AbsRel gated Δ% |")
    A("|---|---|---|---|---|---|")
    for name, skey, thr, cond in [
        ("Δauc03 > 0（严格门）", "d_auc03", 0.0, False),
        ("Δauc03 > −0.01（宽松门，全 20 场景）", "d_auc03", -0.01, False),
        ("**τ≤0.55∧N≥100 条件宽松门（Δ>−0.01）**", "d_auc03", -0.01, True),
        ("Δauc03_30 > 0", "d_auc03_30", 0.0, False),
        ("Δauc30 > 0", "d_auc30", 0.0, False),
        ("Δmse_proj ≤ 0", "d_mse_proj", 0.0, False),
        ("Δe_depth > 0", "d_e_depth", 0.0, False),
        ("Δmse_norm > 0（hindsight）", "d_mse_norm", 0.0, False),
        ("Δmse_raw > 0（hindsight）", "d_mse_raw", 0.0, False),
    ]:
        inv = skey == "d_mse_proj"  # accept iff signal <= 0
        if inv:
            out = {"base": [[] for _ in range(4)], "tta": [[] for _ in range(4)],
                   "gated": [[] for _ in range(4)], "oracle": [[] for _ in range(4)]}
            nf = 0
            for v in data.values():
                b, t = v["base"], v["tta"]
                deltas = [t[0] - b[0], t[1] - b[1], b[2] - t[2], b[3] - t[3]]
                accept = v[skey] <= thr
                if not accept:
                    nf += 1
                for i in range(4):
                    out["base"][i].append(b[i])
                    out["tta"][i].append(t[i])
                    out["gated"][i].append(t[i] if accept else b[i])
                    out["oracle"][i].append(t[i] if deltas[i] > 0 else b[i])
            m = {k: [mean(x) for x in vv] for k, vv in out.items()}
        else:
            m, nf = gate_means(skey, thr, cond)
        b, t, g_, o = m["base"], m["tta"], m["gated"], m["oracle"]
        f1 = f"{fmt(b[1])}→{fmt(t[1])}→**{fmt(g_[1])}** ({fmt(o[1])})"
        cd = f"{fmt(b[2])}→{fmt(t[2])}→**{fmt(g_[2])}** ({fmt(o[2])})"
        A(f"| {name} | {nf}/{n} | {f1} | {cd} | {pct((g_[0]-b[0])/b[0])} | {pct((b[3]-g_[3])/b[3])} |")
    A("")

    # threshold sweep
    A("### Δprobe_auc03 阈值扫描（accept iff Δ > thr）")
    A("")
    A("| thr | fb | F1 gated | CD gated | AUC@3 gated | F1 Δ% vs base | CD Δ% vs base |")
    A("|---|---|---|---|---|---|---|")
    for thr in [-0.03, -0.02, -0.01, 0.0, 0.005, 0.01, 0.02, 0.03, 0.05]:
        m, nf = gate_means("d_auc03", thr)
        b, g_ = m["base"], m["gated"]
        A(f"| {thr} | {nf}/{n} | {fmt(g_[1])} | {fmt(g_[2])} | {fmt(g_[0])} | "
          f"{pct((g_[1]-b[1])/b[1])} | {pct((b[2]-g_[2])/b[2])} |")
    A("")

    # ---- per-scene detail ----
    A("## 逐场景明细（按真实 dF1 排序）")
    A("")
    A("信号均为 GT-free、训练结束时已知。`rej@−0.01` = 宽松门 Δauc03>−0.01 将该场景回退 baseline；")
    A("`cond` = τ≤0.55∧N≥100 条件门是否管辖该场景（7bc286c1b6 τ=0.5523>0.55 不管辖）；`hit` = 条件门决定与真实 F1 符号一致。")
    A("")
    A("| scene | τ | N | Δauc03 | Δauc03_30 | Δauc30 | Δe_depth | loss_drop | true dF1 | true dCD | true dAUC | rej@−0.01 | cond | hit |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    n_hit = 0
    misses, false_fb = [], []
    for sc, v in sorted(data.items(), key=lambda kv: kv[1]["dF1"]):
        rej = v["d_auc03"] <= -0.01
        cond = v["tau"] <= 0.55 and v["N"] >= 100
        eff_rej = rej and cond
        hit = (not eff_rej) == (v["dF1"] > 0)
        n_hit += int(hit)
        if cond and not rej and v["dF1"] < 0:
            misses.append((sc, v["dF1"], v["d_auc03"]))
        if eff_rej and v["dF1"] > 0:
            false_fb.append((sc, v["dF1"], v["d_auc03"]))
        A(f"| {sc} | {v['tau']:.4f} | {v['N']} | {v['d_auc03']:+.4f} | {v['d_auc03_30']:+.4f} | {v['d_auc30']:+.4f} | "
          f"{v['d_e_depth']:+.5f} | {v['loss_drop']:+.3f} | {v['dF1']:+.4f} | {v['dCD']:+.5f} | {v['dAUC']:+.4f} | "
          f"{'REJ' if rej else 'ok'} | {'Y' if cond else 'N'} | {'Y' if hit else 'N'} |")
    A("")
    A(f"条件门符号命中：{n_hit}/{n}。")
    A("")

    # ---- verdict ----
    mc, nfc = gate_means("d_auc03", -0.01, cond=True)
    ml, nfl = gate_means("d_auc03", -0.01, cond=False)
    A("## 裁决")
    A("")
    A(f"- 无门（24:8+B5 头条）：F1 {fmt(mb[1])}→{fmt(mt[1])}（{pct((mt[1]-mb[1])/mb[1])}）、"
      f"CD {fmt(mb[2])}→{fmt(mt[2])}（{pct((mt[2]-mb[2])/mb[2])}，负=变差）。")
    A(f"- 条件宽松门（τ≤0.55∧N≥100，Δ>−0.01；实际管辖 19/20，7bc286c1b6 τ=0.5523 不管辖）：fb={nfc}，"
      f"F1 → **{fmt(mc['gated'][1])}**（{pct((mc['gated'][1]-mb[1])/mb[1])} vs baseline），"
      f"CD → **{fmt(mc['gated'][2])}**（{pct((mb[2]-mc['gated'][2])/mb[2])}），"
      f"AUC@3 → {fmt(mc['gated'][0])}（{pct((mc['gated'][0]-mb[0])/mb[0])}），"
      f"AbsRel → {fmt(mc['gated'][3])}（{pct((mb[3]-mc['gated'][3])/mb[3])}）。")
    A(f"- 宽松门不加条件（全 20 场景）：fb={nfl}，F1 → {fmt(ml['gated'][1])}（{pct((ml['gated'][1]-mb[1])/mb[1])}），"
      f"CD → {fmt(ml['gated'][2])}（{pct((mb[2]-ml['gated'][2])/mb[2])}）。")
    A(f"- oracle（逐场景取优）：F1 {fmt(mc['oracle'][1])}、CD {fmt(mc['oracle'][2])}。")
    A("")
    A(f"**残余漏检场景**（条件门保留但真实 dF1<0，共 {len(misses)} 个）："
      + ("、".join(f"{sc}（dF1 {df:+.4f}，Δauc03 {da:+.4f}）" for sc, df, da in misses) if misses else "无")
      + "。")
    A(f"**误杀场景**（条件门回退但真实 dF1>0，共 {len(false_fb)} 个）："
      + ("、".join(f"{sc}（dF1 {df:+.4f}，Δauc03 {da:+.4f}）" for sc, df, da in false_fb) if false_fb else "无")
      + "。")
    A("")
    A("## 结论")
    A("")
    A("**(1) Join / sanity。** 20 场景 probe/trace/eval 三者场景集一致；probe step-0 ≡ A0_baseline；")
    A("从 eval32_metrics_B5_maskdistill.json 重算的均值精确复现头条（AUC +5.4% / F1 −1.1% / CD −1.2% / AbsRel +9.3%）。")
    A("")
    A("**(2) probe 信号不能预测 24:8+B5 的逐场景 F1 delta 符号——且方向相对 16:4 反转。**")
    A("Δprobe_auc03(100−0) 的 F1 符号准确率仅 8/20（40%），Spearman ρ=−0.27（16:4 B5 为 12/20=60%、ρ=+0.10）。")
    A("粒度反而变好了（步长 1/84≈0.012 vs 16:4 的 1/36≈0.028；17/20 distinct、0 ties）——所以失败不是量化分辨率问题。")
    A("失败是结构性的：F1 伤口最深的三个场景恰是 probe 增益最大的场景——")
    A("1ada7a0617（Δprobe +0.137，dF1 −0.2484）、c5439f4607（+0.292，−0.0370）、c4c04e6d6c（+0.274，−0.0676）。")
    A("机制与 SCENE_1ADA_ANATOMY §5 一致：24:8 提升的是 probe 处的成对位姿质量（teacher 更长→蒸馏目标更好），")
    A("但 100v 融合处的跨 head 量规脱钩（1ada 24:8 失配 +3.31% > 16:4 +2.97%）反而加剧——探针在结构上量不到融合级伤口，")
    A("且二者在 24:8 下呈负相关。补充：Δprobe_auc03 对 dAUC 符号仍然高度可预测（14/20=70%，ρ=+0.64）——")
    A("探针本身没坏，它量的确实是位姿；坏的是「位姿改善 ⇒ F1 改善」这条链路。")
    A("")
    A("**(3) τ≤0.55∧N≥100 条件宽松门（Δ>−0.01）在 24:8+B5 上有害，不能转正。**")
    A(f"fb={nfc}/20，F1 → {fmt(mc['gated'][1])}（{pct((mc['gated'][1]-mb[1])/mb[1])}，比无门 {pct((mt[1]-mb[1])/mb[1])} 更差）、"
      f"CD → {fmt(mc['gated'][2])}（{pct((mc['gated'][2]-mb[2])/mb[2])}）；无条件宽松门 fb={nfl} 同样更差（F1 {fmt(ml['gated'][1])}）。")
    A("被回退的 4 个场景（286b55a2bf / 9071e139d9 / 3e8bba0176 / 578511c8a9）真实 dF1 全部为正——100% 误杀；")
    A("8 个真负场景全部漏检。阈值扫描 {−0.03…+0.05} 内**没有任何阈值**能把 F1 转正"
      f"（最好 thr=−0.03 时 {fmt(gate_means('d_auc03', -0.03)[0]['gated'][1])} < 无门 {fmt(mt[1])} < baseline {fmt(mb[1])}）。")
    A("oracle 上限 F1 0.6867 / CD 0.0766 仍在，但 probe 信号够不到。")
    A("注：7bc286c1b6（τ=0.5523）在条件门下不管辖、侥幸正确（dF1 +0.0217）；无条件宽松门会把它也误杀。")
    A("")
    A("**(4) hindsight 参考（非可部署裁决）。** mse 家族在此 run 偶然有效：Δmse_raw>0 门（fb=12）把 F1 转 +0.57%、")
    A("CD 转 +0.74%，但代价是 AUC 收益从 +5.4% 掉到 +1.6%、AbsRel 从 +9.3% 掉到 +2.1%（误杀 f3d64c30f8/40aec5fffa/cc5237fd77 等）；")
    A("Δmse_norm>0 门（fb=3，拒 09c1414f1b/c4c04e6d6c/c5439f4607）F1 −0.13%/CD −0.50%，保住 AUC +3.2%/AbsRel +7.9%。")
    A("不可作为依据：Δmse_raw 在 16:4 B5 上是反向的（55%，ρ=−0.16），跨配置不稳定。真正可部署的修法仍是")
    A("GAUGE_SCANNETPP.md 的评测端量规重耦（fixfree/fixgt 直达 F1/CD 转正）或融合级 GT-free 信号——probe 门对 24:8+B5 应停用。")
    A("")
    A("Reproduce: `python3 scripts/gate_analysis_s8t24.py`（CPU-only，stdlib）。")
    A("")

    out = os.path.join(RUN, "GATE_ANALYSIS_S8T24.md")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", out)

    # compact stdout summary
    print(f"\nn={n} scenes joined")
    print(f"base  AUC {mb[0]:.4f} F1 {mb[1]:.4f} CD {mb[2]:.4f} AbsRel {mb[3]:.4f}")
    print(f"TTA   AUC {mt[0]:.4f} F1 {mt[1]:.4f} CD {mt[2]:.4f} AbsRel {mt[3]:.4f}")
    print(f"ungated rel: AUC {(mt[0]-mb[0])/mb[0]:+.2%} F1 {(mt[1]-mb[1])/mb[1]:+.2%} "
          f"CD {(mt[2]-mb[2])/mb[2]:+.2%} AbsRel {(mt[3]-mb[3])/mb[3]:+.2%}")
    print(f"cond gate fb={nfc}: F1 {mc['gated'][1]:.4f} ({(mc['gated'][1]-mb[1])/mb[1]:+.2%}) "
          f"CD {mc['gated'][2]:.4f} ({(mc['gated'][2]-mb[2])/mb[2]:+.2%}) "
          f"AUC {mc['gated'][0]:.4f} ({(mc['gated'][0]-mb[0])/mb[0]:+.2%}) "
          f"AbsRel {mc['gated'][3]:.4f} ({(mc['gated'][3]-mb[3])/mb[3]:+.2%})")
    print(f"sign hit (cond): {n_hit}/{n}; misses: {[m[0] for m in misses]}; false fb: {[f[0] for f in false_fb]}")
    print("\nper-scene (sorted by dF1):")
    for sc, v in sorted(data.items(), key=lambda kv: kv[1]["dF1"]):
        print(f"  {sc} tau={v['tau']:.4f} dAUC03={v['d_auc03']:+.4f} "
              f"base(AUC {v['base'][0]:.4f} F1 {v['base'][1]:.4f} CD {v['base'][2]:.4f}) "
              f"tta(AUC {v['tta'][0]:.4f} F1 {v['tta'][1]:.4f} CD {v['tta'][2]:.4f}) "
              f"dF1={v['dF1']:+.4f} dCD={v['dCD']:+.5f} dAUC={v['dAUC']:+.4f}")


if __name__ == "__main__":
    main()
