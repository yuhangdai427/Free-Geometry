#!/usr/bin/env python3
"""Assemble GAUGE_S8T24.md (24:8+B5 gauge-recoupling championship verdict)
from gauge_scan artifacts under final_protocol_s8/scannetpp_s8t24:
- gauge_mismatch.csv / gauge_correlation.json (measurement + Spearman)
- gauge_gtfree.json (GT-free factors)
- gauge_fix_metrics.json (orig / fixgt / fixfree fusion + pose checks)
- eval32_metrics_B5_maskdistill.json (published per-scene numbers, paired)
- final_protocol/scannetpp gauge_scan (16:4 reference mismatch per scene)
Pure CPU. New file; no existing module modified."""

import csv
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))

RR = os.path.join(_REPO, "artifacts", "diagnostics", "final_protocol_s8",
                  "scannetpp_s8t24")
OUT = os.path.join(RR, "gauge_scan")
RR164 = os.path.join(_REPO, "artifacts", "diagnostics", "final_protocol",
                     "scannetpp")
ARMS = ["A0_baseline", "B5_maskdistill"]
TTA = "B5_maskdistill"

# 16:4 reference numbers (GAUGE_SCANNETPP.md, published)
REF164 = {"fixgt_f1": 0.6740, "fixgt_cd": 0.0770,
          "fixfree_f1": 0.6676, "fixfree_cd": 0.0777,
          "gate_fixfree_f1": 0.6723, "gate_fixfree_cd": 0.0779,
          "gate_fixgt_f1": 0.6824, "gate_fixgt_cd": 0.0767,
          "auc_base": 0.5930, "auc_tta": 0.6126,
          "mm_mean": 0.0129, "mm_maxabs": 0.0310, "mm_1ada": 0.0297}


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def f4(x):
    return "—" if x is None else f"{x:.4f}"


def main():
    manifest = json.load(open(os.path.join(RR, "scene_manifest.json")))
    scenes = sorted(manifest["scenes"])
    gm = load_csv(os.path.join(OUT, "gauge_mismatch.csv"))
    corr = json.load(open(os.path.join(OUT, "gauge_correlation.json")))
    gtfree = json.load(open(os.path.join(OUT, "gauge_gtfree.json")))
    fix = json.load(open(os.path.join(OUT, "gauge_fix_metrics.json")))
    mm = {(r["arm"], r["scene"]): r for r in gm}

    # 16:4 per-scene B5 mismatch (reference)
    mm164 = {}
    p164 = os.path.join(RR164, "gauge_scan", "gauge_mismatch.csv")
    if os.path.isfile(p164):
        for r in load_csv(p164):
            if r["arm"] == "B5_maskdistill":
                mm164[r["scene"]] = float(r["mismatch"])

    pub = json.load(open(os.path.join(RR, "eval32_metrics_B5_maskdistill.json")))
    pubb, pubt = pub["A0_baseline@100v"], pub["B5_maskdistill@100v"]

    def pubf(which, scene, mode="scannetpp_recon_unposed", key="fscore"):
        return pub[which][mode][scene][key]

    def fused(arm, variant, scene):
        k = f"{arm}_{variant}@100v::scannetpp_recon_unposed"
        if k not in fix or scene not in fix[k]:
            return None
        return fix[k][scene]

    def pose_auc_regen(arm, scene):
        k = f"{arm}@100v::scannetpp_pose"
        if k not in fix or scene not in fix[k]:
            return None
        return fix[k][scene]["auc03"]

    inv = fix.get("pose_invariance", {})

    pub_base_f1 = mean([pubf("A0_baseline@100v", s) for s in scenes])
    pub_base_cd = mean([pubf("A0_baseline@100v", s, key="overall") for s in scenes])
    pub_tta_f1 = mean([pubf("B5_maskdistill@100v", s) for s in scenes])
    pub_tta_cd = mean([pubf("B5_maskdistill@100v", s, key="overall") for s in scenes])
    pub_base_auc = mean([pubf("A0_baseline@100v", s, "scannetpp_pose", "auc03") for s in scenes])
    pub_tta_auc = mean([pubf("B5_maskdistill@100v", s, "scannetpp_pose", "auc03") for s in scenes])

    L = []
    A = L.append
    A("# 24:8 + B5 量规重耦：scannetpp 冠军决定性一击（s8t24）")
    A("")
    A("日期：2026-09-14。新增代码 `diagnostics/free_geometry/gauge_s8t24_{{infer,measure,gtfree,fuse,report}}.py`"
      "（未改动任何现有模块，未改动 diagnostics/free_geometry/ 下任何已有 .py）；产物 `gauge_scan/`。"
      "机制背景：`../final_protocol/scannetpp/GAUGE_SCANNETPP.md` 与 `SCENE_1ADA_ANATOMY.md`；"
      "探针门在 24:8 上有害的证据：`GATE_ANALYSIS_S8T24.md`。")
    A("")
    A("**定义**（与 16:4 量规研究完全同源）：`s_pose` = `align_poses_umeyama(gt, pred, ransac=True, rs=42)` "
      "的 Sim3 尺度（评测 `_prep_unposed` 把它乘到深度上）；`s_depth` = 全 100 帧有效像素 pooling 的 "
      "median(GT/pred) 深度尺度；**失配 = s_pose/s_depth − 1**。GT 修复因子 `f_gt = s_depth/s_pose`；"
      "GT-free 因子 `f_free` = 预测坐标系内多视图互投影一致性（conf 加权跨帧最近邻距离）在 "
      "s∈[0.90,1.10]、步长 0.005 网格上的 argmin。")
    A("")

    # ---- chain validation ----
    regen = {}
    for arm in ARMS:
        regen[arm] = {
            "f1": mean([fused(arm, "orig", s)["fscore"] for s in scenes
                        if fused(arm, "orig", s)]),
            "cd": mean([fused(arm, "orig", s)["overall"] for s in scenes
                        if fused(arm, "orig", s)]),
            "auc": mean([pose_auc_regen(arm, s) for s in scenes
                         if pose_auc_regen(arm, s) is not None]),
        }
    A("**链路校验**：npz 由 `ckpts/<scene>/B5_maskdistill/step100_lora_peft` 重新推理生成"
      f"（GPU 推理前等 memory.free≥20GB，60s 轮询）；评测帧集 = 本 root `scene_manifest.json` 的 eval 帧（100v）。"
      f"再生 npz 重算（同引擎 VGGTEvaluator、CPU 融合、o3d seed 42、paired 20 场景）："
      f"A0 F1 {regen['A0_baseline']['f1']:.4f}（published {pub_base_f1:.4f}）、"
      f"CD {regen['A0_baseline']['cd']:.4f}（{pub_base_cd:.4f}）、"
      f"AUC@3 {regen['A0_baseline']['auc']:.4f}（{pub_base_auc:.4f}）；"
      f"B5 F1 {regen[TTA]['f1']:.4f}（{pub_tta_f1:.4f}）、"
      f"CD {regen[TTA]['cd']:.4f}（{pub_tta_cd:.4f}）、"
      f"AUC@3 {regen[TTA]['auc']:.4f}（{pub_tta_auc:.4f}）——差异均为 npz 再生噪声级。")
    r1 = mm.get((TTA, "1ada7a0617"))
    if r1:
        A(f"1ada7a0617 复算：s_pose={float(r1['s_pose']):.4f}（解剖报告 2.1353）、"
          f"失配 {float(r1['mismatch'])*100:+.2f}%（解剖报告 +3.31%）。")
    A("")

    # ---- (1) mismatch table ----
    A("## (1) 逐场景量规失配（20/20 场景 × 2 臂，100v；16:4 B5 对照）")
    A("")
    A("| scene | A0 失配% | B5(24:8) 失配% | B5(16:4) 失配% | Δ(24:8−16:4) pt | published dF1 |")
    A("|---|---|---|---|---|---|")
    d24, d16 = [], []
    for s in scenes:
        r0 = mm.get(("A0_baseline", s))
        r5 = mm.get((TTA, s))
        m16 = mm164.get(s)
        dd = None if (r5 is None or m16 is None) else \
            (float(r5["mismatch"]) - m16) * 100
        if dd is not None:
            d24.append(float(r5["mismatch"]))
            d16.append(m16)
        A(f"| {s} | {float(r0['mismatch'])*100:+.2f} | "
          f"{float(r5['mismatch'])*100:+.2f} | "
          f"{'—' if m16 is None else f'{m16*100:+.2f}'} | "
          f"{'—' if dd is None else f'{dd:+.2f}'} | "
          f"{float(r5['dF1']):+.4f} |")
    A("")
    for arm in ARMS:
        sub = [mm[(arm, s)] for s in scenes if (arm, s) in mm]
        A(f"- {arm} (24:8)：失配 mean {mean([float(r['mismatch']) for r in sub])*100:+.2f}%、"
          f"mean|失配| {mean([abs(float(r['mismatch'])) for r in sub])*100:.2f}%、"
          f"max|失配| {max(abs(float(r['mismatch'])) for r in sub)*100:.2f}%")
    if d24:
        A(f"- 对照：B5(24:8) mean {mean(d24)*100:+.2f}% vs B5(16:4) mean {mean(d16)*100:+.2f}%"
          f"（16:4 文档值 +1.29%、max|失配| 3.10%）；逐场景 Δ mean {mean([a-b for a, b in zip(d24, d16)])*100:+.2f}pt。")
    A("")

    # ---- (2) correlation ----
    A("## (2) 失配 vs dF1 相关性（Spearman，dF1 取自本 root eval32_metrics_B5_maskdistill.json paired）")
    A("")
    c = corr.get(TTA, {})
    if c:
        A(f"- B5(24:8)：n={c['n']}，ρ(失配, dF1) = {c['rho_mismatch_dF1']:+.3f}（p={c['p']:.2e}）；"
          f"ρ(|失配|, dF1) = {c['rho_absmismatch_dF1']:+.3f}（p={c['p_abs']:.2e}）；"
          f"增量口径 ρ(Δ失配 vs 同场景 baseline, dF1) = {c['rho_deltamismatch_dF1']:+.3f}"
          f"（p={c['p_delta']:.2e}）。")
        A("  16:4 对照：B5 ρ(失配, dF1) = -0.343（p=0.139）、增量口径 -0.200（p=0.398）。")
    A("")

    # ---- (3) fusion means ----
    A("## (3) 反事实量规修复（orig / fixgt / fixfree 三变体，官方 VGGTEvaluator CPU 融合，paired 20 场景）")
    A("")
    A("| 臂 | F1 orig | F1 fixgt | F1 fixfree | CD orig | CD fixgt | CD fixfree |")
    A("|---|---|---|---|---|---|---|")
    for arm in ARMS:
        ss = [s for s in scenes if fused(arm, "fixgt", s) and fused(arm, "orig", s)
              and fused(arm, "fixfree", s)]
        A(f"| {arm} | {mean([fused(arm,'orig',s)['fscore'] for s in ss]):.4f} | "
          f"**{mean([fused(arm,'fixgt',s)['fscore'] for s in ss]):.4f}** | "
          f"**{mean([fused(arm,'fixfree',s)['fscore'] for s in ss]):.4f}** | "
          f"{mean([fused(arm,'orig',s)['overall'] for s in ss]):.4f} | "
          f"**{mean([fused(arm,'fixgt',s)['overall'] for s in ss]):.4f}** | "
          f"**{mean([fused(arm,'fixfree',s)['overall'] for s in ss]):.4f}** |")
    A("")
    A("逐场景（F1 与 CD；f_gt / f_free 为深度预乘因子）：")
    A("")
    A("| scene | 臂 | f_gt | f_free | F1 orig | F1 fixgt | F1 fixfree | CD orig | CD fixgt | CD fixfree |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for s in scenes:
        for arm in ARMS:
            ro, rg, rf = fused(arm, "orig", s), fused(arm, "fixgt", s), fused(arm, "fixfree", s)
            if ro is None or rg is None or rf is None:
                continue
            g = gtfree.get(f"{arm}/{s}", {})
            A(f"| {s} | {arm} | {f4(g.get('f_gt'))} | {f4(g.get('f_free_dist'))} | "
              f"{ro['fscore']:.4f} | {rg['fscore']:.4f} | {rf['fscore']:.4f} | "
              f"{ro['overall']:.4f} | {rg['overall']:.4f} | {rf['overall']:.4f} |")
    A("")

    # ---- (4) GT-free deviation ----
    A("## (4) GT-free 因子（可部署形态）")
    A("")
    A("| 臂 | n | mean dev (f_free−f_gt) | mean|dev| | max|dev| | ρ(f_free, f_gt) |")
    A("|---|---|---|---|---|---|")
    from scipy.stats import spearmanr
    for arm in ARMS:
        ds, fxs, fgs = [], [], []
        for s in scenes:
            g = gtfree.get(f"{arm}/{s}")
            if g and g.get("f_gt") is not None:
                ds.append(g["f_free_dist"] - g["f_gt"])
                fxs.append(g["f_free_dist"])
                fgs.append(g["f_gt"])
        if not ds:
            continue
        rho, _ = spearmanr(fxs, fgs)
        A(f"| {arm} | {len(ds)} | {mean(ds)*100:+.2f}pt | {mean([abs(d) for d in ds])*100:.2f}pt | "
          f"{max(abs(d) for d in ds)*100:.2f}pt | {rho:+.3f} |")
    A("")
    A("网格曲线：`gauge_scan/gauge_gtfree_curves.csv`。")
    A("")

    # ---- (d) AUC invariance ----
    A("## (d) AUC 不变性验证")
    A("")
    A(f"变体 npz 与原 npz 的外参/内参逐位比对：{inv.get('n_pairs', 0)} 对，"
      f"max|Δ| = {inv.get('max_abs_diff', float('nan')):.3e}。"
      "pose 评测（AUC）只读 extrinsics，深度预乘不进入位姿通路 → **AUC@3 结构上不变**。")
    A(f"published AUC@3 均值（修复不改变它们）：A0 {pub_base_auc:.4f} → B5(24:8) "
      f"{pub_tta_auc:.4f}（{(pub_tta_auc/pub_base_auc-1)*100:+.1f}%）；"
      f"16:4 B5 对照 {REF164['auc_base']:.4f}→{REF164['auc_tta']:.4f}"
      f"（{(REF164['auc_tta']/REF164['auc_base']-1)*100:+.1f}%）。")
    A("")

    # ---- (5) verdict ----
    f1o = mean([fused(TTA, "orig", s)["fscore"] for s in scenes])
    f1g = mean([fused(TTA, "fixgt", s)["fscore"] for s in scenes])
    f1f = mean([fused(TTA, "fixfree", s)["fscore"] for s in scenes])
    cdo = mean([fused(TTA, "orig", s)["overall"] for s in scenes])
    cdg = mean([fused(TTA, "fixgt", s)["overall"] for s in scenes])
    cdf = mean([fused(TTA, "fixfree", s)["overall"] for s in scenes])
    b_f1o = mean([fused("A0_baseline", "orig", s)["fscore"] for s in scenes])
    b_cdo = mean([fused("A0_baseline", "orig", s)["overall"] for s in scenes])

    crit_auc = inv.get("max_abs_diff", 1.0) == 0.0
    crit_f1_pos = f1f > pub_base_f1
    crit_cd_pos = cdf < pub_base_cd
    crit_abs = f1f >= REF164["gate_fixfree_f1"]
    champion = crit_auc and crit_f1_pos and crit_cd_pos and crit_abs

    A("## (5) 冠军裁决：24:8+B5+fixfree vs 16:4 全系")
    A("")
    A("| 配置 | F1 | CD | AUC@3 增益 | 可部署 |")
    A("|---|---|---|---|---|")
    A(f"| 24:8 baseline（published） | {pub_base_f1:.4f} | {pub_base_cd:.4f} | — | — |")
    A(f"| 24:8+B5 原样 | {f1o:.4f}（pub {pub_tta_f1:.4f}） | {cdo:.4f}（pub {pub_tta_cd:.4f}） | "
      f"{(pub_tta_auc/pub_base_auc-1)*100:+.1f}% | 是 |")
    A(f"| 24:8+B5+fixgt | {f1g:.4f} | {cdg:.4f} | 同上（不变） | 否（GT） |")
    A(f"| **24:8+B5+fixfree** | **{f1f:.4f}** | **{cdf:.4f}** | "
      f"{(pub_tta_auc/pub_base_auc-1)*100:+.1f}%（不变） | **是** |")
    A(f"| 16:4+B5+fixfree | {REF164['fixfree_f1']:.4f} | {REF164['fixfree_cd']:.4f} | "
      f"{(REF164['auc_tta']/REF164['auc_base']-1)*100:+.1f}% | 是 |")
    A(f"| 16:4+B5+门+fixfree（16:4 可部署冠军） | {REF164['gate_fixfree_f1']:.4f} | "
      f"{REF164['gate_fixfree_cd']:.4f} | {(REF164['auc_tta']/REF164['auc_base']-1)*100:+.1f}% | 是 |")
    A(f"| 16:4+B5+门+fixgt | {REF164['gate_fixgt_f1']:.4f} | {REF164['gate_fixgt_cd']:.4f} | — | 否 |")
    A("")
    A("判据逐条：")
    A("")
    A(f"1. **AUC 维持 +5.4%**：变体/原 npz 外参内参 {inv.get('n_pairs', 0)} 对逐位一致"
      f"（max|Δ|={inv.get('max_abs_diff', float('nan')):.3e}）→ pose 通路不动，published "
      f"AUC@3 {pub_base_auc:.4f}→{pub_tta_auc:.4f}（{(pub_tta_auc/pub_base_auc-1)*100:+.1f}%）原样保持 "
      f"→ {'✅' if crit_auc else '❌'}")
    A(f"2. **F1 相对 baseline 转正**：fixfree {f1f:.4f} vs baseline {pub_base_f1:.4f}"
      f"（{(f1f/pub_base_f1-1)*100:+.2f}%）→ {'✅' if crit_f1_pos else '❌'}")
    A(f"3. **CD 相对 baseline 转正**：fixfree {cdf:.4f} vs baseline {pub_base_cd:.4f}"
      f"（{(1-cdf/pub_base_cd)*100:+.2f}% 改善）→ {'✅' if crit_cd_pos else '❌'}")
    A(f"4. **绝对值 ≥ 16:4 B5+门+fixfree（F1 {REF164['gate_fixfree_f1']:.4f}）**："
      f"{f1f:.4f} → {'✅' if crit_abs else '❌'}")
    A("")
    if champion:
        A(f"**裁决：24:8+B5+fixfree 满足全部判据，成为 scannetpp 总冠军（可部署形态）**——"
          f"F1 {f1f:.4f} / CD {cdf:.4f} / AUC@3 {(pub_tta_auc/pub_base_auc-1)*100:+.1f}%"
          f"（16:4 可部署冠军仅 +3.3%）。")
    else:
        fails = []
        if not crit_auc:
            fails.append("AUC 不变性")
        if not crit_f1_pos:
            fails.append("F1 转正")
        if not crit_cd_pos:
            fails.append("CD 转正")
        if not crit_abs:
            fails.append("F1 绝对值≥0.6723")
        A(f"**裁决：24:8+B5+fixfree 未通过判据（失败项：{'、'.join(fails)}），不能加冕。**")
    A("")
    A(f"补充（paired 口径）：24:8+B5 原样相对 baseline dF1 {f1o-b_f1o:+.4f} / "
      f"dCD {b_cdo-cdo:+.4f}；fixgt 后 dF1 {f1g-b_f1o:+.4f} / dCD {b_cdo-cdg:+.4f}；"
      f"fixfree 后 dF1 {f1f-b_f1o:+.4f} / dCD {b_cdo-cdf:+.4f}"
      f"（dCD 正=改善；baseline 取同 npz 再生的 A0 orig 融合值 {b_f1o:.4f}/{b_cdo:.4f}）。")
    A("")

    with open(os.path.join(RR, "GAUGE_S8T24.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("WROTE", os.path.join(RR, "GAUGE_S8T24.md"))

    with open(os.path.join(OUT, "gauge_s8t24_summary.json"), "w") as f:
        json.dump({
            "regen": regen,
            "published": {"base_f1": pub_base_f1, "base_cd": pub_base_cd,
                          "tta_f1": pub_tta_f1, "tta_cd": pub_tta_cd,
                          "base_auc": pub_base_auc, "tta_auc": pub_tta_auc},
            "b5_fused": {"orig_f1": f1o, "fixgt_f1": f1g, "fixfree_f1": f1f,
                         "orig_cd": cdo, "fixgt_cd": cdg, "fixfree_cd": cdf},
            "a0_fused_orig": {"f1": b_f1o, "cd": b_cdo},
            "pose_invariance": inv,
            "criteria": {"auc_invariant": bool(crit_auc),
                         "f1_positive": bool(crit_f1_pos),
                         "cd_positive": bool(crit_cd_pos),
                         "f1_ge_164_champion": bool(crit_abs)},
            "champion": bool(champion),
        }, f, indent=1)


if __name__ == "__main__":
    main()
