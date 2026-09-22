#!/usr/bin/env python3
"""Consolidate ALL existing per-scene results into one JSON.

Covers (everything that has per-scene numbers on disk, ALL seed=0):
  omnigeo   : baseline + rkdc TTA @100 (main), @50/@100/@150/@200 (step ablation),
              20-pairs ablation  — dual metrics (pose AUC + depth) per sequence
  omnivideo : baseline + rkdc TTA @100 — 187 sequences
  da3_legacy: 7scenes/eth3d/hiroom/scannetpp — t8s4_lossall (rkdc1h, LoRA 0-39),
              mv13 (rkdc1h, LoRA 13-39), rkdcr (rel arm), rkdc1hc (ctk arm)
  vggt_legacy: 7scenes/eth3d/hiroom/scannetpp — C2M_maskrel / C2M_RKDC1H
              (lossall runs) per-scene AUC@3 + F1
  da3_diag3 : 3-state (theta0/base/rel) per-scene evals on 7 diagnostic scenes

Output: workspace/all_per_scene_results.json
"""
import glob
import json
import os

ROOT = "/root/autodl-tmp/Free-Geometry"
os.chdir(ROOT)

out = {"meta": {
    "note": "所有 run 均为 seed=0（磁盘上不存在 seed1/2 的逐场景结果；"
            "唯一历史 seed-1 训练 artifacts/diagnostics/seed1_20 无评估数据）；"
            "未运行过名为 TCO 的方法——若指其他名称请确认",
    "seed": 0,
    "generated": "2026-09-20",
}}


def load_rows(paths):
    by = {}
    for p in paths:
        if os.path.exists(p):
            for r in json.load(open(p)):
                by[r["seq"]] = r
    return list(by.values())


def pick(row, keys):
    return {k: row[k] for k in keys if k in row}


OMNI_KEYS = None  # all


def omnigeo():
    d = {}
    main = load_rows(["workspace/omnigeo_vggt_rkdc/results_shard0_2.json",
                      "workspace/omnigeo_vggt_rkdc/results_shard1_2.json"])
    d["rkdc_100steps_10pairs_main"] = main
    for tag, paths in [
        ("steps_ablation_s200_ckpt100_150_200",
         ["workspace/omnigeo_vggt_rkdc_s200/results_shard0_3.json",
          "workspace/omnigeo_vggt_rkdc_s200/results_shard1_3.json",
          "workspace/omnigeo_vggt_rkdc_s200/results_shard2_3.json"]),
        ("steps_50_s50", ["workspace/omnigeo_vggt_rkdc_s50/results_shard0_3.json",
                          "workspace/omnigeo_vggt_rkdc_s50/results_shard1_3.json",
                          "workspace/omnigeo_vggt_rkdc_s50/results_shard2_3.json"]),
        ("pairs_20_s100", ["workspace/omnigeo_vggt_rkdc_s100_p20/results_shard0_2.json",
                           "workspace/omnigeo_vggt_rkdc_s100_p20/results_shard1_2.json"]),
    ]:
        rows = load_rows(paths)
        if rows:
            d[tag] = rows
    return {"n_sequences": 49, "protocol": "SelfEvo eval branch, sparse 1/10, "
            "crop518, [0,1]; TTA=rkdc1h LoRA-24layer VGGT", "runs": d}


def omnivideo():
    return {
        "n_sequences": 187,
        "protocol": "同 omnigeo",
        "runs": {"rkdc_100steps_10pairs_main": load_rows(
            ["workspace/omnivideo_vggt_rkdc/results_shard0_3.json",
             "workspace/omnivideo_vggt_rkdc/results_shard1_3.json",
             "workspace/omnivideo_vggt_rkdc/results_shard2_3.json"])},
    }


DA3_KEYS = ["auc03", "auc05", "auc15", "auc30", "abs_rel", "delta125",
            "recon_fscore", "recon_overall", "recon_acc", "recon_comp"]


def da3_legacy():
    d = {}
    runs = [
        ("rkdc1h_lora0_39", "da3_protocol_{ds}_t8s4_lossall"),
        ("rkdc1h_lora13_39_mv13", "mv13_{ds}"),
        ("rkdc1hr_rel_lora0_39", "da3_{ds}_rkdcr"),
    ]
    for ds in ("7scenes", "eth3d", "hiroom", "scannetpp"):
        d[ds] = {}
        for tag, pat in runs:
            p = f"workspace/{pat.format(ds=ds)}/smoke_summary.json"
            if not os.path.exists(p):
                continue
            scenes = json.load(open(p))["scenes"]
            d[ds][tag] = {s: pick(v.get("eval", {}), DA3_KEYS)
                          for s, v in scenes.items() if "eval" in v}
    # ctk arm (7scenes only) + 3-state diag eval
    p = "workspace/da3_7scenes_rkdc1hc/smoke_summary.json"
    if os.path.exists(p):
        scenes = json.load(open(p))["scenes"]
        d["7scenes"]["rkdc1hc_ctk_lora0_39"] = {
            s: pick(v.get("eval", {}), DA3_KEYS)
            for s, v in scenes.items() if "eval" in v}
    return {"protocol": "DA3-GIANT 8:4 loss_all_pos 100步；指标=AUC@3/15/30+"
            "F1/CD+AbsRel（t8s4_lossall/mv13 为 LoRA 范围对照）", "datasets": d}


def da3_diag3():
    p = "workspace/fg_diag_v2_eval.json"
    if not os.path.exists(p):
        return None
    return {"protocol": "三状态逐场景评估（θ₀冻结 / θ_base=rkdc1h / θ_rel=rkdc1hr，"
            "均为重训 all-40 LoRA）",
            "scenes": json.load(open(p))}


def vggt_legacy():
    d = {}
    for ds in ("7scenes", "eth3d", "hiroom", "scannetpp"):
        d[ds] = {}
        for mf in sorted(glob.glob(f"artifacts/diagnostics/final_protocol_lossall/"
                                   f"{ds}/eval32_metrics_*lossall.json")):
            arm = os.path.basename(mf).replace("eval32_metrics_", "") \
                                         .replace("_lossall.json", "")
            try:
                m = json.load(open(mf))
            except Exception:
                continue
            per = {}
            for arm_key, payload in m.items():
                pk = [k for k in payload if "pose" in k]
                rk = [k for k in payload if "recon" in k]
                if not pk:
                    continue
                pose = payload[pk[0]]
                recon = payload[rk[0]] if rk else {}
                for s in pose:
                    if s == "mean":
                        continue
                    per.setdefault(s, {})[arm_key] = {
                        "auc03": pose[s].get("auc03"),
                        "auc15": pose[s].get("auc15"),
                        "auc30": pose[s].get("auc30"),
                        "fscore": recon.get(s, {}).get("fscore"),
                    }
            d[ds][arm] = per
    return {"protocol": "VGGT @504 lossall（maskrel=7s/eth3d/hiroom, RKDC1H=spp）；"
            "AUC@3°/F1，LoRA 24层", "datasets": d}


out["omnigeo"] = omnigeo()
out["omnivideo"] = omnivideo()
out["da3_legacy"] = da3_legacy()
diag = da3_diag3()
if diag:
    out["da3_diag3states"] = diag
out["vggt_legacy"] = vggt_legacy()

path = "workspace/all_per_scene_results.json"
with open(path, "w") as f:
    json.dump(out, f, indent=1)
print(f"written {path} ({os.path.getsize(path)/1e6:.1f} MB)")


def count(x, depth=0):
    if isinstance(x, dict):
        return sum(count(v, depth + 1) for v in x.values())
    if isinstance(x, list):
        return len(x)
    return 0


for k in ("omnigeo", "omnivideo", "da3_legacy", "vggt_legacy"):
    print(k, "runs/datasets:", list(out[k].get("runs", out[k].get("datasets", {})).keys()))
