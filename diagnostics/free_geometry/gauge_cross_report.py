#!/usr/bin/env python3
"""Cross-dataset gauge-recoupling report: aggregates gauge_cross artifacts
(gauge_mismatch.csv + gauge_gtfree.json + gauge_fix_metrics.json) for one
dataset into per-scene and dataset-mean orig/fixgt/fixfree tables, prints
markdown fragments and writes gauge_cross/gauge_cross_summary.json.
Also verifies AUC invariance (variant npz extrinsics bitwise == source npz)
on up to 2 scenes. Pure CPU. New file; no existing module modified.

  python3 diagnostics/free_geometry/gauge_cross_report.py --dataset 7scenes
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

from common import load_manifest  # noqa: E402

FP = os.path.join(_REPO, "artifacts", "diagnostics", "final_protocol")
CHAMPION = {
    "7scenes": ("C2M_MC", "100v", "eval32_metrics_C2M_MC.json"),
    "hiroom": ("C2M_maskrel", "allv", "eval32_metrics_C2M.json"),
    "eth3d": ("C2M_SCL", "allv", "eval32_metrics_C2M_SCL.json"),
}
VARIANTS = ["orig", "fixgt", "fixfree"]


def mean(xs):
    xs = list(xs)
    return float(np.mean(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(CHAMPION))
    args = ap.parse_args()
    ds = args.dataset
    arm_tta, tag, metric_file = CHAMPION[ds]
    RR = os.path.join(FP, ds)
    OUT = os.path.join(RR, "gauge_cross")
    ARMS = ["A0_baseline", arm_tta]

    manifest = load_manifest(os.path.join(RR, "scene_manifest.json"))
    scenes = sorted(manifest["scenes"])

    mm = {}
    with open(os.path.join(OUT, "gauge_mismatch.csv")) as f:
        for row in csv.DictReader(f):
            mm[(row["arm"], row["scene"])] = row
    gf = json.load(open(os.path.join(OUT, "gauge_gtfree.json")))
    f_free = {(v["arm"], v["scene"]): v["f_free_dist"] for v in gf.values()}
    fm = json.load(open(os.path.join(OUT, "gauge_fix_metrics.json")))
    pub = json.load(open(os.path.join(RR, metric_file)))

    def fused(arm, variant, scene):
        key = f"{arm}_{variant}@{tag}::{ds}_recon_unposed"
        v = fm.get(key, {}).get(scene)
        return v if isinstance(v, dict) else None

    def pubm(arm, scene):
        key = f"{'A0_baseline' if arm == 'A0_baseline' else arm_tta}@{tag}"
        return pub[key][f"{ds}_recon_unposed"][scene]

    # ---- AUC invariance spot check: variant npz ext/intr == source npz ----
    auc_check = []
    for scene in scenes[:2]:
        for arm in ARMS:
            src = os.path.join(OUT, "eval32", f"{arm}@{tag}", "model_results",
                               ds, scene, "unposed", "exports", "mini_npz",
                               "results.npz")
            for variant in ("fixgt", "fixfree"):
                dst = src.replace(f"{arm}@{tag}", f"{arm}_{variant}@{tag}")
                if not (os.path.isfile(src) and os.path.isfile(dst)):
                    continue
                a = np.load(src)
                b = np.load(dst)
                de = max(float(np.abs(a[k] - b[k]).max())
                         for k in ("extrinsics", "intrinsics"))
                auc_check.append(de)
    auc_max_dev = max(auc_check) if auc_check else float("nan")

    # ---- per-scene rows ----
    rows = []
    for scene in scenes:
        for arm in ARMS:
            r = mm.get((arm, scene))
            fo = fused(arm, "orig", scene)
            if r is None or fo is None:
                continue
            fg = fused(arm, "fixgt", scene)
            ff = fused(arm, "fixfree", scene)
            rows.append({
                "scene": scene, "arm": arm,
                "mismatch": float(r["mismatch"]),
                "f_gt": float(r["s_depth"]) / float(r["s_pose"]),
                "f_free": f_free.get((arm, scene)),
                "F1_orig": fo["fscore"],
                "F1_fixgt": fg["fscore"] if fg else None,
                "F1_fixfree": ff["fscore"] if ff else None,
                "CD_orig": fo["overall"],
                "CD_fixgt": fg["overall"] if fg else None,
                "CD_fixfree": ff["overall"] if ff else None,
                "F1_pub": pubm(arm, scene)["fscore"],
                "CD_pub": pubm(arm, scene)["overall"],
            })

    # ---- dataset means (strict paired over scenes with all variants) ----
    summary = {"dataset": ds, "arm_tta": arm_tta, "tag": tag,
               "auc_invariance_max_dev": auc_max_dev, "arms": {}}
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm
               and all(r[f"{m}_{v}"] is not None
                       for m in ("F1", "CD") for v in VARIANTS)]
        e = {"n": len(sub)}
        for m in ("F1", "CD"):
            for v in VARIANTS:
                e[f"{m}_{v}"] = mean(r[f"{m}_{v}"] for r in sub)
        e["F1_pub"] = mean(r["F1_pub"] for r in sub)
        e["CD_pub"] = mean(r["CD_pub"] for r in sub)
        e["mismatch_mean"] = mean(r["mismatch"] for r in sub)
        e["mismatch_meanabs"] = mean(abs(r["mismatch"]) for r in sub)
        devs = [r["f_free"] - r["f_gt"] for r in sub
                if r["f_free"] is not None]
        e["ffree_dev_mean_pt"] = mean(devs) * 100
        e["ffree_dev_meanabs_pt"] = mean(abs(d) for d in devs) * 100
        e["ffree_dev_maxabs_pt"] = max((abs(d) for d in devs),
                                       default=float("nan")) * 100
        summary["arms"][arm] = e

    # ---- paired TTA-vs-baseline deltas under each variant ----
    paired = {}
    bmap = {r["scene"]: r for r in rows if r["arm"] == "A0_baseline"}
    tmap = {r["scene"]: r for r in rows if r["arm"] == arm_tta}
    common_scenes = [s for s in scenes if s in bmap and s in tmap
                     and all(bmap[s][f"{m}_{v}"] is not None
                             and tmap[s][f"{m}_{v}"] is not None
                             for m in ("F1", "CD") for v in VARIANTS)]
    for v in VARIANTS:
        paired[f"dF1_{v}"] = mean(tmap[s][f"F1_{v}"] - bmap[s][f"F1_{v}"]
                                  for s in common_scenes)
        paired[f"dCD_{v}"] = mean(bmap[s][f"CD_{v}"] - tmap[s][f"CD_{v}"]
                                  for s in common_scenes)  # +=improved
    summary["paired"] = paired
    summary["paired_n"] = len(common_scenes)

    # regen sanity: fused orig vs published
    summary["orig_vs_pub"] = {
        arm: {"F1_meanabsdev": mean(abs(r["F1_orig"] - r["F1_pub"])
                                    for r in rows if r["arm"] == arm),
              "CD_meanabsdev": mean(abs(r["CD_orig"] - r["CD_pub"])
                                    for r in rows if r["arm"] == arm)}
        for arm in ARMS}

    with open(os.path.join(OUT, "gauge_cross_summary.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)

    # ---- markdown fragments ----
    print(f"\n### {ds} ({arm_tta}@{tag}, n={len(common_scenes)} paired)\n")
    print("| 臂 | 失配 mean% | mean\\|失配\\|% | F1 orig | F1 fixgt | F1 fixfree |"
          " CD orig | CD fixgt | CD fixfree |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        e = summary["arms"][arm]
        print(f"| {arm} | {e['mismatch_mean']*100:+.2f} | "
              f"{e['mismatch_meanabs']*100:.2f} | "
              f"{e['F1_orig']:.4f} | {e['F1_fixgt']:.4f} | {e['F1_fixfree']:.4f} | "
              f"{e['CD_orig']:.4f} | {e['CD_fixgt']:.4f} | {e['CD_fixfree']:.4f} |")
    print(f"\npaired dF1: orig {paired['dF1_orig']:+.4f} -> "
          f"fixgt {paired['dF1_fixgt']:+.4f} -> fixfree {paired['dF1_fixfree']:+.4f}")
    print(f"paired dCD(+为好): orig {paired['dCD_orig']:+.4f} -> "
          f"fixgt {paired['dCD_fixgt']:+.4f} -> fixfree {paired['dCD_fixfree']:+.4f}")
    for arm in ARMS:
        e = summary["arms"][arm]
        print(f"f_free vs f_gt [{arm}]: dev mean {e['ffree_dev_mean_pt']:+.2f}pt "
              f"mean|dev| {e['ffree_dev_meanabs_pt']:.2f}pt "
              f"max|dev| {e['ffree_dev_maxabs_pt']:.2f}pt")
    print(f"orig vs published mean|dev|: "
          + "; ".join(f"{a} F1 {summary['orig_vs_pub'][a]['F1_meanabsdev']:.4f} "
                      f"CD {summary['orig_vs_pub'][a]['CD_meanabsdev']:.4f}"
                      for a in ARMS))
    print(f"AUC invariance spot check max|Delta ext/intr| = {auc_max_dev:.2e}")
    print(f"\nREPORT {ds} DONE", flush=True)


if __name__ == "__main__":
    main()
