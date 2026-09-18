#!/usr/bin/env python3
"""Offline selector evaluation over the real protocol-v2 probe traces.

For every scene trace under workspace/protocol_v2/{run}/probe_trace/ run three
selection strategies and compare their picks against the ground truth (GT)
that actually exists on disk:

  final      always pick the last trace step (the status-quo "use step 100")
  select_old tta_v2.controller.select with the legacy tau_qual qualification
             rule (catastrophic_rel=None)
  select_new select with the redesigned rule (rank by improvement; safety net
             vetoes only above catastrophic_rel=0.5)

EVIDENCE LEVELS (reported per scene x strategy, never conflated):
  gt_known        the chosen step has a measured GT auc03 (final-step evals
                  exist for every scene; intermediate-step GT exists ONLY for
                  vggt facade/electro at steps {0,30,60,100} from
                  workspace/protocol_v2/selector_check/)
  selection_only  the chosen step has NO GT -> we can only report WHAT the
                  rule picked, not whether it was right

GT sources (read-only):
  baselines        workspace/protocol_v2/baselines.json  [model][eth3d][baseline][scene]
  vggt final       workspace/protocol_v2/vggt_eth3d/eval32_metrics_V2_4scenes.json
                   workspace/protocol_v2/vggt_eth3d_rest/eval32_metrics_V2.json
  da3 final        logs/v2_da3_eth3d.log, logs/v2_da3_eth3d_rest.log  ("[scene] eval: auc03=...")
  vggt intermediate workspace/protocol_v2/selector_check/s{0,30,60,100}/metrics.json
  da3 intermediate  workspace/protocol_v2/selector_check_da3.json (expected to be
                   ABSENT while those evals are still running -> recorded as
                   unavailable, not as an error)

Output: workspace/protocol_v2/analysis/selector_offline_eval.json + summary.
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from free_geometry.tta_v2.controller import ControllerConfig, select  # noqa: E402

RUNS = ["da3_eth3d", "da3_eth3d_rest", "vggt_eth3d", "vggt_eth3d_rest"]
GT_FINAL_VGGT = {
    "vggt_eth3d": "eval32_metrics_V2_4scenes.json",
    "vggt_eth3d_rest": "eval32_metrics_V2.json",
}
GT_FINAL_DA3_LOG = {
    "da3_eth3d": "logs/v2_da3_eth3d.log",
    "da3_eth3d_rest": "logs/v2_da3_eth3d_rest.log",
}
INTERMEDIATE_STEPS = (0, 30, 60, 100)


def load_trace(path):
    """Accept both line formats -> [evaluate()-shaped entries] in step order."""
    per_step = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "records" in r:  # whole evaluate() line (DA3)
                per_step[int(r["step"])] = {"step": int(r["step"]),
                                            "records": r["records"]}
            else:  # per-record line (VGGT train_arms)
                step = int(r["step"])
                per_step.setdefault(step, {"step": step, "records": []})
                per_step[step]["records"].append({
                    "pair_id": r["pair_id"], "mask_id": int(r["mask_id"]),
                    "components": r["components"], "total": r.get("total")})
    return [per_step[k] for k in sorted(per_step)]


def load_baselines(path):
    with open(path) as f:
        d = json.load(f)
    out = {}
    for model in ("vggt", "da3"):
        scenes = d.get(model, {}).get("eth3d", {}).get("baseline", {})
        out[model] = {k: v for k, v in scenes.items()
                      if isinstance(v, dict) and not k.startswith("_")}
    return out


def load_vggt_final(path):
    """-> {scene: {"auc03": float, "fscore": float|None}}"""
    with open(path) as f:
        d = json.load(f)
    out = {}
    for arm_blk in d.values():
        pose = arm_blk.get("eth3d_pose", {}) if isinstance(arm_blk, dict) else {}
        for scene, m in pose.items():
            if scene == "mean" or not isinstance(m, dict):
                continue
            if isinstance(m.get("auc03"), (int, float)):
                out[scene] = {"auc03": float(m["auc03"]),
                              "fscore": float(m["fscore"]) if isinstance(m.get("fscore"), (int, float)) else None}
    return out


def load_da3_final(log_path):
    """Parse '[scene] eval: auc03=... fscore=... cd=...' lines."""
    out = {}
    if not os.path.exists(log_path):
        return out
    pat = re.compile(r"^\[(?P<scene>[^\]]+)\] eval: auc03=(?P<auc03>\S+) fscore=(?P<fscore>\S+)")
    with open(log_path, errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            def _f(x):
                try:
                    return float(x)
                except ValueError:
                    return None
            out[m.group("scene")] = {"auc03": _f(m.group("auc03")),
                                     "fscore": _f(m.group("fscore"))}
    return out


def load_vggt_intermediate(root):
    """-> {scene: {step: auc03}} from selector_check/s{step}/metrics.json."""
    out = {}
    for step in INTERMEDIATE_STEPS:
        p = os.path.join(root, f"s{step}", "metrics.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            d = json.load(f)
        for arm_blk in d.values():
            pose = arm_blk.get("eth3d_pose", {}) if isinstance(arm_blk, dict) else {}
            for scene, m in pose.items():
                if scene == "mean" or not isinstance(m, dict):
                    continue
                if isinstance(m.get("auc03"), (int, float)):
                    out.setdefault(scene, {})[step] = float(m["auc03"])
    return out


def load_da3_intermediate(path):
    """-> {scene: {step: auc03}} from selector_check_da3.json
    ({scene: {step(str): {auc03, ...}}}). Absent -> {}."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    out = {}
    for scene, steps in d.items():
        if not isinstance(steps, dict):
            continue
        for step, m in steps.items():
            try:
                s = int(step)
            except (TypeError, ValueError):
                continue
            if isinstance(m, dict) and isinstance(m.get("auc03"), (int, float)):
                out.setdefault(scene, {})[s] = float(m["auc03"])
    return out


def judge(chosen_step, gt_by_step, baseline_auc03):
    """-> (evidence, detail_dict). 'correct' is ONLY defined when the chosen
    step's GT is known: chosen step achieves the best known auc03."""
    if chosen_step not in gt_by_step:
        return "selection_only", {}
    auc = gt_by_step[chosen_step]
    best_step = max(gt_by_step, key=lambda s: gt_by_step[s])
    detail = {
        "auc03": auc,
        "d_auc03_vs_baseline": (auc - baseline_auc03) if baseline_auc03 is not None else None,
        "best_known_step": best_step,
        "best_known_auc03": gt_by_step[best_step],
        "correct": chosen_step == best_step,
        "regret_auc03": gt_by_step[best_step] - auc,
    }
    return "gt_known", detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="workspace/protocol_v2")
    ap.add_argument("--out", default="workspace/protocol_v2/analysis/selector_offline_eval.json")
    args = ap.parse_args()

    baselines = load_baselines(os.path.join(args.root, "baselines.json"))
    gt_final = {}
    for run, fname in GT_FINAL_VGGT.items():
        gt_final.update(load_vggt_final(os.path.join(args.root, run, fname)))
    for run, log in GT_FINAL_DA3_LOG.items():
        for scene, m in load_da3_final(log).items():
            gt_final.setdefault(scene, m)
    gt_inter = load_vggt_intermediate(os.path.join(args.root, "selector_check"))
    inter_da3_path = os.path.join(args.root, "selector_check_da3.json")
    gt_inter_da3 = load_da3_intermediate(inter_da3_path)

    scenes = []
    for run in RUNS:
        model = "da3" if run.startswith("da3") else "vggt"
        trace_dir = os.path.join(args.root, run, "probe_trace")
        for path in sorted(glob.glob(os.path.join(trace_dir, "*.jsonl"))):
            scene = os.path.splitext(os.path.basename(path))[0]
            trace = load_trace(path)
            steps = [int(e["step"]) for e in trace]
            final_step = steps[-1]

            gt_by_step = {}
            if scene in gt_final and gt_final[scene].get("auc03") is not None:
                gt_by_step[final_step] = gt_final[scene]["auc03"]
            if model == "vggt" and scene in gt_inter:
                gt_by_step.update(gt_inter[scene])
            if model == "da3" and scene in gt_inter_da3:
                gt_by_step.update(gt_inter_da3[scene])
            base_auc = baselines.get(model, {}).get(scene, {}).get("auc03")

            strategies = {"final": {"selected_step": final_step,
                                    "fell_back_to_baseline": False,
                                    "improvement": None,
                                    "disqualified": {}}}
            for name, cfg in (("select_old", ControllerConfig(catastrophic_rel=None)),
                              ("select_new", ControllerConfig())):
                try:
                    strategies[name] = select(trace, cfg)
                except Exception as e:  # unscoreable trace: report, don't die
                    strategies[name] = {"selected_step": None,
                                        "fell_back_to_baseline": None,
                                        "improvement": None,
                                        "disqualified": {}, "error": str(e)}

            entry = {"run": run, "model": model, "scene": scene,
                     "trace_steps": steps,
                     "baseline_auc03": base_auc,
                     "gt_known_steps": sorted(gt_by_step),
                     "gt_final_fscore": gt_final.get(scene, {}).get("fscore"),
                     "strategies": {}}
            for name, res in strategies.items():
                chosen = res["selected_step"]
                evidence, detail = ("selection_only", {}) if chosen is None \
                    else judge(chosen, gt_by_step, base_auc)
                entry["strategies"][name] = {
                    "selected_step": chosen,
                    "fell_back_to_baseline": res["fell_back_to_baseline"],
                    "improvement": res["improvement"],
                    "n_disqualified": len(res["disqualified"]),
                    "evidence": evidence,
                    **detail}
            scenes.append(entry)

    # ---- summary
    summary = {"n_scenes": len(scenes),
               "intermediate_gt": {
                   "vggt": {"scenes": sorted(gt_inter), "steps": sorted(INTERMEDIATE_STEPS)},
                   "da3": {"scenes": sorted(gt_inter_da3),
                           "status": "available" if gt_inter_da3 else "ABSENT (evals still running)"}},
               "strategies": {}}
    for name in ("final", "select_old", "select_new"):
        picks = [s["strategies"][name]["selected_step"] for s in scenes]
        fb = sum(1 for s in scenes if s["strategies"][name]["fell_back_to_baseline"])
        gt_known = [s for s in scenes if s["strategies"][name]["evidence"] == "gt_known"]
        correct = sum(1 for s in gt_known if s["strategies"][name].get("correct"))
        sel_only = sum(1 for s in scenes if s["strategies"][name]["evidence"] == "selection_only")
        summary["strategies"][name] = {
            "pick_distribution": dict(sorted(Counter(picks).items(), key=lambda kv: str(kv[0]))),
            "n_fallback": fb,
            "n_gt_known": len(gt_known),
            "n_correct_on_gt_known": correct,
            "n_selection_only": sel_only,
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "scenes": scenes}, f, indent=1)

    print(f"scenes={summary['n_scenes']}  da3 intermediate GT: {summary['intermediate_gt']['da3']['status']}"
          f" scenes={summary['intermediate_gt']['da3']['scenes']}")
    for name, st in summary["strategies"].items():
        print(f"{name:11s} picks={st['pick_distribution']} fallback={st['n_fallback']}"
              f" gt_known={st['n_gt_known']} correct={st['n_correct_on_gt_known']}"
              f" selection_only={st['n_selection_only']}")
    print("\nper-scene (evidence: K=gt_known, S=selection_only):")
    for s in scenes:
        row = " ".join(
            f"{n}={s['strategies'][n]['selected_step']}"
            f"({'K' if s['strategies'][n]['evidence'] == 'gt_known' else 'S'}"
            f"{',FB' if s['strategies'][n]['fell_back_to_baseline'] else ''})"
            for n in ("final", "select_old", "select_new"))
        print(f"  {s['model']:4s} {s['scene']:14s} {row}")
    print(f"\nwrote {args.out}")
    print("NOTE: 'correct' counts are ONLY meaningful on gt_known points "
          "(final step everywhere; {0,30,60,100} for vggt facade/electro). "
          "selection_only rows report what the rule picked — no GT verdict exists.")


if __name__ == "__main__":
    main()
