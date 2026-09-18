#!/usr/bin/env python3
"""Protocol v2 post-dataset analysis + cleanup.

Per (model, dataset) run dir:
  1. Extract per-scene TTA metrics (DA3: smoke_summary.json; VGGT: eval32_metrics_V2.json).
  2. Pair against ARCHIVED baselines from workspace/protocol_v2/baselines.json
     (baselines are NEVER recomputed).
  3. Replay the offline checkpoint selector on probe traces (no retraining).
  4. Write workspace/protocol_v2/analysis/{model}_{dataset}.{md,json}.
  5. Cleanup (ONLY with --cleanup): remove recon/point-cloud exports; prune
     v2 ckpt series to {step0, final, selected} per scene. Prints freed bytes.
     Cleanup is opt-in — analysis must never destroy artifacts by default.
"""
import argparse
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def load_json(p):
    with open(p) as f:
        return json.load(f)


def is_scene_dict(d):
    return isinstance(d, dict) and any(
        k in d for k in ("auc03", "auc3", "fscore", "recon_fscore", "overall", "recon_overall"))


def walk_metrics(node, out):
    """Collect {scene: metrics} from any nesting: a dict whose values are
    scene-level metric dicts."""
    if not isinstance(node, dict):
        return
    if is_scene_dict(node):
        return  # leaf metrics dict itself; parent handles it
    for k, v in node.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and is_scene_dict(v):
            out.setdefault(k, {}).update(v)
        else:
            walk_metrics(v, out)


def get_metric(d, *keys):
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def replay_selector(run_dir):
    """Return {scene: select() result} from probe_trace JSONLs."""
    from free_geometry.tta_v2 import ControllerConfig, select
    out = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "probe_trace", "*.jsonl"))):
        scene = os.path.basename(path)[:-len(".jsonl")]
        by_step = {}
        try:
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    if "records" in r:  # core-style: one line per evaluate()
                        by_step.setdefault(int(r["step"]), []).extend(r["records"])
                    elif "components" in r:  # per-record line (both pipelines)
                        by_step.setdefault(int(r["step"]), []).append(r)
        except Exception as e:
            out[scene] = {"error": f"trace unreadable: {e}"}
            continue
        trace = [{"step": s, "records": by_step[s]} for s in sorted(by_step)]
        if not trace or trace[0]["step"] != 0:
            out[scene] = {"error": "no step-0 record"}
            continue
        try:
            out[scene] = select(trace, ControllerConfig())
        except Exception as e:
            out[scene] = {"error": f"select failed: {e}"}
    return out


def prune_ckpts(run_dir, selected_by_scene):
    """Keep step0 / final(max step) / selected; delete the rest (+_peft dirs)."""
    freed = 0
    kept, deleted = [], []
    for path in glob.glob(os.path.join(run_dir, "ckpts", "**", "v2", "step*_lora.pt"),
                          recursive=True):
        base = os.path.basename(path)
        try:
            step = int(base[len("step"):-len("_lora.pt")])
        except ValueError:
            continue
        scene = path.split(os.sep + "ckpts" + os.sep)[1].split(os.sep)[0]
        sel = selected_by_scene.get(scene, {})
        sel_step = sel.get("selected_step")
        steps_here = [int(os.path.basename(p)[len("step"):-len("_lora.pt")])
                      for p in glob.glob(os.path.join(os.path.dirname(path), "step*_lora.pt"))]
        keep = {0, max(steps_here)}
        if isinstance(sel_step, int):
            keep.add(sel_step)
        if step in keep:
            kept.append(path)
            continue
        for p in (path, path.replace(".pt", "_peft")):
            if os.path.isfile(p):
                freed += os.path.getsize(p)
                os.remove(p)
            elif os.path.isdir(p):
                freed += sum(os.path.getsize(os.path.join(dp, f))
                             for dp, _, fs in os.walk(p) for f in fs)
                shutil.rmtree(p)
        deleted.append(path)
    return freed, kept, deleted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["da3", "vggt"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--baselines", default="workspace/protocol_v2/baselines.json")
    ap.add_argument("--out_dir", default="workspace/protocol_v2/analysis")
    ap.add_argument("--cleanup", action="store_true",
                    help="DESTRUCTIVE: also remove recon exports and prune the "
                         "intermediate v2 ckpt series to {step0, final, "
                         "selected}. Off by default; analysis never deletes "
                         "artifacts unless explicitly asked.")
    args = ap.parse_args()

    # ---- 1. TTA per-scene metrics
    if args.model == "da3":
        src = os.path.join(args.run_dir, "smoke_summary.json")
        if not os.path.exists(src):
            print(f"ANALYSIS SKIP: {src} missing (run failed?)")
            return
        raw = load_json(src)
        tta = {}
        for scene, blk in raw.get("scenes", {}).items():
            ev = blk.get("eval") or {}
            m = {"auc03": get_metric(ev, "auc03"),
                 "fscore": get_metric(ev, "recon_fscore"),
                 "overall": get_metric(ev, "recon_overall")}
            tta[scene] = {k: v for k, v in m.items() if v is not None}
    else:
        src = os.path.join(args.run_dir, "eval32_metrics_V2.json")
        if not os.path.exists(src):
            print(f"ANALYSIS SKIP: {src} missing (run failed?)")
            return
        tta = {}
        walk_metrics(load_json(src), tta)

    # ---- 2. baseline pairing
    base = {}
    try:
        bj = load_json(args.baselines)
        arm_blk = bj.get(args.model, {}).get(args.dataset, {}).get("baseline", {})
        base = {k: v for k, v in arm_blk.items()
                if isinstance(v, dict) and not k.startswith("_")}
    except Exception as e:
        print(f"WARNING: baselines unreadable: {e}")

    rows = []
    for scene in sorted(tta):
        t, b = tta[scene], base.get(scene, {})
        row = {"scene": scene}
        for key, tk, bk in (("auc03", "auc03", "auc03"),
                            ("f1", "fscore", "fscore"),
                            ("chamfer", "overall", "overall")):
            tv, bv = t.get(tk), get_metric(b, bk) if b else None
            if tv is not None:
                row[key] = tv
            if tv is not None and bv:
                row[f"d_{key}_rel"] = (tv - bv) / abs(bv)
                row[f"baseline_{key}"] = bv
        rows.append(row)

    def mean_of(key):
        vals = [r[key] for r in rows if key in r]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "n_scenes": len(rows),
        "mean_auc03": mean_of("auc03"), "mean_d_auc03_rel": mean_of("d_auc03_rel"),
        "mean_f1": mean_of("f1"), "mean_d_f1_rel": mean_of("d_f1_rel"),
        "mean_chamfer": mean_of("chamfer"), "mean_d_chamfer_rel": mean_of("d_chamfer_rel"),
        "degraded": [r["scene"] for r in rows
                     if r.get("d_auc03_rel", 0) < -0.005 or r.get("d_f1_rel", 0) < -0.005],
        "missing_baseline": [r["scene"] for r in rows
                             if "d_auc03_rel" not in r and "d_chamfer_rel" not in r],
    }

    # ---- 3. selector replay
    decisions = replay_selector(args.run_dir)
    n_fb = sum(1 for d in decisions.values() if d.get("fell_back_to_baseline"))
    summary["selector"] = {"n_scenes_with_trace": len(decisions),
                           "n_fallback": n_fb,
                           "n_improved_pick": sum(
                               1 for d in decisions.values()
                               if not d.get("fell_back_to_baseline", True)
                               and d.get("improvement", 0) > 0)}

    # ---- 4. write outputs
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"{args.model}_{args.dataset}")
    with open(stem + ".json", "w") as f:
        json.dump({"model": args.model, "dataset": args.dataset,
                   "run_dir": args.run_dir, "summary": summary,
                   "scenes": rows, "selector": decisions}, f, indent=1)
    with open(stem + ".md", "w") as f:
        f.write(f"# v2 {args.model} {args.dataset}\n\n")
        f.write(f"scenes={summary['n_scenes']} "
                f"dAUC03_rel={summary['mean_d_auc03_rel']} "
                f"dF1_rel={summary['mean_d_f1_rel']} "
                f"dChamfer_rel={summary['mean_d_chamfer_rel']}\n\n")
        f.write("| scene | baseline AUC3 | TTA AUC3 | d% | baseline F1 | TTA F1 | d% | selector |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            sel = decisions.get(r["scene"], {})
            f.write(f"| {r['scene']} | {r.get('baseline_auc03', '')} | "
                    f"{r.get('auc03', '')} | {r.get('d_auc03_rel', '')} | "
                    f"{r.get('baseline_f1', '')} | {r.get('f1', '')} | "
                    f"{r.get('d_f1_rel', '')} | step{sel.get('selected_step', '?')}"
                    f"{' FB' if sel.get('fell_back_to_baseline') else ''} |\n")
        f.write(f"\ndegraded: {summary['degraded']}\n")

    # ---- 5. cleanup (opt-in; metrics already extracted above)
    freed = 0
    if args.cleanup:
        recon = os.path.join(args.run_dir, "recon")
        if os.path.isdir(recon):
            freed += sum(os.path.getsize(os.path.join(dp, f))
                         for dp, _, fs in os.walk(recon) for f in fs)
            shutil.rmtree(recon)
        sel_map = {s: d for s, d in decisions.items() if isinstance(d, dict)}
        f2, kept, deleted = prune_ckpts(args.run_dir, sel_map)
        freed += f2
        print(f"cleanup: freed {freed / 2**20:.0f} MiB "
              f"(recon exports + {len(deleted)} intermediate ckpts, kept {len(kept)})")

    print(f"ANALYSIS {args.model}/{args.dataset}: scenes={summary['n_scenes']} "
          f"dAUC={summary['mean_d_auc03_rel']} dF1={summary['mean_d_f1_rel']} "
          f"dCD={summary['mean_d_chamfer_rel']} degraded={summary['degraded']} "
          f"selector={summary['selector']}")


if __name__ == "__main__":
    main()
