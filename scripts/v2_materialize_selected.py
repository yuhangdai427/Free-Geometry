#!/usr/bin/env python3
"""Protocol v2 checkpoint-selection materializer ("selected" execution chain).

Reads <run_root>/probe_trace/*.jsonl, replays tta_v2.controller.select per
(scene, arm), and mirrors the selected v2 LoRA checkpoint into a shadow tree:

    <run_root>/ckpts_selected/<scene>/<arm>/step{STEP}_lora.pt(+_peft)
        <- hardlink (or copy) of
    <run_root>/ckpts/<scene>/<arm>/v2/step{sel}_lora.pt(+_peft)

so the existing eval pipeline (eval_viewcounts.py --ckpt_root
<run_root>/ckpts_selected --step 100) evaluates the SELECTED model with zero
changes. A selected step of 0 (selector fell back to baseline) materializes
the step-0 checkpoint.

Two probe_trace line formats are accepted:
  - per-record lines (VGGT train_arms): {scene, arm, step, pair_id, mask_id,
    components, total, ...}
  - whole evaluate() lines (core/DA3): {step, records: [{pair_id, mask_id,
    components, ...}, ...]} — grouped under --default_arm, scene from filename.

Source checkpoint existence is a HARD assert (the selector picked a step that
was never checkpointed = pipeline violation). Output:
  <run_root>/ckpts_selected/decisions.json
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from free_geometry.tta_v2.controller import ControllerConfig, select  # noqa: E402


def _read_trace_dir(trace_dir, default_arm):
    """-> {(scene, arm): [evaluate()-shaped entries]} in step order.

    Per-record lines are grouped by their OWN arm field (never merged across
    arms; previously the last line's arm silently won). Whole-line (DA3)
    entries carry no arm and go to default_arm."""
    groups = {}
    for path in sorted(os.listdir(trace_dir)):
        if not path.endswith(".jsonl"):
            continue
        scene = path[:-len(".jsonl")]
        per_key = {}  # (arm, step) -> evaluate()-shaped entry
        with open(os.path.join(trace_dir, path)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if "records" in r:  # whole evaluate() line
                    per_key[(default_arm, int(r["step"]))] = {
                        "step": int(r["step"]), "records": r["records"]}
                    continue
                arm = r.get("arm", default_arm)
                step = int(r["step"])
                e = per_key.setdefault((arm, step), {"step": step, "records": []})
                e["records"].append({
                    "pair_id": r["pair_id"], "mask_id": int(r["mask_id"]),
                    "components": r["components"],
                    "total": r.get("total")})
        by_arm = {}
        for (arm, step), e in per_key.items():
            by_arm.setdefault(arm, {})[step] = e
        for arm, steps in by_arm.items():
            groups.setdefault((scene, arm), []).extend(
                steps[k] for k in sorted(steps))
    return groups


def _link_or_copy(src, dst):
    """Hardlink file/dir (fallback copy). Raises AssertionError when src
    missing — selection without a checkpoint is a pipeline violation."""
    assert os.path.exists(src), f"selected checkpoint source missing: {src}"
    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.makedirs(dst)
        for name in sorted(os.listdir(src)):
            _link_or_copy(os.path.join(src, name), os.path.join(dst, name))
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--step", type=int, default=100,
                    help="target step label in ckpts_selected (default 100, "
                         "matching the eval pipeline's --step)")
    ap.add_argument("--default_arm", default="v2",
                    help="arm label for whole-line traces without an arm field")
    ap.add_argument("--tau_qual", type=float, default=0.05)
    ap.add_argument("--eps_abs", type=float, default=1e-3)
    ap.add_argument("--min_delta", type=float, default=0.005)
    ap.add_argument("--arm", default=None,
                    help="only process trace groups with this arm")
    ap.add_argument("--out_arm", default=None,
                    help="arm dir name under ckpts_selected (default: same as "
                         "the trace's arm; use e.g. X_SEL to avoid eval output "
                         "collision with the final-step pass)")
    args = ap.parse_args()

    trace_dir = os.path.join(args.run_root, "probe_trace")
    assert os.path.isdir(trace_dir), f"no probe_trace dir: {trace_dir}"
    groups = _read_trace_dir(trace_dir, args.default_arm)
    assert groups, f"no probe traces parsed from {trace_dir}"

    cfg = ControllerConfig(tau_qual=args.tau_qual, eps_abs=args.eps_abs,
                           min_delta=args.min_delta)
    out_root = os.path.join(args.run_root, "ckpts_selected")
    os.makedirs(out_root, exist_ok=True)
    decisions = {}
    n_err = 0
    for (scene, arm), trace in sorted(groups.items()):
        if args.arm is not None and arm != args.arm:
            continue
        key = f"{scene}/{arm}"
        try:
            sel = select(trace, cfg)
        except Exception as e:
            decisions[key] = {"error": f"select failed: {e}"}
            print(f"[{key}] select failed: {e}", flush=True)
            n_err += 1
            continue
        sel_step = int(sel["selected_step"])
        src_base = os.path.join(args.run_root, "ckpts", scene, arm, "v2",
                                f"step{sel_step}_lora.pt")
        if not os.path.exists(src_base) and not os.path.exists(
                src_base.replace(".pt", "_peft")):
            # DA3 layout has no arm layer: ckpts/<scene>/v2/stepN_lora.pt
            src_base = os.path.join(args.run_root, "ckpts", scene, "v2",
                                    f"step{sel_step}_lora.pt")
        dst_arm = args.out_arm or arm
        dst_base = os.path.join(out_root, scene, dst_arm,
                                f"step{args.step}_lora.pt")
        _link_or_copy(src_base.replace(".pt", "_peft"),
                      dst_base.replace(".pt", "_peft"))
        pt_src, pt_dst = src_base, dst_base
        if os.path.exists(pt_src):
            _link_or_copy(pt_src, pt_dst)
        decisions[key] = {
            "selected_step": sel_step,
            "fell_back_to_baseline": bool(sel["fell_back_to_baseline"]),
            "improvement": float(sel["improvement"]),
            "n_comparable": int(sel.get("n_comparable", 0)),
            "disqualified": {str(k): v for k, v in sel["disqualified"].items()},
            "src": os.path.relpath(src_base, args.run_root),
            "dst": os.path.relpath(dst_base, args.run_root),
        }
        print(f"[{key}] selected_step={sel_step} "
              f"fell_back={sel['fell_back_to_baseline']} "
              f"improvement={sel['improvement']:.4f} -> {dst_base}", flush=True)
    with open(os.path.join(out_root, "decisions.json"), "w") as f:
        json.dump(decisions, f, indent=2, sort_keys=True)
    print(f"decisions -> {os.path.join(out_root, 'decisions.json')}"
          f" ({len(decisions)} entries, {n_err} errors)", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
