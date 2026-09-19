#!/usr/bin/env python3
"""Prune a RUNNING v2 run dir's intermediate ckpts without breaking selection.

For every scene whose final step100 ckpt exists, replay the selector on the
scene's probe trace (same semantics as protocol_v2_analyze.replay_selector),
then keep only step0 / step100 / selected-step ckpts (both stepN_lora.pt files
and stepN_lora_peft dirs). Scenes still training are skipped.

Usage: python scripts/v2_prune_running.py RUN_DIR
"""
import glob
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main(run_dir):
    from free_geometry.tta_v2 import ControllerConfig, select
    freed = 0
    for trace_path in sorted(glob.glob(os.path.join(run_dir, "probe_trace", "**", "*.jsonl"),
                                       recursive=True)):
        scene = os.path.relpath(trace_path, os.path.join(run_dir, "probe_trace"))[:-len(".jsonl")]
        by_step = {}
        for line in open(trace_path):
            r = json.loads(line)
            if "records" in r:
                by_step.setdefault(int(r["step"]), []).extend(r["records"])
            elif "components" in r:
                by_step.setdefault(int(r["step"]), []).append(r)
        if not by_step or 0 not in by_step:
            continue
        v2dirs = [d for d in glob.glob(os.path.join(run_dir, "ckpts", scene, "**", "v2"),
                                       recursive=True)
                  + [os.path.join(run_dir, "ckpts", scene, "v2")] if os.path.isdir(d)]
        if not v2dirs:
            continue
        v2dir = v2dirs[0]
        steps = {}
        for p in glob.glob(os.path.join(v2dir, "step*_lora.pt")) + \
                 glob.glob(os.path.join(v2dir, "step*_lora_peft")):
            m = re.fullmatch(r"step(\d+)_lora(?:\.pt|_peft)", os.path.basename(p))
            if m:
                steps.setdefault(int(m.group(1)), []).append(p)
        if 100 not in steps:
            continue  # scene still training
        trace = [{"step": s, "records": by_step[s]} for s in sorted(by_step)]
        try:
            dec = select(trace, ControllerConfig())
            sel = int(dec.get("selected_step", 100))
            fb = bool(dec.get("fell_back_to_baseline"))
        except Exception as e:
            print(f"[prune] {scene}: selector replay failed ({e}); keeping 0/100 only")
            sel, fb = 100, False
        keep = {0, 100} if fb else {0, 100, sel}
        removed = []
        for s, paths in steps.items():
            if s in keep:
                continue
            for p in paths:
                try:
                    if os.path.isdir(p):
                        sz = sum(os.path.getsize(f) for f in
                                 glob.glob(os.path.join(p, "**", "*"), recursive=True)
                                 if os.path.isfile(f))
                        shutil.rmtree(p)
                    else:
                        sz = os.path.getsize(p)
                        os.remove(p)
                    freed += sz
                    removed.append(s)
                except OSError:
                    pass
        print(f"[prune] {scene}: sel={sel} fb={fb} kept={sorted(keep)} removed={sorted(set(removed))}")
    print(f"[prune] freed {freed/2**30:.1f} GiB")


if __name__ == "__main__":
    main(sys.argv[1])
