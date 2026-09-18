#!/usr/bin/env python3
"""Unified TTA entry point for the 3-model migration (omega / pi3 / dvlt).

Runs in the da3 env for omega/pi3 and the dvlt env for dvlt (trainer only
reads protocol JSONs — no DA3 imports). Arms:
  a0           frozen baseline (eval export only)
  m_allpos     maskdistill only, ALL-position loss
  rkdc_allpos  maskdistill + 1.5*rkd_centers + 1.0*couple_centers (default)
"""
import argparse
import csv
import json
import os
import sys
import time

ROOT = "/root/autodl-tmp/Free-Geometry"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "free_geometry"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["omega", "pi3", "dvlt"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="scene tags (defaults to all protocol JSONs)")
    ap.add_argument("--arm", default="rkdc_allpos",
                    choices=["a0", "m_allpos", "rkdc_allpos", "maskrel_allpos",
                             "rkdcr_allpos"])
    ap.add_argument("--lora_variant", default="shared",
                    choices=["shared", "two_stage"],
                    help="LoRA variant for DVLT: shared (single LoRA across all "
                         "K loops) or two_stage (early/late LoRA, split at step 6)")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.0, help="0 = model default")
    ap.add_argument("--mask_ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--protocol_dir",
                    default=os.path.join(ROOT, "workspace/fgmig/protocols"))
    ap.add_argument("--output_root", required=True)
    args = ap.parse_args()

    import numpy as np
    import torch
    from free_geometry import trainer
    from free_geometry.adapters import get_adapter

    pdir = os.path.join(args.protocol_dir, args.dataset)
    tags = args.scenes or sorted(f[:-5] for f in os.listdir(pdir) if f.endswith(".json"))
    os.makedirs(args.output_root, exist_ok=True)
    adapter = get_adapter(args.model,
                          **({"lora_variant": args.lora_variant}
                             if args.model == "dvlt" else {}))
    summary = {"model": args.model, "dataset": args.dataset, "args": vars(args),
               "params": None, "scenes": {}}
    trace = []

    for tag in tags:
        scene = tag.replace("__", "/")
        proto = json.load(open(os.path.join(pdir, f"{tag}.json")))
        out_dir = os.path.join(args.output_root, tag, "exports")
        t0 = time.time()
        if args.arm != "a0":
            stats = trainer.train_scene(
                adapter, proto, scene, arm=args.arm, steps=args.steps,
                epochs=args.epochs,
                lr=(args.lr if args.lr > 0 else None),
                seed=args.seed, mask_ratio=args.mask_ratio,
                log_fn=print, trace=trace)
        else:
            adapter.load("cuda")
            stats = {"steps": 0}
        trainer.export_scene_eval(adapter, proto, scene,
                                  os.path.dirname(out_dir))
        summary["scenes"][tag] = {**stats, "time_s": round(time.time() - t0, 1)}
        summary["params"] = adapter.student_label()
        print(f"[{tag}] EXPORTED ({time.time() - t0:.0f}s)", flush=True)

    with open(os.path.join(args.output_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if trace:
        with open(os.path.join(args.output_root, "training_trace.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
            w.writeheader()
            w.writerows(trace)
    print("DONE", args.output_root)


if __name__ == "__main__":
    main()
