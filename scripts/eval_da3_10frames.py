#!/usr/bin/env python3
"""DA3 10-frame/scene re-evaluation (2026-09-20).

New eval protocol: 10 equidistant frames per scene, taken from the N-dispatch
manifest's eval32_frames base — the SAME base and slicing rule as VGGT's 10v
subset (eval_viewcounts: ev[::max(1,len(ev)//10)][:10]), so the two models are
scored on identical frames.

Evaluates BOTH the frozen baseline (zero LoRA) and the adapted model (per-scene
ckpts/<scene>/c2m_final_lora.pt) on those 10 frames, writing
workspace/ndispatch/da3_<ds>/eval10.json.

Usage: python scripts/eval_da3_10frames.py --dataset eth3d [--scenes ...]
"""
import argparse, json, os, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "src"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "..", "diagnostics", "free_geometry"))

import torch
from depth_anything_3.test_time_adaption import protocol_v1 as P
import common as fg_common
from train_da3_protocol import make_dataset, get_scene_data, evaluate_scene


def pick10(ev):
    return ev[::max(1, len(ev) // 10)][:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--run_root", default=None,
                    help="default workspace/ndispatch/da3_<dataset>")
    ap.add_argument("--model_name", default="model_weights/DA3-GIANT-1.1")
    args = ap.parse_args()

    run_root = args.run_root or f"workspace/ndispatch/da3_{args.dataset}"
    manifest = json.load(open(f"workspace/ndispatch/{args.dataset}/scene_manifest.json"))
    scenes = args.scenes or sorted(manifest["scenes"])
    fg_common.set_dataset(args.dataset)
    dataset_obj = make_dataset(args.dataset)

    print(f"loading student: {args.model_name}")
    student = P.create_student(args.model_name)
    student.to("cuda").eval()

    out_path = os.path.join(run_root, "eval10.json")
    results = {}
    if os.path.exists(out_path):
        results = json.load(open(out_path))  # resume-friendly
    for scene in scenes:
        sc = manifest["scenes"][scene]
        frames10 = pick10(sc["eval32_frames"])
        scene_data = get_scene_data(scene)
        rec = {"frames10": frames10, "n_eval": len(frames10)}
        for tag, ckpt in (("baseline", None),
                          ("adaptive", os.path.join(run_root, "ckpts", scene,
                                                    "c2m_final_lora.pt"))):
            if ckpt is not None and not os.path.exists(ckpt):
                print(f"[{scene}] missing {ckpt} — skipping adaptive")
                continue
            P.reset_lora_(student)
            if ckpt is not None:
                student.load_lora_weights(ckpt)
            student.to("cuda").eval()
            t0 = time.time()
            ev = evaluate_scene(student, scene_data, frames10,
                                scene=scene, dataset_obj=dataset_obj,
                                export_dir=os.path.join(run_root, "recon10", scene))
            ev["eval_time_s"] = time.time() - t0
            rec[tag] = {k: v for k, v in ev.items()
                        if isinstance(v, (int, float))}
            print(f"[{scene}] {tag}: auc03={ev['auc03']:.4f} "
                  f"f1={ev.get('recon_fscore', float('nan')):.4f} "
                  f"({len(frames10)} frames, {ev['eval_time_s']:.0f}s)", flush=True)
        results[scene] = rec
        with open(out_path, "w") as f:
            json.dump(results, f, indent=1)
    print(f"-> {out_path} ({len(results)} scenes)")


if __name__ == "__main__":
    main()
