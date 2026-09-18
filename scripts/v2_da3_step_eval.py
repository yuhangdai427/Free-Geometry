#!/usr/bin/env python3
"""Evaluate arbitrary v2 checkpoint steps of DA3 scenes (selector validation).

For each scene x step: reset LoRA (B=0 == baseline), optionally load
<run_root>/ckpts/<scene>/v2/step{N}_lora.pt, then run the same evaluate_scene
used by the training pipeline (pose AUC + recon F1/CD). Writes one JSON:
{scene: {step: metrics}}.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "diagnostics", "free_geometry"))
sys.path.insert(0, os.path.dirname(__file__))

import common as fg_common
from depth_anything_3.test_time_adaption import protocol_v1 as P
from train_da3_protocol import evaluate_scene, get_scene_data, make_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_name", default="model_weights/DA3-GIANT-1.1")
    args = ap.parse_args()

    fg_common.set_dataset(args.dataset)
    dataset_obj = make_dataset(args.dataset)
    student = P.create_student(args.model_name)  # fresh LoRA: B=0 == baseline

    out = {}
    if os.path.exists(args.out):
        out = json.load(open(args.out))
    for scene in args.scenes:
        scene_data = get_scene_data(scene)
        image_files = list(scene_data.image_files)
        proto = P.build_scene_protocol(image_files, scene, dataset=args.dataset,
                                       n_train=10, n_shared=4, teacher_N=8)
        out.setdefault(scene, {})
        for step in args.steps:
            P.reset_lora_(student)
            if step > 0:
                ck = os.path.join(args.run_root, "ckpts", scene, "v2",
                                  f"step{step}_lora.pt")
                assert os.path.exists(ck), f"missing ckpt {ck}"
                student.load_lora_weights(ck)
            ev = evaluate_scene(student, scene_data, proto["eval_frames"],
                                scene=scene, dataset_obj=dataset_obj,
                                export_dir=os.path.join(
                                    args.run_root, "selector_check_recon",
                                    scene, f"step{step}"))
            out[scene][step] = ev
            print(f"[{scene}] step{step}: auc03={ev['auc03']:.4f} "
                  f"fscore={ev.get('recon_fscore', float('nan')):.4f}", flush=True)
            with open(args.out, "w") as f:
                json.dump(out, f, indent=1)
    print("DA3_STEP_EVAL_DONE")


if __name__ == "__main__":
    main()
