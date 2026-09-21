#!/usr/bin/env python3
"""DA3 raymap-camera evaluation (2026-09-21 smoke).

Evaluates pose AUC with the camera solved from the RAY MAP
(model.inference(use_ray_pose=True) -> get_extrinsic_from_camray), for BOTH
the frozen baseline (zero LoRA) and the TTA-adapted model (per-scene
c2m_final_lora.pt). Pure-feature-loss companion eval — no cam_dec numbers.

Usage: python scripts/eval_da3_raypose.py --dataset eth3d --scene delivery_area \
         [--run_root workspace/smoke_raypose]
"""
import argparse, json, os, random, sys, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "src"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "..", "diagnostics", "free_geometry"))

from depth_anything_3.test_time_adaption import protocol_v1 as P
import common as fg_common
from train_da3_protocol import make_dataset, get_scene_data, evaluate_scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="eth3d")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--run_root", default="workspace/smoke_raypose")
    ap.add_argument("--model_name", default="model_weights/DA3-GIANT-1.1")
    # RANSAC in the ray solver samples from the GLOBAL torch RNG when
    # random_seed is None -> re-seed before EVERY inference so runs are
    # bit-reproducible (user requirement: fix the seed dead)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fg_common.set_dataset(args.dataset)
    dataset_obj = make_dataset(args.dataset)
    scene_data = get_scene_data(args.scene)
    files = list(scene_data.image_files)

    proto = P.build_scene_protocol(files, args.scene, dataset=args.dataset,
                                   n_train=10, n_shared=4, teacher_N=16)
    frames = list(proto["eval_frames"])
    print(f"[{args.scene}] N={len(files)} eval_frames={len(frames)} "
          f"teacher_N={proto['teacher_N']}")

    student = P.create_student(args.model_name)
    student.to("cuda").eval()

    out = {"scene": args.scene, "n_eval": len(frames), "teacher_N": proto["teacher_N"],
           "seed": args.seed}
    for tag, ckpt in (("baseline_raypose", None),
                      ("tta_pw0_allpos_raypose",
                       os.path.join(args.run_root, "ckpts", args.scene, "c2m_final_lora.pt"))):
        P.reset_lora_(student)
        if ckpt is not None:
            # frozen-token runs save ONLY the _peft dir (.pt carries the camera
            # token and is written only when it was trainable)
            if not (os.path.exists(ckpt) or os.path.exists(ckpt.replace(".pt", "_peft"))):
                print(f"missing {ckpt} — skip {tag}"); continue
            student.load_lora_weights(ckpt)
        student.to("cuda").eval()
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        random.seed(args.seed)
        t0 = time.time()
        ev = evaluate_scene(student, scene_data, frames, scene=args.scene,
                            dataset_obj=dataset_obj,
                            export_dir=os.path.join(args.run_root, "recon_ray", tag, args.scene),
                            use_ray_pose=True)
        ev["eval_time_s"] = time.time() - t0
        out[tag] = {k: v for k, v in ev.items() if isinstance(v, (int, float))}
        print(f"[{args.scene}] {tag}: AUC@3={ev['auc03']:.4f} "
              f"F1={ev.get('recon_fscore', float('nan')):.4f} "
              f"CD={ev.get('recon_overall', float('nan')):.4f} "
              f"abs_rel={ev.get('abs_rel', float('nan')):.4f} "
              f"({ev['eval_time_s']:.0f}s)", flush=True)

    dst = os.path.join(args.run_root, f"raypose_{args.scene}.json")
    json.dump(out, open(dst, "w"), indent=1)
    print("->", dst)


if __name__ == "__main__":
    main()
