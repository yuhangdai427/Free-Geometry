#!/usr/bin/env python3
"""Evaluate the diag2 3-state checkpoints (theta0/theta_base/theta_rel) per scene:
AUC (pose) + F1 (TSDF recon) + depth metrics, exactly as the protocol eval.

- theta0     : reset_lora_ (B=0 == frozen baseline), same student object
- theta_base : workspace/diag2_{spp,7s}_rkdc1h/ckpts/{scene}/c2m_final_lora.pt
- theta_rel  : workspace/diag2_{spp,7s}_rkdc1hr/ckpts/{scene}/c2m_final_lora.pt

Output: workspace/fg_diag_v2_eval.json (full precision) + recon exports under
workspace/diag2_eval_recon/{scene}/{state}/ (mini_npz; deletable afterwards).
"""
import argparse
import json
import os
import sys
import time

ROOT = "/root/autodl-tmp/Free-Geometry"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "diagnostics", "free_geometry"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import common as fg  # noqa: E402
from depth_anything_3.test_time_adaption import protocol_v1 as P  # noqa: E402
from train_da3_protocol import evaluate_scene, make_dataset  # noqa: E402

RUNS = [
    ("scannetpp", ["09c1414f1b", "1ada7a0617", "21d970d8de", "7bc286c1b6", "9071e139d9"],
     "workspace/diag2_spp_rkdc1h", "workspace/diag2_spp_rkdc1hr"),
    ("7scenes", ["office", "chess"],
     "workspace/diag2_7s_rkdc1h", "workspace/diag2_7s_rkdc1hr"),
]

KEEP = ["auc03", "auc05", "auc15", "auc30", "abs_rel", "delta125",
        "recon_acc", "recon_comp", "recon_overall", "recon_fscore",
        "n_eval_frames", "eval_time_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="workspace/fg_diag_v2_eval.json")
    args = ap.parse_args()

    device = "cuda"
    student = P.create_student("model_weights/DA3-GIANT-1.1", device)

    results = {}
    for dataset, scenes, base_root, rel_root in RUNS:
        fg.set_dataset(dataset)
        dataset_obj = make_dataset(dataset)
        for scene in scenes:
            data = fg.get_scene_data(scene)
            files = list(data.image_files)
            proto = P.build_scene_protocol(files, scene, dataset=dataset,
                                           n_train=10, n_shared=4, teacher_N=8)
            states = {
                "theta0": None,
                "theta_base": os.path.join(base_root, "ckpts", scene, "c2m_final_lora.pt"),
                "theta_rel": os.path.join(rel_root, "ckpts", scene, "c2m_final_lora.pt"),
            }
            results.setdefault(scene, {"dataset": dataset, "states": {}})
            for state, ckpt in states.items():
                P.reset_lora_(student)  # B=0 -> frozen forward
                if ckpt is not None:
                    assert os.path.isfile(ckpt), ckpt
                    student.load_lora_weights(ckpt)
                student.to(device).eval()
                t0 = time.time()
                ev = evaluate_scene(
                    student, data, proto["eval_frames"], max_frames=0,
                    scene=scene, dataset_obj=dataset_obj,
                    export_dir=os.path.join("workspace", "diag2_eval_recon",
                                            scene, state))
                ev["eval_time_s"] = time.time() - t0
                results[scene]["states"][state] = {k: ev[k] for k in KEEP
                                                   if k in ev}
                print(f"[{scene}/{state}] auc03={ev['auc03']:.4f} "
                      f"auc30={ev['auc30']:.4f} f1={ev.get('recon_fscore', float('nan')):.4f} "
                      f"absrel={ev.get('abs_rel', float('nan')):.4f} "
                      f"({ev['eval_time_s']:.0f}s)", flush=True)
                with open(args.out, "w") as f:
                    json.dump(results, f, indent=1)
            torch.cuda.empty_cache()

    print("EVAL DONE ->", args.out)


if __name__ == "__main__":
    main()
