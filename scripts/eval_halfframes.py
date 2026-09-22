#!/usr/bin/env python3
"""Half-frames eval: use the saved camrel facade ckpt, evaluate at 76 (all)
vs 38 (every 2nd frame) vs 19 (every 4th) frames. Tests whether F1 drop is
from cross-frame inconsistency accumulation."""
import sys, os
sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "diagnostics/free_geometry")
import numpy as np, torch
import common as fg_common
from train_da3_protocol import make_dataset, get_scene_data, evaluate_scene
from depth_anything_3.test_time_adaption import protocol_v1 as P

scene = "facade"
fg_common.set_dataset("eth3d")
ds = make_dataset("eth3d")
sd = get_scene_data(scene)
N = len(sd.image_files)
print(f"{scene}: N={N}")

ckpt = "workspace/f1diag_facade/ckpts/facade/c2m_final_lora.pt"
if not os.path.exists(ckpt):
    ckpt_p = ckpt.replace(".pt", "_peft")
    assert os.path.exists(ckpt_p), f"no ckpt at {ckpt} or {ckpt_p}"

student = P.create_student("model_weights/DA3-GIANT-1.1", train_camera_token=False)
P.reset_lora_(student)
student.load_lora_weights(ckpt)
student.to("cuda").eval()

for tag, sel in [("all76", list(range(N))),
                 ("half38", list(range(0, N, 2))),
                 ("quarter19", list(range(0, N, 4)))]:
    ev = evaluate_scene(student, sd, sel, scene=scene, dataset_obj=ds,
                        export_dir=f"workspace/f1diag_facade/halfeval/{tag}")
    print(f"  {tag}({len(sel)}f): AUC={ev['auc03']:.4f} F1={ev['recon_fscore']:.4f} "
          f"prec={ev.get('recon_precision', float('nan')):.4f} recall={ev.get('recon_recall', float('nan')):.4f} "
          f"abs_rel={ev.get('abs_rel', float('nan')):.4f}", flush=True)

# also zero-shot at same frame counts for reference
zs = P.create_teacher("model_weights/DA3-GIANT-1.1").to("cuda").eval()
for tag, sel in [("zs_all76", list(range(N))),
                 ("zs_half38", list(range(0, N, 2)))]:
    ev = evaluate_scene(zs, sd, sel, scene=scene, dataset_obj=ds,
                        export_dir=f"workspace/f1diag_facade/halfeval/{tag}")
    print(f"  {tag}({len(sel)}f): AUC={ev['auc03']:.4f} F1={ev['recon_fscore']:.4f}", flush=True)
