#!/bin/bash
# DTU unified-protocol experiments (2026-09-17):
#   dtu   (22 scenes): recon_unposed (acc/comp/overall in mm) + pose
#   dtu64 (13 scenes): pose only (AUC@3)
# Arms: a0 baseline + rkdc1h (all-position loss, 8:4, 100 steps)
# Failure-tolerant: each task continues on failure.
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/envs/da3/bin/python

DTU_SCENES="scan1 scan4 scan9 scan10 scan11 scan12 scan13 scan15 scan23 scan24 scan29 scan32 scan33 scan34 scan48 scan49 scan62 scan75 scan77 scan110 scan114 scan118"
DTU64_SCENES="scan105 scan114 scan118 scan122 scan24 scan37 scan40 scan55 scan63 scan65 scan69 scan83 scan97"

run_arm () {
  local ds=$1 scenes=$2 arm=$3
  local out=workspace/da3_dtu_unified/${ds}_${arm}
  mkdir -p "$out"
  echo "[dtu] $ds $arm START $(date '+%F %T')"
  if [ "$arm" = "a0" ]; then
    # Baseline: skip training, just eval (--skip_train doesn't exist; use steps=0)
    $PY scripts/train_da3_protocol.py --dataset $ds --scenes $scenes \
        --output_root "$out" --steps 1 --arm pw0 --teacher_N 8 \
        --n_train 1 --mask_ratio 0 --skip_eval \
        > /dev/null 2>&1  # just create the output dir
    # Actually for a0 we need a separate eval-only path; use the baseline scripts
    $PY - "$ds" "$scenes" "$out" <<'PYEOF'
import json, os, sys
import numpy as np
import torch
sys.path.insert(0, 'src')
sys.path.insert(0, 'diagnostics/free_geometry')
sys.path.insert(0, 'scripts')
from depth_anything_3.test_time_adaption import protocol_v1 as P
from train_da3_protocol import evaluate_scene, make_dataset, get_scene_data
import common as fg

ds, scenes_str, out = sys.argv[1], sys.argv[2], sys.argv[3]
scenes = scenes_str.split()
fg.set_dataset(ds)
dataset_obj = make_dataset(ds)
teacher = P.create_teacher('model_weights/DA3-GIANT-1.1', 'cuda')

results = {}
for scene in scenes:
    try:
        data = get_scene_data(scene)
        files = list(data.image_files)
        from depth_anything_3.test_time_adaption.protocol_v1 import build_scene_protocol
        proto = build_scene_protocol(files, scene, dataset=ds, n_train=10,
                                     n_shared=4, teacher_N=8)
        ev = evaluate_scene(teacher, data, proto['eval_frames'],
                           scene=scene, dataset_obj=dataset_obj,
                           export_dir=os.path.join(out, 'recon', scene))
        results[scene] = ev
        print(f"[{scene}] auc03={ev['auc03']:.4f} "
              f"overall={ev.get('recon_overall', float('nan')):.4f}", flush=True)
    except Exception as e:
        print(f"[{scene}] FAILED: {e}", flush=True)
        results[scene] = {"error": str(e)}

with open(os.path.join(out, 'metrics.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"BASELINE DONE: {len(results)} scenes")
PYEOF
  else
    # TTA: normal training + eval
    $PY scripts/train_da3_protocol.py --dataset $ds --scenes $scenes \
        --output_root "$out" --steps 100 --arm rkdc1h --teacher_N 8 \
        --n_train 10 --mask_ratio 0.5 --loss_all_pos \
        >> logs/dtu_${ds}_${arm}.log 2>&1 \
        || { echo "[dtu] $ds $arm FAILED"; return 1; }
  fi
  echo "[dtu] $ds $arm DONE $(date '+%F %T')"
}

# Wait for current experiments to release GPU capacity
echo "[dtu] waiting for fg_518 to finish (GPU capacity) $(date '+%F %T')"
while tmux has-session -t fg_518 2>/dev/null; do sleep 120; done
echo "[dtu] GPU lane free, starting DTU $(date '+%F %T')"

# DTU-49: baseline then TTA
run_arm dtu "$DTU_SCENES" a0
run_arm dtu "$DTU_SCENES" rkdc1h

# DTU-64: baseline then TTA
run_arm dtu64 "$DTU64_SCENES" a0
run_arm dtu64 "$DTU64_SCENES" rkdc1h

echo "[dtu] ALL DONE $(date '+%F %T')"
