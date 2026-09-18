#!/bin/bash
# DTU unified-protocol experiments v2 (fixes: evaluate_scene accepts raw model,
# gt_depth_files conditionally skipped for DTU).
#   dtu   (22 scenes): pose AUC + recon (acc/comp/overall in mm)
#   dtu64 (13 scenes): pose AUC only
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/envs/da3/bin/python

DTU_SCENES="scan1 scan4 scan9 scan10 scan11 scan12 scan13 scan15 scan23 scan24 scan29 scan32 scan33 scan34 scan48 scan49 scan62 scan75 scan77 scan110 scan114 scan118"
DTU64_SCENES="scan105 scan114 scan118 scan122 scan24 scan37 scan40 scan55 scan63 scan65 scan69 scan83 scan97"

# ---- baseline (a0): frozen model, no training ----
run_baseline () {
  local ds=$1 scenes=$2
  local out=workspace/da3_dtu_unified/${ds}_a0
  mkdir -p "$out"
  echo "[dtu] $ds a0 START $(date '+%F %T')"
  $PY - "$ds" "$scenes" "$out" <<'PYEOF'
import json, os, sys
import numpy as np
import torch
sys.path.insert(0, 'src')
sys.path.insert(0, 'diagnostics/free_geometry')
sys.path.insert(0, 'scripts')
from depth_anything_3.test_time_adaption import protocol_v1 as P
from train_da3_protocol import evaluate_scene, make_dataset
import common as fg

ds, scenes_str, out = sys.argv[1], sys.argv[2], sys.argv[3]
scenes = scenes_str.split()
fg.set_dataset(ds)
dataset_obj = make_dataset(ds)
teacher = P.create_teacher('model_weights/DA3-GIANT-1.1', 'cuda')

results = {}
for scene in scenes:
    try:
        data = fg.get_scene_data(scene)
        files = list(data.image_files)
        proto = P.build_scene_protocol(files, scene, dataset=ds, n_train=10,
                                       n_shared=4, teacher_N=8)
        ev = evaluate_scene(teacher, data, proto['eval_frames'],
                           scene=scene, dataset_obj=dataset_obj,
                           export_dir=os.path.join(out, 'recon', scene))
        results[scene] = ev
        rec = {k: round(v, 4) for k, v in ev.items() if k.startswith('recon_')}
        print(f"[{scene}] auc03={ev['auc03']:.4f} {rec}", flush=True)
        del teacher  # keep memory clean between scenes? No, keep it loaded
        teacher = P.create_teacher('model_weights/DA3-GIANT-1.1', 'cuda')
    except Exception as e:
        print(f"[{scene}] FAILED: {e}", flush=True)
        results[scene] = {"error": str(e)}

with open(os.path.join(out, 'metrics.json'), 'w') as f:
    json.dump(results, f, indent=2)
ok = sum(1 for v in results.values() if 'error' not in v)
print(f"BASELINE {ds} DONE: {ok}/{len(results)} scenes OK")
PYEOF
  echo "[dtu] $ds a0 DONE $(date '+%F %T')"
}

# ---- TTA (rkdc1h): normal training + eval ----
run_tta () {
  local ds=$1 scenes=$2
  local out=workspace/da3_dtu_unified/${ds}_rkdc1h
  mkdir -p "$out"
  echo "[dtu] $ds rkdc1h START $(date '+%F %T')"
  $PY scripts/train_da3_protocol.py --dataset $ds --scenes $scenes \
      --output_root "$out" --steps 100 --arm rkdc1h --teacher_N 8 \
      --n_train 10 --mask_ratio 0.5 --loss_all_pos \
      >> logs/dtu_${ds}_rkdc1h_v2.log 2>&1 \
      || { echo "[dtu] $ds rkdc1h FAILED"; return 1; }
  echo "[dtu] $ds rkdc1h DONE $(date '+%F %T')"
}

# ---- paired summary ----
summarize () {
  local ds=$1
  $PY - "$ds" <<'PYEOF'
import json, os, sys
import numpy as np
ds = sys.argv[1]
try:
    b = json.load(open(f'workspace/da3_dtu_unified/{ds}_a0/metrics.json'))
    t = json.load(open(f'workspace/da3_dtu_unified/{ds}_rkdc1h/smoke_summary.json'))['scenes']
    common = [s for s in b if s in t and 'error' not in b[s] and 'eval' in t[s]]
    da = [(t[s]['eval']['auc03']-b[s]['auc03'])/b[s]['auc03']*100 for s in common if b[s]['auc03']>0]
    print(f"{ds}: dAUC={np.mean(da):+.2f}% (n={len(common)})")
    # recon metrics if available
    bo = [b[s].get('recon_overall') for s in common if b[s].get('recon_overall')]
    to = [t[s]['eval'].get('recon_overall') for s in common if t[s]['eval'].get('recon_overall')]
    if bo and to:
        print(f"  baseline overall={np.mean(bo):.3f}mm -> TTA overall={np.mean(to):.3f}mm")
except Exception as e:
    print(f"{ds}: summary error: {e}")
PYEOF
}

run_baseline dtu "$DTU_SCENES"
run_tta dtu "$DTU_SCENES"
summarize dtu

run_baseline dtu64 "$DTU64_SCENES"
run_tta dtu64 "$DTU64_SCENES"
summarize dtu64

echo "[dtu] ALL DONE $(date '+%F %T')"
