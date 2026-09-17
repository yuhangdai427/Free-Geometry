#!/bin/bash
# Full-grid loss-position ablation (masked input + ALL-position loss) over
# 2 models x 4 datasets. DA3: plain 8:4 rkdc1h on 7scenes/scannetpp (eth3d/hiroom
# already done). VGGT: champion arms on the frozen final_protocol manifests
# (16:4), fresh run_root copies so champion ckpts/eval32 are never touched.
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python
export PYTORCH_ALLOC_CONF=expandable_segments:True
DG=diagnostics/free_geometry
RRV=artifacts/diagnostics/final_protocol_lossall

# ---- VGGT: copy frozen manifests into a fresh run_root ----
for ds in 7scenes eth3d hiroom scannetpp; do
  mkdir -p $RRV/$ds
  cp -n artifacts/diagnostics/final_protocol/$ds/scene_manifest.json $RRV/$ds/scene_manifest.json
done

vggt () {
  local ds=$1 arm=$2 vc=$3
  echo "=== [VGGT $ds] arm=$arm loss_all_pos TRAIN $(date '+%F %T') ==="
  $PY $DG/train_arms.py --run_root $RRV/$ds --arms $arm --epochs 10 --seed 0 \
      --no_eval32 --loss_all_pos || { echo "[VGGT $ds] TRAIN FAILED"; return 1; }
  $PY $DG/eval_viewcounts.py --manifest $RRV/$ds/scene_manifest.json \
      --ckpt_root $RRV/$ds/ckpts --run_root $RRV/$ds --step 100 \
      --arms $arm --view_subsets $vc || { echo "[VGGT $ds] EVALVC FAILED"; return 1; }
  $PY $DG/run_eval.py --run_root $RRV/$ds --datas $ds || { echo "[VGGT $ds] RUNEVAL FAILED"; return 1; }
  mv $RRV/$ds/eval32_metrics.json $RRV/$ds/eval32_metrics_${arm}_lossall.json
  echo "=== [VGGT $ds] COMPLETE $(date '+%F %T') ==="
}

da3 () {
  local ds=$1 scenes=$2
  local out=workspace/da3_protocol_${ds}_t8s4_lossall
  echo "=== [DA3 $ds] loss_all_pos TRAIN $(date '+%F %T') ==="
  $PY scripts/train_da3_protocol.py --dataset $ds --scenes $scenes \
      --output_root $out --arm rkdc1h --teacher_N 8 --steps 100 \
      --mask_ratio 0.5 --loss_all_pos > logs/da3_${ds}_lossall.log 2>&1 \
      || { echo "[DA3 $ds] TRAIN FAILED"; return 1; }
  $PY - "$ds" "$out" <<'EOF'
import json, sys
import numpy as np
ds, out = sys.argv[1], sys.argv[2]
s = json.load(open(f'{out}/smoke_summary.json'))['scenes']
ba = json.load(open(f'artifacts/diagnostics/final_protocol/da3_baseline/{ds}_baseline.json'))['scenes']
br = json.load(open(f'artifacts/diagnostics/final_protocol/da3_baseline/{ds}_recon_baseline.json'))['scenes']
da = [(r['eval']['auc03']-ba[sc]['auc03'])/ba[sc]['auc03']*100 for sc, r in s.items() if ba[sc]['auc03'] > 0]
df = [(r['eval']['recon_fscore']-br[sc]['fscore'])/br[sc]['fscore']*100 for sc, r in s.items() if br[sc]['fscore'] > 0]
print(f"[DA3 {ds}] SUMMARY dAUC={np.mean(da):+.2f}% dF1={np.mean(df):+.2f}% (n={len(da)})", flush=True)
EOF
  echo "=== [DA3 $ds] COMPLETE $(date '+%F %T') ==="
}

da3 7scenes "chess fire heads office pumpkin redkitchen stairs"
da3 scannetpp "09c1414f1b 1ada7a0617 21d970d8de 286b55a2bf 38d58a7a31 3e8bba0176 40aec5fffa 578511c8a9 5f99900f09 7831862f02 7bc286c1b6 9071e139d9 acd95847c5 bcd246daf bde1e479ad c4c04e6d6c c5439f4607 cc5237fd77 f3d64c30f8 fb5a96b1a2"
vggt 7scenes  C2M_maskrel 100v
vggt eth3d    C2M_maskrel allv
vggt hiroom   C2M_maskrel allv
vggt scannetpp C2M_RKDC1H 100v
echo "ALL8 DONE $(date '+%F %T')"
