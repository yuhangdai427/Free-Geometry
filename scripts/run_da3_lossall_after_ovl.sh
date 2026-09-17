#!/bin/bash
# Orchestrator #2: wait for orchestrator #1 (eth3d sparse_overlap), then run the
# loss-position ablation: masked student input + ALL-position distill loss
# (--loss_all_pos) on eth3d and hiroom, with paired-vs-baseline summaries.
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python

echo "[orch2] waiting for orch1 (eth3d ovl) $(date '+%F %T')"
while pgrep -f "run_eth3d_ovl_after_t8.sh" > /dev/null; do sleep 60; done

export PYTORCH_ALLOC_CONF=expandable_segments:True

run () {
  local ds=$1; local scenes=$2; local out=$3
  echo "[orch2] $ds loss_all_pos start $(date '+%F %T')"
  "$PY" scripts/train_da3_protocol.py --dataset "$ds" --scenes $scenes \
    --output_root "$out" --arm rkdc1h --teacher_N 8 --steps 100 \
    --mask_ratio 0.5 --loss_all_pos > "logs/da3_${ds}_lossall.log" 2>&1
  echo "[orch2] $ds exit=$? $(date '+%F %T')"
  "$PY" - "$ds" "$out" <<'EOF'
import json, sys
import numpy as np
ds, out = sys.argv[1], sys.argv[2]
s = json.load(open(f'{out}/smoke_summary.json'))['scenes']
ba = json.load(open(f'artifacts/diagnostics/final_protocol/da3_baseline/{ds}_baseline.json'))['scenes']
br = json.load(open(f'artifacts/diagnostics/final_protocol/da3_baseline/{ds}_recon_baseline.json'))['scenes']
da = [(r['eval']['auc03']-ba[sc]['auc03'])/ba[sc]['auc03']*100 for sc, r in s.items() if ba[sc]['auc03'] > 0]
df = [(r['eval']['recon_fscore']-br[sc]['fscore'])/br[sc]['fscore']*100 for sc, r in s.items() if br[sc]['fscore'] > 0]
lines = [f"# DA3 {ds} 8:4: masked input + ALL-position loss (--loss_all_pos)", "",
         f"Mean dAUC@3 = **{np.mean(da):+.2f}%**  dF1 = **{np.mean(df):+.2f}%** (n={len(da)})", "",
         "2x2 matrix references: masked-in/masked-pos (champion) vs clean-in/all-pos (nomask)"]
open(f'{out}/result_summary.md', 'w').write("\n".join(lines))
print(f"[orch2] {ds} SUMMARY dAUC={np.mean(da):+.2f}% dF1={np.mean(df):+.2f}%")
EOF
}

run eth3d "courtyard delivery_area electro facade kicker office pipes playground relief relief_2 terrains" \
    workspace/da3_protocol_eth3d_t8s4_lossall
run hiroom "$(tr '\n' ' ' < workspace/benchmark_dataset/hiroom/selected_scene_list_val.txt)" \
    workspace/da3_protocol_hiroom_t8s4_lossall
echo "[orch2] ALL DONE $(date '+%F %T')"
