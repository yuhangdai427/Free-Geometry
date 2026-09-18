#!/bin/bash
# Protocol v2 overnight lane B: VGGT on eth3d -> [GATE] -> 7scenes -> scannetpp -> hiroom
# NEVER reruns baseline: SKIP_BASELINE=1 everywhere; deltas vs baselines.json.
# Same pilot-gate as lane A.
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
PY=/root/miniconda3/envs/da3/bin/python
DG=diagnostics/free_geometry
V2="--v2_probe --v2_ckpt --v2_rel_weight 1.0 --v2_grad_cap --v2_couple_fix --loss_all_pos --v2_baselines_json workspace/protocol_v2/baselines.json"
GATE=workspace/protocol_v2/GATE_ETH3D_OK
STOPF=workspace/protocol_v2/STOP
ARM=C2M_RKDC1H

mkdir -p logs workspace/protocol_v2

while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | head -1)
  [ "${used:-99999}" -lt 60000 ] && break
  echo "[v2-vggt] waiting for GPU (used ${used}MiB) $(date '+%F %T')"
  sleep 60
done
$PY $DG/train_arms.py --help 2>&1 | grep -q -- '--loss_all_pos' \
  || { echo "[v2-vggt] FATAL: --loss_all_pos missing in train_arms.py"; exit 1; }

view_subset () { case "$1" in eth3d|hiroom) echo allv;; *) echo 100v;; esac; }

run_ds () {
  local ds=$1
  local RR=workspace/protocol_v2/vggt_$ds
  local VS
  VS=$(view_subset "$ds")
  [ -f "$STOPF" ] && { echo "[v2-vggt] STOP file present, abort"; exit 1; }
  mkdir -p "$RR"
  cp "artifacts/diagnostics/final_protocol/$ds/scene_manifest.json" "$RR/scene_manifest.json"
  echo "[v2-vggt] $ds train START $(date '+%F %T')"
  # shellcheck disable=SC2086
  $PY $DG/train_arms.py --run_root "$RR" --arms $ARM --epochs 10 --seed 0 --no_eval32 \
      $V2 --swanlab --swanlab_suffix _v2 \
      >> "logs/v2_vggt_${ds}.log" 2>&1 \
    || { echo "[v2-vggt] $ds train FAILED $(date '+%F %T')"; return 1; }
  SKIP_BASELINE=1 $PY $DG/eval_viewcounts.py --manifest "$RR/scene_manifest.json" \
      --ckpt_root "$RR/ckpts" --run_root "$RR" --step 100 \
      --arms $ARM --view_subsets "$VS" >> "logs/v2_vggt_${ds}.log" 2>&1 \
    || { echo "[v2-vggt] $ds eval FAILED $(date '+%F %T')"; return 1; }
  $PY $DG/run_eval.py --run_root "$RR" --datas "$ds" \
      --experiments "${ARM}@${VS}" >> "logs/v2_vggt_${ds}.log" 2>&1 || return 1
  mv "$RR/eval32_metrics.json" "$RR/eval32_metrics_V2.json"
  rm -rf "$RR"/eval32/*/model_results
  $PY scripts/protocol_v2_analyze.py --model vggt --dataset "$ds" --run_dir "$RR" \
      >> "logs/v2_vggt_${ds}_analysis.log" 2>&1 || true
  echo "[v2-vggt] $ds DONE $(date '+%F %T')"
}

run_ds eth3d
echo "[v2-vggt] eth3d pilot finished, waiting for gate $GATE $(date '+%F %T')"
while [ ! -f "$GATE" ]; do
  [ -f "$STOPF" ] && { echo "[v2-vggt] STOPPED at gate $(date '+%F %T')"; exit 1; }
  sleep 60
done
run_ds 7scenes
run_ds scannetpp
run_ds hiroom
echo "[v2-vggt] ALL DONE $(date '+%F %T')"
