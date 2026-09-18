#!/bin/bash
# Protocol v2 overnight lane A: DA3 on eth3d -> [GATE] -> 7scenes -> scannetpp -> hiroom -> dtu -> dtu64
# NEVER reruns baseline: deltas are computed against workspace/protocol_v2/baselines.json.
# Pilot-gate: after eth3d the lane parks until workspace/protocol_v2/GATE_ETH3D_OK exists
# (created by the operator after pilot analysis), or aborts on workspace/protocol_v2/STOP.
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
PY=/root/miniconda3/envs/da3/bin/python
V2="--v2_probe --v2_ckpt --v2_rel_weight 1.0 --v2_grad_cap --v2_couple_fix --v2_baselines_json workspace/protocol_v2/baselines.json"
GATE=workspace/protocol_v2/GATE_ETH3D_OK
STOPF=workspace/protocol_v2/STOP

mkdir -p logs workspace/protocol_v2

# preflight: GPU must be nearly empty before we start
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | head -1)
  [ "${used:-99999}" -lt 60000 ] && break
  echo "[v2-da3] waiting for GPU (used ${used}MiB) $(date '+%F %T')"
  sleep 60
done
$PY scripts/train_da3_protocol.py --help 2>&1 | grep -q -- '--v2_probe' \
  || { echo "[v2-da3] FATAL: v2 flags missing in train_da3_protocol.py"; exit 1; }

scenes_of () { ls workspace/fgmig/protocols/"$1"/*.json | xargs -n1 basename | sed 's/\.json$//' | tr '\n' ' '; }

run_ds () {
  local ds=$1
  local out=workspace/protocol_v2/da3_$ds
  [ -f "$STOPF" ] && { echo "[v2-da3] STOP file present, abort"; exit 1; }
  echo "[v2-da3] $ds START $(date '+%F %T')"
  # shellcheck disable=SC2086
  $PY scripts/train_da3_protocol.py --dataset "$ds" --scenes $(scenes_of "$ds") \
      --output_root "$out" --steps 100 --arm rkdc1h --teacher_N 8 --n_train 10 \
      --mask_ratio 0.5 --loss_all_pos $V2 \
      --swanlab --swanlab_suffix _v2 \
      >> "logs/v2_da3_${ds}.log" 2>&1 \
    && echo "[v2-da3] $ds TRAIN+EVAL DONE $(date '+%F %T')" \
    || echo "[v2-da3] $ds FAILED $(date '+%F %T') (see logs/v2_da3_${ds}.log)"
  $PY scripts/protocol_v2_analyze.py --model da3 --dataset "$ds" --run_dir "$out" \
      >> "logs/v2_da3_${ds}_analysis.log" 2>&1 || true
}

run_ds eth3d
echo "[v2-da3] eth3d pilot finished, waiting for gate $GATE $(date '+%F %T')"
while [ ! -f "$GATE" ]; do
  [ -f "$STOPF" ] && { echo "[v2-da3] STOPPED at gate $(date '+%F %T')"; exit 1; }
  sleep 60
done
run_ds 7scenes
run_ds scannetpp
run_ds hiroom
run_ds dtu
run_ds dtu64
echo "[v2-da3] ALL DONE $(date '+%F %T')"
