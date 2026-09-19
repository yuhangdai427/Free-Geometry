#!/bin/bash
# Rerun the 6 scannetpp scenes that were trained with the pre-crash old config.
# Their ckpts were deleted 09:00, so the lane skip logic would redo them —
# but the main lane already passed scannetpp; this script is the explicit rerun.
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
PY=/root/miniconda3/envs/da3/bin/python
MAN=workspace/protocol_v2/ab_manifests
V2="--v2_ab_manifest $MAN --v2_probe --v2_ckpt --v2_rel_weight 1.0 --v2_grad_cap --v2_qfeat_off --v2_rel_gate_deg 0 --v2_apply_selection --v2_baselines_json workspace/protocol_v2/baselines.json"
OUT=workspace/protocol_v2/full_da3_scannetpp
LOG=logs/v3_da3_scannetpp_fix6.log

while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | head -1)
  [ "${used:-99999}" -lt 40000 ] && break
  echo "[fix6] waiting for GPU (used ${used}MiB) $(date '+%F %T')" >> "$LOG"; sleep 60
done

echo "[fix6] START $(date '+%F %T')" >> "$LOG"
$PY scripts/train_da3_protocol.py --dataset scannetpp \
    --scenes 09c1414f1b 1ada7a0617 21d970d8de 286b55a2bf 38d58a7a31 3e8bba0176 \
    --output_root "$OUT" --steps 100 --arm rkdc1h --teacher_N 8 --n_train 10 \
    --mask_ratio 0.5 --loss_all_pos $V2 \
    >> "$LOG" 2>&1 \
  && echo "[fix6] DONE $(date '+%F %T')" >> "$LOG" \
  || echo "[fix6] FAILED $(date '+%F %T')" >> "$LOG"

$PY scripts/protocol_v2_analyze.py --model da3 --dataset scannetpp \
    --run_dir "$OUT" --baselines workspace/protocol_v2/baselines.json \
    --out_dir workspace/protocol_v2/analysis >> logs/v3_da3_scannetpp_analysis.log 2>&1 || true
echo "[fix6] ANALYZED $(date '+%F %T')" >> "$LOG"
