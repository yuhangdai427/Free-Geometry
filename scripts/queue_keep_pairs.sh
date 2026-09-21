#!/bin/bash
# Pair-filtered DA3 rerun (courtyard): keep ONLY the mid/benign rel-grad pairs
# P5/P6/P8 (per-pair g_rot means 5.7/5.3/4.8), dropping the spike pairs
# (P1:20.2, P3:13.6) and near-zero pairs (P0/P2/P4, P7, P9).
# Everything else identical to the fgk1 verdict run (deployed B5+rel, u100k1).
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python

for i in $(seq 1 300); do
  if ! pgrep -f "train_pw0_accum|train_arms.py|eval_vggt_cosw1" >/dev/null 2>&1 \
     && grep -q "all done" logs/queue_fgraw.log 2>/dev/null; then
    break
  fi
  sleep 20
done
if pgrep -f "train_pw0_accum|train_arms.py|eval_vggt_cosw1" >/dev/null 2>&1; then
  echo "[kp] trainer STILL running — ABORT (no-parallel)"; exit 1
fi

echo "[kp] DA3 courtyard keep-pairs 5,6,8 $(date '+%F %T')"
$PY scripts/train_pw0_accum.py --scene courtyard --vggt_sync \
    --loss_form huber --half_mode joint --updates 20 --accum 1 \
    --keep_pairs 5,6,8 \
    --out workspace/da3_kp568_courtyard \
    > logs/da3_kp568_courtyard.log 2>&1
echo "[kp] done rc=$? $(date '+%F %T')"
