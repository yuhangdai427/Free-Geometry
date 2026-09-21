#!/bin/bash
# facade with n_train=5 (5 train pairs instead of 10), cosw1 + per-component
# grad logging, 16:4 (facade auto). SERIAL: waits for the fgray raypose queue
# to fully finish first (no-parallel).
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python

for i in $(seq 1 120); do  # up to ~40 min
  if ! pgrep -f "eval_da3_raypose.py|train_pw0_accum.py" >/dev/null 2>&1 \
     && grep -q "facade done" logs/queue_fgray.log 2>/dev/null; then
    break
  fi
  sleep 20
done
if pgrep -f "eval_da3_raypose.py|train_pw0_accum.py" >/dev/null 2>&1; then
  echo "[nt5] trainer/eval STILL running after timeout — ABORT (no-parallel)"; exit 1
fi

echo "[nt5] facade n_train=5 cosw1 grad run $(date '+%F %T')"
$PY scripts/train_pw0_accum.py --scene facade --n_train 5 \
    --updates 50 --accum 1 --loss_form half_mse --half_mode cosw1 \
    --out workspace/cosw1_grad_facade_nt5 \
    > logs/cosw1_grad_facade_nt5.log 2>&1
echo "[nt5] facade n_train=5 done $(date '+%F %T')"
