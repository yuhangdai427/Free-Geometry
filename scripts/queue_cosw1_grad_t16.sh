#!/bin/bash
# Queue slot: cosw1 per-component-grad rerun, courtyard FORCED 16:4.
# Waits for the in-flight queue (courtyard 8:4 -> facade 16:4) to fully finish
# (facade is last; its final "cam_dec" metric line marks completion), then runs.
# Never starts while another train_pw0_accum.py is alive (no-parallel rule).
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python

for i in $(seq 1 150); do  # up to ~50 min
  if ! pgrep -f train_pw0_accum.py >/dev/null 2>&1 \
     && grep -q "cam_dec" logs/cosw1_grad_facade.log 2>/dev/null; then
    break
  fi
  sleep 20
done
if pgrep -f train_pw0_accum.py >/dev/null 2>&1; then
  echo "[queue] a trainer is STILL running after timeout — ABORT (no-parallel)"
  exit 1
fi

echo "[queue] starting courtyard 16:4 cosw1 grad run $(date '+%F %T')"
$PY scripts/train_pw0_accum.py --scene courtyard --teacher_N 16 \
    --updates 50 --accum 1 --loss_form half_mse --half_mode cosw1 \
    --out workspace/cosw1_grad_courtyard_t16 \
    > logs/cosw1_grad_courtyard_t16.log 2>&1
echo "[queue] courtyard t16 done $(date '+%F %T')"
