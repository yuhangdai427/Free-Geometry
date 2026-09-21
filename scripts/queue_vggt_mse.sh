#!/bin/bash
# DA3 + VGGT paradigm with MSE distance term (vggt_mse): enc space, 0.5*d^2
# full-element mean + 2*(1-cos), clip 1.0 (deployed). SERIAL after fgvnoclip.
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python

for i in $(seq 1 150); do
  if ! pgrep -f "train_pw0_accum|train_vggt_pw0_accum" >/dev/null 2>&1 \
     && grep -q "all done" logs/queue_fgvnoclip.log 2>/dev/null; then
    break
  fi
  sleep 20
done
if pgrep -f "train_pw0_accum|train_vggt_pw0_accum" >/dev/null 2>&1; then
  echo "[mse] trainer STILL running — ABORT (no-parallel)"; exit 1
fi

for SC in courtyard facade; do
  echo "[mse] $SC vggt_mse run $(date '+%F %T')"
  $PY scripts/train_pw0_accum.py --scene $SC --half_mode vggt_mse --n_train 5 \
      --updates 50 --accum 1 \
      --out workspace/vggtmse_grad_$SC \
      > logs/vggtmse_grad_$SC.log 2>&1
  echo "[mse] $SC done $(date '+%F %T')"
done
echo "[mse] all done $(date '+%F %T')"
