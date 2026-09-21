#!/bin/bash
# DA3 port of the VGGT paradigm: raw-token space + SmoothL1(beta=1, full mean)
# + 2*(1-cos), n_train=5, courtyard then facade — SERIAL (no-parallel).
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python

if pgrep -f "train_pw0_accum|train_vggt_pw0_accum|eval_da3_raypose|eval_vggt_cosw1" >/dev/null 2>&1; then
  echo "[vport] another job running — ABORT (no-parallel)"; exit 1
fi
for SC in courtyard facade; do
  echo "[vport] $SC vggt-port run $(date '+%F %T')"
  $PY scripts/train_pw0_accum.py --scene $SC --half_mode vggt --n_train 5 \
      --updates 50 --accum 1 \
      --out workspace/vggtport_grad_$SC \
      > logs/vggtport_grad_$SC.log 2>&1
  echo "[vport] $SC done $(date '+%F %T')"
done
echo "[vport] all done $(date '+%F %T')"
