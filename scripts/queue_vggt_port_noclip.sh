#!/bin/bash
# VGGT-paradigm port on DA3 WITHOUT gradient clipping (--clip 0), n_train=5,
# courtyard then facade — SERIAL, waits for fgvport to finish first.
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python

for i in $(seq 1 120); do
  if ! pgrep -f "train_pw0_accum|train_vggt_pw0_accum" >/dev/null 2>&1 \
     && grep -q "all done" logs/queue_fgvport.log 2>/dev/null; then
    break
  fi
  sleep 20
done
if pgrep -f "train_pw0_accum|train_vggt_pw0_accum" >/dev/null 2>&1; then
  echo "[noclip] trainer STILL running — ABORT (no-parallel)"; exit 1
fi

for SC in courtyard facade; do
  echo "[noclip] $SC vggt-port NOCLIP run $(date '+%F %T')"
  $PY scripts/train_pw0_accum.py --scene $SC --half_mode vggt --n_train 5 \
      --updates 50 --accum 1 --clip 0 \
      --out workspace/vggtport_noclip_$SC \
      > logs/vggtport_noclip_$SC.log 2>&1
  echo "[noclip] $SC done $(date '+%F %T')"
done
echo "[noclip] all done $(date '+%F %T')"
