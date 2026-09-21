#!/bin/bash
# Fix-up: DA3 deployed-form vggt_sync rerun with the CORRECT schedule
# (100 updates x 1 pair, matching VGGT) — the earlier fgmgr2 runs accidentally
# used accum=5 (500 visits). Waits for fgmgr2's facade run to drain.
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python

for i in $(seq 1 200); do
  if ! pgrep -f "train_pw0_accum|train_arms.py" >/dev/null 2>&1 \
     && grep -q "all done" logs/queue_fgmgr2.log 2>/dev/null; then
    break
  fi
  sleep 20
done
if pgrep -f "train_pw0_accum|train_arms.py" >/dev/null 2>&1; then
  echo "[k1] trainer STILL running — ABORT (no-parallel)"; exit 1
fi

for SC in courtyard facade; do
  echo "[k1] DA3 vggt_sync u100k1 $SC $(date '+%F %T')"
  $PY scripts/train_pw0_accum.py --scene $SC --vggt_sync \
      --loss_form huber --half_mode joint --updates 100 --accum 1 \
      --out workspace/da3_maskrel_k1_grad_$SC \
      > logs/da3_maskrel_k1_grad_$SC.log 2>&1
  echo "[k1] DA3 $SC done rc=$? $(date '+%F %T')"
done
echo "[k1] all done $(date '+%F %T')"
