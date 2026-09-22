#!/bin/bash
# cwd_camtok WITHOUT camrel: pure token distillation (patch CWD + cos +
# camera-token CWD + cos at all 4 layers), NO decoded-pose loss.
# First 4 eth3d scenes. Queued after fgcc drains.
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python

for i in $(seq 1 200); do
  if ! pgrep -f train_pw0_accum >/dev/null 2>&1 \
     && grep -q "CC DONE" workspace/cam_campaign/SUMMARY.log 2>/dev/null; then
    break
  fi
  sleep 30
done

for SC in courtyard delivery_area electro facade; do
  echo "[ccpure] $SC start $(date '+%F %T')"
  $PY scripts/train_pw0_accum.py --scene $SC --vggt_sync --half_mode cwd_camtok \
      --cwd_tau 0.5 --updates 100 --accum 1 \
      --out workspace/cam_campaign/ccpure/$SC > logs/cam_ccpure_$SC.log 2>&1 \
      && grep "cam_dec" logs/cam_ccpure_$SC.log >> workspace/cam_campaign/SUMMARY.log \
      || echo "[$(date '+%F %T')] CCPURE $SC FAILED" >> workspace/cam_campaign/SUMMARY.log
  rm -rf workspace/cam_campaign/ccpure/$SC/ckpts workspace/cam_campaign/ccpure/$SC/recon
done
echo "[$(date '+%F %T')] ==== CCPURE DONE ====" >> workspace/cam_campaign/SUMMARY.log
