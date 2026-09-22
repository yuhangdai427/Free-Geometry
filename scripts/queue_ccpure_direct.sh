#!/bin/bash
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python
for SC in courtyard delivery_area electro facade; do
  echo "[ccpure] $SC start $(date '+%F %T')"
  $PY scripts/train_pw0_accum.py --scene $SC --vggt_sync --half_mode cwd_camtok \
      --cwd_tau 0.5 --updates 100 --accum 1 \
      --out workspace/cam_campaign/ccpure/$SC > logs/cam_ccpure_$SC.log 2>&1 \
      && grep "cam_dec" logs/cam_ccpure_$SC.log >> workspace/cam_campaign/SUMMARY.log
  rm -rf workspace/cam_campaign/ccpure/$SC/ckpts workspace/cam_campaign/ccpure/$SC/recon
done
echo "[$(date '+%F %T')] ==== CCPURE DONE ====" >> workspace/cam_campaign/SUMMARY.log
