#!/bin/bash
# CWD on patches + CWD on camera token + cos on both + camrel,
# raw space, first 4 eth3d scenes.
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python
for SC in courtyard delivery_area electro facade; do
  echo "[cc] $SC start $(date '+%F %T')"
  $PY scripts/train_pw0_accum.py --scene $SC --vggt_sync --half_mode cwd_camtok \
      --cwd_tau 0.5 --camrel --updates 100 --accum 1 \
      --out workspace/cam_campaign/cc/$SC > logs/cam_cc_$SC.log 2>&1 \
      && grep "cam_dec" logs/cam_cc_$SC.log >> workspace/cam_campaign/SUMMARY.log \
      || echo "[$(date '+%F %T')] CC $SC FAILED" >> workspace/cam_campaign/SUMMARY.log
  rm -rf workspace/cam_campaign/cc/$SC/ckpts workspace/cam_campaign/cc/$SC/recon
done
echo "[$(date '+%F %T')] ==== CC DONE ====" >> workspace/cam_campaign/SUMMARY.log
