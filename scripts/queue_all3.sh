#!/bin/bash
# SmoothL1 + 2cos + CWD (all three patch terms) + camrel, raw space,
# first 4 eth3d scenes only.
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python
MAN=artifacts/diagnostics/final_protocol/eth3d/scene_manifest.json
for SC in courtyard delivery_area electro facade; do
  echo "[all3] $SC start $(date '+%F %T')"
  $PY scripts/train_pw0_accum.py --scene $SC --vggt_sync --half_mode vggt_all3 \
      --cwd_tau 0.5 --camrel --updates 100 --accum 1 \
      --out workspace/cam_campaign/all3/$SC > logs/cam_all3_$SC.log 2>&1 \
      && grep "cam_dec" logs/cam_all3_$SC.log >> workspace/cam_campaign/SUMMARY.log \
      || echo "[$(date '+%F %T')] ALL3 $SC FAILED" >> workspace/cam_campaign/SUMMARY.log
  rm -rf workspace/cam_campaign/all3/$SC/ckpts workspace/cam_campaign/all3/$SC/recon
done
echo "[$(date '+%F %T')] ==== ALL3 DONE ====" >> workspace/cam_campaign/SUMMARY.log
