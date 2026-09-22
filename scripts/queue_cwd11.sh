#!/bin/bash
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python
MAN=artifacts/diagnostics/final_protocol/eth3d/scene_manifest.json
SCENES=$($PY -c "import json;print(' '.join(sorted(json.load(open('$MAN'))['scenes'])))")
for SC in $SCENES; do
  echo "[cwd11] $SC start $(date '+%F %T')"
  if $PY scripts/train_pw0_accum.py --scene $SC --vggt_sync --half_mode vggt_cwd \
      --cwd_tau 0.5 --camrel --updates 100 --accum 1 \
      --out workspace/cam_campaign/cwd/$SC > logs/cam_cwd_$SC.log 2>&1; then
    grep "cam_dec" logs/cam_cwd_$SC.log >> workspace/cam_campaign/SUMMARY.log
  else
    echo "[$(date '+%F %T')] CWD $SC FAILED" >> workspace/cam_campaign/SUMMARY.log
  fi
  rm -rf workspace/cam_campaign/cwd/$SC/ckpts workspace/cam_campaign/cwd/$SC/recon
done
echo "[$(date '+%F %T')] ==== CWD11 ALL DONE ====" >> workspace/cam_campaign/SUMMARY.log
