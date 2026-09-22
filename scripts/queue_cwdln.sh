#!/bin/bash
# CWD-LN variant: CWD + cos on LAYER-NORMED tokens (head_norm space),
# camrel 1/1/0.5, all eth3d scenes. Queued after the CWD-raw queue drains.
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python
MAN=artifacts/diagnostics/final_protocol/eth3d/scene_manifest.json
SCENES=$($PY -c "import json;print(' '.join(sorted(json.load(open('$MAN'))['scenes'])))")

# wait for CWD-raw queue to finish
for i in $(seq 1 200); do
  if ! pgrep -f "train_pw0_accum" >/dev/null 2>&1 \
     && grep -q "CWD11 ALL DONE" workspace/cam_campaign/SUMMARY.log 2>/dev/null; then
    break
  fi
  sleep 30
done

for SC in $SCENES; do
  echo "[cwdln] $SC start $(date '+%F %T')"
  if $PY scripts/train_pw0_accum.py --scene $SC --vggt_sync --half_mode vggt_cwd_ln \
      --cwd_tau 0.5 --camrel --updates 100 --accum 1 \
      --out workspace/cam_campaign/cwdln/$SC > logs/cam_cwdln_$SC.log 2>&1; then
    grep "cam_dec" logs/cam_cwdln_$SC.log >> workspace/cam_campaign/SUMMARY.log
  else
    echo "[$(date '+%F %T')] CWDLN $SC FAILED" >> workspace/cam_campaign/SUMMARY.log
  fi
  rm -rf workspace/cam_campaign/cwdln/$SC/ckpts workspace/cam_campaign/cwdln/$SC/recon
done
echo "[$(date '+%F %T')] ==== CWDLN ALL DONE ====" >> workspace/cam_campaign/SUMMARY.log
