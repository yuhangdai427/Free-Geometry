#!/bin/bash
# Wait for spp100smooth to finish, then run ablation
set -u
cd /root/autodl-tmp/Free-Geometry
for i in $(seq 1 300); do
  if ! pgrep -f "train_pw0_accum" >/dev/null 2>&1 \
     && grep -q "SMOOTH100 ALL DONE" workspace/spp100smooth/SUMMARY.log 2>/dev/null; then
    break
  fi
  sleep 20
done
echo "[ablation] starting $(date '+%F %T')"
/root/miniconda3/envs/da3/bin/python scripts/ablation_train_vs_zero.py > logs/ablation_main.log 2>&1
echo "[ablation] done rc=$? $(date '+%F %T')"
