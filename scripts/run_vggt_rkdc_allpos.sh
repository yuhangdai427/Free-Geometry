#!/bin/bash
# VGGT RKDC1H + all-position loss on 7scenes, eth3d, hiroom
# (scannetpp already done: +5.63/+2.38)
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
DG=diagnostics/free_geometry
RRV=artifacts/diagnostics/final_protocol_lossall
PY=/root/miniconda3/envs/da3/bin/python

for ds in 7scenes eth3d hiroom; do
  arm=C2M_RKDC1H
  out=$RRV/$ds
  echo "[vggt-rkdc] $ds $arm all-pos START $(date '+%F %T')"
  $PY $DG/train_arms.py --run_root $out --arms $arm --epochs 10 --seed 0 \
      --no_eval32 --loss_all_pos >> logs/vggt_${ds}_rkdc_allpos.log 2>&1 \
      || { echo "[vggt-rkdc] $ds TRAIN FAILED"; continue; }
  vc=$([ "$ds" = "7scenes" ] && echo "100v" || echo "allv")
  $PY $DG/eval_viewcounts.py --manifest $out/scene_manifest.json \
      --ckpt_root $out/ckpts --run_root $out --step 100 \
      --arms $arm --view_subsets $vc >> logs/vggt_${ds}_rkdc_allpos.log 2>&1 \
      || { echo "[vggt-rkdc] $ds EVALVC FAILED"; continue; }
  $PY $DG/run_eval.py --run_root $out --datas $ds >> logs/vggt_${ds}_rkdc_allpos.log 2>&1 \
      || { echo "[vggt-rkdc] $ds RUNEVAL FAILED"; continue; }
  mv $out/eval32_metrics.json $out/eval32_metrics_${arm}_lossall.json
  echo "[vggt-rkdc] $ds DONE $(date '+%F %T')"
done
echo "[vggt-rkdc] ALL DONE $(date '+%F %T')"
