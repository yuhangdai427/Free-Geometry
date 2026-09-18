#!/bin/bash
# Bug-fix experiments (2026-09-17):
# 1. DA3 7scenes with UNIFIED protocol (8:4, rkdc1h, all-pos, 10 pairs,
#    LR fix) — same as eth3d/hiroom/scannetpp, no special curriculum
# 2. A2/A3 token mask ablation rerun (eth3d) with FIXED corruption/loss mask
#    separation — previous A2/A3 accidentally masked 100% of tokens
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/envs/da3/bin/python

# ---- 1. DA3 7scenes unified protocol ----
out1=workspace/da3_7scenes_unified
mkdir -p "$out1"
echo "[fix] DA3 7scenes unified (8:4 rkdc1h all-pos LR-fixed) START $(date '+%F %T')"
$PY scripts/train_da3_protocol.py --dataset 7scenes \
    --scenes chess fire heads office pumpkin redkitchen stairs \
    --output_root "$out1" --steps 100 --arm rkdc1h --teacher_N 8 \
    --n_train 10 --mask_ratio 0.5 --loss_all_pos \
    > logs/da3_7scenes_unified.log 2>&1 \
    || echo "[fix] DA3 7scenes unified FAILED"
echo "[fix] DA3 7scenes unified DONE $(date '+%F %T')"

# ---- 2. A2/A3 token mask rerun (fixed separation) ----
SCENES="courtyard delivery_area electro facade kicker office pipes playground relief relief_2 terrains"
for mode in token_shallow token_feat; do
  out2=workspace/da3_eth3d_maskpos_${mode}_v2
  mkdir -p "$out2"
  echo "[fix] $mode v2 (50% corruption, all-pos loss) START $(date '+%F %T')"
  $PY scripts/train_da3_protocol.py --dataset eth3d --scenes $SCENES \
      --output_root "$out2" --steps 100 --arm rkdc1h --teacher_N 8 \
      --mask_ratio 0.5 --mask_mode "$mode" --loss_all_pos \
      > "logs/da3_eth3d_maskpos_${mode}_v2.log" 2>&1 \
      || { echo "[fix] $mode v2 FAILED"; continue; }
  echo "[fix] $mode v2 DONE $(date '+%F %T')"
done
echo "[fix] ALL DONE $(date '+%F %T')"
