#!/bin/bash
# Mask-position ablation on DA3 eth3d (AI plan 2026-09-17 section 7):
#   none          A0: clean input, all-position loss
#   image         A1: ImageNet-mean fill on input pixels (current recipe)
#   token_shallow A2: zero patch tokens right after patch projection
#   token_feat    A3: zero patch tokens at block-12 output (before DA3's
#                    cross-view attention starts at block 13)
# All arms: same Omega (same seed), same supervision (all positions), same
# 8:4 rkdc1h recipe, same eval frames.
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/envs/da3/bin/python
SCENES="courtyard delivery_area electro facade kicker office pipes playground relief relief_2 terrains"

for mode in none image token_shallow token_feat; do
  out=workspace/da3_eth3d_maskpos_${mode}
  mkdir -p "$out"
  echo "[maskpos] $mode START $(date '+%F %T')"
  $PY scripts/train_da3_protocol.py --dataset eth3d --scenes $SCENES \
      --output_root "$out" --steps 100 --arm rkdc1h --teacher_N 8 \
      --mask_ratio 0.5 --mask_mode "$mode" --loss_all_pos \
      > "logs/da3_eth3d_maskpos_${mode}.log" 2>&1 \
      || { echo "[maskpos] $mode FAILED"; continue; }
  echo "[maskpos] $mode DONE $(date '+%F %T')"
done
echo "[maskpos] ALL DONE $(date '+%F %T')"
