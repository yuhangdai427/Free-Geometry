#!/bin/bash
# Adaptive loss (方案1) on eth3d: DA3 + VGGT in parallel
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/envs/da3/bin/python
SCENES="courtyard delivery_area electro facade kicker office pipes playground relief relief_2 terrains"

# DA3 adaptive
out=workspace/da3_adaptive_eth3d
mkdir -p "$out"
echo "[adapt] DA3 eth3d START $(date '+%F %T')"
$PY scripts/train_da3_protocol.py --dataset eth3d --scenes $SCENES \
    --output_root "$out" --steps 100 --arm adaptive \
    --teacher_N 8 --n_shared 4 --n_train 10 \
    --mask_ratio 0.5 --loss_all_pos \
    > logs/da3_adaptive_eth3d.log 2>&1 \
    || echo "[adapt] DA3 FAILED"
echo "[adapt] DA3 eth3d DONE $(date '+%F %T')"
