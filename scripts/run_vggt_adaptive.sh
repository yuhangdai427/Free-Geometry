#!/bin/bash
# VGGT (original) C2M_ADAPTIVE on eth3d
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
DG=diagnostics/free_geometry
RR=artifacts/diagnostics/final_protocol_lossall/eth3d
PY=/root/miniconda3/envs/da3/bin/python

echo "[vggt-adapt] eth3d TRAIN $(date '+%F %T')"
$PY $DG/train_arms.py --run_root $RR --arms C2M_ADAPTIVE \
    --epochs 10 --seed 0 --no_eval32 --loss_all_pos \
    >> logs/vggt_adaptive_eth3d.log 2>&1 \
    || { echo "TRAIN FAILED"; exit 1; }
echo "[vggt-adapt] EVALVC $(date '+%F %T')"
$PY $DG/eval_viewcounts.py --manifest $RR/scene_manifest.json \
    --ckpt_root $RR/ckpts --run_root $RR --step 100 \
    --arms C2M_ADAPTIVE --view_subsets allv \
    >> logs/vggt_adaptive_eth3d.log 2>&1
echo "[vggt-adapt] RUNEVAL $(date '+%F %T')"
$PY $DG/run_eval.py --run_root $RR --datas eth3d \
    >> logs/vggt_adaptive_eth3d.log 2>&1
echo "[vggt-adapt] DONE $(date '+%F %T')"
