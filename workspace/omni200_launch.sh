#!/bin/bash
# OmniGeo 200-step run: 2 shards, ckpt saves + in-place dual-metric eval every 20 steps in [100,200]
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 PYTHONUNBUFFERED=1
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python
CS="--benchmark omni --steps 200 --ckpt_steps 100 120 140 160 180 200 --n_pairs 10 --swanlab"
OUT=workspace/omnigeo_vggt_rkdc_s200
mkdir -p $OUT
touch $OUT/RUN_STARTED
tmux new-session -d -s omni200_s0 "$PY scripts/vggt_omnigeo_tta.py --shard 0/2 $CS --out $OUT > workspace/omni200_s0.log 2>&1"
sleep 20
tmux new-session -d -s omni200_s1 "$PY scripts/vggt_omnigeo_tta.py --shard 1/2 $CS --out $OUT > workspace/omni200_s1.log 2>&1"
echo "launched at $(date '+%F %T')"
