#!/bin/bash
# OmniGeo 200-step FAST: 3 shards, ckpt evals at 100/150/200 (weights still saved per ckpt),
# baseline reused from the 100-step historical results (deterministic, verified identical)
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 PYTHONUNBUFFERED=1
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python
OUT=workspace/omnigeo_vggt_rkdc_s200
CS="--benchmark omnigeo --steps 200 --ckpt_steps 100 150 200 --n_pairs 10 --swanlab --skip_base_eval --base_from workspace/omnigeo_vggt_rkdc/results_shard0_2.json workspace/omnigeo_vggt_rkdc/results_shard1_2.json"
mkdir -p $OUT
touch $OUT/RUN_STARTED
for i in 0 1 2; do
  tmux new-session -d -s omni200_s$i "$PY scripts/vggt_omnigeo_tta.py --shard $i/3 $CS --out $OUT > workspace/omni200_s$i.log 2>&1"
  sleep 15
done
echo "launched 3 shards at $(date '+%F %T')"
