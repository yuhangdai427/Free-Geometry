#!/bin/bash
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 PYTHONUNBUFFERED=1
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python
OUT=workspace/omnivideo_vggt_rkdc
CS="--benchmark omnivideo --steps 100 --ckpt_steps 100 --n_pairs 10 --swanlab --skip_base_eval"
mkdir -p $OUT
for i in 0 1 2; do
  tmux new-session -d -s ov_s$i "$PY scripts/vggt_omnigeo_tta.py --shard $i/3 $CS --out $OUT > workspace/ov_s$i.log 2>&1"
  sleep 15
done
echo "omnivideo launched $(date '+%F %T')"
