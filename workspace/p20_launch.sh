#!/bin/bash
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 PYTHONUNBUFFERED=1
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python
OUT=workspace/omnigeo_vggt_rkdc_s100_p20
CS="--benchmark omnigeo --steps 100 --ckpt_steps 100 --n_pairs 20 --swanlab --skip_base_eval --base_from workspace/omnigeo_vggt_rkdc/results_shard0_2.json workspace/omnigeo_vggt_rkdc/results_shard1_2.json"
mkdir -p $OUT
for i in 0 1; do
  tmux new-session -d -s p20_s$i "$PY scripts/vggt_omnigeo_tta.py --shard $i/2 $CS --out $OUT > workspace/p20_s$i.log 2>&1"
  sleep 15
done
echo "p20 2-shard launched $(date '+%F %T')"
