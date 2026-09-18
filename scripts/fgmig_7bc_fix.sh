#!/bin/bash
# Rerun 7bc286c1b6 (the one scene changed from dense to random) for all
# affected models. Only TTA training (baselines unchanged). Run in parallel.
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True

run_one () {
  local model=$1 label=$2 bin_py=$3
  local out=workspace/fgmig/runs/${model}_scannetpp_7bcfix
  mkdir -p "$out"
  echo "[$label] $model 7bc START $(date '+%F %T')"
  if [ "$model" = "dvlt" ]; then
    export PYTHONPATH=/root/autodl-tmp/dvlt/src
  fi
  $bin_py scripts/train_fg_protocol.py --model $model --dataset scannetpp \
      --scenes 7bc286c1b6 --arm rkdc_allpos \
      --output_root "$out" >> logs/fgmig_7bcfix_$model.log 2>&1 \
      || { echo "[$label] $model FAILED"; return 1; }
  /root/miniconda3/envs/da3/bin/python scripts/fg_eval_from_npz.py \
      --dataset scannetpp --output_root "$out" \
      >> logs/fgmig_7bcfix_$model.log 2>&1 \
      || echo "[$label] metrics FAILED"
  echo "[$label] $model 7bc DONE $(date '+%F %T')"
}

run_one omega "7bc-fix" /root/miniconda3/envs/da3/bin/python &
sleep 5
run_one pi3 "7bc-fix" /root/miniconda3/envs/da3/bin/python &
wait
echo "[7bc-fix] omega+pi3 done"
