#!/usr/bin/env bash
# 20-scene validation eval: 4 configs x (32v/8v/4v), model_results deleted inline
set -x
M20=artifacts/diagnostics/bakeoff_20scenes/scene_manifest.json
S16=artifacts/diagnostics/seq16_20/scene_manifest.json
C20=artifacts/diagnostics/bakeoff_20scenes/ckpts
CS16=artifacts/diagnostics/seq16_20/ckpts

run_one () {
  name=$1; mani=$2; ck=$3; arm=$4; step=$5
  python diagnostics/free_geometry/eval_viewcounts.py \
    --run_root artifacts/diagnostics/v20_$name --manifest $mani \
    --ckpt_root $ck --step $step --arms $arm --view_subsets 32v 8v 4v \
    && python diagnostics/free_geometry/run_eval.py \
    --run_root artifacts/diagnostics/v20_$name --manifest $mani \
    && rm -rf artifacts/diagnostics/v20_$name/eval32/*/model_results
}

run_one md20 $M20 $C20 B5_maskdistill 100
run_one c2m20 $M20 $C20 C2M_maskrel 100
run_one seqc2m20 $S16 $CS16 C2M_maskrel 100
run_one seqmd20 $S16 $CS16 B5_maskdistill 100
echo ALL_EVAL_DONE
