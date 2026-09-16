#!/usr/bin/env bash
set -x
M=artifacts/diagnostics/seed1_20/scene_manifest.json
C=artifacts/diagnostics/seed1_20/ckpts
run_one () {
  name=$1; arm=$2
  python diagnostics/free_geometry/eval_viewcounts.py \
    --run_root artifacts/diagnostics/vs1_$name --manifest $M \
    --ckpt_root $C --step 100 --arms $arm --view_subsets 32v 8v 4v \
    && python diagnostics/free_geometry/run_eval.py \
    --run_root artifacts/diagnostics/vs1_$name --manifest $M \
    && rm -rf artifacts/diagnostics/vs1_$name/eval32/*/model_results
}
run_one md B5_maskdistill
run_one c2m C2M_maskrel
echo ALL_EVAL_DONE
