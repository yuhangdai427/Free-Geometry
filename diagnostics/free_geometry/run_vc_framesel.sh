#!/usr/bin/env bash
# frame-selection grid eval: 5 configs x (32v/8v/4v) x 6 dev scenes, cleanup inline
set -x
SCENES="1ada7a0617 38d58a7a31 7831862f02 7bc286c1b6 9071e139d9 bde1e479ad"

run_one () {
  name=$1; root=$2; step=$3
  python diagnostics/free_geometry/eval_viewcounts.py \
    --run_root artifacts/diagnostics/vf_$name --manifest $root/scene_manifest.json \
    --ckpt_root $root/ckpts --step $step --arms B5_maskdistill --view_subsets 32v 8v 4v \
    && python diagnostics/free_geometry/run_eval.py \
    --run_root artifacts/diagnostics/vf_$name --manifest $root/scene_manifest.json \
    && rm -rf artifacts/diagnostics/vf_$name/eval32/*/model_results
}

run_one pairs5 artifacts/diagnostics/pairs5_v2 100
run_one pairs20 artifacts/diagnostics/pairs20_v2 100
run_one pairs40 artifacts/diagnostics/pairs40_v2 100
run_one seq8 artifacts/diagnostics/seq8_v2 100
run_one seq16s4 artifacts/diagnostics/seq16s4_v2 100
echo ALL_EVAL_DONE
