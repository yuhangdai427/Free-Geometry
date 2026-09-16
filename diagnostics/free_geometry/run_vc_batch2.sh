#!/usr/bin/env bash
# FM/XCO/maskdistill eval batch: 4 configs x (32v/8v/4v) x 6 scenes
set -x
V2M=artifacts/diagnostics/bakeoff_v2_transductive/scene_manifest.json
LITM=artifacts/diagnostics/lit_screen/scene_manifest.json
V2C=artifacts/diagnostics/bakeoff_v2_transductive/ckpts
LITC=artifacts/diagnostics/lit_screen/ckpts

run_one () {
  name=$1; mani=$2; ck=$3; arm=$4; step=$5
  python diagnostics/free_geometry/eval_viewcounts.py \
    --run_root artifacts/diagnostics/vc_$name --manifest $mani \
    --ckpt_root $ck --step $step --arms $arm --view_subsets 32v 8v 4v \
    && python diagnostics/free_geometry/run_eval.py \
    --run_root artifacts/diagnostics/vc_$name --manifest $mani
}

run_one fmzero $V2M $V2C FM_zero 100
run_one fmmean $V2M $V2C FM_mean 100
run_one xco $V2M $V2C XCO 100
run_one maskdistill $LITM $LITC B5_maskdistill 100
echo ALL_EVAL_DONE
