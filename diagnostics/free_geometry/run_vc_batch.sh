#!/usr/bin/env bash
# steps/pairs/ohem eval batch: 5 configs x (32v/8v/4v) x 6 scenes
set -x
V2M=artifacts/diagnostics/bakeoff_v2_transductive/scene_manifest.json
P20M=artifacts/diagnostics/pairs20_v2/scene_manifest.json
V2C=artifacts/diagnostics/bakeoff_v2_transductive/ckpts
P20C=artifacts/diagnostics/pairs20_v2/ckpts
SCENES="1ada7a0617 38d58a7a31 7831862f02 7bc286c1b6 9071e139d9 bde1e479ad"

run_one () { # name manifest ckpt_root arm step
  name=$1; mani=$2; ck=$3; arm=$4; step=$5
  python diagnostics/free_geometry/eval_viewcounts.py \
    --run_root artifacts/diagnostics/vc_$name --manifest $mani \
    --ckpt_root $ck --step $step --arms $arm --view_subsets 32v 8v 4v \
    && python diagnostics/free_geometry/run_eval.py \
    --run_root artifacts/diagnostics/vc_$name --manifest $mani
}

run_one step100 $V2M $V2C C2_b5_rel 100
run_one step200 $V2M $V2C C2_b5_rel 200
run_one step400 $V2M $V2C C2_b5_rel 400
run_one pairs20 $P20M $P20C C2_b5_rel 100
run_one ohem100 $V2M $V2C C2O_ohem 100
echo ALL_EVAL_DONE
