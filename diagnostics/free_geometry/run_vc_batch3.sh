#!/usr/bin/env bash
set -x
V2M=artifacts/diagnostics/bakeoff_v2_transductive/scene_manifest.json
S16M=artifacts/diagnostics/seq16_v2/scene_manifest.json
V2C=artifacts/diagnostics/bakeoff_v2_transductive/ckpts
S16C=artifacts/diagnostics/seq16_v2/ckpts

run_one () {
  name=$1; mani=$2; ck=$3; arm=$4; step=$5
  python diagnostics/free_geometry/eval_viewcounts.py \
    --run_root artifacts/diagnostics/vc_$name --manifest $mani \
    --ckpt_root $ck --step $step --arms $arm --view_subsets 32v 8v 4v \
    && python diagnostics/free_geometry/run_eval.py \
    --run_root artifacts/diagnostics/vc_$name --manifest $mani
}

run_one md25 $V2M $V2C MD25 100
run_one md75 $V2M $V2C MD75 100
run_one mdhi $V2M $V2C MDHI 100
run_one mdlo $V2M $V2C MDLO 100
run_one ss8m $V2M $V2C SS8M 100
run_one seq16c2m $S16M $S16C C2M_maskrel 100
echo ALL_EVAL_DONE
