#!/bin/bash
# Relaunch the 5 failed ndispatch runs (2026-09-21): the 4-lane overnight
# layout OOM-cascaded (~95GB real vs 81GB nominal). 3 lanes now: 1 DA3 + 2 VGGT
# = ~63GB peak. Stuck eval10 watchers (evB hiroom / evB3 scannetpp) pick up
# automatically when the new summaries land.
#   laneR1 (DA3):  hiroom -> scannetpp
#   laneR2 (VGGT): hiroom -> scannetpp
#   laneR3 (VGGT): dtu
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/envs/da3/bin/python
DG=diagnostics/free_geometry
ND=workspace/ndispatch

S_HR=$(tr '\n' ' ' < workspace/benchmark_dataset/hiroom/selected_scene_list_val.txt)
S_SP="09c1414f1b 1ada7a0617 21d970d8de 286b55a2bf 38d58a7a31 3e8bba0176 40aec5fffa 578511c8a9 5f99900f09 7831862f02 7bc286c1b6 9071e139d9 acd95847c5 bcd2436daf bde1e479ad c4c04e6d6c c5439f4607 cc5237fd77 f3d64c30f8 fb5a96b1a2"

da3 () {
  local ds=$1 sc=$2
  local out=$ND/da3_$ds
  rm -rf "$out"; mkdir -p "$out"
  echo "=== [DA3 $ds] RETRY START $(date '+%F %T') ==="
  $PY scripts/train_da3_protocol.py --dataset $ds --scenes $sc \
      --output_root "$out" --steps 100 --arm adaptive_a --pose_weight 1.0 \
      --n_shared 4 --n_train 10 --mask_ratio 0.5 --loss_all_pos \
      > logs/ndispatch_da3_${ds}.log 2>&1 \
      || { echo "=== [DA3 $ds] RETRY FAILED ==="; return 1; }
  echo "=== [DA3 $ds] RETRY DONE $(date '+%F %T') ==="
}

vggt () {
  local ds=$1 vc=$2
  local out=$ND/vggt_$ds
  rm -rf "$out"; mkdir -p "$out"
  cp -f $ND/$ds/scene_manifest.json $out/scene_manifest.json
  echo "=== [VGGT $ds] RETRY TRAIN $(date '+%F %T') ==="
  $PY $DG/train_arms.py --run_root $out --arms C2M_ADAPTIVE \
      --epochs 10 --seed 0 --no_eval32 --loss_all_pos --v2_couple_fix \
      > logs/ndispatch_vggt_${ds}_train.log 2>&1 \
      || { echo "=== [VGGT $ds] RETRY TRAIN FAILED ==="; return 1; }
  $PY $DG/eval_viewcounts.py --manifest $out/scene_manifest.json \
      --ckpt_root $out/ckpts --run_root $out --step 100 \
      --arms C2M_ADAPTIVE --view_subsets $vc \
      > logs/ndispatch_vggt_${ds}_evalvc.log 2>&1 \
      || { echo "=== [VGGT $ds] RETRY EVALVC FAILED ==="; return 1; }
  $PY $DG/run_eval.py --run_root $out --datas $ds \
      > logs/ndispatch_vggt_${ds}_runeval.log 2>&1 \
      || echo "=== [VGGT $ds] RETRY RUNEVAL FAILED (continuing) ==="
  mv $out/eval32_metrics.json $out/eval32_metrics_C2M_ADAPTIVE_ndispatch.json 2>/dev/null
  echo "=== [VGGT $ds] RETRY DONE $(date '+%F %T') ==="
}

case "${1:-}" in
  laneR1) da3 hiroom "$S_HR"; da3 scannetpp "$S_SP" ;;
  laneR2) vggt hiroom "10v allv"; vggt scannetpp "10v 100v" ;;
  laneR3) vggt dtu "10v 49v" ;;
  *) echo "usage: $0 laneR1|laneR2|laneR3"; exit 1 ;;
esac
echo "=== LANE ${1} ALL DONE $(date '+%F %T') ==="
