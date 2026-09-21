#!/bin/bash
# N-dispatch unified runs (2026-09-20): adaptive_a (cfix) + auto teacher_N
# (N>=64 -> 16:4 else 8:4) on DA3 + VGGT x {7scenes, eth3d, hiroom, scannetpp, dtu}.
# 4 GPU lanes (~2x17.5GB DA3 + 2x23GB VGGT = 81GB < 97GB).
#   laneA (DA3):  eth3d -> dtu
#   laneB (DA3):  hiroom -> 7scenes -> scannetpp
#   laneC (VGGT): 7scenes -> eth3d -> dtu
#   laneD (VGGT): hiroom -> scannetpp
# VGGT eval includes the 10v subset (10-frame/scene eval) alongside the legacy set.
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/envs/da3/bin/python
DG=diagnostics/free_geometry
ND=workspace/ndispatch
mkdir -p logs

S_7S="chess fire heads office pumpkin redkitchen stairs"
S_ET="courtyard delivery_area electro facade kicker office pipes playground relief relief_2 terrains"
S_HR=$(tr '\n' ' ' < workspace/benchmark_dataset/hiroom/selected_scene_list_val.txt)
S_SP="09c1414f1b 1ada7a0617 21d970d8de 286b55a2bf 38d58a7a31 3e8bba0176 40aec5fffa 578511c8a9 5f99900f09 7831862f02 7bc286c1b6 9071e139d9 acd95847c5 bcd246daf bde1e479ad c4c04e6d6c c5439f4607 cc5237fd77 f3d64c30f8 fb5a96b1a2"
S_DT="scan1 scan4 scan9 scan10 scan11 scan12 scan13 scan15 scan23 scan24 scan29 scan32 scan33 scan34 scan48 scan49 scan62 scan75 scan77 scan110 scan114 scan118"

da3 () {  # ds, scenes
  local ds=$1 sc=$2
  local out=$ND/da3_$ds
  mkdir -p "$out"
  echo "=== [DA3 $ds] START $(date '+%F %T') ==="
  $PY scripts/train_da3_protocol.py --dataset $ds --scenes $sc \
      --output_root "$out" --steps 100 --arm adaptive_a --pose_weight 1.0 \
      --n_shared 4 --n_train 10 --mask_ratio 0.5 --loss_all_pos \
      > logs/ndispatch_da3_${ds}.log 2>&1 \
      || { echo "=== [DA3 $ds] TRAIN FAILED ==="; return 1; }
  echo "=== [DA3 $ds] DONE $(date '+%F %T') ==="
}

vggt () {  # ds, view_subsets
  local ds=$1 vc=$2
  local out=$ND/vggt_$ds
  mkdir -p "$out"
  cp -f $ND/$ds/scene_manifest.json $out/scene_manifest.json
  echo "=== [VGGT $ds] TRAIN START $(date '+%F %T') ==="
  $PY $DG/train_arms.py --run_root $out --arms C2M_ADAPTIVE \
      --epochs 10 --seed 0 --no_eval32 --loss_all_pos --v2_couple_fix \
      > logs/ndispatch_vggt_${ds}_train.log 2>&1 \
      || { echo "=== [VGGT $ds] TRAIN FAILED ==="; return 1; }
  echo "=== [VGGT $ds] EVALVC START $(date '+%F %T') ==="
  $PY $DG/eval_viewcounts.py --manifest $out/scene_manifest.json \
      --ckpt_root $out/ckpts --run_root $out --step 100 \
      --arms C2M_ADAPTIVE --view_subsets $vc \
      > logs/ndispatch_vggt_${ds}_evalvc.log 2>&1 \
      || { echo "=== [VGGT $ds] EVALVC FAILED ==="; return 1; }
  $PY $DG/run_eval.py --run_root $out --datas $ds \
      > logs/ndispatch_vggt_${ds}_runeval.log 2>&1 \
      || echo "=== [VGGT $ds] RUNEVAL FAILED (continuing) ==="
  mv $out/eval32_metrics.json $out/eval32_metrics_C2M_ADAPTIVE_ndispatch.json 2>/dev/null
  echo "=== [VGGT $ds] DONE $(date '+%F %T') ==="
}

case "${1:-}" in
  laneA) da3 eth3d "$S_ET"; da3 dtu "$S_DT" ;;
  laneB) da3 hiroom "$S_HR"; da3 7scenes "$S_7S"; da3 scannetpp "$S_SP" ;;
  laneC) vggt 7scenes "10v 100v"; vggt eth3d "10v allv"; vggt dtu "10v 49v" ;;
  laneD) vggt hiroom "10v allv"; vggt scannetpp "10v 100v" ;;
  *) echo "usage: $0 laneA|laneB|laneC|laneD"; exit 1 ;;
esac
echo "=== LANE ${1} ALL DONE $(date '+%F %T') ==="
