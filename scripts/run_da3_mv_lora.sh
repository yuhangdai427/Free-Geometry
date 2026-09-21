#!/bin/bash
# DA3 rerun with CORRECT LoRA scope (blocks 13-39, multiview transformer only).
# Both 8:4 and 16:4, all 4 datasets, all-position supervision, LR fixed.
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/envs/da3/bin/python

run () {
  local ds=$1 scenes=$2 tn=$3 tag=$4
  local out=workspace/da3_mv_${ds}_t${tn}s4_${tag}
  mkdir -p "$out"
  echo "[mv] $ds t=$tn START $(date '+%F %T')"
  $PY scripts/train_da3_protocol.py --dataset $ds --scenes $scenes \
      --output_root "$out" --steps 100 --arm rkdc1h \
      --teacher_N $tn --n_shared 4 --n_train 10 \
      --mask_ratio 0.5 --loss_all_pos \
      > logs/da3_mv_${ds}_t${tn}.log 2>&1 \
      || { echo "[mv] $ds t=$tn FAILED"; return 1; }
  echo "[mv] $ds t=$tn DONE $(date '+%F %T')"
}

SCENES_7S="chess fire heads office pumpkin redkitchen stairs"
SCENES_ET="courtyard delivery_area electro facade kicker office pipes playground relief relief_2 terrains"
SCENES_HR=$(tr '\n' ' ' < workspace/benchmark_dataset/hiroom/selected_scene_list_val.txt)
SCENES_SP="09c1414f1b 1ada7a0617 21d970d8de 286b55a2bf 38d58a7a31 3e8bba0176 40aec5fffa 578511c8a9 5f99900f09 7831862f02 7bc286c1b6 9071e139d9 acd95847c5 bcd2436daf bde1e479ad c4c04e6d6c c5439f4607 cc5237fd77 f3d64c30f8 fb5a96b1a2"

# 8:4 first (fastest to compare with old results)
run 7scenes "$SCENES_7S" 8  allpos
run eth3d   "$SCENES_ET" 8  allpos
# Then 16:4
run 7scenes "$SCENES_7S" 16 allpos
run eth3d   "$SCENES_ET" 16 allpos
run hiroom  "$SCENES_HR" 8  allpos
run hiroom  "$SCENES_HR" 16 allpos
run scannetpp "$SCENES_SP" 8  allpos
run scannetpp "$SCENES_SP" 16 allpos
echo "[mv] ALL DONE $(date '+%F %T')"
