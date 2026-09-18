#!/bin/bash
# Token-mask experiments on VGGT + DA3, all 4 datasets (2 × 4 = 8 cells).
# Protocol: 8:4, rkdc1h (VGGT=C2M_RKDC1H), all-position loss, 100 steps.
# VGGT: feature-level masking (hook on aggregator.patch_embed output)
# DA3: token_shallow mode (hook on vit.patch_embed output)
# scannetpp first as sanity check, then remaining 3 datasets.
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/envs/da3/bin/python
DG=diagnostics/free_geometry

# ---- DA3 token_shallow (already implemented) ----
run_da3 () {
  local ds=$1
  local scenes=$2
  local out=workspace/da3_${ds}_token_shallow
  mkdir -p "$out"
  echo "[tok] DA3 $ds START $(date '+%F %T')"
  $PY scripts/train_da3_protocol.py --dataset $ds --scenes $scenes \
      --output_root "$out" --steps 100 --arm rkdc1h --teacher_N 8 \
      --n_train 10 --mask_ratio 0.5 --mask_mode token_shallow --loss_all_pos \
      > logs/da3_${ds}_token_shallow.log 2>&1 \
      || { echo "[tok] DA3 $ds FAILED"; return 1; }
  echo "[tok] DA3 $ds DONE $(date '+%F %T')"
}

# ---- VGGT token masking (new --mask_mode token) ----
run_vggt () {
  local ds=$1
  local vc=$([ "$ds" = "7scenes" ] || [ "$ds" = "scannetpp" ] && echo "100v" || echo "allv")
  local RR=artifacts/diagnostics/final_protocol_t8/$ds
  local out=artifacts/diagnostics/final_protocol_tokenmask/$ds
  mkdir -p "$out"
  cp -n $RR/scene_manifest.json $out/ 2>/dev/null || true
  echo "[tok] VGGT $ds START $(date '+%F %T')"
  $PY $DG/train_arms.py --run_root $out --arms C2M_RKDC1H \
      --epochs 10 --seed 0 --no_eval32 --loss_all_pos --mask_mode token \
      >> logs/vggt_${ds}_tokenmask.log 2>&1 \
      || { echo "[tok] VGGT $ds TRAIN FAILED"; return 1; }
  $PY $DG/eval_viewcounts.py --manifest $out/scene_manifest.json \
      --ckpt_root $out/ckpts --run_root $out --step 100 \
      --arms C2M_RKDC1H --view_subsets $vc >> logs/vggt_${ds}_tokenmask.log 2>&1 \
      || { echo "[tok] VGGT $ds EVALVC FAILED"; return 1; }
  $PY $DG/run_eval.py --run_root $out --datas $ds >> logs/vggt_${ds}_tokenmask.log 2>&1 \
      || { echo "[tok] VGGT $ds RUNEVAL FAILED"; return 1; }
  mv $out/eval32_metrics.json $out/eval32_metrics_token.json
  echo "[tok] VGGT $ds DONE $(date '+%F %T')"
}

# ---- scannetpp first (sanity check) ----
SPP_SCENES="09c1414f1b 1ada7a0617 21d970d8de 286b55a2bf 38d58a7a31 3e8bba0176 40aec5fffa 578511c8a9 5f99900f09 7831862f02 7bc286c1b6 9071e139d9 acd95847c5 bcd2436daf bde1e479ad c4c04e6d6c c5439f4607 cc5237fd77 f3d64c30f8 fb5a96b1a2"

echo "=== scannetpp sanity check ==="
run_da3 scannetpp "$SPP_SCENES" &
DA3_PID=$!
run_vggt scannetpp &
VGGT_PID=$!
wait $DA3_PID $VGGT_PID
echo "=== scannetpp done ==="

# ---- remaining 3 datasets serially ----
for ds in 7scenes eth3d hiroom; do
  case $ds in
    7scenes) SC="chess fire heads office pumpkin redkitchen stairs" ;;
    eth3d)   SC="courtyard delivery_area electro facade kicker office pipes playground relief relief_2 terrains" ;;
    hiroom)  SC="$(tr '\n' ' ' < workspace/benchmark_dataset/hiroom/selected_scene_list_val.txt)" ;;
  esac
  run_da3 $ds "$SC" &
  run_vggt $ds &
  wait
done

echo "[tok] ALL DONE $(date '+%F %T')"
