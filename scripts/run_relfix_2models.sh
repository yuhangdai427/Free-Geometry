#!/bin/bash
# Rel-loss corrected experiments: VGGT + DA3, all 4 datasets.
# Arm: rkdcr = maskdistill + 1.5·rkd_huber + 1.0·couple + 1.0·rel_corrected
# Mask: image (standard) — tests whether corrected rel improves rkdc
# Protocol: 8:4, all-position, 100 steps
# scannetpp first as sanity check.
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/envs/da3/bin/python
DG=diagnostics/free_geometry

# ---- DA3 with rkdcr arm (need to add to protocol_v1) ----
run_da3 () {
  local ds=$1 scenes=$2
  local out=workspace/da3_${ds}_rkdcr
  mkdir -p "$out"
  echo "[rel] DA3 $ds START $(date '+%F %T')"
  $PY scripts/train_da3_protocol.py --dataset $ds --scenes $scenes \
      --output_root "$out" --steps 100 --arm rkdc1h --teacher_N 8 \
      --n_train 10 --mask_ratio 0.5 --loss_all_pos \
      > logs/da3_${ds}_rkdcr.log 2>&1 \
      || { echo "[rel] DA3 $ds FAILED"; return 1; }
  echo "[rel] DA3 $ds DONE $(date '+%F %T')"
}

# ---- VGGT with rkdc + corrected rel (need arm name) ----
run_vggt () {
  local ds=$1
  local vc=$([ "$ds" = "7scenes" ] || [ "$ds" = "scannetpp" ] && echo "100v" || echo "allv")
  local RR=artifacts/diagnostics/final_protocol_t8/$ds
  local out=artifacts/diagnostics/final_protocol_relfix/$ds
  mkdir -p "$out"
  cp -n $RR/scene_manifest.json $out/ 2>/dev/null || true
  echo "[rel] VGGT $ds START $(date '+%F %T')"
  # Use C2M_RKDC1H (existing arm; rel fix is in the loss code for maskrel only
  # — for now test the corrected rkdc without rel, then add rel later)
  $PY $DG/train_arms.py --run_root $out --arms C2M_RKDC1H \
      --epochs 10 --seed 0 --no_eval32 --loss_all_pos \
      >> logs/vggt_${ds}_relfix.log 2>&1 \
      || { echo "[rel] VGGT $ds TRAIN FAILED"; return 1; }
  $PY $DG/eval_viewcounts.py --manifest $out/scene_manifest.json \
      --ckpt_root $out/ckpts --run_root $out --step 100 \
      --arms C2M_RKDC1H --view_subsets $vc >> logs/vggt_${ds}_relfix.log 2>&1 \
      || { echo "[rel] VGGT $ds EVALVC FAILED"; return 1; }
  $PY $DG/run_eval.py --run_root $out --datas $ds >> logs/vggt_${ds}_relfix.log 2>&1 \
      || { echo "[rel] VGGT $ds RUNEVAL FAILED"; return 1; }
  mv $out/eval32_metrics.json $out/eval32_metrics_relfix.json
  echo "[rel] VGGT $ds DONE $(date '+%F %T')"
}

# ---- scannetpp first ----
SPP_SCENES="09c1414f1b 1ada7a0617 21d970d8de 286b55a2bf 38d58a7a31 3e8bba0176 40aec5fffa 578511c8a9 5f99900f09 7831862f02 7bc286c1b6 9071e139d9 acd95847c5 bcd2436daf bde1e479ad c4c04e6d6c c5439f4607 cc5237fd77 f3d64c30f8 fb5a96b1a2"

echo "=== scannetpp ==="
run_da3 scannetpp "$SPP_SCENES" &
run_vggt scannetpp &
wait
echo "=== scannetpp done ==="

# ---- remaining datasets ----
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

echo "[rel] ALL DONE $(date '+%F %T')"
