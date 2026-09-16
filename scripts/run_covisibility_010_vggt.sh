#!/usr/bin/env bash
set -euo pipefail

# Strict sparse-view protocol: up to five distinct scenes per dataset, each
# contributing only unique, pairwise <= 0.1 overlap frames.  Ineligible scenes
# are omitted; they are never padded with repeated cameras.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/da3/bin/python}"
DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="$DEVICE"
SELECTION_SOURCE="${SELECTION_SOURCE:-$REPO_ROOT/artifacts/covisibility_all_datasets_0025/nonoverlap_010_counts.jsonl}"
SELECTION_ROOT="${SELECTION_ROOT:-$REPO_ROOT/artifacts/covisibility_010_strict_top5}"
SELECTION="$SELECTION_ROOT/selected_scenes.jsonl"
MANIFEST_ROOT="${MANIFEST_ROOT:-$REPO_ROOT/artifacts/covisibility_010_strict_top5/sequences}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$REPO_ROOT/checkpoints/covisibility_010_strict_top5_vggt}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/results/covisibility_010_strict_top5_vggt}"
MODEL_NAME="${MODEL_NAME:-$REPO_ROOT/model_weights/VGGT-1B}"

"$PYTHON_BIN" "$REPO_ROOT/scripts/build_strict_covisibility_010_top5.py" \
  --input "$SELECTION_SOURCE" --output "$SELECTION" \
  --report "$SELECTION_ROOT/selection_report.json" --datasets eth3d scannetpp \
  --threshold 0.1 --max-scenes 5

for dataset in eth3d scannetpp; do
  "$PYTHON_BIN" "$REPO_ROOT/scripts/generate_covisibility_010_sequences.py" --selection "$SELECTION" --output "$MANIFEST_ROOT/${dataset}_train.jsonl" --datasets "$dataset" --split train --samples-per-scene 10 --epochs 3 --seed 30
  "$PYTHON_BIN" "$REPO_ROOT/scripts/generate_covisibility_010_sequences.py" --selection "$SELECTION" --output "$MANIFEST_ROOT/${dataset}_eval.jsonl" --datasets "$dataset" --split eval --samples-per-scene 1 --seed 43
  "$PYTHON_BIN" "$REPO_ROOT/scripts/train_vggt_covisibility_010.py" --sequence-manifest "$MANIFEST_ROOT/${dataset}_train.jsonl" --dataset "$dataset" --samples_per_scene 10 --epochs 3 --batch_size 1 --num_workers 2 --image_size 504 --model_name "$MODEL_NAME" --output_layers 4 11 17 23 --lora_rank 32 --lora_alpha 32 --lora_layers_start 0 --lr 1e-5 --lr_scheduler cosine --warmup_ratio 0.15 --eta_min 1e-8 --weight_decay 1e-5 --cf_weight 2.0 --cf_topk 4 --cf_num_ref_samples 256 --cf_num_shared_samples 256 --cf_selection_mode mixed --use_cf_distance --cf_distance_weight 1.0 --save_interval 0 --log_interval 1 --output_dir "$CHECKPOINT_ROOT/$dataset"
done

"$PYTHON_BIN" "$REPO_ROOT/scripts/benchmark_vggt_covisibility_010.py" --sequence-manifest "$MANIFEST_ROOT/eth3d_eval.jsonl" --selection "$SELECTION" --datasets eth3d --model-name "$MODEL_NAME" --lora-eth3d "$CHECKPOINT_ROOT/eth3d/latest_lora.pt" --output-root "$RESULT_ROOT/eth3d" --delete-arm-results
"$PYTHON_BIN" "$REPO_ROOT/scripts/benchmark_vggt_covisibility_010.py" --sequence-manifest "$MANIFEST_ROOT/scannetpp_eval.jsonl" --selection "$SELECTION" --datasets scannetpp --model-name "$MODEL_NAME" --lora-scannetpp "$CHECKPOINT_ROOT/scannetpp/latest_lora.pt" --output-root "$RESULT_ROOT/scannetpp" --delete-arm-results
