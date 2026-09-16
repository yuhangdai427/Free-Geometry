#!/usr/bin/env bash
set -euo pipefail

# Two LoRAs total: one trained on both hard protocols per dataset. Every
# protocol is still evaluated separately on its own fixed hard windows.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/da3/bin/python}"
MODEL="$ROOT/model_weights/VGGT-1B"
ART="$ROOT/artifacts/hard_view_subsets_7scenes_hiroom"
CKPT="$ROOT/checkpoints/hard_view_subsets_7scenes_hiroom"
OUT="$ROOT/results/hard_view_subsets_7scenes_hiroom"

common=(--samples_per_scene 10 --epochs 3 --batch_size 1 --num_workers 2 --image_size 504
  --model_name "$MODEL" --output_layers 4 11 17 23 --lora_rank 32 --lora_alpha 32 --lora_layers_start 0
  --lr 1e-5 --lr_scheduler cosine --warmup_ratio 0.15 --eta_min 1e-8 --weight_decay 1e-5
  --patch_huber_weight 1.0 --patch_huber_cos_weight 2.0 --cf_weight 2.0 --cf_topk 4
  --cf_num_ref_samples 256 --cf_num_shared_samples 256 --cf_selection_mode mixed --use_cf_distance
  --cf_distance_weight 1.0 --cf_distance_chunk_size 16 --cf_distance_type l2 --cf_distance_temperature 1.0
  --cf_distance_mode kl --cf_distance_huber_beta 0.5 --cf_d1_weight 1.0 --cf_d2_weight 1.0 --cf_d3_weight 0.0
  --save_interval 0 --log_interval 1 --seed 30)

complete_train() { [[ -s "$1/latest_lora.pt" && -s "$1/telemetry.json" ]]; }

for ds in 7scenes hiroom; do
  run="$CKPT/$ds"; mkdir -p "$run"
  if ! complete_train "$run"; then
    "$PYTHON_BIN" scripts/train_vggt_covisibility_010.py --sequence-manifest "$ART/$ds/train_combined.jsonl" --dataset "$ds" "${common[@]}" --output_dir "$run"
  fi
  for criterion in texture occlusion; do
    result="$OUT/$ds/$criterion"
    "$PYTHON_BIN" scripts/benchmark_vggt_covisibility_010.py --selection "$ART/$ds/source_frames.jsonl" \
      --sequence-manifest "$ART/$ds/$criterion/eval.jsonl" --datasets "$ds" --model-name "$MODEL" \
      --lora-7scenes "$run/latest_lora.pt" --lora-hiroom "$run/latest_lora.pt" --output-root "$result" \
      --recon-unposed --delete-arm-results
    "$PYTHON_BIN" scripts/summarize_hard_view_ci.py --window-metrics "$result/window_metrics.json" --dataset "$ds"
  done
  for criterion in texture occlusion; do
    result="$OUT/$ds/${criterion}_16v"
    "$PYTHON_BIN" scripts/benchmark_vggt_covisibility_010.py --selection "$ART/$ds/source_frames.jsonl" \
      --sequence-manifest "$ART/$ds/$criterion/eval_16v.jsonl" --datasets "$ds" --model-name "$MODEL" \
      --lora-7scenes "$run/latest_lora.pt" --lora-hiroom "$run/latest_lora.pt" --output-root "$result" \
      --recon-unposed --arms base_16v lora_16v --delete-arm-results
  done
done

echo "[done] $(date -u +%FT%TZ)"
