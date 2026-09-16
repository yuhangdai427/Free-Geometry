#!/usr/bin/env bash
set -euo pipefail

# E6: nested teacher/student view-count ablation, trained independently per dataset.
# Run from any directory. Benchmark constants resolve workspace/ relative to /root.

STAGE="${1:-prepare}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd /root
export CUDA_VISIBLE_DEVICES=1
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/da3/bin/python}"

# Keep training randomness separate from the established 43/44/45 evaluation seeds.
TRAIN_SEEDS=(30 31 32)
EVAL_SEEDS=(43 44 45)
DATASETS=(eth3d 7scenes)
SAMPLES_PER_SCENE=10
EPOCHS=3
LR=1e-5
IMAGE_SIZE=504
MODEL_NAME="${MODEL_NAME:-${REPO_ROOT}/model_weights/VGGT-1B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/checkpoints/e6_nested_views}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/e6_nested_views}"
INDICES_ROOT="${INDICES_ROOT:-${REPO_ROOT}/artifacts/e6_nested_views_indices}"

# Identical for every E6 run except dataset, training seed, num_views, and student indices.
COMMON_TRAIN_ARGS=(
  --samples_per_scene "${SAMPLES_PER_SCENE}" --epochs "${EPOCHS}" --lr "${LR}"
  --batch_size 2 --num_workers 2 --image_size "${IMAGE_SIZE}" --model_name "${MODEL_NAME}"
  --output_layers 4 11 17 23 --lora_rank 32 --lora_alpha 32 --lora_layers_start 0
  --lr_scheduler cosine --warmup_ratio 0.15 --eta_min 1e-8 --weight_decay 1e-5
  --patch_huber_weight 1.0 --patch_huber_cos_weight 2.0 --patch_huber_delta 1.0
  --cf_weight 2.0 --cf_topk 4 --cf_num_ref_samples 256 --cf_num_shared_samples 256
  --cf_angle1_weight 1.0 --cf_angle2_weight 1.0 --cf_angle3_weight 1.0 --cf_selection_mode mixed
  --use_cf_distance --cf_distance_weight 1.0 --cf_distance_chunk_size 16 --cf_distance_type l2
  --cf_distance_temperature 1.0 --cf_distance_mode kl --cf_distance_huber_beta 0.5
  --cf_d1_weight 1.0 --cf_d2_weight 1.0 --cf_d3_weight 0.0 --save_interval 0 --log_interval 1
)

prepare_indices() {
  local dataset
  for dataset in "${DATASETS[@]}"; do
    local index_dir="${INDICES_ROOT}/${dataset}"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/generate_e6_view_indices.py" --dataset "${dataset}" --seeds "${TRAIN_SEEDS[@]}" \
      --samples_per_scene "${SAMPLES_PER_SCENE}" --pool_size 16 --skip_insufficient --prefix train --output_dir "${index_dir}"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/generate_e6_view_indices.py" --dataset "${dataset}" --seeds "${EVAL_SEEDS[@]}" \
      --samples_per_scene 1 --pool_size 32 --use_all_if_insufficient --prefix eval --output_dir "${index_dir}"
  done
}

train_one() {
  local dataset="$1" config="$2" seed="$3" teacher_views="$4"; shift 4
  local run_dir="${OUTPUT_ROOT}/${dataset}/${config}/seed${seed}"
  local batch_size=2
  if [[ "${teacher_views}" -eq 16 ]]; then
    # 16-view teacher forwards exceed the safe 8-view memory envelope at batch size 2.
    batch_size=1
  fi
  mkdir -p "$(dirname "${run_dir}")"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/train_vggt.py" --dataset "${dataset}" "${COMMON_TRAIN_ARGS[@]}" \
    --seed "${seed}" --num_views "${teacher_views}" --batch_size "${batch_size}" "$@" \
    --nested_indices_file "${INDICES_ROOT}/${dataset}/train_seed${seed}_p16.jsonl" \
    --output_dir "${run_dir}" 2>&1 | tee "${run_dir}.log"
}

evaluate_one() {
  local dataset="$1" config="$2" train_seed="$3" eval_seed="$4" views="$5"
  local checkpoint="${OUTPUT_ROOT}/${dataset}/${config}/seed${train_seed}/latest_lora.pt"
  local work_dir="${RESULTS_ROOT}/${dataset}/${config}/train_seed${train_seed}/eval_seed${eval_seed}/${views}v"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/benchmark_multiview_all_datasets.py" \
    --model_family vggt --datasets "${dataset}" --all_scenes --view_counts "${views}" --seed "${eval_seed}" \
    --model_name "${MODEL_NAME}" --image_size "${IMAGE_SIZE}" --lora_path "${checkpoint}" \
    --work_dir "${work_dir}" --view_indices_file "${INDICES_ROOT}/${dataset}/eval_seed${eval_seed}_p32.jsonl"
}

train_all() {
  local dataset seed
  for dataset in "${DATASETS[@]}"; do
    for seed in "${TRAIN_SEEDS[@]}"; do
      train_one "${dataset}" C1 "${seed}" 4
      train_one "${dataset}" C2 "${seed}" 8
      train_one "${dataset}" C3 "${seed}" 16
      train_one "${dataset}" C4 "${seed}" 8 --student_indices 0 4
      train_one "${dataset}" C5 "${seed}" 16 --student_indices 0 4 8 12
    done
  done
}

evaluate_all() {
  local dataset config train_seed eval_seed views
  for dataset in "${DATASETS[@]}"; do
    for config in C1 C2 C3 C4 C5; do
      for train_seed in "${TRAIN_SEEDS[@]}"; do
        for eval_seed in "${EVAL_SEEDS[@]}"; do
          for views in 8 16 32; do
            evaluate_one "${dataset}" "${config}" "${train_seed}" "${eval_seed}" "${views}"
          done
        done
      done
    done
  done
}

case "${STAGE}" in
  prepare) prepare_indices ;;
  train) train_all ;;
  evaluate) evaluate_all ;;
  report) "${PYTHON_BIN}" "${REPO_ROOT}/scripts/summarize_e6_view_ablation.py" --checkpoints_root "${OUTPUT_ROOT}" --results_root "${RESULTS_ROOT}" ;;
  *) echo "Usage: $0 {prepare|train|evaluate|report}" >&2; exit 2 ;;
esac
