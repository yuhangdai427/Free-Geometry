#!/usr/bin/env bash
set -euo pipefail

# Resumable E6 launcher. It preserves completed runs and uses GPU 1 only.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd /root
export CUDA_VISIBLE_DEVICES=1
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/da3/bin/python}"

TRAIN_SEEDS=(30 31 32)
EVAL_SEEDS=(43 44 45)
# Override for a separate, resumable E6 dataset extension, e.g. "scannetpp".
DATASETS=(${DATASETS_OVERRIDE:-eth3d 7scenes})
TRAIN_LR="${TRAIN_LR:-1e-5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/checkpoints/e6_nested_views}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/e6_nested_views}"
INDICES_ROOT="${INDICES_ROOT:-${REPO_ROOT}/artifacts/e6_nested_views_indices}"
MODEL_NAME="${MODEL_NAME:-${REPO_ROOT}/model_weights/VGGT-1B}"
FUSION_WORKERS="${FUSION_WORKERS:-4}"

COMMON_TRAIN_ARGS=(
  --samples_per_scene 10 --epochs 3 --lr "${TRAIN_LR}" --num_workers 2 --image_size 504 --model_name "${MODEL_NAME}"
  --output_layers 4 11 17 23 --lora_rank 32 --lora_alpha 32 --lora_layers_start 0
  --lr_scheduler cosine --warmup_ratio 0.15 --eta_min 1e-8 --weight_decay 1e-5
  --patch_huber_weight 1.0 --patch_huber_cos_weight 2.0 --patch_huber_delta 1.0
  --cf_weight 2.0 --cf_topk 4 --cf_num_ref_samples 256 --cf_num_shared_samples 256
  --cf_angle1_weight 1.0 --cf_angle2_weight 1.0 --cf_angle3_weight 1.0 --cf_selection_mode mixed
  --use_cf_distance --cf_distance_weight 1.0 --cf_distance_chunk_size 16 --cf_distance_type l2
  --cf_distance_temperature 1.0 --cf_distance_mode kl --cf_distance_huber_beta 0.5
  --cf_d1_weight 1.0 --cf_d2_weight 1.0 --cf_d3_weight 0.0 --save_interval 0 --log_interval 1
)

is_complete_run() {
  local run_dir="$1"
  [[ -s "${run_dir}/latest_lora.pt" && -s "${run_dir}/telemetry.json" ]] || return 1
  # A checkpoint is written each epoch; only skip runs with all three epochs recorded.
  "${PYTHON_BIN}" - "${run_dir}/telemetry.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    telemetry = json.load(f)
raise SystemExit(0 if len(telemetry.get("epochs", [])) == 3 else 1)
PY
}

train_one() {
  local dataset="$1" config="$2" seed="$3" teacher_views="$4"; shift 4
  local run_dir="${OUTPUT_ROOT}/${dataset}/${config}/seed${seed}"
  if is_complete_run "${run_dir}"; then
    echo "[SKIP train] ${dataset}/${config}/seed${seed}: complete"
    return
  fi
  local batch_size=2
  if [[ "${teacher_views}" -eq 16 ]]; then batch_size=1; fi
  mkdir -p "$(dirname "${run_dir}")"
  echo "[TRAIN] ${dataset}/${config}/seed${seed}: teacher=${teacher_views}, batch_size=${batch_size}"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/train_vggt.py" --dataset "${dataset}" "${COMMON_TRAIN_ARGS[@]}" \
    --seed "${seed}" --num_views "${teacher_views}" --batch_size "${batch_size}" "$@" \
    --nested_indices_file "${INDICES_ROOT}/${dataset}/train_seed${seed}_p16.jsonl" --output_dir "${run_dir}" \
    2>&1 | tee "${run_dir}.log"
}

is_complete_eval() {
  local dataset="$1" work_dir="$2" views="$3"
  [[ -s "${work_dir}/${views}v/metric_results/${dataset}_pose.json" && -s "${work_dir}/${views}v/metric_results/${dataset}_recon_unposed.json" ]]
}

prune_eval_artifacts() {
  local dataset="$1" work_dir="$2" views="$3"
  local eval_dir="${work_dir}/${views}v"
  # Never remove predictions until both metric reports exist and are non-empty.
  if is_complete_eval "${dataset}" "${work_dir}" "${views}"; then
    find "${eval_dir}/model_results" "${eval_dir}/visualizations" -depth -delete 2>/dev/null || true
    echo "[PRUNE eval] ${eval_dir}: retained metric JSON and summary only"
  else
    echo "[KEEP eval artifacts] ${eval_dir}: metrics incomplete" >&2
    return 1
  fi
}

evaluate_one() {
  local dataset="$1" config="$2" train_seed="$3" eval_seed="$4" views="$5"
  local checkpoint="${OUTPUT_ROOT}/${dataset}/${config}/seed${train_seed}/latest_lora.pt"
  local work_dir="${RESULTS_ROOT}/${dataset}/${config}/train_seed${train_seed}/eval_seed${eval_seed}/${views}v"
  if is_complete_eval "${dataset}" "${work_dir}" "${views}"; then
    prune_eval_artifacts "${dataset}" "${work_dir}" "${views}"
    echo "[SKIP eval] ${dataset}/${config}/train${train_seed}/eval${eval_seed}/${views}v: complete"
    return
  fi
  echo "[EVAL] ${dataset}/${config}/train${train_seed}/eval${eval_seed}/${views}v"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/benchmark_multiview_all_datasets.py" \
    --model_family vggt --datasets "${dataset}" --all_scenes --view_counts "${views}" --seed "${eval_seed}" \
    --model_name "${MODEL_NAME}" --image_size 504 --lora_path "${checkpoint}" --work_dir "${work_dir}" \
    --view_indices_file "${INDICES_ROOT}/${dataset}/eval_seed${eval_seed}_p32.jsonl" \
    --num_fusion_workers "${FUSION_WORKERS}"
  prune_eval_artifacts "${dataset}" "${work_dir}" "${views}"
}

for dataset in "${DATASETS[@]}"; do
  for seed in "${TRAIN_SEEDS[@]}"; do
    train_one "${dataset}" C1 "${seed}" 4
    train_one "${dataset}" C2 "${seed}" 8
    train_one "${dataset}" C3 "${seed}" 16
    train_one "${dataset}" C4 "${seed}" 8 --student_indices 0 4
    train_one "${dataset}" C5 "${seed}" 16 --student_indices 0 4 8 12
  done
done

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

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/summarize_e6_view_ablation.py" \
  --checkpoints_root "${OUTPUT_ROOT}" --results_root "${RESULTS_ROOT}" --datasets "${DATASETS[@]}"
