#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# DA3 leave-one-dataset-out cross-domain experiment.
#
# Held-out eval dataset: 7scenes
# Training datasets: eth3d + scannetpp + hiroom
#
# This trains one shared Free-Geometry adapter on the three source domains,
# evaluates it on 7scenes, and renders summary figures from metrics.json.
# =============================================================================

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PYTHONUNBUFFERED=1

is_positive_integer() {
    [[ "${1:-}" =~ ^[1-9][0-9]*$ ]]
}

if ! is_positive_integer "${OMP_NUM_THREADS:-}"; then
    export OMP_NUM_THREADS=1
fi
if ! is_positive_integer "${MKL_NUM_THREADS:-}"; then
    export MKL_NUM_THREADS=1
fi
if ! is_positive_integer "${OPENBLAS_NUM_THREADS:-}"; then
    export OPENBLAS_NUM_THREADS=1
fi
if ! is_positive_integer "${NUMEXPR_NUM_THREADS:-}"; then
    export NUMEXPR_NUM_THREADS=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

MODEL_NAME=${MODEL_NAME:-depth-anything/DA3-GIANT-1.1}
TRAIN_DATASETS=${TRAIN_DATASETS:-"eth3d scannetpp hiroom"}
TEST_DATASET=${TEST_DATASET:-7scenes}
EXP_NAME=${EXP_NAME:-lodo_7scenes_train_eth3d_scannetpp_hiroom}

OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/${EXP_NAME}}
BENCHMARK_ROOT=${BENCHMARK_ROOT:-./workspace/${EXP_NAME}}
FIGURE_DIR=${FIGURE_DIR:-./results/${EXP_NAME}/figures}

NUM_VIEWS=${NUM_VIEWS:-8}
MAX_FRAMES_LIST=${MAX_FRAMES_LIST:-"4 8"}
LORA_SEEDS=${LORA_SEEDS:-44}
BASELINE_SEEDS=${BASELINE_SEEDS:-44}
LORA_EPOCHS=${LORA_EPOCHS:-latest}

# Mixed-domain training defaults. Override from the environment for ablations.
SAMPLES_PER_SCENE=${SAMPLES_PER_SCENE:-1}
SEEDS_LIST=${SEEDS_LIST:-none}
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-2}
LR=${LR:-5e-5}
ETA_MIN=${ETA_MIN:-1e-8}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-32}

# Feature + CF settings.
CF_WEIGHT=${CF_WEIGHT:-2.0}
CF_TOPK=${CF_TOPK:-4}
CF_NUM_REF=${CF_NUM_REF:-256}
CF_NUM_SHARED=${CF_NUM_SHARED:-256}

PATCH_HUBER_WEIGHT=${PATCH_HUBER_WEIGHT:-1.0}
PATCH_HUBER_COS_WEIGHT=${PATCH_HUBER_COS_WEIGHT:-2.0}
PATCH_HUBER_DELTA=${PATCH_HUBER_DELTA:-1.0}

USE_FINETUNE=${USE_FINETUNE:-0}
FINETUNE_LR=${FINETUNE_LR:-5e-6}

CF_DIST_WEIGHT=${CF_DIST_WEIGHT:-1.0}
CF_DIST_TEMP=${CF_DIST_TEMP:-10.0}
CF_D1_WEIGHT=${CF_D1_WEIGHT:-1.0}
CF_D2_WEIGHT=${CF_D2_WEIGHT:-1.0}
CF_D3_WEIGHT=${CF_D3_WEIGHT:-0.0}
CF_DIST_MODE=${CF_DIST_MODE:-kl}

finetune_args=()
if [ "${USE_FINETUNE}" = "1" ]; then
    LR=${FINETUNE_LR}
    finetune_args=(--finetune)
fi

seed_args=()
effective_samples="${SAMPLES_PER_SCENE}"
if [ "${SEEDS_LIST}" != "none" ]; then
    seed_args=(--seeds_list ${SEEDS_LIST})
    effective_samples="$(wc -w <<< "${SEEDS_LIST}")"
fi

train() {
    echo ""
    echo "============================================================"
    echo "Training DA3 LODO adapter"
    echo "  Train datasets: ${TRAIN_DATASETS}"
    echo "  Held-out eval:  ${TEST_DATASET}"
    echo "  Output:         ${OUTPUT_DIR}"
    echo "  Samples/scene:  ${effective_samples}"
    echo "  Seeds list:     ${SEEDS_LIST} (set SEEDS_LIST=none to use random samples)"
    echo "  Epochs:         ${EPOCHS}"
    echo "  Batch size:     ${BATCH_SIZE}"
    echo "  LR:             ${LR}"
    echo "  LoRA rank:      ${LORA_RANK}, alpha: ${LORA_ALPHA}"
    echo "============================================================"

    python -u ./scripts/train_da3.py \
        --datasets ${TRAIN_DATASETS} \
        --samples_per_scene "${SAMPLES_PER_SCENE}" \
        "${seed_args[@]}" \
        "${finetune_args[@]}" \
        --model_name "${MODEL_NAME}" \
        --num_views "${NUM_VIEWS}" \
        --patch_huber_weight "${PATCH_HUBER_WEIGHT}" \
        --patch_huber_cos_weight "${PATCH_HUBER_COS_WEIGHT}" \
        --patch_huber_delta "${PATCH_HUBER_DELTA}" \
        --cf_weight "${CF_WEIGHT}" \
        --cf_topk "${CF_TOPK}" \
        --cf_num_ref_samples "${CF_NUM_REF}" \
        --cf_num_shared_samples "${CF_NUM_SHARED}" \
        --cf_angle1_weight 1.0 \
        --cf_angle2_weight 1.0 \
        --cf_angle3_weight 1.0 \
        --cf_selection_mode mixed \
        --use_cf_distance \
        --cf_distance_weight "${CF_DIST_WEIGHT}" \
        --cf_distance_temperature "${CF_DIST_TEMP}" \
        --cf_distance_mode "${CF_DIST_MODE}" \
        --cf_d1_weight "${CF_D1_WEIGHT}" \
        --cf_d2_weight "${CF_D2_WEIGHT}" \
        --cf_d3_weight "${CF_D3_WEIGHT}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --lr "${LR}" \
        --lora_rank "${LORA_RANK}" \
        --lora_alpha "${LORA_ALPHA}" \
        --lr_scheduler cosine \
        --warmup_ratio 0.15 \
        --eta_min "${ETA_MIN}" \
        --weight_decay 1e-5 \
        --output_dir "${OUTPUT_DIR}"
}

benchmark_lora_epoch() {
    local epoch=$1
    local max_frames=$2
    local lora_path="${OUTPUT_DIR}/epoch_${epoch}_lora.pt"
    local work_dir="${BENCHMARK_ROOT}/lora_epoch${epoch}/frames_${max_frames}/${TEST_DATASET}"

    if [ ! -f "${lora_path}" ]; then
        echo "WARNING: LoRA weights not found at ${lora_path}. Skipping."
        return 0
    fi

    echo ""
    echo "============================================================"
    echo "Benchmarking held-out ${TEST_DATASET}"
    echo "  Epoch:      ${epoch}"
    echo "  Max frames: ${max_frames}"
    echo "  Seeds:      ${LORA_SEEDS}"
    echo "============================================================"

    python -u ./scripts/benchmark_da3.py \
        --lora_path "${lora_path}" \
        --base_model "${MODEL_NAME}" \
        --lora_rank "${LORA_RANK}" \
        --lora_alpha "${LORA_ALPHA}" \
        "${finetune_args[@]}" \
        --datasets "${TEST_DATASET}" \
        --modes pose recon_unposed \
        --max_frames "${max_frames}" \
        --seeds ${LORA_SEEDS} \
        --work_dir "${work_dir}"
}

benchmark_lora() {
    shopt -s nullglob
    local lora_files=( "${OUTPUT_DIR}"/epoch_*_lora.pt )
    shopt -u nullglob

    if [ "${#lora_files[@]}" -eq 0 ]; then
        echo "ERROR: No LoRA epoch weights found under ${OUTPUT_DIR}/" >&2
        return 1
    fi

    local all_epochs=()
    local file
    for file in "${lora_files[@]}"; do
        local base
        base="$(basename "${file}")"
        if [[ "${base}" =~ ^epoch_([0-9]+)_lora\.pt$ ]]; then
            all_epochs+=( "${BASH_REMATCH[1]}" )
        fi
    done
    mapfile -t all_epochs < <(printf '%s\n' "${all_epochs[@]}" | sort -n)

    local epochs=()
    if [ "${LORA_EPOCHS}" = "latest" ]; then
        epochs=( "${all_epochs[-1]}" )
    elif [ "${LORA_EPOCHS}" = "all" ]; then
        epochs=( "${all_epochs[@]}" )
    else
        # Space-separated epoch ids, for example LORA_EPOCHS="1 2".
        read -r -a epochs <<< "${LORA_EPOCHS}"
    fi

    local epoch
    local max_frames
    for epoch in "${epochs[@]}"; do
        for max_frames in ${MAX_FRAMES_LIST}; do
            benchmark_lora_epoch "${epoch}" "${max_frames}"
        done
    done
}

benchmark_baseline() {
    local max_frames
    local seed

    echo ""
    echo "============================================================"
    echo "Benchmarking DA3 baseline on held-out ${TEST_DATASET}"
    echo "  Max frames: ${MAX_FRAMES_LIST}"
    echo "  Seeds:      ${BASELINE_SEEDS}"
    echo "============================================================"

    for max_frames in ${MAX_FRAMES_LIST}; do
        for seed in ${BASELINE_SEEDS}; do
            local work_dir="${BENCHMARK_ROOT}/baseline/frames_${max_frames}/${TEST_DATASET}/seed${seed}"
            echo "  [Baseline] max_frames=${max_frames}, seed=${seed}"
            python -u -c "
import json
import os
import sys
import torch

sys.path.insert(0, 'src')
from depth_anything_3.api import DepthAnything3
from depth_anything_3.bench.evaluator import Evaluator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
api = DepthAnything3.from_pretrained('${MODEL_NAME}').to(device)
work_dir = '${work_dir}'
evaluator = Evaluator(
    work_dir=work_dir,
    datas=['${TEST_DATASET}'],
    modes=['pose', 'recon_unposed'],
    max_frames=${max_frames},
    seed=${seed},
)
evaluator.infer(api)
metrics = evaluator.eval()
evaluator.print_metrics(metrics)
os.makedirs(work_dir, exist_ok=True)
with open(os.path.join(work_dir, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)
"
        done
    done
}

figures() {
    mkdir -p "${FIGURE_DIR}"

    local latest_metrics=""
    shopt -s nullglob
    local metrics_files=( "${BENCHMARK_ROOT}"/lora_epoch*/frames_*/"${TEST_DATASET}"/metrics.json )
    shopt -u nullglob
    if [ "${#metrics_files[@]}" -gt 0 ]; then
        latest_metrics="$(printf '%s\n' "${metrics_files[@]}" | sort -V | tail -n 1)"
    fi

    if [ -z "${latest_metrics}" ]; then
        echo "ERROR: No metrics.json found under ${BENCHMARK_ROOT}. Run benchmark_lora first." >&2
        return 1
    fi

    echo ""
    echo "============================================================"
    echo "Rendering figures"
    echo "  Metrics: ${latest_metrics}"
    echo "  Output:  ${FIGURE_DIR}"
    echo "============================================================"

    python -u ./scripts/plot_da3_lodo_metrics.py \
        --metrics "${latest_metrics}" \
        --dataset "${TEST_DATASET}" \
        --out_dir "${FIGURE_DIR}"
}

usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  train               - Train one adapter on eth3d+scannetpp+hiroom"
    echo "  benchmark_lora      - Evaluate all saved epochs on held-out 7scenes"
    echo "  benchmark_baseline  - Evaluate original DA3 on held-out 7scenes"
    echo "  figures             - Generate figures from latest held-out metrics.json"
    echo "  all                 - Train + benchmark_lora + benchmark_baseline + figures (default)"
    echo ""
    echo "Useful overrides:"
    echo "  EPOCHS=5 LORA_EPOCHS=all LORA_SEEDS=\"43 44 45\" BASELINE_SEEDS=\"43 44 45\" MAX_FRAMES_LIST=\"16 32\" $0 all"
}

command="${1:-all}"
case "${command}" in
    train)
        train
        ;;
    benchmark_lora)
        benchmark_lora
        ;;
    benchmark_baseline)
        benchmark_baseline
        ;;
    figures)
        figures
        ;;
    all)
        train
        benchmark_lora
        benchmark_baseline
        figures
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Unknown command: ${command}" >&2
        usage
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "Done"
echo "  Checkpoints: ${OUTPUT_DIR}/"
echo "  Benchmarks:  ${BENCHMARK_ROOT}/"
echo "  Figures:     ${FIGURE_DIR}/"
echo "============================================================"
