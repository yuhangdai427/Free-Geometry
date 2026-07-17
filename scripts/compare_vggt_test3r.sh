#!/usr/bin/env bash
set -e

# Baseline VGGT vs VGGT-Test3R comparison.
# Test3R defaults mirror the original Test3R mv_recon settings:
# prompt=32, epochs=1, lr=1e-5, accum_iter=2, all N^3 triplets by default.
#
# Full run:
#   nohup bash scripts/compare_vggt_test3r.sh > logs/compare_vggt_test3r_4datasets.log 2>&1 &

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "/opt/conda/etc/profile.d/conda.sh"
fi

conda activate da3

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$(pwd)/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$(pwd)/.cache/huggingface/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$(pwd)/.cache/huggingface/hub}"

MODEL_NAME="${MODEL_NAME:-facebook/vggt-1b}"
WORK_DIR="${WORK_DIR:-./workspace/vggt_test3r_comparison_4datasets}"
IMAGE_SIZE="${IMAGE_SIZE:-504}"
DATASETS="${DATASETS:-hiroom eth3d 7scenes scannetpp}"
MODES="${MODES:-pose recon_unposed}"
SEEDS="${SEEDS:-43 44 45}"
VIEW_COUNTS="${VIEW_COUNTS:-4 8}"
NUM_FUSION_WORKERS="${NUM_FUSION_WORKERS:-4}"

conda run -n da3 python ./scripts/compare_vggt_test3r.py \
    --base_model "${MODEL_NAME}" \
    --datasets ${DATASETS} \
    --modes ${MODES} \
    --seeds ${SEEDS} \
    --view_counts ${VIEW_COUNTS} \
    --num_fusion_workers "${NUM_FUSION_WORKERS}" \
    --image_size "${IMAGE_SIZE}" \
    --work_dir "${WORK_DIR}" \
    "$@"
