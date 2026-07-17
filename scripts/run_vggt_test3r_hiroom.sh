#!/usr/bin/env bash
set -e

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
WORK_DIR="${WORK_DIR:-./workspace/vggt_test3r_hiroom_allframes}"
IMAGE_SIZE="${IMAGE_SIZE:-504}"

conda run -n da3 python ./scripts/run_vggt_test3r.py \
    --base_model "${MODEL_NAME}" \
    --datasets hiroom \
    --modes pose recon_unposed \
    --seeds 43 \
    --view_counts 0 \
    --max_triplets 100 \
    --triplet_batch_size 4 \
    --image_size "${IMAGE_SIZE}" \
    --work_dir "${WORK_DIR}" \
    "$@"
