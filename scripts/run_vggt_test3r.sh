#!/usr/bin/env bash
set -e

# VGGT-test3r runner.
# Mirrors scripts/run_vggt.sh by using facebook/vggt-1b, but keeps Hugging Face
# cache files in this repo instead of /root/.cache.

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$(pwd)/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$(pwd)/.cache/huggingface/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$(pwd)/.cache/huggingface/hub}"

MODEL_NAME="${MODEL_NAME:-facebook/vggt-1b}"
WORK_DIR="${WORK_DIR:-./workspace/vggt_test3r}"
IMAGE_SIZE="${IMAGE_SIZE:-504}"

python ./scripts/run_vggt_test3r.py \
    --base_model "${MODEL_NAME}" \
    --datasets 7scenes \
    --modes pose recon_unposed \
    --seeds 43 44 45 \
    --view_counts 4 8 \
    --image_size "${IMAGE_SIZE}" \
    --work_dir "${WORK_DIR}" \
    "$@"
