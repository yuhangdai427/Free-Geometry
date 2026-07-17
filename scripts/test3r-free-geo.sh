#!/usr/bin/env bash
set -e

# Test3R-style VGGT Free-Geometry LoRA TTA.
# This does not use the VGGT-test3r prompt path. It freezes the original VGGT
# weights, inserts LoRA into the alternating frame/global encoder blocks, and
# updates only LoRA weights with Test3R triplet point-head consistency.

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$(pwd)/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$(pwd)/.cache/huggingface/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$(pwd)/.cache/huggingface/hub}"

MODEL_NAME="${MODEL_NAME:-facebook/vggt-1b}"
WORK_DIR="${WORK_DIR:-./workspace/vggt_test3r_free_geo}"
IMAGE_SIZE="${IMAGE_SIZE:-504}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-1}"
ACCUM_ITER="${ACCUM_ITER:-2}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_LAYERS_START="${LORA_LAYERS_START:-0}"
LORA_TARGET="${LORA_TARGET:-attention_mlp}"

python ./scripts/test3r-free-geo.py \
    --base_model "${MODEL_NAME}" \
    --datasets 7scenes \
    --modes pose recon_unposed \
    --seeds 43 44 45 \
    --view_counts 4 8 \
    --image_size "${IMAGE_SIZE}" \
    --work_dir "${WORK_DIR}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --accum_iter "${ACCUM_ITER}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --lora_layers_start "${LORA_LAYERS_START}" \
    --lora_target "${LORA_TARGET}" \
    "$@"
