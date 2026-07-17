#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/da3"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "/opt/conda/etc/profile.d/conda.sh"
fi

conda activate da3

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$ROOT_DIR/.cache/huggingface"
export HF_HUB_CACHE="$ROOT_DIR/.cache/huggingface/hub"

mkdir -p "$HF_HUB_CACHE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting VGGT download"
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"

hf download facebook/vggt-1b --cache-dir "$HF_HUB_CACHE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished VGGT download"
