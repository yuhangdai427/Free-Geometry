#!/usr/bin/env bash
set -euo pipefail

# Isolated DTU-64 pose ablation: train LoRA only, with the camera token frozen.
# It writes separate outputs and never mutates the earlier full-run results.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/da3/bin/python}"
MODEL_NAME="${MODEL_NAME:-depth-anything/DA3-GIANT-1.1}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/da3_dtu64_lora_only_ablation}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${REPO_ROOT}/workspace/da3_dtu64_lora_only_ablation}"
GPU_ID="${GPU_ID:-1}"
EVAL_SEED="${EVAL_SEED:-43}"

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
[[ -x "${PYTHON_BIN}" ]] || { echo "Python executable not found: ${PYTHON_BIN}" >&2; exit 1; }

"${PYTHON_BIN}" -u scripts/train_da3.py \
    --dataset dtu64 --samples_per_scene 10 --model_name "${MODEL_NAME}" --num_views 8 \
    --freeze_camera_token \
    --patch_huber_weight 1.0 --patch_huber_cos_weight -2.0 --patch_huber_delta 1.0 \
    --cf_weight 2.0 --cf_topk 4 --cf_num_ref_samples 256 --cf_num_shared_samples 256 \
    --cf_angle1_weight 1.0 --cf_angle2_weight 1.0 --cf_angle3_weight 1.0 --cf_selection_mode mixed \
    --use_cf_distance --cf_distance_weight 1.0 --cf_distance_temperature 10.0 --cf_distance_mode kl \
    --cf_d1_weight 1.0 --cf_d2_weight 1.0 --cf_d3_weight 0.0 \
    --epochs 3 --batch_size 4 --num_workers 2 --lr 1e-4 \
    --lora_rank 32 --lora_alpha 32 --lr_scheduler cosine --warmup_ratio 0.15 --eta_min 1e-7 \
    --weight_decay 1e-5 --log_interval 1 --output_dir "${OUTPUT_DIR}"

LORA_PATH="${OUTPUT_DIR}/epoch_2_lora.pt"
if [[ ! -f "${LORA_PATH}" ]]; then
    # A frozen camera token leaves no tensor to serialize in the LoRA sidecar.
    # Reuse the metadata checkpoint so the PEFT-only adapter is loadable.
    CHECKPOINT_PATH="${OUTPUT_DIR}/epoch_2.pt"
    [[ -f "${CHECKPOINT_PATH}" ]] || { echo "Missing checkpoint: ${CHECKPOINT_PATH}" >&2; exit 1; }
    ln -sfn "epoch_2.pt" "${LORA_PATH}"
fi
[[ -f "${LORA_PATH}" ]] || { echo "Missing LoRA checkpoint: ${LORA_PATH}" >&2; exit 1; }
"${PYTHON_BIN}" -u scripts/benchmark_da3.py \
    --lora_path "${LORA_PATH}" --base_model "${MODEL_NAME}" --lora_rank 32 --lora_alpha 32 \
    --datasets dtu64 --modes pose --max_frames 64 --seed "${EVAL_SEED}" \
    --work_dir "${WORKSPACE_ROOT}"

"${PYTHON_BIN}" - "${WORKSPACE_ROOT}" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
path = root / "metric_results/dtu64_pose.json"
data = json.loads(path.read_text())
scenes = {key: value for key, value in data.items() if key != "mean"}
if len(scenes) != 13 or not all(math.isfinite(values.get(key, float("nan"))) for values in scenes.values() for key in ("auc03", "auc30")):
    raise SystemExit(f"Invalid pose metrics: {path}")
print(f"Verified LoRA-only DTU-64 pose: {path}")
PY

# Preserve compact metrics/checkpoints/logs, not large per-view inference exports.
RAW_DIR="${WORKSPACE_ROOT}/model_results"
if [[ -d "${RAW_DIR}" ]]; then
    SIZE="$(du -sb "${RAW_DIR}" | awk '{print $1}')"
    find "${RAW_DIR}" -depth -delete
    echo "Removed verified raw model results: ${RAW_DIR} (${SIZE} bytes)"
fi
