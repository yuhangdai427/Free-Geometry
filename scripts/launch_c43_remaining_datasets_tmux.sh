#!/usr/bin/env bash
set -euo pipefail

# Launch a C=43 sequence in one tmux session on GPU 0.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${SESSION_NAME:-freegeo_c43}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/workspace/c43_remaining_datasets/logs}"
RUNNER="${REPO_ROOT}/scripts/run_c43_remaining_datasets.sh"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/da3/bin/python}"
COMMAND="${1:-all}"

if ! command -v tmux >/dev/null; then
    echo "tmux is required but was not found." >&2
    exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -f "${REPO_ROOT}/model_weights/VGGT-1B/model.pt" ]]; then
    echo "Missing local VGGT weights: ${REPO_ROOT}/model_weights/VGGT-1B/model.pt" >&2
    exit 1
fi
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION_NAME}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/c43_sequence_${STAMP}.log"

CMD="cd '${REPO_ROOT}' && export CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com DATA_ROOT='${REPO_ROOT}/workspace/benchmark_dataset' MODEL_NAME='${REPO_ROOT}/model_weights/VGGT-1B' PYTHON_BIN='${PYTHON_BIN}'; '${RUNNER}' '${COMMAND}' 2>&1 | tee '${LOG_FILE}'"
tmux new-session -d -s "${SESSION_NAME}" "bash -o pipefail -lc $(printf '%q' "${CMD}")"

echo "Started tmux session: ${SESSION_NAME}"
echo "Log: ${LOG_FILE}"
echo "Attach: tmux attach -t ${SESSION_NAME}"
echo "Follow: tail -f ${LOG_FILE}"
