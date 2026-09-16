#!/usr/bin/env bash
set -euo pipefail

# Leave the active full evaluation untouched.  Poll its tmux session and only
# start the 32-view evaluation after the full job has exited.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FULL_SESSION="${FULL_SESSION:-da3_dtu_full_gpu1}"
POLL_SECONDS="${POLL_SECONDS:-60}"

cd "${REPO_ROOT}"
while tmux has-session -t "${FULL_SESSION}" 2>/dev/null; do
    echo "$(date -u +%FT%TZ) waiting for ${FULL_SESSION}; no GPU work started"
    sleep "${POLL_SECONDS}"
done

echo "$(date -u +%FT%TZ) ${FULL_SESSION} finished; starting 32-view evaluation"
exec env CUDA_VISIBLE_DEVICES=1 GPU_ID=1 PYTHON_BIN=/root/miniconda3/envs/da3/bin/python \
    bash scripts/run_da3_dtu_eval32_gpu1.sh all
