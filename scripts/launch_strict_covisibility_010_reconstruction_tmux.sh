#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
SESSION="${SESSION:-strict_covisibility_010_reconstruction_gpu0}"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  exit 0
fi
tmux new-session -d -s "$SESSION" \
  "cd '$REPO_ROOT' && CUDA_VISIBLE_DEVICES=0 bash scripts/run_strict_covisibility_010_reconstruction.sh 2>&1 | tee '$LOG_DIR/${SESSION}.log'"
echo "Started tmux session: $SESSION"
echo "Log: $LOG_DIR/${SESSION}.log"
