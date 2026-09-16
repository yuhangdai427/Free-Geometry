#!/usr/bin/env bash
set -euo pipefail

# Launch the audited low-coverage experiment on GPU0.  This is deliberately
# separate from the legacy padded 0.1 run and writes no output below its paths.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
SESSION="${SESSION:-strict_covisibility_010_top5_gpu0}"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" \
  "cd '$REPO_ROOT' && CUDA_VISIBLE_DEVICES=0 bash scripts/run_covisibility_010_vggt.sh 2>&1 | tee '$LOG_DIR/${SESSION}.log'"
echo "Started tmux session: $SESSION"
echo "Log: $LOG_DIR/${SESSION}.log"
