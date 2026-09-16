#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$REPO_ROOT/logs/hard_view_gpu0_memory.csv"
mkdir -p "$(dirname "$LOG")"
if [[ ! -s "$LOG" ]]; then
  echo "timestamp_utc,memory_used_mib,memory_total_mib,gpu_util_percent" > "$LOG"
fi
while tmux has-session -t hard_view_experiments_gpu0 2>/dev/null; do
  timestamp="$(date -u +%FT%TZ)"
  nvidia-smi --id=0 --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', ' -v timestamp="$timestamp" '{print timestamp "," $1 "," $2 "," $3}' >> "$LOG"
  sleep 30
done
