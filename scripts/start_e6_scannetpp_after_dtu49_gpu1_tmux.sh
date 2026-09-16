#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${SESSION_NAME:-e6_scannetpp_nested_views_gpu1}"
RUNNER="${REPO_ROOT}/scripts/launch_e6_scannetpp_after_dtu49_gpu1.sh"

tmux has-session -t "${SESSION_NAME}" 2>/dev/null && {
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  exit 1
}
tmux new-session -d -s "${SESSION_NAME}" "exec ${RUNNER}"
echo "Started waiting session: ${SESSION_NAME}"
echo "It will start ScanNet++ E6 only after da3_dtu49_per_scene_lora_gpu1 exits."
