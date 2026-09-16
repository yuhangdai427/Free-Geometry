#!/usr/bin/env bash
set -euo pipefail

# Reclaim each completed benchmark's exports as soon as its metric JSON is
# complete and finite.  Never remove the active experiment's model_results.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${REPO_ROOT}/workspace/da3_dtu_full_gpu1}"
FULL_SESSION="${FULL_SESSION:-da3_dtu_full_gpu1}"
POLL_SECONDS="${POLL_SECONDS:-30}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/da3/bin/python}"

cd "${REPO_ROOT}"

cleanup_if_verified() {
    local root="$1" metric_file="$2" expected_scenes="$3" metric_keys="$4"
    local raw_dir="${root}/model_results"
    [[ -d "${raw_dir}" ]] || return 0
    if ! "${PYTHON_BIN}" - "${root}" "${metric_file}" "${expected_scenes}" "${metric_keys}" <<'PY'
import json
import math
import sys
from pathlib import Path

root, metric_file, expected_scenes, metric_keys = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4].split(',')
path = root / 'metric_results' / metric_file
if not path.is_file():
    raise SystemExit(1)
data = json.loads(path.read_text())
scenes = {key: value for key, value in data.items() if key != 'mean'}
if len(scenes) != expected_scenes:
    raise SystemExit(1)
for values in scenes.values():
    for key in metric_keys:
        value = values.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise SystemExit(1)
PY
    then
        # Metrics are not ready yet; leave the active export untouched.
        return 0
    fi
    local size
    size="$(du -sb "${raw_dir}" | awk '{print $1}')"
    find "${raw_dir}" -depth -delete
    echo "$(date -u +%FT%TZ) removed verified raw results: ${raw_dir} (${size} bytes)"
}

while true; do
    cleanup_if_verified "${WORKSPACE_ROOT}/dtu64/baseline_seed43" "dtu64_pose.json" 13 "auc03,auc30"
    cleanup_if_verified "${WORKSPACE_ROOT}/dtu64/free_geometry_epoch2_seed43" "dtu64_pose.json" 13 "auc03,auc30"
    cleanup_if_verified "${WORKSPACE_ROOT}/dtu/baseline_seed43" "dtu_recon_unposed.json" 22 "acc,comp,overall"
    cleanup_if_verified "${WORKSPACE_ROOT}/dtu/free_geometry_epoch2_seed43" "dtu_recon_unposed.json" 22 "acc,comp,overall"
    if ! tmux has-session -t "${FULL_SESSION}" 2>/dev/null; then
        echo "$(date -u +%FT%TZ) ${FULL_SESSION} finished; cleanup monitor exiting"
        exit 0
    fi
    sleep "${POLL_SECONDS}"
done
