#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Local CUDA 12.8 matches torch; caller may override on a different machine.
export CUDA_HOME="${CUDA_HOME:-/localhdd02/yuhang/envs/lbw}"
export CPATH="${CPATH:+$CPATH:}$CUDA_HOME/targets/x86_64-linux/include"
export LIBRARY_PATH="${LIBRARY_PATH:+$LIBRARY_PATH:}$CUDA_HOME/targets/x86_64-linux/lib"
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-4}"
exec .venv/bin/python -u scripts/run_comparison.py "$@"
