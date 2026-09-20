#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DA3_PYTHON="${DA3_PYTHON:-/localhdd02/yuhang/envs/da3/bin/python}"
if [[ ! -x .venv/bin/python ]]; then
  "$DA3_PYTHON" -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip install --disable-pip-version-check 'kornia>=0.7' pytest
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
mkdir -p weights/hub/checkpoints
# Existing matcher files are loaded locally. Only missing matcher weights download.
TORCH_HOME="$PWD/weights" .venv/bin/python -c 'import self_geometry; from lightglue import SuperPoint, LightGlue; SuperPoint(max_num_keypoints=2048); LightGlue(features="superpoint")'
