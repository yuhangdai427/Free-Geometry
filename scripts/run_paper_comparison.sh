#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Runtime optimizations preserve frame counts, method schedules and evaluation.
exec bash scripts/run_comparison.sh --config configs/comparison.yaml \
  --root artifacts/paper_comparison_50updates --set checkpointing=false "$@" --protocol paper
