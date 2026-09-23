#!/bin/bash
# Final tuning lane: VGGT dtu64 orig u50, then VGGT dtu orig u25 (sequential).
set -u
cd /localhdd02/yuhang/code/free_geometry_latest
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash scripts/run_vggt_orig_variant.sh dtu64 50 orig_u50
bash scripts/run_vggt_orig_variant.sh dtu 25 orig_u25
echo "==== FINAL LANE ALL DONE ===="
