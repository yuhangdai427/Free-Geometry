#!/bin/bash
# Wait for the heavy dtu lane (and any u50 work) to finish, then run the dtu64
# gated campaign alone with allocator fragmentation mitigation.
set -u
while systemctl --user is-active --quiet fgdt-vggtg-dtu.service \
   || systemctl --user is-active --quiet fgdt-da3u50.service; do
  sleep 60
done
sleep 30
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec bash /localhdd02/yuhang/code/free_geometry_latest/scripts/run_vggt_gated_campaign.sh dtu64
