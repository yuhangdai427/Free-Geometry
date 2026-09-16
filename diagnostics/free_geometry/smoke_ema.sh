#!/bin/bash
# Smoke: E2 arms (plain) + C2_b5_rel with --ema_teacher 1. OOM -> wait 60s, retry (max 3).
set -uo pipefail
cd /root/autodl-tmp/Free-Geometry
D=diagnostics/free_geometry
RR=artifacts/diagnostics/smoke_ema

run_with_retry() {
  local log="$1"; shift
  for i in 1 2 3; do
    "$@" > "$log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then echo "OK $log"; return 0; fi
    if grep -qi "out of memory" "$log"; then
      echo "OOM in $log (attempt $i), sleeping 60s"; sleep 60
    else
      echo "FAILED $log rc=$rc (non-OOM)"; return $rc
    fi
  done
  echo "FAILED $log after retries"; return 1
}

run_with_retry "$RR/smoke_e2_arms.log" \
  python3 $D/train_arms.py --run_root $RR --scenes 1ada7a0617 \
    --arms E2_out C2_E2 --max_steps 3 --no_ckpt

run_with_retry "$RR/smoke_ema_c2.log" \
  python3 $D/train_arms.py --run_root $RR --scenes 1ada7a0617 \
    --arms C2_b5_rel --ema_teacher 1 --max_steps 3 --no_ckpt

echo "SMOKE_ALL_DONE"
