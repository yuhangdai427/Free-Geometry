#!/bin/bash
# Watchdog for the fgov overnight queue: every 15 min check the tmux session,
# disk, FAILED markers; restart the queue if its session died prematurely.
PATLOG=logs/overnight_patrol.log
cd /root/autodl-tmp/Free-Geometry
while true; do
  {
    echo "=== patrol $(date '+%F %T') ==="
    if grep -q "==== ALL DONE ====" workspace/overnight/SUMMARY.log 2>/dev/null; then
      echo "queue COMPLETE — patrol exiting"
      break
    fi
    if ! tmux has-session -t fgov 2>/dev/null; then
      echo "fgov session DEAD — relaunching"
      tmux new-session -d -s fgov \
        'bash /root/autodl-tmp/Free-Geometry/scripts/overnight_rawrel.sh 2>&1 | tee -a logs/queue_fgov.log'
    else
      echo "fgov alive"
    fi
    FREE=$(df --output=avail -B1G /root/autodl-tmp | tail -1 | tr -d ' ')
    echo "disk free: ${FREE}G"
    if [ "${FREE:-99}" -lt 5 ]; then
      echo "LOW DISK — purging overnight ckpts/eval32 (metrics retained)"
      rm -rf workspace/overnight/*/ckpts workspace/overnight/*/eval32 2>/dev/null
    fi
    NF=$(grep -c "FAILED" workspace/overnight/SUMMARY.log 2>/dev/null || echo 0)
    echo "FAILED cells so far: $NF"
    tail -3 workspace/overnight/SUMMARY.log 2>/dev/null
  } >> $PATLOG 2>&1
  sleep 900
done
