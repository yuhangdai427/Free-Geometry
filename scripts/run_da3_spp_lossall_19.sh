#!/bin/bash
# Orchestrator #3: after the 8-cell chain, rerun DA3 scannetpp loss_all_pos on
# all 20 scenes (the earlier failure was a typo'd scene id "bcd246daf"; the real
# scene is bcd2436daf and its data is intact).
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python

echo "[orch3] waiting for 8-cell chain $(date '+%F %T')"
while pgrep -f "run_lossall_8cells.sh" > /dev/null; do sleep 60; done

export PYTORCH_ALLOC_CONF=expandable_segments:True
echo "[orch3] DA3 scannetpp (20 scenes) start $(date '+%F %T')"
"$PY" scripts/train_da3_protocol.py --dataset scannetpp \
  --scenes 09c1414f1b 1ada7a0617 21d970d8de 286b55a2bf 38d58a7a31 3e8bba0176 40aec5fffa 578511c8a9 5f99900f09 7831862f02 7bc286c1b6 9071e139d9 acd95847c5 bcd2436daf bde1e479ad c4c04e6d6c c5439f4607 cc5237fd77 f3d64c30f8 fb5a96b1a2 \
  --output_root workspace/da3_protocol_scannetpp_t8s4_lossall \
  --arm rkdc1h --teacher_N 8 --steps 100 \
  --mask_ratio 0.5 --loss_all_pos > logs/da3_scannetpp_lossall.log 2>&1
echo "[orch3] exit=$? $(date '+%F %T')"
"$PY" - <<'EOF'
import json
import numpy as np
s = json.load(open('workspace/da3_protocol_scannetpp_t8s4_lossall/smoke_summary.json'))['scenes']
ba = json.load(open('artifacts/diagnostics/final_protocol/da3_baseline/scannetpp_baseline.json'))['scenes']
br = json.load(open('artifacts/diagnostics/final_protocol/da3_baseline/scannetpp_recon_baseline.json'))['scenes']
da = [(r['eval']['auc03']-ba[sc]['auc03'])/ba[sc]['auc03']*100 for sc, r in s.items() if ba[sc]['auc03'] > 0]
df = [(r['eval']['recon_fscore']-br[sc]['fscore'])/br[sc]['fscore']*100 for sc, r in s.items() if br[sc]['fscore'] > 0]
open('workspace/da3_protocol_scannetpp_t8s4_lossall/result_summary.md', 'w').write(
    "# DA3 scannetpp 8:4 loss_all_pos (20 scenes)\n\n"
    f"Mean dAUC@3 = **{np.mean(da):+.2f}%**  dF1 = **{np.mean(df):+.2f}%** (n={len(da)})\n")
print(f"[orch3] SUMMARY dAUC={np.mean(da):+.2f}% dF1={np.mean(df):+.2f}% (n={len(da)})")
EOF
echo "[orch3] ALL DONE $(date '+%F %T')"
