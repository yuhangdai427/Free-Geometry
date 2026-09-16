#!/bin/bash
# SelfEvo 采样 × 四数据集 × {10,20}对 全矩阵调度器（<=2 路并行，防 OOM）
set -u
cd /root/autodl-tmp/Free-Geometry
D=artifacts/diagnostics/final_protocol/da3_baseline
LOG=$D/selfevo_matrix.log
export PYTORCH_ALLOC_CONF=expandable_segments:True

run_one() {
    local ds="$1"; local np="$2"; local out="workspace/da3_se_${ds}_p${np}"
    local scenes
    case "$ds" in
        7scenes) scenes="chess fire heads office pumpkin redkitchen stairs";;
        eth3d) scenes="courtyard delivery_area electro facade kicker office pipes playground relief relief_2 terrains";;
        *) scenes=$(python3 -c "import json; print(' '.join(sorted(json.load(open('artifacts/diagnostics/final_protocol/$ds/scene_manifest.json' if '$ds'!='scannetpp' else 'artifacts/diagnostics/final_protocol/scannetpp_v3/scene_manifest.json'))['scenes'])))");;
    esac
    echo "=== SE ${ds} p${np} start $(date '+%F %T') ===" >> "$LOG"
    python3 scripts/train_da3_protocol.py --dataset "$ds" --scenes $scenes \
        --output_root "$out" --steps 100 --arm rkdc1h \
        --ratio_mix "8:4,16:4,24:8" --selfevo --n_train "$np" --early_stop >> "$LOG" 2>&1 \
        || echo "SE ${ds} p${np} FAILED" >> "$LOG"
    echo "=== SE ${ds} p${np} done $(date '+%F %T') ===" >> "$LOG"
}

for ds in 7scenes eth3d hiroom scannetpp; do
    for np in 10 20; do
        while [ $(pgrep -f "^python3 scripts/train_da3_protocol.py" | wc -l) -ge 2 ]; do sleep 30; done
        run_one "$ds" "$np" &
    done
done
wait
echo "=== selfevo matrix ALL DONE $(date '+%F %T') ===" >> "$LOG"
