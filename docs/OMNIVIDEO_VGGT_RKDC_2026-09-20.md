# VGGT + rkdc TTA 在 OmniVideo benchmark（SelfEvo eval 协议）— 2026-09-20

与 OmniGeo 实验完全同协议（`scripts/vggt_omnigeo_tta.py --benchmark omnivideo`）：
187 序列 × 81 帧，sparse 1/10 = 每序列 8 帧评估，518 crop、[0,1] 输入、
全对相对位姿角度误差 AUC@{5,15,30} + 深度（整序列 scale&shift 对齐），一次前向双指标。
TTA = rkdc1h 100 步（LoRA r32×24 层 aggregator、8:4、50% 遮挡、全位置 loss）。
baseline 为冻结 VGGT-1B 同协议评估。运行：3 shard，00:45–04:40（约 4 小时），
期间 tmux 会话第 5 次被外部杀掉（收尾阶段），断点续跑 + 逐序列落盘保证 187/187 完整。

## 结果（187 序列均值）

| 指标 | baseline | TTA(rkdc 100步) | Δ |
|---|---|---|---|
| 位姿 AUC@5° | 68.05 | **70.12** | **+2.07** |
| 位姿 AUC@15° | 83.60 | 84.40 | +0.80 |
| 位姿 AUC@30° | 89.78 | 90.15 | +0.37 |
| Racc@30° | 100.00 | 100.00 | 0（旋转已解满） |
| 深度 Abs Rel | 0.1475 | **0.1431** | **−0.0044（−3.0% 相对）** |
| 深度 δ<1.25 | 0.823 | **0.831** | +0.8pp |
| 深度 Log RMSE | 0.330 | 0.326 | −0.004 |

逐序列 AUC15：升>1pp 61 / 降>1pp 47 / 平 79；单序列波动大（±14~24pp 双向）。

## 解读

1. **OmniVideo 上 TTA 仍为正收益**，但量级小于 OmniGeo（AUC@5 +2.1 vs +2.8）：
   OmniVideo 的冻结基线强得多（AUC@5 68 vs 45；旋转 Racc@30 直接 100%），
   headroom 小，符合"基线越差收益越大"的跨数据集规律。
2. 收益集中在**严阈值**（AUC@5 +2.1 > AUC@15 +0.8 > AUC@30 +0.4）——TTA 主要
   精修小误差对（平移方向精度），不修复大失败。
3. 深度小幅同向改善（AbsRel −3% 相对、δ1.25 +0.8pp），无冲突迹象。
4. 两个 benchmark 合并结论：rkdc TTA 在 OmniWorld 两个子集上位姿+深度双正收益，
   且协议代码同一套、零改动跨数据集。

## 数据

- 逐序列全精度：`workspace/omnivideo_vggt_rkdc/results_shard{0,1,2}_3.json`（187 序列
  × base/tta/d 三列齐全）+ `overall.json`
- swanlab：项目 `free-geometry-omni-video`（实验 vggt_rkdc_s100_shard0/2 等）
