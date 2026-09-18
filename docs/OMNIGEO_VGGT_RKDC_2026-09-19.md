# VGGT + rkdc TTA 在 OmniGeo benchmark（SelfEvo eval 协议）— 2026-09-19

## 实验

模型 VGGT-1B；数据 OmniGeo（OmniWorld `benchmark/geometric_prediction`，49 序列 ×384 帧，
C2W GT + 深度 GT）。评测**逐行遵循 Self-Evolve eval 分支协议**（`relpose/eval_angle.py` +
`videodepth/` gamegeo 条目）：

- sparse 1/10 帧采样（含端点，~38 帧/序列），crop 模式宽 518（14 倍数，中心裁剪）
- 位姿：全对相对位姿误差（旋转四元数角 + 平移方向角，180° 歧义处理），
  Racc/Tacc/AUC@{5,15,30}°
- 深度：预测 CUBIC resize 到 GT 尺寸，整序列 scale&shift 对齐（median 初始化 1000 步），
  gt>0 掩码，Abs Rel / Sq Rel / RMSE / Log RMSE / δ1.25^{1,2,3}
- **一次前向同时产出两套指标**（baseline 与 TTA 各一次前向，无重复推理）

TTA（我们的统一 protocol，适配 518）：teacher 8 帧 / student 4 帧（slots 0,2,4,6），
50% 14×14 图像块遮挡，loss = maskdistill(全位置, teacher-conf 加权) + 1.5×RKD-Huber(δ=0.2)
+ 1.0×couple；LoRA r32 仅 24 层 aggregator（multi-view 部分）；100 步 AdamW lr3e-5
wd1e-5 clip1.0 warmup15%+cosine；全部 seed 化。

## 结果（49 序列均值；baseline = 冻结 VGGT，TTA = rkdc 训 100 步）

| 指标 | baseline | TTA | Δ |
|---|---|---|---|
| 位姿 AUC@5° | 45.24 | **48.08** | **+2.84** |
| 位姿 AUC@15° | 64.74 | **67.04** | **+2.30** |
| 位姿 AUC@30° | 72.70 | 74.69 | +1.99 |
| Racc@30 / Tacc@30 | 92.49 / 84.97 | 93.68 / 86.11 | +1.18 / +1.14 |
| 深度 Abs Rel | 0.1803 | **0.1611** | **−0.019（−10.6% 相对）** |
| 深度 δ<1.25 | 0.760 | **0.795** | **+3.5pp** |
| 深度 Log RMSE | 0.460 | 0.449 | −0.010 |

逐序列：AUC15 提升>1pp 的 **29/49**，下降>1pp 仅 4/49；深度 AbsRel 改善 **38/49**。
最大提升 +14.2pp（0598713c0db0）；最大回退 −10.3pp（2421ec63b890，1 例）。

## 结论

1. **rkdc TTA 在 OmniGeo 上位姿与深度同时正收益**，且是该 benchmark 协议下的首次
   TTA 报告；新数据集（合成游戏场景，明显 OOD）上无需改动 loss 即有效。
2. 收益分布不均：基线越差的序列收益越大（17.9→31.2、65.5→79.7），基线强的序列持平
   ±1pp —— 与"headroom 主导"的历史规律一致。
3. 例外序列 2421ec63b890（−10pp）值得单查（可能与视角组合歧义同类）。

## 协议备注（重要）

- SelfEvo eval 分支的预处理输出 **[0,1] 无 ImageNet 归一化**——对 vanilla VGGT 验证过：
  归一化反而使平移方向崩（AUC≈0），[0,1] 是正确口径；与论文协议一致。
- 数据与代码：`scripts/vggt_omnigeo_tta.py`；结果（全精度逐序列+两套指标）
  `workspace/omnigeo_vggt_rkdc/results_shard{0_2,1_2}.json/.csv` + `_summary.json`；
  swanlab 项目 `free-geometry-omnigeo`（每序列 base/tta AUC15 与差值）。
- 49 序列 ×（baseline 前向+TTA 100 步+TTA 前向，双指标同前向）双 shard 共 80 分钟。
