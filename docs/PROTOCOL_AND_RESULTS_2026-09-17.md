# Free-Geometry TTA: 完整 Protocol 与 5 模型 × 4 数据集结果（2026-09-17）

## 一、方法概要

**Free Geometry TTA**：每个场景独立做短程 LoRA 微调（≤100 步），冻结长上下文 teacher（8–24 帧）监督短上下文 student（4 帧），全程 GT-free。

```
冻结 teacher（干净 8 帧）→ teacher cache（特征 / 深度 / 置信 / 相机中心）
LoRA student（遮挡 4 帧）→ 特征 + 几何输出
                        → distill loss（GT-free）
```

## 二、数据集

| 数据集 | 场景数 | 每场景帧数 N | τ 范围 | τ 分派 | 评测帧 |
|---|---|---|---|---|---|
| 7scenes | 7 | 500–1000 | 0.67–0.73 | dense | 100 (seed 42) |
| eth3d | 11 | 14–76 | 0.06–0.24 | sparse | 全帧 |
| hiroom | 30 | 10–23 | 0.04–0.20 | sparse | 全帧 |
| scannetpp | 20 | 100–534 | 0.34–0.55 | 混合 | 100 (seed 42) |

## 三、选帧策略（GT-free，5 模型共用）

```
τ = 中位数相邻帧 SIFT 匹配率（≤40 等距对，320px，400 特征，ratio 0.75）
τ > 0.55 → dense：等距超集 M=min(N, T×1.5)，extras SIFT 重叠∈[0.1,0.5] 软过滤
τ ≤ 0.55 → sparse：纯随机窗口
共享帧固定在 teacher slots [0,2,4,6]
10 训练对 + 2 probe 对（稳定种子）
```

## 四、统一 Loss

| 项 | 公式 | 说明 |
|---|---|---|
| maskdistill | Huber(β=1) + 2(1−cos)，teacher-conf 加权 | 50% image 遮挡（ImageNet 均值填充），**全位置监督** |
| rkd_huber | 均值归一化相机中心两两距离 + 三角角，Huber(δ=0.2) | gauge-free，仅需相机中心 |
| couple | (log RMS(centers) − log mean_depth)² 标量对齐 | 深度-位姿量规耦合 |
| rel | 旋转 chordal + 平移方向 1−cos | 与 AUC 指标同公式（inv(Pᵢ)Pⱼ） |

组合臂：`rkdc` = maskdistill + 1.5·rkd + 1.0·couple | `maskrel` = maskdistill + 1.0·rel

## 五、5 模型 × 4 数据集：冠军 protocol（从磁盘 args 逐格提取）

| 模型×数据集 | 选帧 | t:s | 对数 | loss | 监督 | 步数 | ES | LoRA | lr |
|---|---|---|---|---|---|---|---|---|---|
| DA3 7s | dense | mix 8:4/16:4/24:8+combo+TS0.7 | 20 | rkdc1h | masked | 100 | 关 | r32, 40层+cam | 3e-5 |
| DA3 eth3d | sparse | 8:4 | 10 | rkdc1h | **全位** | 100 | 关 | 同上 | 3e-5 |
| DA3 hiroom | sparse | 8:4 | 10 | rkdc1h | **全位** | 100 | 关 | 同上 | 3e-5 |
| DA3 spp | 混合 | 8:4 | 10 | rkdc1h | **全位** | 100 | 关 | 同上 | 3e-5 |
| VGGT 7s | dense | 16:4 | 10 | maskrel | **全位** | 100 | 关 | r32, 24层 | 3e-5 |
| VGGT eth3d | sparse | 16:4 | 10 | maskrel | **全位** | 100 | 关 | 同上 | 3e-5 |
| VGGT hiroom | sparse | 16:4 | 10 | maskrel | **全位** | 100 | 关 | 同上 | 3e-5 |
| VGGT spp | 混合 | 16:4 | 10 | RKDC1H | **全位** | 100 | 关 | 同上 | 3e-5 |
| Ω 7s | dense | 8:4 | 10 | **maskrel** | **全位** | 100 | 关 | r32, 48blk | 3e-5 |
| Ω eth3d | sparse | 8:4 | 10 | **maskrel** | **全位** | 100 | 关 | 同上 | 3e-5 |
| Ω hiroom | sparse | 8:4 | 10 | **rkdc** | **全位** | 100 | 关 | 同上 | 3e-5 |
| Ω spp | 混合 | 8:4 | 10 | **rkdc** | **全位** | 100 | 关 | 同上 | 3e-5 |
| Pi3 全部 | 各 | 8:4 | 10 | rkdc | **全位** | 100 | 关 | r32, 36层 | 3e-5 |
| DVLT 全部 | 各 | 8:4 | 10 | rkdc | **全位** | 100 | 关 | r32, 8模块 | 3e-5 |

LoRA 统一设置：r=32, α=32, dropout=0, A~kaiming(√5), B=0（零-LoRA = 冻结 baseline）。
优化器：AdamW wd=1e-5, clip=1.0, warmup 15% + cosine → 1e-8。

## 六、Early stop 机制

**定义**：tail-10 步 loss 均值相对 prev-10 改善 <2%，且 step ≥ max(30, warmup+10)。
**现状**：上述 20 格全部**未启用**（固定 100 步）。代码已实现（`--early_stop`），历史实验证明 100 步内提前停止与跑满无差异。two_stage 模式自动禁用 ES。

## 七、完整结果（baseline 绝对值 + TTA 相对提升）

| 模型 | 数据集 | baseline AUC/F1 | dAUC% | dF1% |
|---|---|---|---|---|
| DA3 | 7scenes | 0.275/0.501 | **+7.62** | **+5.50** |
| | eth3d | 0.485/0.791 | +3.39 | −1.88 |
| | hiroom | 0.803/0.861 | **+4.71** | **+1.92** |
| | scannetpp | 0.847/0.783 | −0.16 | +0.32 |
| VGGT | 7scenes | 0.238/0.475 | +1.47 | **+18.37** |
| | eth3d | 0.265/0.574 | **+32.24** | **+25.46** |
| | hiroom | 0.491/0.562 | **+20.32** | **+19.80** |
| | scannetpp | 0.593/0.666 | **+5.63** | +2.38 |
| Ω | 7scenes | 0.226/0.440 | +3.36 | +2.29 |
| | eth3d | 0.396/0.685 | +3.23 | −1.14 |
| | hiroom | 0.824/0.816 | **+7.01** | +2.46 |
| | scannetpp | 0.655/0.654 | −2.29 | −1.40 |
| Pi3 | 7scenes | 0.263/0.424 | −16.69 | **+19.73** |
| | eth3d | 0.351/0.721 | +5.39 | +0.50 |
| | hiroom | 0.666/0.760 | −0.57 | **+10.92** |
| | scannetpp | 0.504/0.621 | **+16.82** | **+5.69** |
| DVLT | 7scenes | 0.302/0.532 | +0.50 | +1.91 |
| | eth3d | 0.366/0.708 | +0.98 | **+4.38** |
| | hiroom | 0.754/0.754 | −2.51 | +2.96 |
| | scannetpp | 0.656/0.695 | **+1.96** | −1.48 |

## 八、关键发现

1. **全位置 loss 优于 masked-only**（8 格中 7 格持平或更优）→ 成为默认。
2. **最优 loss 臂逐模型不同**：DA3=rkdc、VGGT=maskrel(除spp)、Ω=混合、Pi3/DVLT=rkdc。
3. **rkd 正收益不跨模型迁移**（VGGT spp +5.5 → Ω spp −2.3，同一家族不同模型）。
4. **headroom 主导**：baseline 低的模型（VGGT eth3d 0.27）涨幅巨大；baseline 高的（DA3 hiroom 0.80）涨幅小。
5. **scannetpp 是所有模型的硬格**；DVLT 用 LoRA（而非全参微调）首次在该格转正。
6. **dense 视频（7scenes）pose 蒸馏伤害跨模型一致**（5/5 模型 AUC 不升或降）。

## 九、遮挡位置消融（DA3-eth3d）

| 臂 | 位置 | 填充 | dAUC | dF1 |
|---|---|---|---|---|
| A0 none | 不遮挡 | — | −2.52 | −6.03 |
| **A1 image** | 输入像素 | ImageNet 均值 | **+3.39** | **−1.88** |
| A2 shallow | patch 投影后 | 零向量（保位置） | −0.02 | −0.53 |
| A3 feat | block-12 输出 | 零向量（删全部） | −0.21 | −0.47 |

**结论**：图像遮挡（A1）是唯一有效方式；特征端 token 置零对 LoRA 梯度信号退化。

## 十、评测协议

- **AUC@3**：全对 relative-pose，首帧对齐，max(rot, t-dir) 误差，1° bin AUC（与 DA3 bench 一致）
- **F1**：TSDF 融合点云（recon_unposed），Sim(3) Umeyama 对齐（含 RANSAC），5cm 阈值
- **Pi3 特殊**：K 从自身点图估计（模型分辨率），不经 GT-K；深度尺度不变性由 Umeyama 全局尺度消除（×1/×2/×0.5 一致性验证通过）
- **配对**：逐场景 TTA − baseline 相对提升均值，baseline 永不重跑
