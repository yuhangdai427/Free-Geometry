# Loss 分量趋势与占比分析（2026-09-18）

## 分析方法

从 training_trace.csv 提取各 loss 分量的 first-10 / last-10 步均值，
计算降幅和加权后占总 loss 的比例。VGGT 的 maskdistill 分量未单独记录
（extra dict 只含 `mask_ratio`），通过 `total − Σ(分量×权重)` 推断。

## 一、DA3 rkdc1hr（rkdc + 修正 rel）— rel 有害

### scannetpp（天花板格）

| 分量 | first 10 | last 10 | 降幅 | 加权后 | 占比 |
|---|---|---|---|---|---|
| **rel_rot** | **0.535** | **0.444** | **17%** | **0.444** | **54.6%** |
| maskdistill | 0.217 | 0.204 | 6% | 0.204 | 25.1% |
| rel_tdir | 0.114 | 0.086 | 24% | 0.086 | 10.6% |
| couple | 0.037 | 0.034 | 8% | 0.034 | 4.2% |
| rkd_sh_a ×1.5 | 0.024 | 0.018 | 23% | 0.028 | 3.4% |
| rkd_sh_d ×1.5 | 0.015 | 0.012 | 21% | 0.018 | 2.2% |

**问题**：rel_rot 是其他分量的 10-50 倍，占总 loss 55%，且降幅仅 17%。
梯度被 rel_rot 的噪声主导。

### 7scenes（rel 发散）

| 分量 | first 10 | last 10 | 降幅 | 占比 |
|---|---|---|---|---|
| maskdistill | 0.177 | 0.144 | 19% | 66.6% |
| **rel_rot** | **0.019** | **0.056** | **−189%** | **25.9%** |
| rel_tdir | 0.018 | 0.009 | 52% | 3.9% |
| rkd/couple | ~0.005 | ~0.003 | 40-46% | 3.6% |

**问题**：rel_rot 从 0.019 涨到 0.056（×2.9），**发散**。
结果：−0.23/−0.27（从 +5.45/+4.92 翻负）。

## 二、DA3 rkdc1h（无 rel）— 健康

| 数据集 | maskdistill 占比 | rkd+couples 占比 | 各分量降幅 |
|---|---|---|---|
| 7scenes | **95.0%** | 5.0% | 20-60% |
| eth3d | 50.9% | 49.1% | 10-67%（couple 降最多） |
| hiroom | **95.2%** | 4.8% | 17-79% |
| scannetpp | **72.5%** | 27.5% | 7-24% |

**结论**：无 rel 时所有分量健康收敛，maskdistill 主导。

## 三、VGGT maskrel（冠军臂）— rel 完美收敛

| 数据集 | maskdistill 占比 | rel_rot 占比 | rel_rot 降幅 | 结果 |
|---|---|---|---|---|
| 7scenes | **98.7%** | 0.4% | **96.5%**（0.033→0.001） | +1.47/+18.37 ✓ |
| eth3d | 62.1% | 18.5% | **91.4%**（1.28→0.11） | +32.24/+25.46 ✓ |
| hiroom | 83.3% | 10.1% | **89.9%**（0.31→0.03） | +20.32/+19.80 ✓ |

**关键发现**：VGGT 相机 head **有能力学好相对位姿**——rel_rot 从 0.03-1.28 降到
0.001-0.11（90%+ 降幅），同时 maskdistill 始终主导。

## 四、VGGT RKDCR1H（rkdc + rel 混合）— rel 与 rkdc 竞争

| 数据集 | maskdistill 占比 | rel_rot 占比 | rel_rot 降幅 | 问题 |
|---|---|---|---|---|
| scannetpp | 48.6% | **33.3%** | 33.7% | 不充分收敛 |
| 7scenes | **90.8%** | 0.4% | 90.9% | 与 maskrel 一致 |
| **eth3d** | 24.4% | **48.1%** | **仅 21.3%** | **rel 主导但不收敛** |

**结论**：rkdc 与 rel 混合时，eth3d/scannetpp 上 rel_rot 降幅从 maskrel 臂的
91%/90% 退化到 21%/34%——中心约束与旋转约束在梯度上互相竞争。

## 五、核心诊断：为什么 VGGT 和 DA3 不能统一

```
VGGT maskrel:  rel_rot 初始大 → 90%+ 降幅 → 有效信号 → 涨点
DA3 rkdc+rel:  rel_rot 初始大 → 17% 或发散 → 噪声/冲突 → 降点

根本原因：DA3 相机 head 已饱和（teacher≈student，step-0 rel 残差≈0.008），
         rel 的梯度信号在饱和条件下是纯噪声；且 rel 量级是 rkd 的 10-50 倍。
```

### 统一规则候选

用 **step-0 rel_rot 值** 做臂选择：
- step-0 rel_rot > 0.3 → VGGT 型（相机 head 有改进空间）→ 用 maskrel
- step-0 rel_rot < 0.05 → DA3 型（相机 head 饱和）→ 用 rkdc

## 六、已修复的 Bug

1. **VGGT 形状广播**（2026-09-18）：`t[:,i]` 是 `[B,3]`，与 `[B,3,1]` 相减
   广播成 `[B,3,3]` 矩阵 → normalize 对矩阵每行而非向量。
   修复：`t[:,i].unsqueeze(-1) - Rr @ t[:,j].unsqueeze(-1)`，加 assert。
   影响：仅 train_arms.py 的 loss_pose_rel（DA3/统一管线无此问题）。
2. **VGGT rel 公式**（2026-09-18）：从 `inv(E_i)@E_j` 改为 `E_i@inv(E_j)`。
3. **DA3 LR schedule**：warmup/cosine 按实际步数而非 n_train×epochs。
4. **Token mask 分离**：corruption_mask 与 loss_mask 独立变量。

## 数据文件位置

| 实验 | trace 路径 |
|---|---|
| DA3 scannetpp rkdc1hr | `workspace/da3_scannetpp_rkdcr/training_trace.csv` |
| DA3 7scenes rkdc1hr | `workspace/da3_7scenes_rkdcr/training_trace.csv` |
| DA3 各数据集 rkdc1h | `workspace/da3_protocol_*_t8s4_lossall/training_trace.csv` |
| VGGT relfix (RKDCR1H) | `artifacts/diagnostics/final_protocol_relfix/*/training_trace.csv` |
| VGGT lossall (maskrel) | `artifacts/diagnostics/final_protocol_lossall/*/training_trace.csv` |
