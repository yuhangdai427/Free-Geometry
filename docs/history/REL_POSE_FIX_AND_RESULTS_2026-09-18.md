# Rel-Pose Loss 修正与统一 Protocol 进展（2026-09-18）

## 一、Rel-Pose 公式修正

### 问题

旧公式对 w2c 输入使用 `inv(E_i) @ E_j` 构造相对位姿——这在数学上对应 `c2w_i @ w2c_j`，
不是物理正确的相机间变换。

### 修正（三处代码同步修复）

```
文件                                        行号
src/free_geometry/losses.py                 loss_rel_pose()
src/depth_anything_3/.../protocol_v1.py     loss_pose_rel()
diagnostics/free_geometry/train_arms.py     loss_pose_rel()
```

| | 旧公式（错误） | 新公式（正确） |
|---|---|---|
| 相对旋转 | `R_i^T @ R_j` | **`R_i @ R_j^T`** |
| 相对平移 | `R_i^T @ (t_j - t_i)` | **`t_i - R_i @ R_j^T @ t_j`** |
| 物理含义 | `E_i^{-1} @ E_j` | `E_i @ E_j^{-1}` = T_{i←j} |

### Gauge 不变性验证

```
同一几何（相机 i 原点、相机 j 沿 x 1 单位），换世界坐标系（旋转 90°+平移）：
新公式 loss = 0.00000000 ✓
```

## 二、新增 Arm

### DA3: `rkdc1hr`（rkdc1h + 修正 rel）

```bash
--arm rkdc1hr --rel_weight 1.0
# loss = maskdistill + 1.5·rkd_huber + 1.0·couple + 1.0·rel_corrected
```

### VGGT: `C2M_RKDCR1H`

```bash
--arms C2M_RKDCR1H
# 同上组成
```

### 统一管线: `rkdcr_allpos`

```bash
--arm rkdcr_allpos
# 同上组成
```

## 三、修正后初步结果（实验进行中）

### scannetpp

| 模型 | 无 rel (rkdc) | 有 rel (rkdcr) | Δ |
|---|---|---|---|
| DA3 | −0.16 / +0.32 | −0.52 / +0.21 | rel 微伤 |
| VGGT | +5.50 / +2.22 | **+5.95** / +0.77 | AUC 微升, F1 降 |

### 7scenes

| 模型 | 无 rel (rkdc) | 有 rel (rkdcr) | Δ |
|---|---|---|---|
| DA3 | **+5.45 / +4.92** | −0.23 / −0.27 | **rel 有害** |

### 初步结论

修正后的 rel 在 DA3-7scenes 上**仍然有害**（与旧公式结论一致）。这说明 rel 的问题
不在公式方向，而在于 DA3 的相机 head 输出已经与 teacher 高度一致（饱和），rel 项
引入的额外约束在饱和状态下是噪声。

## 四、其他修复（本 session）

1. **LR schedule**: warmup 和 cosine 按实际步数（`n_steps`）而非 `n_train × epochs`
2. **Token mask 分离**: `corruption_mask`（遮挡用）与 `loss_mask`（监督用）独立
3. **DTU gt_depth**: addict Dict 的 `hasattr` 检查失效 → 改为值类型检查
4. **DTU a0**: `evaluate_scene` 接受 raw `DepthAnything3`（不再要求 `.da3` 包装）
5. **VGGT 分辨率**: 回退到 504（DA3 benchmark 统一口径）

## 五、实验配置汇总

| 实验 | 模型 | 臂 | 遮挡 | 协议 |
|---|---|---|---|---|
| rel-fix | DA3 | rkdc1hr | image 50% | 8:4, 全位置, 100步, lr 3e-5 |
| rel-fix | VGGT | C2M_RKDCR1H | image 50% | 8:4, 全位置, 100步, lr 3e-5 |
| DTU | DA3 | rkdc1h | image 50% | 8:4, 全位置, 100步, lr 3e-5 |
