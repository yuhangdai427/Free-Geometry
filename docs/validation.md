# 验证记录（2026-09-20）

代码覆盖 **DA3-Giant + VGGT × 5 数据集 × 3 seed**。540 次全量适配尚未启动；以下为验证运行，不称为论文完整数值复现。可机读记录见 [validation_results.json](validation_results.json)。

## 已完成的检查

- 数据预检：90/90 场景、4,555 张官方选择的图像；包含 DTU 的 RGB、相机、mask、ObsMask、Plane、GT 点云，无缺失。
- 全量 dry-run：540 个唯一 `(model, dataset, scene, seed)` 任务，合并成 180 个共享冻结模型的场景 worker。
- CPU/逻辑测试：24 项通过，覆盖几何梯度、LoRA、FAN/GD/DWA、任务枚举、失败继续、seed 重置、配置科学计数法、缓存隔离、DTU 对齐/分阶段评测和完整覆盖统计。
- 双模型全数据集短适配：每个数据集取官方首场景，抽 6 帧、2 步，两个模型各跑 seed 0/1/2，**30/30 次成功**。
- 速度配置短适配：DTU scan1，同样 6 帧、2 步，两个模型各三个 seed，**6/6 次成功**。
- 模型加载验证：跳过临时随机初始化后，两模型的深度、置信度、相机内外参与原加载路径的真实 baseline 逐值相同；零 LoRA 输出亦逐值相同。真实反向传播的梯度有限且非零。DA3 为 40 个 QKV、15,728,640 参数；VGGT 为 72 个 QKV、18,874,368 参数。
- DTU 实际评测：两个模型均完成 baseline/适配的 posed + unposed 评测；DA3 将 GPU 融合与 CPU 距离评分拆分后，全部指标与原先合并评测 **完全一致**。
- DA3 正式长度单场景：ETH3D courtyard 全部 38 帧、50 次优化，速度配置适配阶段 117.97 秒、峰值 47.23 GiB，选中第 46 次更新后的权重。该时长包括适配阶段匹配/初始化/写盘，不包括 worker 的模型加载、baseline 推理和独立评测。

完整 38 帧 / 50 步 DA3 courtyard 的官方评测也已完成（单场景验证，非数据集均值）：

| 指标 | Baseline | Self-Geometry |
|---|---:|---:|
| AUC@3 | 0.5225 | 0.5282 |
| F1 unposed | 0.7473 | 0.7630 |
| F1 posed | 0.8380 | 0.8587 |

## 速度与显存

RTX PRO 6000 96 GB，13 帧 × 378 × 504，真实模型、三个反向梯度计算及 AdamW 更新。每配置预热 1 步、计时 3 步；使用合成标量损失覆盖深度/相机路径，**这是训练算子微基准，不是完整场景耗时**，不含 LightGlue、评测或 checkpoint 写盘。

| 模型 | 激活 checkpoint 开 + 普通 AdamW | checkpoint 关 + fused AdamW | 微基准加速 | 参考 / 速度配置峰值显存 |
|---|---:|---:|---:|---:|
| DA3-Giant | 2.784 s/步 | 2.020 s/步 | 1.38× | 21.17 / 52.17 GiB |
| VGGT | 2.523 s/步 | 1.748 s/步 | 1.44× | 16.08 / 45.81 GiB |

13 帧对应默认 FAN 的最大训练批次（目标 + 12 个角度桶）。更接近正方形的输入可能占用更多显存。速度配置同时将恢复 checkpoint 写盘从每 5 步改为每 25 步，并复用三个 seed 的模型加载、图像和匹配；这些节省没有混入上述微基准比例。

## 恢复与随机状态

`isolated_preparation_v2` 保证构建模型或匹配缓存不会推进训练 RNG。每个 seed 重新生成 LoRA；共享资源不会继承其他 seed 的适配参数或优化器。

`check_resume.py --model da3|vggt` 从同一个第 5 步 checkpoint 分叉，对比连续执行与新进程恢复后的第 6 步，检查 loss 历史、历史最佳参数和完整 LoRA 权重。当前 fused AdamW 的两模型恢复检查均通过：DA3 最大参数差 5.44e-9，VGGT 3.73e-9，容差均为 1e-7；loss 历史完全相同。原始记录在 validation_results.json 的 resume_checks 中。CUDA backward 不保证独立运行逐位相同；完整 benchmark 通过三个 seed 报告实际波动。

## 保留的先前 VGGT 诊断

下列为旧 RNG 协议、单场景 50 步结果，保留在 artifacts/main，不混入新全量统计：

| 数据集 / 场景 | AUC@3 baseline → adapted | F1 unposed baseline → adapted |
|---|---:|---:|
| ETH3D / courtyard | 0.2361 → 0.2451 | 0.3216 → 0.3801 |
| 7Scenes / chess | 0.1882 → 0.2025 | 0.3953 → 0.3684 |
| ScanNet++ / 09c1414f1b | 0.2178 → 0.2684 | 0.6172 → 0.6260 |
| HiRoom / 20241230/828738/cam_sampled_08 | 0.5231 → 0.0066 | 0.7694 → 0.1098 |

HiRoom 的退化不触发筛场景、挑 seed 或换参数。主损失与超参数见 [method.md](method.md)。
