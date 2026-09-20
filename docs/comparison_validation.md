# Comparison validation

本页区分接口正确性、短 smoke 和复现指标。代码支持全量 2160 个结果格；这些验证不代表全量 benchmark 已完成，也不保证移植方法在每个场景改善 baseline。

## Test3R 接口与原生 point head

作者固定版本为 `a2eb94bc716df27521f053d417fcf6afa9870ff5`。核对入口包括 `dust3r/inference.py`（损失、优化器、累积）、`eval/mv_recon/data.py`（有序三元组）、`dust3r/model.py`（提示注入）。本仓库是面向 DA3/VGGT 的移植，不是作者已经支持这两个模型的实现。

当前默认 VGGT Test3R 加载原生预训练 point head，只在 pointmap loss 使用；DA3 使用预测深度和内参反投影。两者最终统一导出 depth/pose 供 DA3 benchmark 评测。

`scripts/validate_comparison_ports.py` 使用真实权重验证：

| 检查 | DA3 | VGGT |
|---|---:|---:|
| 与此前 baseline 的 depth/conf/extrinsics/intrinsics 最大误差 | 0 | 0 |
| TCO 冻结编码器首次/重复缓存输出最大误差 | 0 | 0 |
| 移除适配器后的 baseline 最大误差 | 0 | 0 |
| Test3R 可训练提示参数 | 589824 | 753664 |
| Test3R 提示梯度范数（有限且非零） | 0.00592017 | 27.5950 |

此外，VGGT 原生 point head 的 activation checkpoint 路径完成 3 帧、2 updates；fast 路径完成三个 seed。参数/优化器在 seed 间重新初始化，冻结模型和 RGB 缓存共享。

## 断点恢复

`scripts/check_comparison_resume.py` 在真实 GPU 上从第一次 optimizer update 的 checkpoint 分叉，验证参数和优化器恢复完全一致、恢复后下一次前向 loss 一致，并记录后续参数差异。

| 模型 / 方法 | 状态精确恢复 | 下一步 loss 最大差异 | 后续参数最大差异 |
|---|---|---:|---:|
| DA3 / Test3R | 是 | 0 | 3.80653e-6 |
| VGGT / Test3R native | 是 | 0 | 4.85987e-6 |
| DA3 / TCO | 是 | 0 | 7.30722e-4 |
| VGGT / TCO | 是 | 0 | 7.19653e-4 |

**恢复状态检查通过不等于优化轨迹逐位一致。** CUDA BF16/attention/renderer 反向计算有数值差异，TCO 的差异更大。这里没有把后续参数等价检查写成通过，也没有宣称精确复现不中断轨迹。原始记录位于本机 `artifacts/comparison_resume_v2` 和 `artifacts/comparison_native_resume`。

## 五数据集 smoke

固定首场景：ETH3D `courtyard`、7Scenes `chess`、ScanNet++ `09c1414f1b`、HiRoom `20241230/828738/cam_sampled_08`、DTU `scan1`。

```bash
bash scripts/run_comparison.sh --root artifacts/all_dataset_smoke \
  --config configs/comparison_fast.yaml --first-only \
  --set max_frames=3 --set iterations=2 --set tco_steps=2 \
  --set test3r_max_updates=2 --skip-evaluation
.venv/bin/python scripts/check_comparison_smoke.py --root artifacts/all_dataset_smoke
```

这一轮两模型 × 四方法 × 五数据集 × 三 seed 的训练/导出 **120/120 格完成，四种方法的全部场景 worker 均正常退出**；baseline 的三个 seed 共享确定性预测。检查训练、导出完整性、有限正深度和官方位姿指标，不执行 120 次三维融合评测。120/120 格的导出完整性和官方位姿指标检查也通过（指标有限，不要求优于 baseline）。实测快照见 [comparison_validation.json](comparison_validation.json)。

## 已完成的三维评测与退化

ETH3D courtyard 三帧、seed 0 已对两模型 × 四方法运行官方位姿及 posed/unposed 三维评测（`artifacts/comparison_smoke`）。其中早期 VGGT Test3R 使用 depth 变体；之后默认 native 版本另在 `artifacts/test3r_native_smoke` 完成 seed 0 同样评测。

VGGT Test3R **native 和 depth 变体在这个短 smoke 均退化，posed/unposed F1=0**；native AUC@3 为 0.222222，baseline 为 0.666667。训练及导出没有异常，但不能据此称效果正常。

补齐五数据集后，VGGT Test3R native 的三帧、两次更新位姿结果如下。四个数据集首场景出现退化；ScanNet++ 首场景的 baseline 已为 0。结果原样保留，不能把这些短测解释为完整原版日程的最终效果。

| 首场景所属数据集 | baseline AUC@3 | Test3R seed 0 / 1 / 2 |
|---|---:|---|
| ETH3D | 0.666667 | 0.222222 / 0.222222 / 0.222222 |
| 7Scenes | 0.222222 | 0.111111 / 0.111111 / 0.111111 |
| ScanNet++ | 0 | 0 / 0 / 0 |
| HiRoom | 0.444444 | 0.333333 / 0.333333 / 0.333333 |
| DTU | 1.000000 | 0.888889 / 0.777778 / 0.888889 |

DTU 的 DA3/VGGT Self-Geometry baseline/adapted 六帧短测已运行官方完整评测；DA3 上拆分 GPU 融合与 CPU 评分得到的指标和原始串行流程一致。新方法的五数据集 smoke 不冒充五数据集全量三维评测。

## 较大场景运行与速度

ETH3D courtyard 全部 38 帧，seed 0，真实模型实测：

| 模型 / 方法 | 更新数 | 适配耗时（秒） | 峰值 allocated 显存（GiB） |
|---|---:|---:|---:|
| DA3 / TCO | 40 | 163.84 | 43.35 |
| VGGT / TCO | 40 | 165.07 | 31.03 |
| DA3 / Test3R | 50 | 53.25 | 31.44 |
| VGGT / Test3R native | 50 | 43.89 | 26.99 |

耗时含适配设置和最终导出，不含最初加载权重/baseline/GT 评测；不是端到端 benchmark 时间。Test3R 的 50 updates 是显式预算变体。另有三帧完整 N³ × 2 epochs 的两模型三 seed 检查（当时 VGGT 使用 depth 变体），不能用它代替当前 native 的全量日程结果。

加速包括冻结编码器缓存、跨 seed 复用模型/RGB/baseline、Test3R 独立 pair batching、按损失只运行必要 heads、CPU 评测与 GPU 训练重叠。TCO 保留 activation checkpoint。单元测试 31 项通过，含独立 seed、失败继续和 DTU 融合/评分拆分检查。
