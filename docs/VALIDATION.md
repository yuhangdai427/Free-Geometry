# Unified Protocol 验证报告

日期：2026-09-19。重构基线：`b647686481c6846ad2bd2a3d0d69dd7c5eb82c23`。

## 已实际运行

本地环境：macOS 26.6.2 / Apple Silicon，Python 3.10.11，PyTorch 2.14.0，NumPy 2.2.6，Pillow 12.3.0，OpenCV 5.0.0。无 CUDA、真实模型权重或数据集。

| 检查 | 实际结果 |
|---|---|
| `pytest tests -q` | **85 passed，5 skipped**；跳过项全部为真实 GPU 模型验收 |
| `ruff check src/free_geometry tests/unified` | 通过 |
| `git diff --check` | 通过 |
| `python -m build --wheel` | 成功生成 wheel |
| wheel 安装到独立目录，再从仓库外导入 | `free_geometry`、DA3、VGGT 导入成功，CLI help 可用 |
| 已安装 CLI 的 prepare → inspect | 真正读取临时图像、保存与校验 manifest；10 个不同训练 shared |
| 原生 DA3/VGGT Python 类导入 | 成功，无需应用、3DGS 或训练权重 |
| DA3 不同长宽比图像预处理 | 共同 crop 及逐帧 image transform 检查通过；未加载模型 |

增加 CPU CI，执行回归测试、静态检查及打包。上表是本机结果；远程 CI 状态以 PR checks 为准。

## 自动测试覆盖

- **选帧**：51 组帧数/比例组合，各检查 100 个 seed；覆盖 N=8–20、49/50、大场景、非默认比例、τ=0.6、坏图、SIFT 关闭、组合不足、shared 去重、extras 最小重叠、独立 slots 和 manifest 损坏。
- **Loss**：Teacher=Student 的几何零误差，identity R 有限梯度，近零 T 基线，RKD 六视角的 60 个三角项及相似变换不变性，Couple 固定支持与非法输出。五项独立梯度与总梯度可加性测试通过。
- **A/B**：同输入 q=1，分歧增大降权、不同项不串权、Teacher/q 无梯度；B 无法拟合时保留合法 A、q=1 并记录错误。
- **DVLT 解算**：已知旋转、内参、中心的合成射线恢复；非零有限梯度、方向有限差分一致；退化射线明确拒绝。测试使用自定义最小特征向量 backward，无 RANSAC。
- **训练控制**：小型真实 LoRA 参数模型运行统一循环。连续训练与中断恢复的参数、逐步日志完全一致；改变 probe 频率不改变训练参数；冻结 Teacher 不变、Student 重置；配置/权重内容不匹配拒绝恢复；早停后恢复不重复训练或 probe。
- **评测对齐**：逆 crop、有效区域和内参同步变换，原生导出不被原地修改。
- **工程边界**：五适配器按需注册、核心无 diagnostics/旧 protocol 依赖、prepare 导入不加载 torch/模型。

上述小模型验证不能证明真实 DA3/VGGT/Omega/Pi3/DVLT 的 LoRA 反传已通过。

## 尚未运行，不计为通过

1. 五模型真实权重的各项 loss → 实际 LoRA 梯度、短程 optimizer 更新、checkpoint 导出与标准指标链。
2. DA3/VGGT 在固定 ETH3D 场景上的完整 100 次更新与基线对照。
3. 大场景真实图像 SIFT/dense 训练、显存与缓存性能测量。
4. DVLT 真实噪声射线上的训练稳定性，以及新可微解算与官方 RANSAC 的旋转差异。
5. 真实数据上的位姿 AUC、深度指标和 TSDF 重建评测。

**因此本分支可供代码审查和 CUDA 验收，尚不能声称五模型训练可用性或指标提升已经得到实验确认。**

## CUDA 验收入口

准备本地 JSON（五个模型键必须齐全）：

```json
{
  "da3": {"config": "/configs/da3.yaml", "manifest": "/manifests/scene.json"},
  "vggt": {"config": "/configs/vggt.yaml", "manifest": "/manifests/scene.json"},
  "omega": {"config": "/configs/omega.yaml", "manifest": "/manifests/scene.json"},
  "pi3": {"config": "/configs/pi3.yaml", "manifest": "/manifests/scene.json"},
  "dvlt": {"config": "/configs/dvlt.yaml", "manifest": "/manifests/scene.json"}
}
```

配置使用默认五项 loss、正确的本地权重/source、CUDA device，与 manifest 的 sampling 一致。

```bash
FG_GPU_SPECS=/configs/gpu_specs.json pytest tests/unified/test_gpu_models.py -v
```

该验收逐项检查真实 LoRA 梯度，再运行两次更新和 checkpoint 导出。完整 100 步及 GT 指标对照另按 README 的四命令执行；短程 smoke 不代替完整实验。
