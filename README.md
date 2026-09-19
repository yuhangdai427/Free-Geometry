# Free Geometry · Unified Protocol

逐场景、无 GT 的多视角 LoRA 适配。**一个训练循环，五个模型适配器：DA3、VGGT、VGGT-Omega、Pi3、DVLT。**

本分支以 `b647686` 的 protocol2 为基础，替换历史上按模型、数据集不断叠加的训练分支。研究报告和原始结果保留，旧实现可从该提交查看。历史报告中的指标不代表这一版统一协议的效果。

## 安装

Python 3.10–3.13；真实训练使用 CUDA 版 PyTorch。先按服务器 CUDA 环境安装 PyTorch，再安装项目：

```bash
pip install -e '.[sift,models,benchmark,dev]'
fg --help
```

只做选帧和 CPU 正确性测试可安装 `.[sift,dev]`。`prepare` 不加载 torch、模型或 GT。模型权重不会随安装自动下载。

Omega、Pi3、DVLT 使用各自官方源码与依赖；在配置中指定 `model.source` 或预先安装相应包。DA3/VGGT 源码随本项目打包。DA3 的应用/视频和 3DGS 功能分别属于 `app`、`gs` 可选依赖，不是训练入口的必需依赖。

## 四个命令

以下路径是示例，替换为实际数据和权重路径。所有步骤应使用同一配置。

```bash
# 1. 生成固定帧任务；同一 manifest 可供五模型使用
fg prepare --dataset eth3d --data-root /data/eth3d --scene courtyard \
  --config configs/default.yaml --output runs/manifests/courtyard.json

# 2. 查看帧 ID、shared slots、A/B 重叠和覆盖次数
fg inspect runs/manifests/courtyard.json

# 3. 训练：独立场景、独立输出目录
fg train --manifest runs/manifests/courtyard.json --config configs/default.yaml \
  --model da3 --weights /models/DA3-GIANT --output runs/da3/courtyard

# 4. 同帧比较冻结基线与适配结果；--data-root 才启用 GT 评测
fg evaluate --manifest runs/manifests/courtyard.json --config configs/default.yaml \
  --model da3 --weights /models/DA3-GIANT --baseline \
  --output runs/eval/da3/courtyard/base --data-root /data/eth3d
fg evaluate --manifest runs/manifests/courtyard.json --config configs/default.yaml \
  --model da3 --weights /models/DA3-GIANT \
  --checkpoint runs/da3/courtyard/checkpoints/final.pt \
  --output runs/eval/da3/courtyard/final --data-root /data/eth3d
```

其他数据集：`hiroom`、`7scenes`、`scannetpp`、`dtu`、`dtu64`。普通图像目录使用 `--dataset images`。HiRoom 重建 GT 若位于其他位置，评测时指定 `--gt-root`。`--skip-reconstruction` 仅计算位姿/可用深度指标。评测帧为有效帧中固定 seed 42 的 100 帧，少于 100 时使用全部有效帧。

配置未知键或错误类型直接报错。`--set section.key=value` 只覆盖给定字段，例如：

```bash
# 全部 loss 基础权重为 1；单独关闭 T
fg train ... --set loss.translation=0
# 关闭 A/B 调权，或只关闭 feature 的 A/B 调权
fg train ... --set reliability.enabled=false
fg train ... --set reliability.feature=false
# 可选 probe 决策模式；默认仍为固定步数
fg train ... --set probe.decision=true
# 减少 / 关闭输入遮挡，不改变全位置监督定义
fg train ... --set train.input_mask_ratio=0
```

这些缩略命令中的 `...` 表示补上完整的 manifest、模型和输出参数。五个单项 loss 配方分别为 `feature.yaml`、`rotation.yaml`、`translation.yaml`、`rkd.yaml`、`couple.yaml`。组合配方还包括 `configs/feature.yaml`、`feature_couple.yaml`、`rkdc.yaml`、`no_ab_weighting.yaml`。配置文件是默认值的覆盖，不支持隐式多层继承。

恢复训练使用原输出目录和相同参数，并加 `--resume RUN/checkpoints/step000040.pt`。总步数、学习率、loss、模型、manifest 必须与原运行一致；不能通过修改总步数改变已开始的 cosine 计划。成功 optimizer update 才计步；AMP 溢出不推进步数或 schedule，连续三次失败终止场景。

## 固定默认协议

- **Teacher 8 → Student 4**，10 组训练 + 2 组 probe；组间允许复用图像，Teacher 内禁止重复。
- `N_valid < 50` 完全不调用 SIFT；否则仅 `τ > 0.6` 使用分层随机 dense 候选，其余直接随机 shared。
- A/B 共享本组 Student 图像，extras 尽量不交叠。N=8 时复用相同缓存，明确标记没有额外核查信息。
- 输入约 50% patch 遮挡；所有有效 patch 监督。固定 Teacher/帧任务，每次访问更新 mask。
- Feature、Rotation-Huber、Translation-direction、RKD-Huber、Couple 默认全开，权重各为 1。
- A/B 按每项自己的监督量分别软降权，固定阈值；不使用全局 `geo_w`，不依赖 student 残差。
- AdamW，lr `3e-5`，weight decay `1e-5`，15% warmup + cosine，全局梯度裁剪 1.0。
- 默认 100 次更新，probe 每 20 步只监测；没有自动场景门控、隐藏 ramp 或 baseline 回退。

详细公式、接口和边界见 [统一协议](docs/UNIFIED_PROTOCOL.md)。删除范围及历史兼容性见 [迁移说明](docs/MIGRATION.md)。

## 模型配置边界

| model.name | 权重形式 | 原生特征读出与 LoRA |
|---|---|---|
| da3 | DA3 giant 本地目录或 Hub ID | DPT `[19,27,33,39]`，MV blocks 13–39，camera token 可训练 |
| vggt | VGGT-1B 本地目录或 Hub ID | DPT `[4,11,17,23]`，24 层 frame/global blocks |
| omega | 本地 `.pt` 文件 | 原生 dense-head LN，24 层 frame/inter-frame blocks |
| pi3 | 包含 `model.safetensors` 的本地目录 | point-decoder projection/LN，36 层 trunk decoder |
| dvlt | 官方目录或 Hub ID | depth/ray decoder LN，共享 recurrent block |

所有 adapter 冻结预测头，保留对输入的梯度。模型差异只存在于适配器；新增架构需要提供同一输出与 checkpoint 接口，不新增训练循环。

DVLT 的训练位姿使用本版新增的可微 DLT/IRLS/QR，评测仍使用官方 RANSAC。导出记录两者旋转差异；它们不是等价算法。本机已做合成恢复和有限差分测试，真实模型/真实场景训练尚须 CUDA 验收。

## 输出与验证

每次运行保存 `config.json`、`manifest.json`、`preprocessing.json`、`startup.json`、`training.jsonl`、`probe.jsonl`、checkpoint 和 `summary.json`。日志区分原始 loss、A/B 加权 loss、最终贡献；无效监督及失败场景不会伪装成零损失成功。

```bash
pytest tests -q
ruff check src/free_geometry tests/unified
python -m build --wheel
```

真实 GPU 验收：准备包含五模型配置/manifest 路径的 JSON，设置 `FG_GPU_SPECS` 后运行 `pytest tests/unified/test_gpu_models.py`。未提供资源时明确跳过五项 GPU 测试；不会将其计为训练验证通过。完整 ETH3D 100 步对照与大场景 dense 评测仍需要真实数据与权重。

[验证报告](docs/VALIDATION.md) 区分已经运行的测试与待完成的 GPU 实验。

## 历史资料

[研究报告目录](docs/history/) 和 `workspace/` 中的原始结果用于追溯。标准多视角 benchmark 和可视化工具保留在 `scripts/`；旧格式 checkpoint 仍通过原模型封装读取，新 checkpoint 使用 `fg evaluate`，不会在旧格式上悄悄改变含义。

项目源自 *Free Geometry: Refining 3D Reconstruction from Longer Versions of Itself*。模型代码保留各自原有版权与许可证声明；项目许可证见 [LICENSE](LICENSE)。
