# Self-Geometry：DA3-Giant + VGGT

独立实现 [Self-Geometry 正文与附录](https://arxiv.org/abs/2608.10708v2) 的几何测试时适配，使用本地模型和 DA3 benchmark。支持 **2 模型 × 5 数据集 × 3 seed**，每场景 50 步。作者训练代码尚未公开，公式映射和固定假设见 [方法说明](docs/method.md)。单场景退化保留，进程失败记录后继续其余任务。

## 直接启动全量

本机 96 GB GPU 的速度配置：

```bash
cd /localhdd02/yuhang/code/self_geometry
bash scripts/run_full.sh --config configs/fast.yaml --root artifacts/full_fast
```

后台运行，与上面的前台命令二选一：

```bash
mkdir -p artifacts/full_fast
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_full.sh \
  --config configs/fast.yaml --root artifacts/full_fast \
  > artifacts/full_fast/launcher.log 2>&1 < /dev/null &
```

默认就是 `--models da3 vggt --datasets eth3d 7scenes scannetpp hiroom dtu --seeds 0 1 2`，不用逐项填写。`scripts/experiments.py` 和 `scripts/pipeline.py` 也是双模型入口。

| 数据集 | 每模型场景数 | 两模型、三 seed 的适配次数 | 重建指标 |
|---|---:|---:|---|
| ETH3D | 11 | 66 | posed / unposed F1 ↑ |
| 7Scenes | 7 | 42 | posed / unposed F1 ↑ |
| ScanNet++ | 20 | 120 | posed / unposed F1 ↑ |
| HiRoom | 30 | 180 | posed / unposed F1 ↑ |
| DTU（论文外补充） | 22 | 132 | posed / unposed overall distance ↓，mm |
| 合计 | 90 | **540** | 另均记录 AUC@1/3/30 |

seed 0/1/2 控制 LoRA 初始化、dropout 和训练随机性。帧列表固定采用官方 DA3 seed 42、每场景最多 100 帧；DTU 默认全部 49 帧及官方参考帧顺序。三次运行均重新初始化 LoRA、优化器与随机状态。

## 检查、恢复与查看结果

```bash
# 检查五个数据集的现有 RGB、相机和 GT
.venv/bin/python scripts/preflight.py

# 写出 540 项计划及 180 条场景 worker 命令，不训练
bash scripts/run_full.sh --config configs/fast.yaml --root artifacts/full_fast --dry-run

# 进度及活跃任务
.venv/bin/python scripts/status.py --root artifacts/full_fast

# 刷新两个模型各自的均值、标准差和失败列表
.venv/bin/python scripts/summarize_full.py --root artifacts/full_fast
```

中断或失败后重新执行同一条启动命令即可：完成的训练直接跳过，未完成训练从 `last.pt` 恢复，失败场景重试。每个场景及 seed 的错误、耗时都保存在日志/JSON；有限的退化指标不算运行错误，也不阻止队列继续。

配置、模型、数据范围或 seed 列表改变时换一个 `--root`，防止混入不同实验。默认配置使用激活 checkpoint，显存占用较低：

```bash
bash scripts/run_full.sh --root artifacts/full_dual
```

输出目录为 `ROOT/{da3,vggt}/seed_{0,1,2}/DATASET/SCENE/`。根目录的 `REPORT.md` / `summary.json` / `failures.json` 是总汇总，各 seed 内有逐场景 `scenes.csv`。统计顺序为每数据集的完整场景等权均值，再取三个 seed 的 **mean ± 样本标准差（ddof=1）**。任何 seed 缺场景，就不输出该数据集的完整跨 seed 统计。DTU 的距离与其他数据集的 F1 分开报告，不混算。

## 已做的运行优化

- **一个模型/场景一个 GPU 进程，连续跑三个独立 seed**：540 次适配只启动 180 个场景 worker；冻结模型、预处理 RGB、baseline、LightGlue 匹配、初始化阈值共用，LoRA 与优化器每次重置。
- 同一模型/场景的 baseline 推理和评测只计算一次，复用到其他 seed；原本 540 次 baseline 评测降到 180 次。适配后各 seed 都独立评测。
- 严格加载已有权重，跳过随后会被 checkpoint 覆盖的大规模随机参数初始化；不分配 VGGT point/track 和 DA3 Gaussian 等未使用分支。
- BF16 backbone、FP32 几何与 heads、fused AdamW。`fast.yaml` 关闭激活重算，并将较大的恢复 checkpoint 从每 5 步改为每 25 步，减少网络盘写入；50 步优化、损失和 best checkpoint 选择规则不变。
- 最多两路 CPU 评测与 GPU 训练重叠。DTU 的 GPU 融合与训练串行，CPU 距离评测与后续训练重叠。`--eval-workers 1` 可进一步限制 CPU 并发。

速度配置用更多显存，并在中断后最多重放 24 步；默认配置最多重放 4 步。实测速度和显存记录见 [验证记录](docs/validation.md)，不把微基准加速比例当作整套 benchmark 加速比例。

默认可复用旧 `artifacts/main` 中兼容的 VGGT baseline / 匹配；`--no-reuse` 关闭旧实验缓存导入，同场景三 seed 的固定资源共享仍保留。不同模型使用各自原生预处理，不跨模型复用预测或匹配。旧版适配结果保留，但不混进本次新 RNG 协议的统计。

## 本地环境与模型

- VGGT：`/localhdd02/yuhang/weights/VGGT-1B/model.pt`。
- DA3-Giant 1.1：`/localhdd02/yuhang/weights/DA3-GIANT-1.1/model.safetensors`，同时读取同目录 `config.json`。
- 数据：`/localhdd02/yuhang/datasets/DA3-BENCH`。
- SuperPoint / LightGlue：`weights/hub/checkpoints/`。
- 本机 `.venv` 继承 `/localhdd02/yuhang/envs/da3`；安装入口 `bash scripts/bootstrap.sh`。VGGT、DA3、LightGlue 的源码和许可证在 `vendor/`，不依赖相邻 Free-Geometry 仓库，不重复下载已有模型或数据。

换路径可使用 `--da3-weights /path/model.safetensors --vggt-weights /path/model.pt --set data_root=/path/DA3-BENCH`。数据、权重、环境和实验输出不提交 Git。

DA3 接入的是可微模型本体，不只是 DA3 的 benchmark：40 个 QKV LoRA 模块、15,728,640 参数；VGGT 为 72 个模块、18,874,368 参数。GT 仅在评测时读取，训练不读取 GT 相机、深度或 mask。

## 单场景及调试

```bash
.venv/bin/python -m pytest -q tests

# 双模型 × 五个首场景 × 三 seed，短小跑，仅检查训练链路
bash scripts/run_full.sh --root artifacts/smoke_new --first-only \
  --set max_frames=6 --set iterations=2 --skip-evaluation

# DA3 单场景；VGGT 将 --model 改为 vggt，并换输出目录
.venv/bin/python scripts/run.py baseline --model da3 --dataset dtu --scene scan1 --output artifacts/dtu_da3
.venv/bin/python scripts/run.py adapt --model da3 --dataset dtu --scene scan1 --output artifacts/dtu_da3 --resume
.venv/bin/python scripts/run.py evaluate --model da3 --dataset dtu --scene scan1 --output artifacts/dtu_da3 --stage adapted
```

`--first-only`、减少帧数/迭代或 `--skip-evaluation` 的运行不会被汇总为完整 benchmark。先前可选歧义排查工具位于 `scripts/diagnostics/`，不进入默认全量队列。
