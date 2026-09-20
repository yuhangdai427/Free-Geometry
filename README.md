# DA3 + VGGT：几何测试时适配复现

独立实现 [Self-Geometry 正文与附录](https://arxiv.org/abs/2608.10708v2) 的几何测试时适配，使用本地模型和 DA3 benchmark。支持 **2 模型 × 5 数据集 × 3 seed**，每场景 50 步。作者训练代码尚未公开，公式映射和固定假设见 [方法说明](docs/method.md)。单场景退化保留，进程失败记录后继续其余任务。

当前默认 **双模型 × baseline / Self-Geometry / TCO × 五数据集 × 三 seed（0、1、2）**，共 1620 个场景结果格。Test3R 已按用户要求停止，保留代码与已完成结果，但不再默认运行。方法来源、移植细节、原版和快速日程区别见 [对比方法说明](docs/comparisons.md)。

**正式对齐原论文的入口**：`bash scripts/run_paper_comparison.sh --root artifacts/paper_comparison_without_test3r`。
它固定检查最多 100 帧、Self-Geometry 50 步和完整 AUC / posed+unposed 重建评测，拒绝 3 帧 / 2 步或跳过评测的 smoke 配置。
训练与评测使用同一场景帧列表；三 seed 指训练随机性，抽帧仍是官方 seed 42。
可选 Test3R 代码保留的设置为：与 Self-Geometry 同为 **50 次优化器更新**；每次累积 4 个三元组，共处理 **200 次三元组**，达到预算即停止。仍从有序 N³ 总体无放回抽取最多 1000 个三元组，按固定顺序取前缀训练。该额外对照对齐更新次数，不表示两种方法每步计算量相同。训练/评测帧数、Self-Geometry 50 步与完整评测不变。`--set test3r_max_updates=null --set test3r_max_triplets=null` 可在另一个 root 恢复 Test3R 全遍历。
本轮 smoke 只用 seed 0、每数据集首场景；全量三 seed 命令留待正式实验。单卡调度默认 `--gpu-workers 2 --eval-workers 1`：Test3R 按显存预算并发，其余未知/大显存训练独占 GPU，CPU 评测同时进行。
主机 RAM 由 Linux cgroup v2 硬限制：所有 runner 共享 **72 GiB**，64 GiB 开始回收；单个训练/评测子任务默认 **32 GiB**，任务组禁用 swap。超限会终止该子任务、保留失败记录并继续其他场景；不会缩帧或修改指标。需要 systemd user manager 和可写的 delegated cgroup，配置失败直接退出。详见 [RAM 诊断与限制](docs/ram_diagnostic.md)。
具体对应和实物输出审计见 [train/eval 协议](docs/protocol_alignment_audit.md)。

```bash
# 全量 baseline / Self-Geometry / TCO：训练 seed 0、1、2，抽帧 seed 42
bash scripts/run_paper_comparison.sh --root artifacts/paper_comparison_without_test3r --seeds 0 1 2

# 仅检查全量计划，不启动训练
bash scripts/run_paper_comparison.sh --root artifacts/paper_comparison_without_test3r --dry-run
```

可选 Test3R 50-update 设置是用户指定的额外对照，报告明确记录该预算；`comparison_fast.yaml` 保留旧的直接抽取 200 个三元组的采样方式。3 帧 / 2 步 smoke 指标不作为正式效果结论。

本机环境已安装；新机器先按下方本地环境说明创建 `.venv`，再运行 `bash scripts/bootstrap_comparison.sh` 获取固定提交的作者源码及 gsplat。已有模型和数据直接复用。接口、断点恢复和 smoke 的实测及退化记录见 [对比验证](docs/comparison_validation.md)。下方 `run_full.sh` 继续作为 **仅 Self-Geometry + baseline** 的双模型优化入口。

## 仅 Self-Geometry：直接启动全量

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
