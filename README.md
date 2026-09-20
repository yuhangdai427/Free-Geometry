# DA3 + VGGT：几何测试时适配复现

使用本地权重、五个数据集和 DA3 benchmark，独立复现 Self-Geometry，并移植 TCO/Test3R 对照。
**当前协议：每模型/方法/场景适配一次，评测使用 42、43、44 三种抽帧，最多 100 帧；相同有序图片只评一次。**
训练 RNG 为 0，三个评测 seed 不会重复训练。论文未明确报告三 seed 平均，这属于用户指定扩展。

完整的训练/评测边界、官方来源、额外数据集设置及启动命令见 [当前协议](docs/current_protocol.md)。
Self-Geometry 公式映射见 [方法说明](docs/method.md)，对照移植来源见 [对比说明](docs/comparisons.md)。

```bash
# 全量：双模型 × baseline / Self-Geometry / TCO × 五数据集（不跑 Test3R）
bash scripts/run_train_once.sh --root artifacts/train_once_eval_three

# 仅生成计划
bash scripts/run_train_once.sh --root artifacts/train_once_eval_three --dry-run

# 首场景 smoke：完整训练预算，单次评测抽帧
bash scripts/run_train_once.sh --root artifacts/train_once_smoke --first-only --eval-seeds 42
```

TCO 使用独立稀疏训练清单：ETH3D/DTU 每 5 帧、7Scenes 每 200 帧。
ScanNet++/HiRoom 没有官方 TCO 设置，明确标为均匀最多 10 帧训练的扩展。
适配后重新用已保存参数推理最多 100 个评测帧。Self-Geometry 每步使用 FAN 子集；
Test3R 已按用户要求移出默认全量队列，保留历史结果和可选实现。

TCO 独占 GPU，纯 CPU 评测可重叠。主机 RAM 总硬上限 72 GiB，子任务默认 32 GiB，
无 swap；超限记录该场景失败。已有结果、checkpoint 和错误均保留，重跑相同命令可恢复。
旧入口的 `--seeds` 是训练 RNG，默认已改为仅 0；不能用它表达三个评测抽帧。
旧文档中的 seed 0/1/2 多次训练计划是历史方案，以当前协议为准。

输出分为 `ROOT/training` 和 `ROOT/evaluation`；后者每个 seed 的 `evaluation_ref.json`
记录 checkpoint 身份与 canonical 结果路径，`summary.json` 报告均值、标准差、缺失场景及
真正计算的评测数。缺场景不输出完整数据集均值。

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
