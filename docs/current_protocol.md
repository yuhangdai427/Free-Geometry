# 当前训练与评测协议（2026-09-20）

本仓库是论文方法及官方对照的移植，不是 Self-Geometry 尚未公开的作者训练代码。
最新约定：每个模型/方法/场景只适配一次（训练 RNG 0），保存该场景的参数；
评测对同一个 checkpoint 使用抽帧 seed 42、43、44，最多 100 帧。
这三次是输入采样变化，不是训练三份模型；也不是三个互不相关的重复实验。
论文没有明确报告三 seed 平均，三次评测是用户指定扩展。
当前全量仅运行 baseline、Self-Geometry、TCO；按用户要求排除 Test3R，保留其历史记录。

## 每种方法的训练输入

| 方法 | 一次更新的输入 | 可训练参数 | 预算 |
|---|---|---|---|
| baseline | 无训练 | 无 | 冻结推理 |
| Self-Geometry | FAN 目标帧 + 每个活跃角度桶一个源帧 | QKV LoRA | 50 次迭代，按几何损失选 best |
| Test3R | `(i,j)` 与 `(i,k)` 两个独立双图输入 | encoder prompts | 4 个三元组累计一次更新，50 次更新 |
| TCO | 下表中的稀疏训练集合 | decoder LoRA | 40 步；DTU 50 步，保存 final |

Self-Geometry 初始化候选帧池及 Test3R 三元组候选池目前固定采用训练抽帧 seed 42
的至多 100 帧；它们的训练前向分别用 FAN 子集/双图，而非 100 帧 batch。
评测抽帧 seed 43/44 不影响已保存的训练参数、候选池或最佳 checkpoint 选择。

## TCO 稀疏训练与最多 100 帧评测分离

| 数据集 | 训练选择（在评测抽帧之前进行） | 当前首场景训练/评测帧数 |
|---|---|---|
| ETH3D | TCO 原步长 5 | 8 / 38 |
| 7Scenes | TCO 原步长 200 | 5 / 100 |
| DTU | TCO 原步长 5 | 10 / 49 |
| ScanNet++ | 官方未覆盖：均匀最多 10 帧扩展 | 10 / 100 |
| HiRoom | 官方未覆盖：均匀最多 10 帧扩展 | 10 / 23 |

保留 DA3 benchmark 的场景定义。特别是 7Scenes 仍用 seq-01（stairs 为 seq-02），
不切换到 TCO 自身的 test-sequence 划分；这里移植的是官方稀疏采样规则。
ETH3D 保留 DA3 的无效图像过滤，DTU 保留 DA3 的参考帧顺序。
训练分辨率设置为 TCO 的 518，使用各架构的 RGB 预处理；评测为 DA3 接口的 504。
训练原始 RGB 清单和完整配置分别存在 `tco_training/manifest.json` 与其 checkpoint 中，
适配结束后清除训练图像编码缓存，重新加载已保存的 LoRA，对评测图像重新推理。
不能把训练视图预测填充成 100 帧，不能把训练 encoder 缓存套到新图像。

沿用 TCO 官方 Adam、rank 4/alpha 16、decoder 投影/FFN LoRA、旋转角度损失、
归一化平移 L1 和 2DGS 光度目标。相机先验按 Self-Geometry 改为冻结模型预测，
默认不使用 GT 内参/深度，内参先验权重为 0。这是 GT-free 对照，不能声称与
使用 GT pose/intrinsics 的 TCO 原论文整套实验完全一致。

## 评测种子去重

`evaluate_seeds.py` 使用有序 RGB 文件清单（含大小和修改时间）及同一 checkpoint
判定复用。图片相同只保留一个 canonical evaluation；每个 seed 保存引用。
不足或恰好 100 帧并全部输入时，42/43/44 均引用同一结果，不重复模型推理或融合。
超过 100 帧且抽样变化时，重新推理新图像并使用与之匹配的 GT 元数据评测。
跨 seed 先对每个数据集场景等权平均，再求均值及样本标准差；缺场景不输出完整均值。
汇总同时报告 `unique_evaluations`，不把重复引用宣称为独立实验。

## 启动

```bash
# 全量：DA3/VGGT × baseline / Self-Geometry / TCO × 五数据集；每场景一次适配，三种评测抽帧
bash scripts/run_train_once.sh --root artifacts/train_once_eval_three

# 相同正式训练预算的首场景 smoke，只评测抽帧 seed 42
bash scripts/run_train_once.sh --root artifacts/train_once_smoke --first-only --eval-seeds 42

# 只检查全量计划，不运行训练
bash scripts/run_train_once.sh --root artifacts/train_once_eval_three --dry-run

# 修正后的 TCO 当前补跑（单训练 seed、单评测 seed）
bash scripts/run_paper_comparison.sh --root artifacts/paper_protocol_tco_sparse_train \
  --methods tco --seeds 0 --first-only --reuse-model-root artifacts/paper_protocol_first/baseline \
  --gpu-workers 1 --eval-workers 1

```

旧 runner 的 `--seeds` 仍明确表示训练 RNG；默认已改为仅 0。不要传 `--seeds 42 43 44`
表达三次评测，使用新入口的 `--eval-seeds`。旧 0/1/2 三次训练计划仅作历史记录。
TCO 独占 GPU；默认单 CPU evaluator。所有实验共享主机 RAM 72 GiB 硬上限，
子任务默认 32 GiB。失败场景留记录，其他场景继续；不再补跑 Test3R。
ETH3D 融合按帧加载原分辨率图像及深度；体素、截断、掩码、对齐及指标保持官方设置。

## 来源

- [Self-Geometry IV-A](https://arxiv.org/html/2608.10708v2#S4.SS1)
- [TCO §4.1.1 采样](https://arxiv.org/html/2604.03878v1#S4.SS1.SSS1)
- [TCO 固定版本](https://github.com/cvlab-stonybrook/TCO/tree/65387333877723cf99086f7f29089379dd50c59a)
- [先前 100 帧训练 OOM 及来源更正](tco_gpu_memory.md)
