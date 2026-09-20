# Baseline / Test3R / Self-Geometry / TCO

正式入口：`bash scripts/run_paper_comparison.sh`，校验论文训练/评测设置；自定义变体入口为 `bash scripts/run_comparison.sh`。默认 DA3-Giant、VGGT，五个数据集，seed 0/1/2，90 场景：共 **2160 个模型/方法/场景/seed 结果格**。其中 baseline 有 540 个结果格，只需推理及评测 180 次；三个适配方法合计 1620 次独立训练。输出结构 `ROOT/METHOD/MODEL/seed_N/DATASET/SCENE/{baseline,adapted}`。

## 来源与移植边界

| 方法 | 来源 | 实现方式 |
|---|---|---|
| baseline | 本地 VGGT / DA3 原始权重 | 原生 RGB 预处理、冻结模型全场景前向 |
| Self-Geometry | [正文及附录](https://arxiv.org/html/2608.10708v2) | 本仓库公式实现，见 [method.md](method.md) |
| Test3R | [作者仓库](https://github.com/nopQAQ/Test3R/tree/a2eb94bc716df27521f053d417fcf6afa9870ff5)、[论文](https://arxiv.org/abs/2506.13750) | 原版支持 DUSt3R；本仓库移植 encoder prompts 和同参考图 pointmap L1 |
| TCO | [作者仓库](https://github.com/cvlab-stonybrook/TCO/tree/65387333877723cf99086f7f29089379dd50c59a)、[正文与补充材料](https://arxiv.org/html/2604.03878v1) | 直接调用固定版本 `tco_loss.py` / `objective.py` / `constraints.py`，适配本地模型接口 |

作者代码克隆至忽略提交的 `external/`。`bash scripts/bootstrap_comparison.sh` 安装 gsplat 1.5.3 并克隆上述固定提交；已有源码只检查，绝不覆盖。不下载模型或数据。TCO 作者仓库的模块、许可证信息随克隆保留；本仓库不重新给作者代码指定许可证。

所有方法共用 BF16 backbone、FP32 heads/几何路径；这是本地统一模型接口的数值设置，不承诺与上游默认混合精度逐位等价。可用 `--set precision=fp32` 关闭 BF16，显存和耗时会增加。

原论文只比较 Free-Geometry 和 TCO；这里的 Test3R 是额外对照。两个模型上的 Test3R 和 DA3 上的 TCO 是明确的架构移植，不能宣称与作者支持模型的代码逐位等价，也不保证重现论文表格数值。

## Test3R

对有序三元组 `(i,j,k)`，分别输入 `(i,j)` 和 `(i,k)`，比较参考图 i 的相机坐标 pointmap，损失为所有 XYZ 分量的平均绝对误差，不加 GT、confidence 或其他几何约束。VGGT 默认使用预训练的原生 point head，坐标系是输入对中第一张图的相机坐标系（[VGGT 论文](https://jytime.github.io/data/VGGT_CVPR25.pdf)）。DA3 没有同类独立 point head，因此由其预测深度与内参反投影构造 pointmap。`--set test3r_vggt_points=depth` 保留 VGGT 的反投影对照变体，需使用不同 root；不会与 native 默认结果混合。无论训练 pointmap 来自哪个 head，最终评测统一使用原生 depth + pose，沿用 DA3 benchmark。

只学习 32 个深层视觉提示，所有原始参数冻结。VGGT 提示放在 DINO 图像编码器的 24 个 block 中，DA3 放在首个跨视图注意力之前的 13 个 block 中。保留原生 CLS/register/position embedding，提示置于前方；末层去掉提示，再进入各自原生多视图网络。按作者代码，首个提示经过 block 0、1；其后逐层替换；不分配作者代码中未参与前向的最后一个提示切片。提示在不同图像间共享、初始化为零。新增零提示仍会改变 attention 的归一化，因此与零初始化 LoRA 不同，插入提示后的输出不保证等于未插入提示的 baseline。

默认 `comparison.yaml`：AdamW，lr=1e-5，betas=(0.9,0.95)，weight_decay=0，累积 4 个三元组后更新，2 epochs；按用户要求，从含重复索引的有序 N³ 总体中无放回抽取最多 1000 个三元组，固定顺序两轮复用；N³ < 1000 时使用全部总体。seed 控制抽样与顺序。100 帧时为 2000 次三元组呈现、500 次 optimizer updates；每场景保存实际抽样索引到 `adapted/triplets.json`，日志及 complete.json 记录总体、上限和实际预算。保留作者 epoch 边界的梯度累积行为，最后未满一次更新的梯度不额外更新参数。checkpoint 保存累积梯度，支持中途恢复。

优化：VGGT 原生 pointmap 训练只运行 aggregator + point head，跳过该损失不需要的 camera/depth heads；最终导出仅运行 depth/pose。整数索引代替 N³ 个图像三元组对象；将独立图像对按 batch 合并前向，`test3r_pair_batch` 是每次合并的三元组数，默认 2；累积批次的 loss 乘以对应样本权重，保持等效目标。可设 1 降低显存。BF16 导致的批次舍入差异不承诺逐位一致。

**快速变体** `comparison_fast.yaml` 明确限制 Test3R 为 50 次 optimizer updates，每次累积 4 个三元组。这是更改训练预算的变体，不能作为原版遍历日程的等价加速。100 帧原版需处理 2,000,000 个三元组；因此入口先完成另外三个方法，再运行 Test3R。当前默认是用户指定的 1000-triplet 上限，不等同于原版全遍历，报告标明 `triplet_cap_1000_per_epoch`。需全遍历使用 `--set test3r_max_triplets=null` 并更换 root；`comparison_fast.yaml` 仍单独标明 `budget_variant`。

## TCO

按 Self-Geometry IV-A，将冻结模型的预测相机作为辅助先验，训练不读取 GT。默认仅 pose prior，intrinsics 权重 0；`tco_intrinsics_weight=0.01` 可额外使用冻结预测内参先验，是显式变体。Self-Geometry 未公开其 TCO 移植代码，也未明确是否同时约束预测内参；默认 pose-only 是据 IV-A 固定的解释。原 TCO 使用 GT pose/intrinsics 的设置不混入此基准。

直接调用作者 2D Gaussian Splatting photometric objective：固定半径比例 0.5、cos 权重、confidence opacity、可见性阈值，`num_view_groups=100`（最多 100 帧时每轮随机一个源视图渲染至全部视图）。保留作者 ED visibility pass 和 RGB pass，没有替换为 grid-sample photometric loss。pose 项：cosine rotation + 2× normalized-translation L1，双方在参考相机坐标系对齐（`pose_type=rel`）。

VGGT：冻结 DINO 和任务 heads，仅 frame/global decoder 上 QKV、attention projection、FFN LoRA。DA3：冻结前 13 个纯图像 block 和 heads，在 block 13–39 的同类投影上加入 LoRA；SwiGLU 对应 w12/w3。rank=4、alpha=16、dropout=0，Adam，无 weight decay，clip=1，使用最终迭代参数。

| 数据集 | steps | lr | photometric weight |
|---|---:|---:|---:|
| ETH3D | 40 | 5e-4 | 0.2 |
| 7Scenes | 40 | 1e-3 | 0.2 |
| DTU | 50 | 2e-4 | 1.0 |
| ScanNet++ | 40 | 1e-3 | 0.2 |
| HiRoom | 40 | 1e-3 | 0.2 |

前三项来自作者启动脚本/附录；后两项没有作者对应数据集参数，固定采用 7Scenes 室内设置，未用 GT 调参。`tco_steps/tco_lr/tco_photo_weight` 的显式覆盖写入计划和 complete.json。

优化：冻结编码器输出在全场景及三个 seed 间缓存，仅重复可训练 decoder；TCO 默认保留 activation checkpoint，避免全 100 帧使用 Self-Geometry 的低视图 fast 设置。CUDA renderer 第一次调用需要编译，后续运行使用缓存；本机入口选择已安装的 CUDA 12.8 与 torch 匹配。

## 全量、恢复与错误

```bash
# Self-Geometry 论文日程 + Test3R 最多 1000 三元组 × 2 epochs，全部 2160 格
bash scripts/run_paper_comparison.sh --root artifacts/paper_comparison_triplets1000

# 全场景、双模型、四方法、三 seed；Test3R 明确采用 50-update 预算
bash scripts/run_comparison.sh --config configs/comparison_fast.yaml --root artifacts/comparison_fast

# 仅生成计划；不占 GPU
bash scripts/run_comparison.sh --config configs/comparison_fast.yaml --root artifacts/comparison_fast --dry-run

# 查看进度和汇总
.venv/bin/python scripts/status_comparison.py --root artifacts/comparison_fast
.venv/bin/python scripts/summarize_comparison.py --root artifacts/comparison_fast
```

模型、RGB、baseline 和评测缓存跨 seed 共用，适配参数和优化器独立重置。CPU 评测与 GPU 训练重叠；DTU GPU 融合串行，CPU 距离评测并行。单场景/seed 失败记日志后继续，有限退化照实保留。重跑相同命令恢复 `last.pt` 并跳过完成任务；改变配置需换 root。

每数据集先对全部场景等权平均，再跨 seed 计算均值和样本标准差。缺场景不输出完整跨 seed 均值；缩帧/首场景调试不算全量复现。原始配置、训练状态、完整失败列表均落盘。全量命令已准备，不自动占用 GPU 启动 1620 次训练。

断点恢复保存参数、Adam 状态、随机状态和未更新的累积梯度。GPU BF16/attention/渲染反传不保证逐位确定性；恢复后继续优化可能与不中断运行有数值差异。验证单独检查恢复状态和下一步前向损失，后续参数差异照实报告，见 [comparison_validation.md](comparison_validation.md)。

## 接续当前验证

`queue_comparison.py` 在前一 matrix 的训练和评测进程结束后启动命令；前一轮个别场景失败仍继续后续方法。当前 Test3R 双模型、五个首场景、seed 0 的上限验证单独保存至 `artifacts/paper_protocol_test3r_1000`，复用 `artifacts/paper_protocol_first/baseline`。分别查看两个 root 的 `SELECTED_RESULTS.md`，不会把旧的未设上限记录改成新协议。

```bash
.venv/bin/python scripts/queue_comparison.py \
  --after artifacts/paper_protocol_first --root artifacts/paper_protocol_test3r_1000 -- \
  bash scripts/run_paper_comparison.sh --root artifacts/paper_protocol_test3r_1000 \
  --methods test3r --seeds 0 --first-only \
  --reuse-model-root artifacts/paper_protocol_first/baseline
```
