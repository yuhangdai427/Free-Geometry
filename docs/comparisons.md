# Baseline / Test3R / Self-Geometry / TCO

当前入口为 `bash scripts/run_train_once.sh`：每场景训练 RNG 0 适配一次，评测抽帧 seed 42/43/44；相同图片去重。TCO 稀疏训练和最多 100 帧评测已分离，Test3R 已恢复补评测。以 [当前协议](current_protocol.md) 为准；旧三次独立训练计划不再作为默认。

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

默认 `comparison.yaml`：AdamW，lr=1e-5，betas=(0.9,0.95)，weight_decay=0，累积 4 个三元组后更新。**按用户最新要求，最多 50 次 optimizer updates，与 Self-Geometry 对齐更新数**。从含重复索引的有序 N³ 总体无放回抽取最多 1000 个三元组，seed 固定抽样与顺序；在当前各数据集首场景上，取该序列前 200 个三元组完成 50 次更新即停止，不再执行完原先两轮的 500 次更新。每场景保存抽样序列到 `adapted/triplets.json`，日志及 complete.json 记录上限、实际 microsteps 和 updates。checkpoint 保存累积梯度，支持同一预算下恢复。两种方法每次更新的输入及损失不同，更新数相同不代表 FLOPs 或样本数相同。

优化：VGGT 原生 pointmap 训练只运行 aggregator + point head，跳过该损失不需要的 camera/depth heads；最终导出仅运行 depth/pose。整数索引代替 N³ 个图像三元组对象；将独立图像对按 batch 合并前向，`test3r_pair_batch` 是每次合并的三元组数，默认 2；累积批次的 loss 乘以对应样本权重，保持等效目标。可设 1 降低显存。BF16 导致的批次舍入差异不承诺逐位一致。

历史 `comparison_fast.yaml` 也是 50 次更新，但直接抽取 200 个三元组；当前默认保留 1000 个候选的样本序列并消费前 200 个，二者不混用 root。报告标明 `updates_cap_50`。需恢复全遍历时，使用 `--set test3r_max_updates=null --set test3r_max_triplets=null` 并更换 root。仅清除更新上限会恢复此前 1000 三元组 × 2 epochs 的日程。

## TCO

按 Self-Geometry IV-A，将冻结模型的预测相机作为辅助先验，训练不读取 GT。默认仅 pose prior，intrinsics 权重 0；`tco_intrinsics_weight=0.01` 可额外使用冻结预测内参先验，是显式变体。Self-Geometry 未公开其 TCO 移植代码，也未明确是否同时约束预测内参；默认 pose-only 是据 IV-A 固定的解释。原 TCO 使用 GT pose/intrinsics 的设置不混入此基准。

直接调用作者 2D Gaussian Splatting photometric objective：固定半径比例 0.5、cos 权重、confidence opacity、可见性阈值，`num_view_groups=100`（最多 100 帧时每轮随机一个源视图渲染至全部视图）。保留作者 ED visibility pass 和 RGB pass，没有替换为 grid-sample photometric loss。pose 项：官方默认 angle rotation + 2× normalized-translation L1（旧版本使用 cosine），双方在参考相机坐标系对齐（`pose_type=rel`）。

VGGT：冻结 DINO 和任务 heads，仅 frame/global decoder 上 QKV、attention projection、FFN LoRA。DA3：冻结前 13 个纯图像 block 和 heads，在 block 13–39 的同类投影上加入 LoRA；SwiGLU 对应 w12/w3。rank=4、alpha=16、dropout=0，Adam，无 weight decay，clip=1，使用最终迭代参数。

| 数据集 | steps | lr | photometric weight |
|---|---:|---:|---:|
| ETH3D | 40 | 5e-4 | 0.2 |
| 7Scenes | 40 | 1e-3 | 0.2 |
| DTU | 50 | 2e-4 | 1.0 |
| ScanNet++ | 40 | 1e-3 | 0.2 |
| HiRoom | 40 | 1e-3 | 0.2 |

前三项来自作者启动脚本/附录；后两项没有作者对应数据集参数，固定采用 7Scenes 室内设置，未用 GT 调参。`tco_steps/tco_lr/tco_photo_weight` 的显式覆盖写入计划和 complete.json。

优化：冻结编码器只在固定稀疏训练集合内缓存，仅重复可训练 decoder；最终评测清除缓存、恢复保存的 LoRA，重新对最多 100 帧推理。TCO 保留 activation checkpoint。训练采样与分辨率见 [当前协议](current_protocol.md)。CUDA renderer 第一次调用需要编译，后续运行使用缓存；本机入口选择已安装的 CUDA 12.8 与 torch 匹配。

## 全量、恢复与错误

新入口、训练和评测种子的区别、相同图片的去重及当前 smoke 命令统一维护于
[当前协议](current_protocol.md)。旧 `--seeds 0 1 2` 是历史独立训练方案，不再是
用户要求的全量协议。Test3R 当前单次训练为 50 次更新，不是原版完整双 epoch 遍历。
适配失败不替换成 baseline，缺失指标不填 0；有限退化如实记录，其他场景继续。

全体任务共享 72 GiB RAM 硬限制，每个子任务默认 32 GiB，当前 Test3R 补评测
为 48 GiB。ETH3D 融合逐帧加载原分辨率 RGBD，保持原官方 TSDF 参数与指标；
各模式融合成功分别保存身份标记，恢复时复用已完成模式。

## 单卡并发

`run_full.py` 默认两个场景进程、一个 CPU 评测进程；显存准入文件
`artifacts/gpu_admission.json` 由进程锁保护，跨方法 runner 共用，按先到顺序调度。
本机 96 GiB 卡预留 90 GiB 预算：已知 504 分辨率、至多 100 帧、pair batch 至多 2
的 Test3R 根据缓存 baseline 的实际图像尺寸申请显存；378×504 时 DA3 42 GiB、
VGGT 38 GiB，方形输入相应增加。它们依据既往实测峰值加余量，仍需监控实际峰值。
未知配置、Self-Geometry、TCO 和 DTU 融合申请全部预算。这样两个适合的 Test3R
场景可同时训练，高显存任务不会与它们重叠；CPU 指标计算独立并行。
这是单张卡的吞吐优化，不保证两倍加速，不改变 seed、输入、更新数和指标。
`--gpu-workers 1` 可关闭同一 runner 内的场景并发。未使用此准入器的外部 GPU
进程不在调度范围内。`queue_comparison.py` 仍可用于需要整轮顺序执行的任务。
