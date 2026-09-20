# 方法、协议与未公开细节

本仓库是 arXiv:2608.10708v2 的独立复现，非作者发布实现。查阅日期：2026-09-20。
正文、附录：https://arxiv.org/html/2608.10708v2
作者仓库：https://github.com/CMLab-Korea/Self-Geometry （当前仅项目展示，代码待发布）。

## 已公开设定与实现对应

| 来源 | 方法 | 本仓库 |
|---|---|---|
| III-B / Eq.2 | 源深度反投影、源到目标相对位姿、像素重投影 L2 距离 | geometry.project / residuals |
| III-B / Eq.3, S.II | Fundamental matrix 与 sqrt(Sampson distance) | geometry.residuals |
| Eq.4 | 将 MVC 梯度投影到 EC 梯度的正交补，每步无条件执行 | optimization.project_gradients / combine_gradients |
| III-C / Eq.5–6 | 旋转角分桶、最多活跃桶、最大熵目标选择 | optimization.fan_setup |
| Table VI | 初始化固定目标，桶内 sequential order | optimization.fan_sample |
| S.6 | 0.85 SSIM + 0.15 L1 的光度一致性 | geometry.photo_residual |
| S.7 | 原始深度梯度的图像边缘加权平滑 | geometry.smoothness |
| S.8 | baseline confidence 前 50% 像素的深度锚定 | training.losses_for_subset |
| S.9–10 | 1.345 × 1.4826 × median(abs(residual)) 的 Huber | geometry.threshold / huber |
| S.11–12 | 五项 DWA，T=1，ratio clamp [0.5,2]，静态权重均 1 | optimization.dwa |
| S.1 | 50 步，AdamW，5e-5，wd .05，cosine，5% warmup，终点 1e-8，clip 5 | configs/paper.yaml |
| S.V | QKV LoRA，rank=64，alpha=64，dropout=.1 | model.inject_lora |
| Algorithm 1 | 最小 sqrt(robust MVC × robust EC) checkpoint | training.adapt |

## 未公开或存在歧义：明确固定的假设

1. 使用 SuperPoint + LightGlue（2048 点，LightGlue 官方默认阈值），在处理后 RGB 上提取，关闭内部 resize；所有无序帧对缓存一次。源/目标方向按需要交换坐标。
2. 每步根据当前预测先 EC 后 MVC，各保留残差较低的 90%；有效性检查排除负深度、越界重投影、非有限值及退化平移。过滤是无梯度掩码。过滤百分位不是作者提供的值。
3. FAN 使用完整 [0°,180°] 的 15° 桶（12 桶）。Table VI 写作“15° (9 bins)”，与角度定义不一致；compat9 对照将前八桶设为 15°，末桶收纳 [120°,180°]。每桶每步取一个源，循环遍历原始帧序；平局选择最低帧索引。
4. 主 LoRA 注入范围是 aggregator 下全部 QKV：图像编码器及 frame/global attention。camera head 是否也包含在“every attention block”中不明确，作为独立对照。记录模块列表与可训练参数量核对 Table VII 的 18.87M。
5. 主梯度方向以 Eq.4 为准。Table IV 的梯度方向标签可被反向理解，保留反向对照。投影在全部 LoRA 参数的联合向量上进行，不是逐参数投影，也不是只在冲突时投影。
6. SSIM 使用 3×3 窗口和 C1=.01²、C2=.03²；仅使用有效逆向 warp 像素，未添加未披露的 auto-mask、min-reprojection 或深度一致性遮挡规则。
7. 辅助损失针对固定目标帧；匹配监督聚合该目标与本步源帧的有效对应点。BDC 直接比较原始预测深度，不引入未公开的子集尺度对齐。完整 baseline 与子集预测可能具有不同尺度，报告该限制。
8. 主损失阈值在 baseline 全帧对过滤后额外去掉最高 10% 残差，再按 S.9 估计；PC 阈值在初始化目标到所有源的 baseline warp 残差上估计并固定；BDC 每步重估。阈值 stop-gradient，下限 1e-6。论文未明确 PC 的更新频率。
9. DWA 前两次有效更新等权，之后使用未乘静态或 DWA 权重的 robust losses。分母下限 1e-12；零项通过论文的 ratio clamp 处理。
10. 50×5% warmup 离散化为向上取整 3 步。默认 AdamW betas=(.9,.999)、eps=1e-8；base 权重 FP32，backbone BF16 autocast，几何和梯度累积 FP32。训练激活 checkpoint 为 non-reentrant。
11. 原始 VGGT benchmark 预处理：最长边 504、各边四舍五入到 14 倍数、缩小 INTER_AREA / 放大 INTER_CUBIC。518 作为分辨率对照。
12. 最佳 checkpoint 必须与测量其 loss 的参数一致，因此保存 update 前模型，日志记录准确 update 数。最后一个 update 后权重保存在 last.pt，best.pt 是论文选择规则对应的已评估状态。

## DA3 benchmark

完整保留本地官方 DA3 源码快照，原路径与 SHA256 见 vendor/SNAPSHOT.json。使用其 dataset.get_data、Evaluator._sample_frames、fuse3d、eval3d 及 pose 工具。抽帧最多 100 张、random.seed(42)、排序后的采样索引，保留 ETH3D 原始过滤名单。

场景数：ETH3D 11，7Scenes 7，ScanNet++ 20，HiRoom 30，额外 DTU 22；共 90。manifest.json 固定 RGB 路径、文件尺寸与时间戳。GT 只由 prepare/evaluate 使用，训练只读取图像 manifest，永不读取 gt_meta.npz。posed 与 unposed 重建都用同一个图像预测，GT 相机只在官方 posed 融合路径替换；尺度、mask、TSDF、采样点数、阈值保留官方定义。

官方 compute_pose 未输出 AUC@1，因此额外调用相同 align_to_first_camera、se3_to_relative_pose_error、calculate_auc_np(max_threshold=1)。F1 为原始 [0,1] 单位；数据集内部 scene macro，四数据集等权。缺失/失败单列，不将部分覆盖均值称为完整复现。

## 两模型、五数据集、三 seed 队列

全量固定主方法，seed 0/1/2 均覆盖 90 场景，每模型 270 次、总共 540 次适配。场景内均值 → 跨 seed 均值及样本标准差（ddof=1），任一 seed 缺场景就不输出该数据集的完整跨 seed 统计。进程失败记录后继续；有限但退化的指标作为有效结果保留，不用 GT 筛场景或挑配置。

DTU 沿用官方 22 场景、49 帧、index 33 参考帧置首，以及原版 mask、相机尺度对齐、融合和 ObsMask/Plane 裁剪协议。GT mask/相机只用于评测。抽帧调试时，评测 get_data 的 RGB、相机和 mask 统一按保存的 manifest 重排。默认全帧时内容与官方 loader 相同。

DTU 输出 posed/unposed 的 acc、comp、overall（mm，低者好），不构造 F1，也不与四数据集 F1 平均。**保留 DA3 原始字段约定**：本地官方 `_evaluate_reconstruction` 返回 `(pred→GT, GT→pred, overall)`，但 `eval3d` 将前两项分别命名为 `comp` 与 `acc`，与通常命名方向相反；overall 始终是两项的平均。本仓库不私自改名，以便与 DA3 原输出直接对照。

每场景重新设置训练 seed，加载冻结模型和构建/读取匹配缓存用 RNG 保存恢复包裹，保证缓存命中与否不影响随后 LoRA 初始化与 dropout。配置 `rng_protocol=isolated_preparation_v2` 显式区分旧诊断版本，旧版已完成适配不移入此次全量；兼容 baseline 与匹配仍可复用。原始诊断和退化结果留在 artifacts/main，不因结果差而删除。

可选变体与旧诊断脚本保存在 configs/variants 和 scripts/diagnostics，默认全量不运行这些变体。

## 验证结果与恢复边界

- 四个首场景的预处理与原有 `BaseVGGT._load_images` 逐像素完全相同，见 artifacts/protocol_check.json。
- 实测可训练参数 18,874,368，72 个 QKV 模块；零 LoRA 与原模型的深度、置信度、相机内外参逐值相同，见 artifacts/runtime_check.json。
- 使用同一第 5 步状态分叉进行真实 GPU 恢复检查，恢复后 loss 历史完全相同，6 步末参数最大差异约 4.33e-9（验证容差 1e-7），见 artifacts/resume_fork_check/result.json。
- CUDA FlashAttention / grid_sample backward 可能存在浮点非确定性。两次独立初始化运行的 bitwise-equivalence 检查未通过，原始结果保留在 artifacts/resume_check；这不作为恢复损坏证据。固定 seed 保证采样与 RNG 序列，不能保证所有 GPU 原子归约逐位相同，重复 seed 实验用于报告实际波动。
- 为避免网络盘成为瓶颈，last.pt 每 5 步及结束时原子保存，包含 optimizer、完整五项 loss 历史、RNG 与历史最佳参数；重启删除未持久化的日志步骤后重放最多 4 步。best.pt 在完成时导出；部分训练的最佳状态在 last.pt 的 best_checkpoint 中。
- 另与本地历史 **FP32 / 100 帧 chess baseline** 对照：图像名单与顺序完全相同。当前 BF16 主干的 AUC@3 / AUC@30 / unposed F1 与 FP32 的绝对差分别约 +0.00135 / +0.00052 / +0.00103，见 artifacts/legacy_baseline_comparison.json。数值差异明确保留，不宣称混合精度与 FP32 逐位等价。


## DA3-Giant 模型适配

使用服务器已有 DA3-GIANT-1.1/model.safetensors 及同目录 config.json；这是本地可用权重版本，论文没有给出可核对的权重哈希。加载本地官方 DepthAnything3Net，不调用带 inference_mode 的 API forward，以便相机和深度损失回传。使用 safetensors 的 tied-weight 元数据恢复共享权重，严格检查所有键，不随机填补缺失参数；不分配未用的 Gaussian 分支。

QKV LoRA 覆盖 backbone 的全部 40 层，包括最初单帧层；不沿用相邻 Free-Geometry 仅选部分层、训练 camera token 的设置。新增参数 15,728,640，对应论文 Table VII 的 15.73M。VGGT 保持 72 个 QKV、18,874,368 参数。两模型其余参数均冻结。

DA3 图像处理直接调用官方 InputProcessor 的 upper_bound_resize（504）；匹配和光度损失使用原始 [0,1] RGB，ImageNet normalization 仅在 DA3 forward 入口执行。相机采用原生可微 decoder，保留原生深度/sky 后处理；ref_view_strategy=first 与 DA3 benchmark 默认一致，FAN 目标在每个训练子集中置首。没有引入 Free-Geometry 蒸馏目标。

## 运行优化及边界

- 每个模型/场景只启动一次 GPU worker，在同一冻结模型上独立重置三个 seed 的 LoRA/优化器/RNG；540 次适配只需 180 次模型加载。图像、baseline、LightGlue 对应点及确定性初始化阈值共用；训练结果分 seed 保存。
- 完整且严格加载 checkpoint 前，跳过 nn.init 的临时随机参数填充；最终权重逐项来自 checkpoint，加载前后的输出一致性用真实 baseline 验证。
- 同一模型/场景的 baseline 推理及两模式评测只做一次，将不可变结果复用于其他 seed。不跨模型复用预测或匹配，因为两模型的原生 resize 路径不同。
- 使用 BF16 backbone、FP32 heads/几何及 fused AdamW。默认仍做激活 checkpoint；configs/fast.yaml 关闭激活重算并减少 checkpoint 写盘（每 25 步），以显存与恢复重放时间换速度。50 次优化、损失、FAN 和 best checkpoint 选择完全不变。
- CPU 评测与下一个场景的 GPU 训练重叠；DTU GPU 融合与训练串行，官方 CPU 距离评测单独进程运行并与下一场景训练重叠；拆分结果已核对一致。原官方评测代码和阈值没有因提速而换成近似评测。
- fast 配置若进程被终止，最多重放 24 步；默认配置最多重放 4 步。CUDA 算子与 fused 优化器不承诺逐位确定性。
