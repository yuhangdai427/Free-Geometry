# Unified Protocol v1

这是当前唯一有效的训练协议。所有配置由 `free_geometry.config.ProtocolConfig` 定义；配置文件和命令行使用同一命名。历史报告不覆盖此文档。

## 输入与帧任务

`SceneSource(scene, image_files, frame_ids, dataset)` 只含图像信息。数据发现与 GT 指标读取分离；GT 文件是否存在不影响训练抽样。排序使用自然帧号顺序，数据集发现只识别 RGB 文件。

`FrameManifest` 是带版本和 SHA256 指纹的 JSON。保存有效 frame IDs、原始路径、坏文件原因、采样配置、τ、SIFT 调用标志、训练/probe 任务和评测帧。所有图像索引均指向验证后的有效帧列表。A/B 各有独立 `slots_a/slots_b`，消费方不能重建或猜测 slots。

随机分支直接采样 shared。Dense 分支将场景分成 `min(N,max(dense_candidates,2T−S))` 个连续桶，每桶随机取一张，再采样 shared/extras。仅去重 shared 的成员集合，不按 A∪B 去重。组内输入按 `floor(i*T/S)` 位置插入 shared，extras 按帧序填剩余槽位；共享首帧在 A/B 中保持同一参考位置。

N<T 明确跳过；组合总数不足明确失败；重试耗尽不补重复任务。合法条件下 extras 最小重叠 `max(0,2T−S−N)`。默认 10+2 shared 集合互不相同。

## 模型与预处理

`ModelAdapter` 提供 `load`、`reset_student`、`prepare_images`、`predict`、`export_eval`、`trainable_state`、`load_trainable_state`。模型按需导入。Hub 权重先固定为本地快照；本地权重逐文件计算 SHA256，恢复时核对内容。Student 每场景重新从基础权重构造，因此 LoRA 和可训练 camera token 都被重置。

`ModelOutput` 包含 frame IDs、`readouts [1,S,P,C]`、`depth/conf/valid [S,H,W]`、`ext_w2c [S,3,4|4,4]`、`centers [S,3]` 和 patch 网格。读出必须位于实际冻结预测头的原生空间；冻结参数不等于 detach Student 输入。

DA3 沿用原生 InputProcessor，但对完整场景一次性固定共同 crop。其他模型使用固定长边/patch 对齐 resize 与右下 padding；像素支持随图像保存。原生默认分辨率 Omega=416，其他=504。遮挡使用归一化零对应的颜色，不改变 token 数。所有 A/B/Student 使用同一预处理结果。保存每帧原图到模型空间的变换矩阵；GT 评测前按逆变换恢复完整图像画布，裁剪外区域保持无效，同时变换预测内参。

输入图片先保存在 CPU；Teacher cache 在 CPU 固定保存，当前任务才搬到训练设备。缓存完成后 Teacher 移到 CPU。适配器前向采用确定性 eval 行为，LoRA 仍可求导；dropout 固定为 0。

## 五项 loss

所有 Teacher 张量及 q 在 cache 准备阶段 detach。有效 Teacher A 监督缺失时拒绝任务；局部非法单元仅从该项排除。B 缺失或非法时对应 q=1，availability=0，含义是没有核查证据。

- **Feature**：每 patch `mean_channel(SmoothL1_beta1) + 2*(1−cos)`；乘 Teacher confidence patch 权重及该层该 patch 的 q，对有效 patch 平均，再对 readout 层平均。confidence 归一化到有效位置均值 1；q 不重新归一化。padding 不参与。
- **R**：`Qij=Ri Rjᵀ`，`z=||Qs−Qt||²_F`，`δ=sin(10°)`。`z≤8δ²` 直接使用 z，其余使用 `16δ*(sqrt(z/8)−δ/2)`，逐边加权平均。避免 identity residual 的 sqrt(0) 梯度。
- **T**：`bij=ti−Qij tj`，方向损失 `1−cos(bs,bt)`。Teacher baseline 必须大于 `1e−12` 且至少为有效边平均 baseline 的 `1e−3`。不会把零基线当方向证据。
- **RKD**：对全部无向边的中心距离按有效均值归一化；对每个顶点及其余点的全部二元组合计算三角余弦，避免只取前三个邻居。距离与夹角分别使用 Huber δ=0.2、各自平均后相加。S=2 时只有距离项。
- **Couple**：`(log(spread_s/mean_depth_s)−log(spread_t/mean_depth_t))²`。spread 是中心去均值后的 RMS；固定 Teacher A 的有限正深度、真实像素区域和 confidence 下 5% 排除后的支持。两端使用完全相同像素。空支持、退化 spread 或 Student 支持内非法深度均显式失败，不能移除 Student 坏像素获得更低损失。

`LossBundle` 区分 raw、weighted、contributions、counts。只有五个顶层权重参与求和；RKD 距离/夹角日志不会被再次计入 total。

## A/B 权重

`q=1/(1+(d/threshold)^2)`，缓存前固定；不使用场景中位数、不读取 Student 残差、不变更训练任务。

| 项 | d | 阈值 |
|---|---|---|
| Feature | 对应 readout patch 的 1−cos | 0.1 |
| R | 相对旋转角差 | 20° |
| T | 相对平移方向角差 | 20° |
| RKD distance | 两边归一化中心距离差绝对值 | 0.2 |
| RKD angle | 对应三角余弦差绝对值 | 0.2 |
| Couple | 相同 A 支持上的 log(spread/mean_depth) 差绝对值 | log(1.25) |

数值舍入误差被视为零分歧。阈值是明确的工程初值，不是已验证最优值；大上下文引起的正常 Feature 改变也可能被降权，因此保留关闭每项 q 的消融。N=8 同上下文 q=1，但 `ab_informative=false`。

## 优化、probe 和 checkpoint

一个循环、一个 total.backward、一个全局裁剪。基础学习率 3e−5，weight decay 1e−5，15% warmup（起始为基础 lr 的 1%）后 cosine 到 1e−8；极短测试计划也不会生成零长度调度器。成功更新才增加 step；访问顺序与 mask 使用独立的确定性 seed。

Probe 每任务两张固定 mask，step 0、每 20 次更新及最终步执行。保存/恢复所有 RNG 和模型模式；无反传。默认不选择模型。probe 使用相同五项公式和基础权重，q=1，不把角度数值和 loss 混加。

显式 `probe.decision=true` 时，step≥20 的最低有效 score 为 selected，精确平分选更早一步。显著改善阈值 0.5%；连续三次不显著改善、step≥40 时停止。非法 probe 不可成为候选；无合法候选报错；step 0 不是自动回退目标。最终和 selected 均保留。

Checkpoint 保存全部可训练参数，以及 optimizer、AMP、RNG、epoch/cursor、policy、计划进度、模型身份、配置和 manifest 指纹。学习率由已完成更新数与原配置恢复。恢复使用原运行目录；恢复旧 checkpoint 时截掉被替代的日志尾部，已早停的运行不会重新执行更新。保存采用临时文件 + 原子替换。

## DVLT 解算

原生 ray grid 使用 `align_corners=True` 下采样。训练用归一化齐次 DLT，求 9×9 normal matrix 的最小特征向量，三轮角残差软加权（阈值 0.2 rad），通过 QR、符号修正得到 proper c2w rotation 和内参；中心来自 ray origins 的均值，转成 w2c。

最小特征向量使用只依赖目标特征值间隔的解析 backward，避免其他奇异值重复造成无关梯度奇异。秩不足、零射线和非法旋转不返回伪 identity。关键求解用 double，输出保持可微；不使用 RANSAC 或 Student detach。

标准评测仍走官方 RANSAC，导出训练解算与官方旋转差异。合成正确性不代表真实有噪射线上的效果已验证；必须完成真实权重 GPU 验收。
