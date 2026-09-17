# TTA 协议迁移执行计划：VGGT-Ω / Pi3 / Déjà View (DVLT)

- 日期：2026-09-17（今晚执行）
- 基线：仓库 `main` @ `19eae1a`
- 范围：将现有 DA3 GT-free per-scene TTA 协议（`protocol_v1.py` / `train_da3_protocol.py`）迁移到三个新模型：VGGT-Omega（retrained `vggt_omega_1b_416_reproduce.pt`）、原始 π³（非 Pi3X）、NVIDIA DVLT
- 说明：以下为源码层面的迁移方案。模块结构为已核实事实；LoRA 安插位置与读出选择为迁移建议，推荐配置的收益仍需验证。

**总体建议：先迁移 VGGT-Ω，再迁移 Pi3，最后迁移 Déjà View。统一的是"自监督任务和 loss 接口"，不是层号，也不是给所有模型硬塞同一个 LayerNorm。**

另外，有两个值得在大规模实验前解决的问题：**现有 `rel` loss 的坐标约定不一致，以及 DVLT 官方推理/位姿后处理中的断梯度操作**。这两个问题会直接影响迁移是否真的在训练目标模块。

---

## 一、先给你一张直接用于迁移的表

下面的 LoRA 路径都相对于各模型的原始核心模块，不包含外层 PEFT wrapper 前缀。

| 模型                   | 第一版 LoRA 安插位置                                                                | 用来计算 patch loss 的特征                                 | LayerNorm 如何处理                                                           |
| -------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------ |
| **VGGT-Ω**           | `aggregator.frame_blocks.{i}` 和 `aggregator.inter_frame_blocks.{i}`，`i=0…23` | 聚合器第 `[4,11,17,23]` 层的 local/global 拼接特征，去除特殊 token | **直接使用已有的 `dense_head.norm`**，与当前 VGGT 方案最接近                             |
| **Pi3，默认 large**     | 主干 `decoder.{i}`，`i=0…35`；不是三个任务分支的 decoder                                  | 主干最后两层 `[34,35]` 拼接后的特征，进入 `point_decoder` 的读出空间    | 没有对应 DPT 的共享入口 LN；建议取 **`point_decoder.projects → blocks[0].norm1`** 的输出 |
| **Déjà View / DVLT** | `recurrent_blocks.0.frame_attn` 和 `.global_attn`，同一组 LoRA 随循环重复使用            | **最后一次循环**的状态经过冻结的 depth/ray decoder 后的 patch 特征    | 分别取已有的 **`depth_decoder.norm`、`ray_decoder.norm` 输出**，不要将它们直接套在主干状态上     |

VGGT-Ω 的四个 tap 和 `dense_head.norm` 是官方前向的实际结构；Pi3 的 `[34,35]` 是其默认 36 层主干实际拼接的位置；DVLT 的 decoder LN 则位于分支内的 transformer blocks 之后。上表的**模块结构是已核实事实，选择这些位置作为蒸馏目标是我的迁移建议**。

三个模型第一版都可以先使用这些子模块：

```text
attn.qkv
attn.proj
mlp.fc1
mlp.fc2
```

建议先沿用你现在的 `r=32, alpha=32, dropout=0`，但**限定在表中的主干范围内**，冻结图像编码器、预测头、原有 LN 和特殊 token。不要全模型按 `"qkv"`、`"proj"` 后缀匹配，否则很容易把单帧编码器和预测头也一起改了。这个新模型初始化策略不要求你回头修改当前 DA3 已有的 camera-token 配置。

---

## 二、你今天这版代码，迁移时真正应该保留什么

### 1. 现在的任务是"少视角 + 图像遮挡"，不是 MAE 式删除 token

你的 DA3 实现会在输入图像上按 patch 网格随机置零，teacher 看干净图像，student 看被遮挡的共享帧；之后在四个 head tap 的 LN 特征上，**只对被遮挡位置计算 Huber + cosine loss**。代码中的 50% 是独立随机采样的期望比例，并非每帧严格选中一半 patch。

因此，新模型应保留这样的任务：

```text
同一个 backbone 的冻结 teacher：
N 个干净视角 → teacher 的共享帧特征

同一个 backbone 的 LoRA student：
M 个共享视角，图像 patch 遮挡 50% → student 特征

在相同 frame ID、相同 patch 坐标处计算蒸馏 loss
```

这里不是让 VGGT 去教 Pi3。**每个新模型都用自己的原始 checkpoint 作为 teacher，再适配它自己的 student。**

还有一个重要区别：不能为了省计算，先缓存干净图像的 DINO 特征，再把其中一半 feature 置零，声称等价于现在的 image masking。那样改成了**编码后 feature masking**，应该单独作为实验，而不是静默替换现有任务。

### 2. "Post-LN 蒸馏"不等于修改 transformer 的 Pre-LN / Post-LN 结构

你现在调用的是已有的、冻结的 head LayerNorm。例如 VGGT 的：

```text
aggregator feature
    → 去掉特殊 token
    → depth_head.norm
    → loss
```

它不是新训练一个 LN 来吸收误差，也不是把每个 transformer block 改成 Post-LN。`modeling.py` 中的 `to_norm()` 就是在复用预测头已有的 norm。

**迁移原则：不改变原始预测路径，只导出适合比较的特征。**

这也意味着统一接口最好叫 `distill_readouts()`，而不只是 `head_norm()`：Pi3 所需的读出操作是"投影 + 原有 LN"，DVLT 则需要经过一小段冻结 decoder。

---

## 三、VGGT-Ω：最接近现有实现，应该最先做

### 3.1 它与当前 VGGT 的主要结构差异

官方实现仍然有 24 组 frame/inter-frame attention，并在 `[4,11,17,23]` 缓存拼接后的 2048 维特征。但迁移时有几个实际变化：

| 项目              | VGGT-Ω 中的实际情况                                     |
| --------------- | ------------------------------------------------- |
| 跨帧模块名           | `inter_frame_blocks`，不是 `global_blocks`           |
| Patch size      | **16**                                            |
| 特殊 token 前缀     | 1 个 camera + 16 个 register，`patch_token_start=17` |
| 深度头名称           | **`dense_head`**                                  |
| Camera head 返回值 | 直接返回 pose tensor，不是迭代预测列表                         |
| 聚合器返回列表         | 保留绝对层号；未缓存的层是 `None`                              |

另外，第 `[2,6,9,14,20]` 个 inter-frame block 只混合 camera/register token，patch token 在该次 inter-frame 更新中旁路通过。不能把所有 inter-frame block 都理解成普通的全 patch global attention。

### 3.2 LoRA 具体放哪里

建议第一版覆盖：

```text
aggregator.frame_blocks.0…23
    .attn.qkv
    .attn.proj
    .mlp.fc1
    .mlp.fc2

aggregator.inter_frame_blocks.0…23
    .attn.qkv
    .attn.proj
    .mlp.fc1
    .mlp.fc2
```

这些 attention/MLP 子模块名在官方实现中存在。

**一个容易忽略的细节：**Ω 的 `qkv` 不是完全普通的 Linear，而是带有 key-bias masking 的 `LinearKMaskedBias`。包装 LoRA 时应保留原模块的前向行为：

```text
output = original_module(x) + LoRA_delta(x)
```

不要仅复制它的 `weight/bias`，然后重建成普通 Linear，把原来的 bias mask 丢掉。官方 attention 会实际调用 `self.qkv(x)`，因此可以在这个调用边界接入适配器；但零 LoRA 的前向一致性必须测试。

### 3.3 Loss 与 LN 的具体位置

直接沿用四层方案：

```python
# 结构示意；feats 是聚合器保留绝对层号的输出列表。
for layer in [4, 11, 17, 23]:
    patch_features = feats[layer][:, :, patch_token_start:, :]
    readout = model.dense_head.norm(patch_features.float())
    # 在 readout 上计算共享帧、masked patch 的 loss。
```

这里**只经过一次原有 LN**。官方 dense head 本来就是先去掉特殊 token，再做 LN、投影和多尺度融合，所以这个位置与现有 VGGT 最容易建立对应关系。

相机监督还要注意：Ω 的 camera head 会联合处理**所有输入帧的 camera/register token**。所以 teacher 应当先用完整 N 帧跑 camera head，再抽取共享帧的 pose；不能先抽成 M 帧再跑，否则 teacher 的相机目标也变成了短上下文预测。

**建议首轮配置：**8→4、50% image masking、四层 native-LN maskdistill；跑通后分别加入 RKD/couple 和修正后的 rel。

---

## 四、Pi3：不能照搬"四层 DPT"，也没有同样的 camera token

### 4.1 实际结构

原始 Pi3 默认 large 模型的结构是：

```text
DINOv2 单帧编码器
    ↓
36 个主干 decoder blocks
    偶数层：frame attention
    奇数层：global attention
    ↓
拼接 h34 和 h35
    ↓
 ┌─────────────────┬─────────────────┬──────────────────┐
 point_decoder      conf_decoder      camera_decoder
    ↓                   ↓                  ↓
 point_head          conf_head          camera_head
```

注意这里的 36 层是 **18 对 local/global block**，不是 36 对。它在主干前加了 5 个 register token，但没有 VGGT 那样独立的 `camera_token` 参数；相机分支最终读取的是 patch 特征。

### 4.2 LoRA 具体放哪里

第一版只放在：

```text
decoder.0…35
    .attn.qkv
    .attn.proj
    .mlp.fc1
    .mlp.fc2
```

先冻结：

```text
encoder
register_token
point_decoder / point_head
conf_decoder / conf_head
camera_decoder / camera_head
```

这里特别容易混淆：**主干叫 `decoder`，任务分支也带 decoder 名字。第一版只训练前者的 LoRA。**

这样你测试的是"适配多视角融合表示"，而不是同时微调几何读出器。冻结分支参数不代表在 student 上给分支加 `no_grad()`；梯度仍然需要穿过冻结分支回到主干。

### 4.3 Loss 放在哪里？LN 放在哪里？

Pi3 的 `TransformerDecoder` 实际执行的是：

```text
projects
    → 多个 transformer blocks
    → linear_out
```

**没有与你的 DPT 对应的共享入口 LN，也没有这个类自身的最终 LN。**因此不能写一个不存在的 `point_decoder.norm`，也不能把 VGGT 的 LN 参数搬过来。

我建议第一版采用下面的读出：

$$
h=\operatorname{concat}(h_{34},h_{35}),
\qquad
\phi_{\mathrm{Pi3}}(h)
=
\texttt{point\_decoder.blocks[0].norm1}
\bigl(\texttt{point\_decoder.projects}(h)\bigr).
$$

然后去掉前 5 个 register token，计算 masked patch loss。

这样做的理由是：**比较的是点图分支真实使用的、经已有投影与归一化的特征，而不是随便挑一个中间层。**但严格命名应当是"projected + native-LN readout"，不要称其为原模型已有的"DPT head-input post-LN"。

Pi3 的归一化位置值得做一个很小的二选一实验：

| 方案        | 实现                                                | 用途              |
| --------- | ------------------------------------------------- | --------------- |
| **建议主方案** | `projects → blocks[0].norm1`                      | 使用点图分支真实的投影与 LN |
| 对照        | 对 `concat(h34,h35)` 做仅用于 loss 的无参数 `F.layer_norm` | 检查分支投影是否真正有帮助   |

第二种的 LN **只出现在 loss 分支**，不进入原始预测路径，也不训练 affine 参数。

### 4.4 几何输出如何接入现有 loss

Pi3 官方输出可以明确转换为：

```python
depth = outputs["local_points"][..., 2]
centers = outputs["camera_poses"][..., :3, 3]  # c2w 的平移就是相机中心
w2c = inverse(outputs["camera_poses"])
confidence = sigmoid(outputs["conf"][..., 0])
```

官方说明明确 `camera_poses` 是 c2w，`conf` 是需要 sigmoid 的原始 logits；不能直接把 raw confidence 当非负 loss 权重。

由此，**RKD/couple 很容易接入，rel 则必须先统一坐标约定**。不建议把 camera-token KD 作为 Pi3 的默认迁移组件——它本来就没有对应的独立 token，而且你的研究笔记也记录了该类监督在现有模型上的 AUC/F1 取舍问题。

---

## 五、Déjà View：LoRA 共享、循环次数固定，并绕开两个断梯度点

### 5.1 LoRA 应该共享，而不是每个循环一份

DVLT 的核心是同一个模块循环执行：

```text
DINOv2 features
    ↓
[frame attention → global attention] × K
    ↓
最终状态 zK
    ↓
ray decoder / depth decoder
```

在共享模式下，实际主干模块就是 `recurrent_blocks[0]`，其内部有 `frame_attn` 和 `global_attn`。

因此 LoRA 放在：

```text
recurrent_blocks.0.frame_attn
    .attn.qkv / .attn.proj / .mlp.fc1 / .mlp.fc2

recurrent_blocks.0.global_attn
    .attn.qkv / .attn.proj / .mlp.fc1 / .mlp.fc2
```

**不要展开成 K 套 LoRA。**第一版也不要训练时间条件模块、depth-scaling gates 或特殊 token。

同样的 rank 在 DVLT 上对应的独立可训练参数会少很多，所以实验应报告**实际独立 LoRA 参数量**，不能只报告 rank 就认为训练容量相同。

### 5.2 Teacher/student 第一版使用相同 K

建议：

$$
\text{teacher}: N\text{ views},K\text{ loops};
\qquad
\text{student}: M\text{ views},K\text{ loops}.
$$

K 从所选 checkpoint 的评测配置读取，两端完全一致。

第一轮不要同时做 `teacher K 大、student K 小`，否则性能变化同时包含"更多视角监督"和"更多循环监督"，难以判断到底是哪一个起作用。

此外，DVLT 的训练路径会采样循环次数，而推理路径使用固定次数。**仅仅调用 `student.train()`，可能就把你以为固定的计算图改成了随机循环训练。**

### 5.3 第一个断梯度点：`forward_inference()`

官方 `forward_inference()` 带有 `@torch.no_grad()`，而模型处于 eval 模式时，常规 `forward()` 会走这个函数。

所以这两种做法都不够：

```text
直接使用 forward_inference 做 student forward
仅设置 model.eval()，以为仍能正常反传
```

建议新增 **`forward_adapt()`**：

* 复用官方 `forward_inference()` 的前向步骤；
* 不带 `no_grad` 装饰；
* 使用固定 K 的 `_solve_inference()`；
* 保持 dropout/drop-path 关闭；
* 正常保留主干及冻结预测分支中的 autograd。

这里是"使用推理时的确定性计算图进行适配"，不是"在无梯度推理接口上训练"。

### 5.4 Patch loss 与 LN 的位置

我建议只监督最后一次循环的状态，不在第一轮加入每个循环的中间监督：

```text
zK
 ├─ depth_decoder.proj_in
 │    → depth_decoder.blocks
 │    → depth_decoder.norm       ← depth 特征 loss
 │
 └─ ray_decoder.proj_in
      → ray_decoder.blocks
      → ray_decoder.norm         ← ray 特征 loss
```

两条分支都去掉特殊 token，只在 masked patch 上计算 loss，再对两条分支取平均。

官方这两个 LN 位于各自分支 transformer 之后，维度是 decoder 的通道数，不是主干通道数。因此**不能直接调用 `depth_decoder.norm(zK)`**。应该导出真实前向经过该 norm 的输出。

这能让同一组主干 LoRA 同时收到 depth 和 ray 两个实际几何读出方向的监督；是否优于仅监督 depth，作为后续小消融验证。

### 5.5 第二个断梯度点：ray → pose，但 RKD/couple 可以直接解决

DVLT 官方评测后处理通过 `rays_to_pose()` 得到相机，其中包含 RANSAC，并明确使用了 `detach()` 和 `no_grad()`。不能直接把其输出接到 student 的 pose loss，然后认为该 loss 会训练主干。

**不过，RKD/couple 不需要完整 pose，它们只需要相机中心。**

官方相机中心就是下采样 ray origin 的加权平均：

$$
C_i
=
\frac{\sum_p w_{ip}\,o_{ip}}
     {\sum_p w_{ip}+\epsilon}.
$$

因此可以沿用官方的下采样方式和权重定义，**只将"求相机中心"这段单独写成可微操作**，student 端不 detach ray origins。默认官方后处理不使用 depth confidence 加权时，直接使用均匀权重即可。这样前向数值可以与官方相机中心对齐，同时梯度能够回到 ray decoder 和 LoRA。

这使得 DVLT 可以自然使用：

```text
maskdistill + camera-center RKD + center/depth couple
```

**完整 `rel` 暂时不要机械照搬。**它还需要可微的相机旋转路径。官方虽然有额外的 camera MLP 分支，但用它训练、用 ray-fitting 评测是两条不同路径，必须明确区分，不能默认二者等价。

---

## 六、Loss 迁移前，建议先修正的坐标与统计问题

### 6.1 现有 `rel` 的输入与公式约定不一致

这是我检查代码时发现的最重要的 loss 问题。

`protocol_v1.py` 的 `loss_pose_rel()` 标明输入是 **w2c**，调用链也确实通过 `decode_pose_w2c()` 提供 w2c；但内部计算使用的是：

$$
R_i^\top R_j,\qquad R_i^\top(t_j-t_i).
$$

这组形式对应把输入当成 c2w，而不是 w2c。

若约定：

$$
x_i=R_iX_w+t_i,
$$

那么从相机 \(i\) 到相机 \(j\) 的正确变换是：

$$
T_{j\leftarrow i}=T_jT_i^{-1},
$$

即：

$$
R_{j\leftarrow i}=R_jR_i^\top,\qquad
t_{j\leftarrow i}=t_j-R_jR_i^\top t_i.
$$

我另外做了一个独立 NumPy 单测：给同一组相机更换世界坐标系，旧公式产生了非零 teacher/student discrepancy，而上述 w2c 公式接近数值零。这个测试验证的是公式，不是完整训练效果。

**建议保留历史实现为 `rel_legacy`，新实现叫 `rel_w2c_v2`。**不要直接覆盖后再将新结果与旧 `maskrel` 档案混为一谈。

### 6.2 RKD/couple 应改成"相机中心接口"

建议将核心函数改成：

```text
loss_rkd_centers(centers_s, centers_t, ...)
loss_couple(centers_s, depth_s, centers_t, depth_t, ...)
```

然后由 adapter 负责产生 centers：

| 模型     | centers 的来源                          |
| ------ | ------------------------------------ |
| VGGT-Ω | 解码 w2c 后计算 `−Rᵀt`                    |
| Pi3    | c2w 的平移列                             |
| DVLT   | 与官方后处理对齐的 ray-origin 均值，student 保留梯度 |

这样 loss 不再承担各模型的坐标转换，也不用为了 DVLT 人造一份不可微的完整 pose。

另一个小问题是：你当前 `couple` 对 teacher 的深度统计使用了 confidence 分位数过滤，而 student 侧没有使用相同过滤。建议新版本让两边在**同一个 teacher-derived 有效像素集合**上统计 mean depth，避免仅仅因为统计区域不同就产生误差。这个修改也应单独版本化。

### 6.3 第一轮不要把所有 loss 全部叠上去

建议按照下面的实验顺序：

| 配置                           | 用途                      | 首轮适用                      |
| ---------------------------- | ----------------------- | ------------------------- |
| **M：只有 maskdistill**         | 验证 LoRA、读出位置、mask 与梯度链路 | 三个模型                      |
| **M + 1.5 RKD + 1.0 couple** | 迁移你现有 RKDC1H 的核心思想      | 三个模型；DVLT 用可微 ray centers |
| **M + rel_w2c_v2**           | 检查相对位姿监督是否额外有效          | 先 Ω、Pi3；DVLT 后续单独接旋转路径    |

权重 `1.5/1.0` 是沿用当前配置的**起点**，不是已经证明适用于新模型的最优权重。

你的统一协议笔记已经记录：`rel`、camera-token KD 和多种 GT-free 门控信号的效果并不跨场景稳定。因此我不建议直接把"Ω 应该用 maskrel，Pi3 应该用 RKD"写成确定结论，也不建议以"训练 loss 降了"作为几何一定改善的接受条件。

---

## 七、Mask 和选帧：第一轮尽量保持任务不变

### 7.1 50% mask 保留，但这些参数必须显式化

建议把 masking 抽成统一组件，至少记录：

```text
mask_ratio
mask_space = image
patch_size
fill_value / normalization_space
mask_seed
```

**Patch size 从模型读取。**Ω 是 16，Pi3/DVLT 是 14；不能把你目前 DA3 的 `504 + patch14` 配置直接用于 Ω。共享图像应先完成一致的 resize/crop，再从 teacher 输入张量抽出 student 子集，之后只给 student 做 mask。

填充值也要统一定义：当前 DA3 是在**已经 ImageNet-normalized 的图像**上置零，代表原始 RGB 空间的均值颜色；三个新模型则在内部执行归一化，输入 RGB 直接置零代表黑色。两者不是同一种遮挡。

我的建议是：新统一协议明确采用 **normalized-zero masking**，对应输入 RGB 填 ImageNet mean；同时保留旧配置用于复现，不把填充值变化隐藏在模型迁移里。

### 7.2 第一轮固定 8→4，不同时扩展所有采样策略

建议先用同一份固定 manifest：

```text
teacher：8 个视角
student：其中 4 个共享视角
teacher_slots：[0, 2, 4, 6]
image masking：50%
```

这样三个模型的差异主要来自架构和读出位置，而不是训练任务也随模型改变。

对于 Ω 和 DVLT，还应确保 **teacher 的第一帧就是 student 的第一帧**，因为它们存在 first-view 与其他视角不同的特殊 token 处理。Pi3 不依赖这种参考帧，但仍必须按照 frame ID 正确 gather，不能假设"第几个 tensor 就一定对应同一张图"。

选帧层可以先保留你目前的 SIFT/τ 逻辑。一个实现细节是：DA3 当前低 τ 分支实际执行的是 `rng.sample(range(N), teacher_N)`，属于全场景随机抽样，不是严格连续滑窗。迁移时应以代码与冻结 manifest 为准，不按文档里的 "window" 名称重新实现一套。

**暂时不自动启用 endpoint-anchored、two-stage 和 ratio-mix。**你的笔记已经记录这些策略在稀疏场景和长连续视频上的行为不同，不能因为换模型就默认一起打开。

等 8→4 跑通，再扩展到 16→4 或 24→8。届时还要泛化 RKD 的三角形枚举：当前 DA3 实现在每个顶点只使用 `rest` 中前三个相机生成角度项，4-view 时合理覆盖，但 student 变成 8-view 后并不是全体三角关系。

### 7.3 第一版不用额外增加一次 clean-student 前向

你当前 DA3 的 RKDC1H 是从**同一次 masked-student forward**获得特征、相机和深度，再计算各项 loss。

建议首轮维持这一点。后续如果发现 pose loss 对遮挡过于敏感，再单独比较：

```text
masked forward → feature loss
clean forward  → geometry loss
```

这是一个有意义的实验，但它增加计算量，也改变任务定义，不应该混入最初的架构迁移。

---

## 八、代码应该怎样组织，才不会继续复制三套训练器

我不建议把 `protocol_v1.py` 再复制三份。可以新建一个模型中立的核心：

```text
src/free_geometry/
├── adapters/
│   ├── base.py
│   ├── vggt_omega.py
│   ├── pi3.py
│   └── dvlt.py
├── losses/
│   ├── masked_feature.py
│   ├── camera_relation.py
│   └── gauge_coupling.py
├── masking.py
├── sampling.py
├── teacher_cache.py
└── trainer.py
```

每个 adapter 只负责几个明确职责：

```text
preprocess()
inject_lora()
forward_with_readouts()
extract_camera_centers()
extract_differentiable_pose()  # 不支持时明确标记，不能返回假梯度
export_for_evaluation()
```

`forward_with_readouts()` 的统一输出可以是：

```text
readouts:      dict[str, Tensor[B, V, P, C]]
depth:         Tensor[B, V, H, W]
confidence:    非负 teacher 权重所需的置信度
centers:       Tensor[B, V, 3]，student 必须可微
w2c:           可选；并记录是否可微、来自哪个分支
view_ids:      原始帧 ID
patch_hw:      patch 网格
```

**`readouts` 的数量不要求都是 4。**Ω 可以是四层，Pi3 是一个 point-branch 读出，DVLT 是 depth/ray 两个读出；统一 loss 对读出取平均即可，不要为凑四层而监督没有对应预测用途的位置。

Teacher cache 至少绑定模型/权重版本、预处理、按顺序排列的 frame IDs、共享映射、读出定义，以及 DVLT 的 K。缓存 teacher 的共享帧目标即可；不要复用会随着 LoRA 更新而失效的 student 特征。

另外，Pi3 原始输出没有直接给出统一评测器可能期待的 intrinsics。若你的 TSDF 路径需要 K，必须显式接入并验证内参估计；不能简单改几个输出字段名就认为评测链已经等价。

---

## 九、我会怎样安排最小实验，避免一开始就扫完整矩阵

### 第一阶段：先做"迁移是否正确"，不看涨点

至少通过下面这些检查：

| 检查              | 通过条件                                            |
| --------------- | ----------------------------------------------- |
| **零 LoRA 一致性**  | 无 mask 时，新 adapter 与官方原始前向输出在数值容差内一致            |
| **梯度链路**        | 单次 backward 能到 LoRA；冻结头和原有 LN 参数不更新             |
| **视角/patch 对齐** | 同一共享图像预处理完全一致，masked 位置与 loss 索引一致，特殊 token 不混入 |
| **几何单测**        | w2c/c2w 转换正确；RKD/couple 的预期尺度、坐标不变性成立           |
| **DVLT 专项**     | LoRA 独立模块数量不随 K 增长；centers 与官方后处理数值对齐且保留梯度      |

注意初始化时 LoRA-B 为零，不能要求第一次 backward 中 LoRA-A 和 LoRA-B 的梯度都非零；应检查整体路径和后续更新，而不是写错误的断言。

### 第二阶段：小规模比较 loss 和读出

使用一份预先固定的开发场景集合，覆盖稀疏图像和连续视频，而不是只选看起来会改善的场景。

* **Ω：**比较 M、M+RKD/couple、M+修正 rel。
* **Pi3：**先确定两个候选读出哪个更合理，再比较上述 loss，避免同时扫 LoRA、LN、loss 三个维度。
* **DVLT：**先比较 M 和 M+RKD/couple；完整 rel 暂不作为首轮必要项。

训练起点可以沿用你现有的 **100 steps、LR `3e-5`、warmup 15%、WD `1e-5`、clip 1.0**，但将其视为待验证的统一起点。

### 第三阶段：再扩展到四数据集与长序列评测

最终评测沿用你这版仓库的 **≥100 帧时评测 100 views，否则 all views**，不要迁移时又退回只报告训练用 4 帧的结果。记录 adaptation/evaluation 的帧交集，并保持与已声明协议一致。

除均值外，重点看每场景涨跌、最差一部分场景的退化幅度，以及新增前向/读出带来的显存和时间开销。你目前真正需要的是"迁移后稳定改善"，而不只是三个模型都能把 masked feature loss 降下来。

---

**最后归纳成第一版实施决策：**

**VGGT-Ω 直接复用四层 native head-LN 蒸馏；Pi3 改为最后两层拼接后的 point-branch 投影/LN 读出；DVLT 使用共享主干 LoRA、固定循环次数，以及最终 depth/ray 分支 LN 读出。三个模型先跑通 maskdistill，再以统一的 camera-center 接口加入 RKD/couple；现有 rel 先修正坐标约定，DVLT 的完整相对位姿监督另行处理。**

这条路线改动边界清楚，也能把"方法不适合该模型"与"接错层、mask 不一致、梯度被切断"区分开。
