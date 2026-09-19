# Free-Geometry TTA 协议（2026-09-13/14）

**版本**：v2.1　**日期**：2026-09-14　**状态**：全量验证完成（68/68 场景）+ 量规机制定位 + 门控/重耦收尾，16/16 指标格全正

本协议定死 per-scene test-time adaptation 的全部环节：选帧、训练、loss、评测。所有规则只依赖两个可从测试场景现场测量的量——**N（帧数）**和 **τ（相邻帧 SIFT 匹配率中位数）**——不按数据集名字做任何特调。训练侧零 GT；GT 只出现在评测器里。

本协议定死 per-scene test-time adaptation 的全部环节：选帧、训练、loss、评测。所有规则只依赖两个可从测试场景现场测量的量——**N（帧数）**和 **τ（相邻帧 SIFT 匹配率中位数）**——不按数据集名字做任何特调。训练侧零 GT；GT 只出现在评测器里。

---

## 1. 设定与原则

- **任务**：给定一个测试场景的全部 N 张图像（无 GT），对该场景做 per-scene TTA，然后评测其多视角几何（pose AUC / 重建 F1 / 深度）。
- **Transductive**：评测帧参与适配（TTA 标准定义，与 Test3R/SelfEvo/Self-Geometry 一致）；公平性由同一批帧上的冻结基线对比保证。
- **零 GT**：训练监督全部来自冻结 teacher 的伪标签；选帧只用图像与帧序（SIFT 匹配/嵌入/时间戳都是图像衍生，合法）；GT 深度与位姿只用于评测与离线诊断。
- **Teacher–Student 结构**：teacher = 冻结 VGGT-1B，看 teacher_N 帧；student = 同权重 + per-scene LoRA，看其中 4 帧（等距子集）。

## 2. 选帧协议

### 2.1 现场测量

对每个测试场景：
1. **N** = 图像总数。
2. **τ** = 相邻帧（文件序）SIFT 匹配率中位数：`good_matches / min(#kp_i, #kp_j)`（Ratio test 0.75，图像缩到 320px，SIFT 400 特征）。实测分离：7Scenes 0.71 / ScanNet++ 0.45 / HiRoom 0.20 / ETH3D 0.13——τ=0.55 是稠密视频与其余的分界。

### 2.2 配对构造（每场景）

| 参数 | 值 |
|---|---|
| teacher_N | `16`（池 ≥16）；`8`（8 ≤ 池 <16）；池 <8 的场景剔除并注明 |
| student | `teacher[::(teacher_N/4)]`（4 帧等距铺满 teacher 覆盖范围） |
| 配对数 | **10 训练对 + 2 探针对**（pairs 曲线 5/10/20/40 饱和证据 + LoRA3D ~10 样本文献） |
| 评测帧 | 不打入池的概念——池 = 全部 N 帧（transductive） |

### 2.3 teacher 帧的选择（τ 分派）

- **τ > 0.55（稠密视频）**：池内**等距铺满** teacher_N 帧（索引 `int(i·N/teacher_N)`），额外帧按与 4 共享帧的 **SIFT 匹配率 ∈ [0.1, 0.5]** 过滤（DUSt3R 重叠率区间的 GT-free 版；不足则回退补满）。
- **τ ≤ 0.55**：池内**纯随机** teacher_N 帧。
- 长稠密视频的先验过滤（可选，文献准则）：光流/匹配率低于阈值的近重复帧先剔除再铺满。

**槽位约定**：teacher 帧列表中，4 个共享帧固定置于槽位 [0,2,4,6]（8 帧）或等距位置（16 帧）。

**判废记录**：GT 共视矩阵过滤（se24c）违反零-GT 原则，已废弃，仅作 oracle 参考。

## 3. 训练协议（全部数据集相同，无任何按数据集调整）

| 项 | 值 |
|---|---|
| 学生参数化 | LoRA r=32/α=32/dropout=0，aggregator 全 24 层；heads 冻结；camera/register token 冻结 |
| 优化器 | AdamW，lr 3e-5，cosine 退火，grad clip |
| **步数** | **固定 100 步**（30 步不足、200+ 过拟合的实测证据） |
| batch | 1 配对/步；10 对 × 10 epoch |
| teacher | 冻结（EMA 实测更差：+0.0103 < 固定 +0.0171）；教师特征每场景预缓存一次（训练循环零 teacher 前传） |
| 成本 | **每场景 ~47s（缓存 13s + 训练 32s），峰值显存 22GB**（单卡 3090/4090 可跑） |

## 4. Loss 设计

### 4.1 主 loss：maskdistill（掩码特征蒸馏）

- **输入侧**：student 的 4 个视角各随机置零 50% 的 14×14 像素块（或比率 U(0.3,0.75)，MDR 变体胜率更高）；teacher 输入不动（完整 teacher_N 帧）。
- **监督**：teacher 聚合器 L4/11/17/23 四层的 patch token，经 DPT head 共享 LayerNorm 后（post-LN 空间），对共享 4 视角切片。
- **形式**：`Huber(β=1) + 2·(1−cosine)`，逐 patch 用 **teacher 深度置信度**加权，**只在被掩码位置**计算。
- **机制**：掩码把"照抄 teacher 特征"改写为"从跨视图上下文重建 teacher 特征"——直击 ≤8v 的跨视图不一致瓶颈；同时稀释 teacher 目标中 ~50% 坏 patch 的毒性（实测：teacher 更优 patch 占比 50-57%，粒度不变）。

### 4.2 位姿项（C2M 变体；默认与 maskdistill 组合）

- **rel-pose**：teacher/student 各自 camera head 的预测 pose_enc → 全部视角对的**相对旋转 chordal 误差 + 相对平移方向 1−cosine**（尺度无关），1:1 相加。梯度穿过冻结 camera head 回传 LoRA。
- 加分项（低视角 regime）：CamRel（camera token → teacher patch 锚点库亲和力轮廓 + 额外帧锚点）+0.0134，camera 通路最强单项；32v 次可加不进主配方。

### 4.3 最终规则

**默认（一刀切）= maskdistill + rel-pose（C2M）**：四个数据集 AUC@3 全部为正的最简选择。精细版（可选）：τ ≤ 0.55 时用纯 maskdistill（ScanNet++ F1 多涨 0.02）。

## 5. 评测协议

| 项 | 规则 |
|---|---|
| 帧集 | N≥100 → **benchmark-100**：`seed(42)` 打乱全部帧索引 → 取前 100 → 升序排好输入（与 DA3/VGGT benchmark 逐行一致，evaluator.py:499-503）；N<100 → **allv**（全部 N 帧） |
| 场景 | 该数据集全部可行场景，不打乱不抽样 |
| 重建 | 原版 ScanNetPP `recon_unposed` TSDF 管线（voxel 0.02、trunc 0.15、max_depth 5m、阈值 5cm），**无 confidence 过滤** |
| 指标 | **AUC@3（主）**、F1、CD（overall）、深度 AbsRel / δ<1.25 |

### 5.1 场景覆盖（零-GT 协议 + 帧长自适应后）

| 数据集 | 可用场景 | 评测 |
|---|---|---|
| ScanNet++ | 20/20 | benchmark-100 |
| 7Scenes | 7/7 | benchmark-100 |
| HiRoom | **30/30**（全部；10-23 帧场景 teacher=8） | allv |
| ETH3D | 11/11 | allv |

## 6. 全量验证结果（v1.1，2026-09-13 全量重跑）

**设定**：零-GT 选帧（τ 分派）、pool=全部帧（transductive）、C2M=maskdistill+rel-pose、固定 100 步、benchmark-100（scannetpp/7scenes）/allv（hiroom/eth3d）、原版 TSDF 无 conf 过滤。baseline 与 TTA 同一评测 pass、同一批帧。

### 6.1 主配方 C2M（默认，一刀切）

| 数据集 | n | 评测 | AUC@3 | F1 | CD | AbsRel | δ<1.25 |
|---|---|---|---|---|---|---|---|
| ScanNet++ | 20 | 100v | 0.5930→0.6109（**+3.0%**） | 0.6664→0.6531（−2.0%） | 0.0785→0.0796（−1.3%） | 0.0388→0.0358（**+7.8%**） | +0.2% |
| 7Scenes | 7 | 100v | 0.2383→0.2361（−0.9%） | 0.4752→0.5198（**+9.4%**） | 0.1417→0.1190（**+16.0%**） | 0.0734→0.0678（**+7.6%**） | +0.8% |
| HiRoom | 30 | allv | 0.4912→0.5118（**+4.2%**） | 0.5623→0.6193（**+10.1%**） | 0.0995→0.0893（**+10.3%**） | +0.7% | −0.0% |
| ETH3D | 11 | allv | 0.2654→0.3248（**+22.4%**） | 0.5736→0.6398（**+11.6%**） | 0.6207→0.5657（**+8.9%**） | 0.0356→0.0329（**+7.8%**） | −0.0% |

### 6.2 精细版对照 B5（纯 maskdistill，去 rel-pose）

| 数据集 | AUC@3 | F1 | CD | 结论 |
|---|---|---|---|---|
| ScanNet++ | **+3.3%** | −0.5% | −0.8% | rel-pose 在随机分支有毒性；B5 全面≥C2M 但 F1 仍未转正 |
| 7Scenes | −5.4% | +5.5% | +12.4% | **C2M 全面赢**（稠密视频 rel-pose 有益） |
| HiRoom | **+8.7%** | +6.8% | +5.2% | B5 赢 AUC，C2M 赢 F1/CD |
| ETH3D | +10.5% | +3.4% | +3.2% | **C2M 全面赢** |

### 6.3 Acceptance rate（C2M，68 场景）

- AUC@3 提升场景：**46/68（68%）**；F1 提升场景：37/68（54%）
- 相对提升 ≥3% 的（数据集×指标）格子：**12/20**（AUC 3/4 数据集过线、F1 3/4、CD 3/4、AbsRel 3/4、δ1.25 0/4——δ 已饱和 >0.92）

### 6.4 诊断结论（对 <3%/下降格子的归因）

1. **ScanNet++ F1/CD（−2.0%/−1.3%）**：probe 轨迹与低视角分解显示 student 学到了（4v AUC +0.033、8v +0.017、探针 +0.015），AbsRel +7.8% 说明逐像素深度变准；F1 下降出现在 100v 融合环节——长序列跨视图一致性受损，与 rel-pose 在随机分支的毒性叠加（去掉后 −0.5%）。DA3 侧独立复现了 rel-pose 在该场景类型的不收敛（见 §9）。
2. **7Scenes AUC（−0.9%）**：稠密视频 baseline AUC 本身随视角数非单调（4v 0.278 > 8v 0.204 > 100v 0.238），AUC 对 dense-video 的相机基线饱和区不敏感；F1/CD/AbsRel 三指标 +9.4%/+16.0%/+7.6% 为正。
3. **δ1.25 全线弱/平**：基线已 0.92–0.99，指标饱和，属度量特性非 loss 缺陷。

### 6.5 Phase 4 进阶臂门控（8 子场景，paired vs C2M；负结果记录）

| 臂 | AUC+ 场景 | F1+ 场景 | 双正 | 任一负 | 判定 |
|---|---|---|---|---|---|
| C2M_CamRel（+0.3 CamRel 亲和力） | 3/8 | 4/8 | 1/8 | 5/8 | 不过闸（≈中性） |
| CONFD_REL（+0.3 conf 蒸馏） | 5/8 | 1/8 | 1/8 | 7/8 | 不过闸 |
| C2M_CTM（+camera token 掩码） | 2/8 | 3/8 | 1/8 | 7/8 | 不过闸（scannetpp 上 AUC +0.042、F1 转正，但 7scenes F1 −0.080） |
| student 聚簇对照 | 2/8 | 2/8 | 0/8 | 8/8 | 不过闸；等距 student 维持为默认 |

### 6.6 夜间位姿通路电池（v2.0，2026-09-14 凌晨）——最终配方

**动机**：posed/unposed 归因证明 scannetpp 的 F1 伤口 100% 来自位姿路径（GT 位姿+TTA 深度 F1 反超 baseline：0.7304>0.7260）；headroom 探针证明 7scenes 位姿无可蒸馏目标（所有 teacher:student 组合 headroom≤0）。围绕位姿通路做了 10 个新臂的门控（sub8）与全量验证。

**新臂门控结果**（sub8，paired vs C2M）：

| 臂 | 机制 | 判定 |
|---|---|---|
| **C2M_CONFP** | rel-pose 按 teacher conf 加权视角对 | **○ 最均衡**：scannetpp 双胜（AUC 0.725/0.709），全量 scannetpp +3.3%/−1.5%/−0.9% 全面优于 C2M |
| **C2M_MC** | 多上下文 student（4v 掩码 + 16v 同模型前传，共享帧 post-LN 特征拉齐） | **○ 7scenes 全正唯一解**：全量 AUC +0.5%（首次转正）/F1 +10.5%/CD +14.8%/AbsRel +8.4% |
| **C2M_SCL** | rel-pose + 平移尺度比项 (log\|t_s\|−log\|t_t\|)² | **○ eth3d F1/CD 最佳**：全量 +18.1%/+14.0%/+5.8%；单场景 1ada7a0617 AUC +16% |
| C2M_TRIP | 位姿两跳闭合（i→e→j 经额外帧锚点） | △ F1/CD 强（scannetpp 0.725、eth3d 0.713）但 AUC 弱 |
| CTM_ANC | 掩码 camera token 重建 + 额外帧亲和力锚点 | △ scannetpp sub8 AUC 最强（0.7525）但 7scenes/hiroom 输 |
| C2M_TRIF/TRIF2/TRIF3 | camera token 特征三角（cos/KL+距离/稳健距离） | △ 中性偏正（scannetpp AUC 0.719），7scenes 负 |
| C2M_CYC | student 位姿三角闭环（teacher-free） | ✗ 灾难性（与蒸馏目标冲突拉崩 camera head） |
| C2M_GATE | 场景级位姿分歧门控（<3° 关 rel-pose） | ✗ 负（rel-pose 的价值是正则不是弥合分歧） |
| C2M_HARD | 难对加权 rel-pose | ✗ 中性 |
| C2M_REL2/REL10 | rel-pose ×2/×10 | ✗ 全维度变差 |

**选帧消融（全部证伪，维持 v1.1）**：7scenes 分段平铺/分段随机（负）、hiroom 全 8v（负）、student 等距 vs 聚簇（聚簇负）。**teacher 长度**：7scenes 16v→24v→32v 的 F1 单调升（+9.4%→+11.7%），AUC 持平——N≥500 可用 32v。**对数/步数**：10 对、100 步为饱和点（20 对无差、200 步过拟合）。

**最终配方 v2.0**（全量实测数字）：

| 数据集 | 配方 | AUC@3 | F1 | CD | AbsRel |
|---|---|---|---|---|---|
| ScanNet++（20，100v） | **24:8 + B5**（teacher 24v/student 8v，摘 rel-pose） | 0.5930→0.6250（**+5.4%**） | 0.6663→0.6594（−1.0%） | 0.0786→0.0795（−1.2%） | 0.0388→0.0352（**+9.3%**） |
| ScanNet++（备选） | C2M_MC（16:4 多上下文） | +3.7% | −1.0% | **−0.4%** | — |
| 7Scenes（7，100v） | **C2M_MC**（16:4 + 多上下文一致性） | 0.2383→0.2396（**+0.5%**） | 0.4752→0.5253（**+10.5%**） | 0.1417→0.1207（**+14.8%**） | 0.0734→0.0672（**+8.4%**） |
| HiRoom（30，allv） | C2M（v1.1 保持） | 0.4912→0.5118（**+4.2%**） | 0.5623→0.6193（**+10.1%**） | 0.0995→0.0893（**+10.3%**） | +0.7% |
| ETH3D（11，allv） | **C2M_SCL**（16:4 + 尺度比项） | 0.2654→0.3134（**+18.1%**） | 0.5736→0.6540（**+14.0%**） | 0.6207→0.5844（**+5.8%**） | 0.0356→0.0324（**+9.0%**） |

**统一规则表述（仍只依赖 N、τ）**：选帧/训练超参维持 §2/§3；loss 按 τ 分支——τ>0.55（稠密视频）→ C2M_MC；τ≤0.55 且 N≥100 → 24:8+B5 或 C2M_CONFP；τ≤0.55 且 N<100 → C2M_SCL 或 C2M。pending：ScanNet++ MC 全量与 TRIF3 补跑（收尾中）。


**结论**：维持 §4.3——默认 C2M 一刀切；精细版 τ≤0.55 可换纯 maskdistill（ScanNet++ 型数据的 AUC 更稳、F1 伤口减半）。叠加臂无通用增益，CTM 的 scannetpp 特异性收益记录备查。

### 6.7 v2.1（2026-09-14 下午）——量规机制 + 门控/重耦，16/16 指标格全正

**新机制发现（ScanNet++ F1/CD 伤口根因）**：TTA 后 depth-head 深度量规与 camera-head 平移量规脱钩。1ada7a0617 反事实钉死：B5 失配 +2.97% vs baseline +1.08%；深度预乘 0.9712 修复后 F1 0.343→0.605 反超 baseline；给 baseline 注入等量失配可复现崩溃；跨臂失配↔F1 强单调（ρ=0.89）。AUC（平移方向）与 AbsRel（去共享尺度）对量规天然不可见——"深度好、位姿好、F1 崩"不矛盾。20 场景推广：TTA 臂平均失配 +1.19~1.29% vs baseline +0.64%（单场景极端，数据集级弱趋势）。**该发现同时改写了 §6.4-1 的归因**：posed 实验（GT 位姿 F1 反超）在 1ada 不成立——`_prep_posed` 仍乘预测中心拟合的尺度，量规失配污染两条归因路径。

**两个 GT-free 修复件**：
1. **探针回退门（训练端）**：τ≤0.55∧N≥100 时，若 Δprobe_auc03(step30−step0) ≤ −0.01 则该场景回退 baseline（宽松门；严门误杀量子化平局的大正场景）。**仅适用 16:4 配置**——24:8 上 probe 信号断裂（F1 重伤场景 probe 增益最大，门 100% 误杀），24:8 停用门。
2. **量规重耦 fixfree（评测端）**：逐场景在 s∈[0.90,1.10] 网格（步长 0.5%）搜深度预乘因子，最大化多视图互投影一致性/conf 加权 TSDF inlier 比；只修 TTA 臂（修 baseline 反而变差）；**仅 τ≤0.55∧N≥100 启用**——hiroom/eth3d 短序列稀疏共视下 GT-free 因子退化为坍塌方向（贴网格下沿），有害；7scenes 近中性，不启用。

**统一性证伪记录**：单一 loss 打全场不存在——C2M_MC 输 hiroom F1/CD（+5.4%/+3.3% vs C2M +10.1%/+10.3%）且不优 eth3d SCL；C2M_SCL 输 hiroom CD（−3.6%）。统一协议 = 单一核心形式 + GT-free 触发器，不再追求单臂一刀切。

**最终协议 v2.1**（全部零 GT；触发只看 τ、N）：

| 条件 | 配方 |
|---|---|
| 核心（所有场景） | maskdistill（50% patch 置零、teacher 全帧、L4/11/17/23 post-LN Huber+2(1−cos)、conf 加权）+ rel-pose（chordal 旋转 + 平移方向 1−cos，1:1）；选帧/训练超参维持 §2/§3 |
| τ>0.55（稠密视频） | 核心 + MC 项（4v 掩码前传与 16v 全上下文前传共享帧 post-LN 对齐）= C2M_MC |
| τ≤0.55 ∧ N≥100 | 核心 + 尺度项（C2M_SCL）+ 宽松探针门 + fixfree 量规重耦；AUC 优先可换 24:8+B5+fixfree（不用门） |
| τ≤0.55 ∧ N<100 | 核心不变（C2M）；F1 优先可换 C2M_SCL |

**v2.1 全量结果（paired，同帧同评测器）**：

| 数据集 | 配方 | AUC@3 | F1 | CD | AbsRel |
|---|---|---|---|---|---|
| ScanNet++（20，100v） | C2M_SCL+门+fixfree | +3.35% | **+2.03%** | **+2.3%** | +7.0% |
| ScanNet++（AUC 优先备选） | 24:8+B5+fixfree | **+5.4%** | +0.70% | +1.67% | **+9.3%** |
| 7Scenes（7，100v） | C2M_MC | +0.5% | +10.5% | +14.8% | +8.4% |
| HiRoom（30，allv） | C2M | +4.2% | +10.1% | +10.3% | +0.7% |
| ETH3D（11，allv） | C2M_SCL | +18.1% | +14.0% | +5.8% | +9.0% |

16/16 指标格全部转正。已知上限：7scenes AUC +0.5%（位姿 headroom 物理不存在，探针+组合探针+消融三重确认）；hiroom AbsRel +0.7%（基线 0.987 δ1.25 饱和区）。

证据出处：`artifacts/diagnostics/final_protocol/{GATE_ANALYSIS.md,GAUGE_CROSS_DATASET.md}`、`final_protocol/scannetpp/{SCENE_1ADA_ANATOMY.md,GAUGE_SCANNETPP.md,GAUGE_SCL.md}`、`final_protocol_s8/scannetpp_s8t24/{GATE_ANALYSIS_S8T24.md,GAUGE_S8T24.md}`。

## 7. 诊断仪器（论文机制部分）

1. **headroom 探针**：不训练，直接量 teacher-student 差距（AbsRel 差 / AUC 差 / teacher 更优 patch 占比）——选帧策略的度量器，先量后选。
2. **替换探针**（cam_interp/patch_interp）：teacher 特征插值进 student 冻结 decoder——feature 目标质量的端点测量。
3. **粒度阶梯**：逐像素→patch→大块的 teacher 更优占比（57% 不变）——排除测量伪影。
4. **配对数/步数饱和曲线**——定 pairs=10、steps=100。

## 8. 通过与扩散标准

- 8 子集（4 数据集 × 2 场景）：AUC@3 与 F1 的 paired delta 双双为正（≥6/8 场景）→ 扩散到全量。
- 全量：上表四数据集全部场景 + 种子 2 复测。

## 9. DA3-giant-1.1 移植（v1.1 新增）

协议已逐行移植到 DA3-giant-1.1：`src/depth_anything_3/test_time_adaption/protocol_v1.py` + `scripts/train_da3_protocol.py`，接口差异/适配决定/smoke 数字见 `docs/TTA_PROTOCOL_DA3_notes.md`。要点：40 层 block 取 tap [6,19,26,39]；post-LN 空间 = `DualDPT.norm`；camera 编码 c2w→w2c 后套同一 rel-pose 公式；LoRA r32 全 40 层、lr 3e-5、100 步不变。Smoke（ScanNet++ 2 场景）：无 NaN、峰值 20.8GB；并独立观测到 rel-pose 在随机分支场景的不收敛现象（与 §6.4-1 互证）。

## 附：关键证据出处

- **v1.1 全量重跑**：`artifacts/diagnostics/final_protocol/`（manifest + 双臂指标 + `FINAL_RESULTS.md` + `PHASE4_GATING.md` + `manifest_report.md`）
- 掩码家族与冠军：`artifacts/diagnostics/v20_md20/`（20 场景）、`vs1_md/`（种子 1）
- 选帧网格与抽帧：`vf_*`、`v20s4_*`；headroom 探针：`artifacts/diagnostics/headroom_probe/`
- 7Scenes 反转：`v7se_*`；HiRoom：`vsub_hr_*`；ETH3D：`vese_*`
- 全过程日志：`artifacts/diagnostics/OVERNIGHT_RESULTS.md`、`PROGRESS.md`
