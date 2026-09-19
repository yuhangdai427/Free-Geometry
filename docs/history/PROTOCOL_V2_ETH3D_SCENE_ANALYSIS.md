# Protocol v2 等权版 ETH3D 试点：逐场景失败归因与机制分析

日期：2026-09-19 凌晨试点（logs/v2_{da3,vggt}_eth3d*.log，run_dir = workspace/protocol_v2/{da3,vggt}_eth3d{,_rest}）
分析口径：AUC = auc03（与日志 `d_auc03_rel`、用户通报一致）；F1 = recon_unposed fscore。均值按 11 场景算术平均。

## 0. 先纠正四件事（读数前必须知道）

1. **今晚 eval 的全部是 step100 final ckpt**。两条 lane 的启动脚本（scripts/run_v2_lane_da3.sh、run_v2_lane_vggt.sh）都没有 `--v2_apply_selection`；日志里的 `selector replay` 是**只读回放**（train_da3_protocol.py:526-541 注释原文 "read-only"），选中 ckpt 未加载、未重 eval。VGGT 侧 eval_viewcounts 固定 `--step 100`（eval4 日志明确加载 step100_lora_peft）。**selector 今晚一次机会都没真正行使**。
2. **"等权版" = 没有 A/B 上下文、没有 reliability 权重**。两个脚本的 V2 变量只有 `--v2_probe --v2_ckpt --v2_rel_weight 1.0 --v2_grad_cap --v2_couple_fix --loss_all_pos`；没有 `--v2_ab_manifest`（DA3）/ `--v2_ab`（VGGT），训练 trace 全部 11 场景 geo_w 列空。protocol_v1.py:1096-1100 的 geo_w 降权路径今晚一次都没走过。
3. **DA3 今晚没有 rel 场景门**。脚本没传 `--v2_rel_gate_deg`（代码默认 30，但试点构建的日志里全程无 gate 输出、trace 无 rot_deg_unmasked_median 列），且全部 11 场景 step1 的 v2 rel 都在训练（rest 4 场景 trace 的 v2_rel_rot 有值）——**rel 分支在所有场景强制开启，包括 courtyard**。VGGT 有 pose_gate（train_arms.py:3108-3114）：baseline student 与 8-view teacher 的 unmasked 相对旋转中位 <3° 时 rel 硬关。
4. **workspace/protocol_v2/analysis/da3_eth3d.{json,md} 是旧 mv13 run 的分析**（json 里 run_dir=workspace/mv13_eth3d），数字与今晚 pilot 不符，已被日志取代，建议删掉以免明早误读。

DA3 均值复算：AUC +3.59%、F1 −1.89%（与通报 +3.6/−1.9 一致）。VGGT 均值复算：AUC +17.3%、F1 +19.8%（通报 +15.9/+18.2，略有出入，差异大概率来自通报版 baseline 集合不同；本报告全部用 baselines.json 重算）。

## 1. 总表（b = baseline，v2 = step100）

### DA3（AUC=auc03，F1=fscore；cd=chamfer overall，abs_rel 逐帧深度）

| 场景 | N | tau | AUC b→v2 | dAUC% | F1 b→v2 | dF1% | abs_rel b→v2 (d%) | cd b→v2 (d%) | selector 回放 | 分类 |
|---|---|---|---|---|---|---|---|---|---|---|
| courtyard | 38 | .130 | .5220→.4248 | **−18.6** | .7505→.8100 | +7.9 | .0125→.0124 (−0.7) | .181→.158 (−12.3) | 选 step10, imp .0068 | **灾难（pose 轴）** |
| delivery_area | 40 | .237 | .3684→.4697 | **+27.5** | .4378→.3541 | **−19.1** | .0326→.0323 (−1.1) | .825→.846 (+2.6) | fell_back | **pose 升 / 融合漂移** |
| electro | 39 | .117 | .6815→.6757 | −0.9 | .7731→.7389 | −4.4 | .0230→.0228 (−0.9) | 2.337→2.478 (+6.0) | fell_back | 无变化偏负 |
| facade | 76 | .161 | .4536→.4632 | +2.1 | .7005→.6844 | −2.3 | .0136→.0136 (+0.2) | .234→.243 (+4.2) | fell_back | 无变化偏负 |
| kicker | 31 | .079 | .4659→.4695 | +0.8 | .9983→.9989 | +0.1 | .0101→.0097 (−3.6) | .043→.044 (+2.3) | fell_back | 无变化（F1 天花板） |
| office | 26 | .121 | .3200→.3251 | +1.6 | .9965→.9974 | +0.1 | .0097→.0096 (−1.4) | .060→.060 (−0.0) | 选 step40, imp .0863 | 健康微升 |
| pipes | 14 | .089 | .5201→.5714 | +9.9 | .8098→.8202 | +1.3 | .0139→.0139 (+0.2) | .176→.176 (−0.1) | 选 step20, imp .0391 | 健康改善 |
| playground | 32 | .062 | .2581→.3038 | +17.7 | .7274→.7311 | +0.5 | .0425→.0432 (+1.6) | .486→.428 (−12.0) | 选 step60, imp .2547 | 健康改善 |
| relief | 19 | .137 | .7271→.7329 | +0.8 | .8359→.8304 | −0.7 | .0239→.0246 (+3.1) | .181→.180 (−0.8) | fell_back | 无变化 |
| relief_2 | 20 | .197 | .4386→.4211 | −4.0 | .7569→.7038 | −7.0 | .0120→.0121 (+0.7) | .191→.233 (+21.9) | 选 step100, imp .5291 | **双降（轻灾难）** |
| terrains | 42 | .244 | .5830→.5974 | +2.5 | .9093→.9357 | +2.9 | .0135→.0134 (−0.5) | .113→.105 (−7.9) | 选 step50, imp .0602 | 健康改善 |

### VGGT（eval = step100 @allv，eval32_metrics_V2*.json）

| 场景 | pose_gate（unmasked 中位） | AUC b→v2 | dAUC% | F1 b→v2 | dF1% | selector 回放 | 分类 |
|---|---|---|---|---|---|---|---|
| courtyard | 1（65.18°） | .2451→.3684 | **+50.3** | .3108→.3204 | +3.1 | 选 step100, imp .6928 | **健康大幅改善** |
| delivery_area | 1（32.79°） | .2068→.2496 | +20.7 | .2963→.2734 | −7.7 | 选 step100, imp .7042 | pose 升 / 融合漂移 |
| electro | 0（2.24°，rel 关） | .4026→.3882 | −3.6 | .6575→.6498 | −1.2 | fell_back | 无变化偏负 |
| facade | 1（10.41°） | .3642→.3396 | −6.7 | .6017→.6082 | +1.1 | 选 step60, imp .2892 | 无变化偏负（pose） |
| kicker | 0（2.44°，rel 关） | .5921→.6179 | +4.4 | .9987→.9972 | −0.1 | fell_back | 无变化偏升 |
| office | 0（1.88°，rel 关） | .2431→.2328 | −4.2 | .9933→.9900 | −0.3 | 选 step30, imp .1447 | 无变化偏负 |
| pipes | 0（2.44°，rel 关） | .3590→.3810 | +6.1 | .8101→.8185 | +1.0 | fell_back | 健康微升 |
| playground | 1（6.89°） | .1512→.1075 | **−28.9** | .7188→.6454 | −10.2 | fell_back | **灾难（pose 轴）** |
| relief | 0（2.28°，rel 关） | .1715→.2827 | +64.8 | .2283→.5861 | +156.7 | 选 step40, imp .4861 | **健康大幅改善（F1 恢复）** |
| relief_2 | 1（25.77°） | .0842→.0825 | −2.1 | .0230→.0368 | +59.7 | 选 step90, imp .6869 | F1 恢复 / AUC 平 |
| terrains | 1（54.22°） | .0991→.1878 | **+89.5** | .6705→.7766 | +15.8 | fell_back | **健康大幅改善** |

分类判据（本报告使用）：**灾难** = 单轴跌幅 ≥10%（AUC 或 F1）；**pose 升/融合漂移** = AUC 升 ≥5% 且 F1 降 ≥5%；**双降** = 两轴均跌 ≥3%；**健康改善** = AUC 升 ≥5% 且 F1 不降；**无变化** = 两轴都在 ±3%（偏负/偏升给标注）。阈值取整为读数方便，场景归属对 ±2pp 不敏感。

## 2. 逐场景小节

probe rot_deg 列为 probe_trace/*.jsonl step0 的 4 条 record（probe0×mask0/1, probe1×mask0/1），单位度，语义 = **masked student 与 frozen 8-view teacher 的相对旋转（边均值）**；last = 该场景最后一个 probe step（step100）。"pair 均值" = 两 mask 平均。couple 为组级标量（相机中心+深度联合，couple_robust）。

### 2.1 courtyard —— 本试点的核心反向场景（DA3 −18.6% vs VGGT +50.3%）

| | DA3 | VGGT |
|---|---|---|
| probe rot step0（p0m0/p0m1/p1m0/p1m1） | 97.2 / 72.3 / 5.8 / 60.6（pair 均值 **84.7°/33.2°**，与通报一致） | 61.2 / 65.9 / 43.8 / 52.5（pair 均值 63.6°/48.2°） |
| probe rot last | 92.8 / 75.6 / 3.7 / 59.4（**未收敛**） | 1.9 / 1.9 / 3.3 / 8.9（**全收敛**） |
| couple step0→last | 0.0736→0.1084（**+47%，反向恶化**） | 0.1665→0.0089（−95%） |
| feature step0→last | 0.393→0.379 | 0.606→0.386 |
| rkd step0→last | 0.110→0.096 | 0.119→0.031 |
| step1 grad_norm | **132.96（全场景最大，次大 relief 29.6）** | — |
| 结果 | AUC −18.6% / F1 +7.9% / abs_rel −0.7% | AUC +50.3% / F1 +3.1% |

机制（Q2 的完整答案）：

- **两边起点一样糟糕**：courtyard 上两个模型的 baseline 都与 8-view teacher 有巨大位姿分歧（DA3 masked 84.7/33.2°；VGGT unmasked pose_gate 中位 65.18°、masked pair 均值 63.6/48.2°）。这不是"DA3 的 student 特别差"——VGGT 的分歧同样远超 30° 门限。
- **区别在于 teacher 目标是否自洽、student 是否能到达**。VGGT 的 teacher 在 courtyard 上**全局自洽**：两个 probe pair 的分歧只有 15.4°，同一 pair 两个固定 mask 间分歧 4.7/8.6°，训练后 rot 一路收到 2–9°，feature/rkd/couple 全部同步改善——8-view teacher 位姿确实优于 VGGT 自己的 baseline 全局位姿（0.245→0.368），追对了方向。
- DA3 的 8-view teacher 在 courtyard 上**互不兼容（破产的不是"位姿精度"而是"自洽性"）**：probe0 与 probe1 的分歧 51.5°；同一 pair 换一张固定 mask 分歧 24.8°（probe0）/ **54.8°**（probe1：5.8° vs 60.6°）——同一个 teacher 对同一组帧，换 8-view 上下文、换输入 mask，给出完全不同的相对位姿。DA3 student 从头到尾到不了这个不一致的目标（rot 终值 92.8/75.6/3.7/59.4，和 step0 几乎一样），rel 梯度（权重 1.0）持续存在。
- **毒害发生在前 10 步无保护期**。grad cap 的 C_R 在前 10 次 update 之后才标定（protocol_v1.py:1353-1355；trace 里 v2_C_R 前 10 行全 nan），courtyard step1 的 grad_norm=132.96（对比 delivery_area 1.25、kicker 0.015）。等 cap 生效时 camera token/LoRA 已被 conflicting rel 梯度打偏；eval 的干净前向（38 帧）位姿 −18.6%，而深度侧 maskdistill 正常工作（abs_rel 持平略降、F1 +7.9%）——**典型的"pose 被 rel 毒化、depth 无恙"剖面**。
- couple 的反向恶化（+47%）是旁证：四个 probe 分量里只有 couple 带全局几何（相机中心）信息，它显示 student 没有走向 teacher，而是被拉向"第三方"位置——噪声累积而非收敛。

### 2.2 delivery_area —— 双模型同型：AUC 大涨、F1 大跌（DA3 +27.5%/−19.1%，VGGT +20.7%/−7.7%）

| | DA3 | VGGT |
|---|---|---|
| probe rot step0 | 3.8/1.2/1.5/1.5（pair 均值 2.5°/1.5°） | 59.4/61.6/64.0/64.7（两 pair 高度一致 ~62°） |
| probe rot last | 2.8/1.8/1.6/1.7 | 6.2/3.7/19.4/18.6（大体收敛） |
| couple step0→last | **5.9e-3→1.4e-3（−76%）** | 0.450→0.052 |
| pose_gate / rel | rel 强制开（无门） | pose_gate=1（32.79°），rel 开 |
| 深度侧 | abs_rel .0326→.0323（−1.1%）；cd +2.6% | e_depth .0074→—（probe mse 全面降） |

机制（Q3 的完整答案）：

- **F1 大跌不是深度图精度崩了**：DA3 abs_rel 基本不动（−1.1%），chamfer 只 +2.6%。F1（TSDF 融合 f-score）从 .438 掉到 .354，而 baseline 的 recall 本来就是全场景最低（0.364，cloud 最薄）。**跌幅远大于 chamfer 涨幅 ⇒ 退化集中在融合覆盖率/召回**：maskdistill 把深度抹得更平滑（对 teacher 的组级一致性 couple 反而改善 76%），无纹理地面/道路的结构细节被抹掉，融合后薄云更薄。
- **"couple 改善但 F1 大降"正说明 couple 的分辨率不够**：couple 是组级标量（相机中心+深度图聚合），平滑但无结构的深度可以让 couple 很好、abs_rel 很好，同时 fusion recall 崩。今晚 probe 里没有任何一个分量带 per-region 结构信息——**这个失败今晚从头到尾无任何前置信号**（probe rot 1–4° 贴地、feature/rkd 微动、selector 干脆 fell_back）。
- AUC 大涨的来源：DA3 probe rot 已在 1–4°（floor effect，probe 看不见 pose 改善）；VGGT 侧是 rel 真把 ~62° 的 student-teacher 分歧吃掉了（→4–19°）。两个模型的 pose 收益都是真的，但融合层为深度平滑买单。

### 2.3 playground —— VGGT 的唯一灾难场景（−28.9%/−10.2%），DA3 同场景却 +17.7%

| | DA3 | VGGT |
|---|---|---|
| probe rot step0 | 28.7/40.7/14.9/8.7（pair 分歧 22.9°） | 1.2/1.0/8.7/11.2（pair 分歧 8.8°） |
| probe rot last | 21.3/32.0/16.4/8.6（未收敛） | 1.3/1.0/**10.1/15.2（probe1 恶化）** |
| rkd step0→last | 0.184→0.136 | **0.0845→0.1267（+50%，恶化）** |
| couple step0→last | 1.966→0.604 | 0.203→0.185（平） |
| trace rel（train pair 均值） | pair 间撕裂 spread 1.95：pair7 均值 1.96、pair6 1.27 到终点不降；pair4 0.015 已收敛 | 1.37→0.043（train pair 上"学会"了） |
| pose_gate / rel | rel 强制开 | pose_gate=1（**6.89°**，刚过 3° 阈值），rel 开 |
| selector | 选 step60（imp .2547） | **fell_back**（rkd 恶化 >τ 把所有候选 disqualify） |
| probe_auc03（GT 探针，4 帧） | —（DA3 无此文件） | .111→.111→.111（全程持平） |
| e_depth | — | .0117→.0127（变差） |

机制：VGGT 的 student-teacher 初始分歧其实很小（6.89° 中位）——**3° pose_gate 把它留在了"开"，但这点 gap 不是"可学的 headroom"，而是 per-pair teacher 噪声**。rel 在 10 个 train pair 上把 student 钉向各家 8-view teacher 的相对位姿（trace rel 1.37→0.04），这些 per-pair 目标拼不出一个全局一致的位姿；probe pair1 的 rkd/rot 在训练后反而恶化、GT 探针 probe_auc03 原地不动、全局 eval AUC 崩 28.9%。**与 courtyard-DA3 同一病型（per-pair teacher 全局不自洽），只是 VGGT 能收敛到不一致的目标，DA3 收敛不到**。危险区是 pose_gate ∈ (3°, ~10°)：小到不足以构成真信号、又被 3° 阈值放行。selector 今晚若被应用（fell_back→step0）反而能救这个场景——见 §5。

DA3 同场景 +17.7% AUC：probe rot 从 29/41° 收到 21/32°（部分收敛）、couple −69%，属于"teacher 目标半自洽、学生吃下一半"的良性版本。

### 2.4 relief_2 —— DA3 双降（−4.0%/−7.0%），VGGT F1 恢复（+59.7%）

| | DA3 | VGGT |
|---|---|---|
| probe rot step0 | 44.7/46.2/1.4/30.0（**pair 分歧 29.7°**；probe1 内部 mask 分歧 28.5°） | 59.3/56.4/50.9/49.5（两 pair 一致 ~55°） |
| probe rot last | **45.3**/22.7/0.8/1.1（probe0 始终到不了） | 17.5/8.2/5.9/6.5（部分收敛） |
| couple step0→last | 4.324→0.492（−89%） | 1.167→0.129 |
| cd / chamfer | .191→.233（**+21.9%**） | — |
| selector | 选 step100（imp .5291，全场最高） | 选 step90（imp .6869） |

机制：DA3 侧 probe0 的 teacher 目标从头到尾不可达（45.3°），pair 间 30° 撕裂说明两个 probe 上下文给出的位姿互相矛盾；couple/rkd 的大幅"改善"（4.32→0.49）主要是学生放弃追 pose、退回深度侧拟合——chamfer +21.9% 是真金白银的退化，**selector 却把 step100 选为全场最佳（imp .5291），probe 与 GT 完全反向**。VGGT 侧 teacher 两 pair 一致（分歧 7.7°），rel 吃下大部分分歧，F1 从 0.023 的崩盘中恢复（+59.7%），AUC 微降 2.1%（追到一半的目标仍有全局残差）。

### 2.5 electro —— 双模型同向偏负（DA3 −0.9%/−4.4%，VGGT −3.6%/−1.2%）

- 共同背景：39/45 帧（6 帧被过滤）、baseline 云本来就残（DA3 comp 4.53、recall .663；VGGT comp 4.73）。
- DA3：couple 1.34→0.13（−90%）、feature/rkd 改善，selector fell_back——但 eval 的是 step100，probe 的全面"改善"换来 F1 −4.4%、chamfer +6%。又一例 probe-GT 轻反向。
- VGGT：pose_gate=0（2.24°，rel 关）——**rel 关的情况下 pose 仍 −3.6%**：feature 蒸馏通过共享 trunk 拖累了 pose head；probe1 rot 30.2/24.8→21.6/11.5（改善）与 eval AUC 反向。

### 2.6 facade —— 双模型 probe 都有高位不收敛分量，结果互异

- DA3：probe0 24.6/28.3° 到终点 26.1/27.2°（纹丝不动），结果 +2.1%/−2.3%，fell_back。
- VGGT：probe1 42.8/41.3° → 34.4/19.1°（收敛一半），pose_gate=1（10.41°，危险区上沿），结果 −6.7%/+1.1%。
- 机制同 playground（轻量版）：per-pair teacher 半自洽，rel 追一半，全局 pose 小跌。

### 2.7 terrains —— VGGT 最大涨幅（+89.5%）与 selector 最大误判同框

- VGGT：pose_gate=1（54.22°），probe rot 50.0/17.6/9.1/25.2 → 3.0/9.2/8.9/6.0，feature/rkd 改善；**唯独 couple 0.125→0.142（+13%）**。controller 的 tau_qual=0.05 单分量一票否决 ⇒ 所有 step 被 disqualify ⇒ **fell_back**——selector 若被应用会把 +89.5% 的最佳场景退回 baseline。
- 旁注：step0 同一 pair 两 mask 分歧 32.4°（50.0 vs 17.6，全场最大 mask-split）——mask 敏感性大 ≠ 会失败，VGGT 照样收敛大胜；**任何用 mask-split 做硬 veto 的规则都会误伤 terrains**。
- DA3：+2.5%/+2.9%，probe1 mask1 有 32.5°→27.5° 的顽固分量，整体无碍。

### 2.8 relief —— VGGT F1 +156.7%（rel 关、纯深度侧恢复）；DA3 无变化

- VGGT pose_gate=0（2.28°）rel 关，feature 0.355→0.291、rkd 0.029→0.004、couple 0.0046→0.0011，probe_auc03 .167→.361；F1 从 .228 的深坑恢复到 .586。**counter 例证：rel 不是 VGGT 涨点的必要条件**。
- DA3：rel 强制开但 step0 rel_rot 高达 1.26（trace）→ 0.0006 完全收敛，结果 +0.8%/−0.7% 无变化；per-pair 撕裂存在（trace rel pair 均值 spread 2.41，pair5 2.15→2.88 发散、pair1 收敛到 0.003）但未酿成事故。

### 2.9 kicker / office / pipes —— 健康基线区

- 三场景 probe rot 全程 0.4–3.3°，双模型一致；kicker/office 的 F1 在 .99 天花板，pipes VGGT +6.1%/+1.0%（probe_auc03 0→.028 同步）。
- office VGGT −4.2% AUC 是唯一刺点：rel 关、probe 全改善（feature .327→.303、rkd .024→.016、couple .0124→.0046），GT pose 仍小跌——纯 feature 蒸馏的 trunk 泄漏，幅度小。

## 3. pair 间撕裂全景（Q5）

probe step0 rot_deg 的三种分歧（度）：

| 场景 | 模型 | probe0 vs probe1 pair 分歧 | mask 分歧 p0 | mask 分歧 p1 | trace per-pair rel spread（10 train pairs 全程均值） | 结局 |
|---|---|---|---|---|---|---|
| courtyard | DA3 | **51.5** | 24.8 | **54.8** | —（主 7 场景无 trace） | AUC −18.6 灾难 |
| courtyard | VGGT | 15.4 | 4.7 | 8.6 | 1.48 | AUC +50.3 |
| delivery_area | DA3 | 1.0 | 2.7 | 0.0 | — | AUC +27.5 / F1 −19.1 |
| delivery_area | VGGT | 3.8 | 2.2 | 0.7 | 1.50 | AUC +20.7 / F1 −7.7 |
| playground | DA3 | 22.9 | 11.9 | 6.2 | **1.95**（pair7 1.96 不降） | AUC +17.7 |
| playground | VGGT | 8.8 | 0.3 | 2.5 | 1.50 | AUC −28.9 灾难 |
| relief_2 | DA3 | **29.7** | 1.5 | 28.5 | 0.73 | AUC/F1 双降 |
| relief_2 | VGGT | 7.7 | 2.9 | 1.4 | 1.21 | F1 +59.7 |
| terrains | VGGT | 16.6 | **32.4** | 16.1 | 1.17 | AUC +89.5 |
| facade | DA3 | 14.3 | 3.7 | 10.6 | — | 平 |
| facade | VGGT | **29.5**（probe1 更差） | 1.3 | 1.4 | 1.03（couple spread 1.20） | AUC −6.7 |
| electro | VGGT | 23.4 | 2.2 | 5.4 | 1.18 | AUC −3.6 |
| relief | DA3 | 0.1 | 1.3 | 2.1 | 2.41（pair5 发散 2.15→2.88） | 平 |
| office | VGGT | 0.7 | 0.3 | 2.0 | **1.99**（pair3 2.00、pair7 1.91 双高） | AUC −4.2 |
| 其余（kicker/office/pipes DA3、relief/kicker/pipes/office VGGT） | | <2 | <2 | <2 | <0.2 | 平/微升 |

读法：

- **pair 分歧（probe0 vs probe1）>~20° 是 DA3 灾难/双降场景的共性**（courtyard 51.5、relief_2 29.7、playground 22.9）；VGGT 的大涨场景 pair 分歧都不大（courtyard 15.4、relief_2 7.7、terrains 16.6、delivery_area 3.8）。**它是"per-pair teacher 全局不自洽"的直接读数，也是今晚最有价值的 scene-level 前置信号**。
- **mask 分歧大不再是可靠的坏信号**：DA3 courtyard 的 54.8 确实伴随灾难，但 VGGT terrains 32.4 / delivery_area 2.2 都是大涨场景。mask-split 只对 DA3 有诊断力（DA3 masked-forward pose 不稳是病；VGGT mask 鲁棒）。
- **trace 的 per-pair rel spread >~1.9 的场景（playground DA3 1.95、office VGGT 1.99、relief DA3 2.41）都对应结局最差的一组**——train-pair 级的 rel 撕裂与 eval 结局相关，但 relief DA3 是"撕裂但无害"的反例（rel 虽撕裂，rot 起点小、追不动学生）。单看 spread 会误报 relief。
- DA3 主 7 场景**没有 training_trace.csv 落盘**（只有 rest 4 场景有），明早队列必须先修 trace 落盘，否则 pair 级审计缺半边。

## 4. 机制覆盖矩阵（Q4）

三个现有机制今晚的真实状态：**gate**：DA3 未部署（等权版无此 flag 的构建），VGGT 只有 3° pose_gate；**selector**：只回放未应用；**reliability（AB）**：未部署。"能拦"= 若在今晚配置下生效且阈值不变，有明确信号触发；"存疑"= 依赖未测量的量或会误伤；"拦不住"= 机制原理上无法感知该失败。

| 失败场景 | rel 场景门（30° unmasked） | selector（应用后假想） | AB reliability（若部署） | 结论 |
|---|---|---|---|---|
| DA3 courtyard（AUC −18.6） | **存疑**：gate 量今晚未测；若 DA3 unmasked 也饱和（≈teacher），gate 不触发；若触发可关 rel 止损 | **拦不住**：probe 把"靠近 teacher"记为改善（step10 imp .0068 被选），teacher 方向错时方向性盲区是原理性的 | **部分可拦**：courtyard 的 A/B 分歧预计大（probe pair 分歧 51.5°），q_rot 会降权矛盾边；但系统性偏差（非噪声）A/B 看不见 | 现有三机制没有一个是"确定能拦" |
| DA3 delivery_area（F1 −19.1） | 拦不住：pose 侧门不管深度 | **帮倒忙**：fell_back 会把 +27.5% 的 AUC 收益退回 baseline（probe floor effect 看不见 pose 改善） | 拦不住：F1 细节丢失是平滑伪影，不是 A/B 噪声 | **无信号覆盖**（abs_rel/couple/chamfer 全部微动） |
| DA3 relief_2（双降） | 存疑同上（probe0 45° 不可达提示 teacher 不自洽，但门量的是 student-teacher 分歧，未必 >30°） | **帮倒忙**：step100 被选为全场最佳（imp .5291），恰是 GT 最差的 ckpt | 同 courtyard，部分可拦 | selector 方向性盲区 |
| DA3 electro / facade（轻负） | 拦不住（分歧 <30°） | fell_back 正好规避（未应用所以没生效） | 噪声量级小，基本不触发 | 轻微，应用 selector 可部分规避 |
| VGGT playground（AUC −28.9） | **拦不住**：pose_gate 中位 6.89° ≪ 30°；30° 门无效，3° pose_gate 反而放行 | **能拦**：rkd +50% 恶化 >τ 触发全 disqualify → fell_back → step0 无损（今晚未应用故未生效） | **拦不住**：初始分歧小，A/B 噪声预计小，q≈1 | selector 是唯一救场机制，但靠的是"分量恶化"而非方向判断 |
| VGGT terrains（+89.5，误伤风险） | **会误伤**：54.22° > 30° → rel 被关 → 全场最佳涨幅消失；同理 courtyard 65.2°、delivery_area 32.8° | **会误伤**：couple +13% >τ 全灭 → fell_back → 最佳涨幅消失 | 若 A/B 分歧小则 q≈1 无影响 | **门与 selector 的现有阈值都会杀死最佳场景** |
| VGGT delivery_area（F1 −7.7） | 同 DA3，拦不住（F1 是融合层问题） | 选 step100，AUC 收益保留，F1 照跌 | 拦不住（同 DA3） | 无信号覆盖 |
| VGGT electro/office/facade（轻负） | 拦不住（rel 关时仍跌：feature 泄漏） | fell_back/step30/step60，office electro 或可避，facade 照跌 | 拦不住 | 轻微 |

**现有机制原理性拦不住的失败（明早报告必须写明）**：

1. **teacher 系统性偏差（方向错但自洽）**：selector 的参照物就是 teacher，"越像 teacher 越好"是它的定义；GT-free 前提下它不可能知道 teacher 错了。courtyard-DA3 与 playground-VGGT（还有 relief_2-DA3 step100）都是这个盲区。
2. **per-pair teacher 全局不自洽**：三个机制里只有 AB reliability 沾边（且只测噪声、不测系统偏差），gate 量的是 student-teacher 距离（不是 teacher 自身一致性），selector 逐 record 对 step0 比（不跨 pair 比）。**今晚没有一个机制直接测量"两个 probe pair 的 teacher 是否互相矛盾"——而这是 §3 里最有力的前置信号**。
3. **深度细节/融合覆盖率退化（F1 型）**：probe 的 depth 侧分量（couple 组级标量、feature 对 teacher 的 patch 相似度）都不含结构信息；abs_rel/chamfer 在事后也几乎不动。delivery_area 型失败是全盲区的。
4. **feature 蒸馏经共享 trunk 对 pose 的泄漏**（VGGT electro/office rel 关仍跌）：无机制监测 pose head 的被动漂移（probe rot 只测 teacher 分歧，不测绝对质量——probe_auc03 只在 VGGT 有且只在 4 帧探针对上算）。
5. **前 10 步 grad-cap 未标定期**（courtyard DA3 gn=133）：C_R 在 10 次 update 后才校准，冲突梯度在最有毒的场景恰好最大——门/reliability 都没有"开场保护"概念。

## 5. 对明早全量队列（v3 lane）的建议

v3 脚本已比今晚多：`--v2_ab_manifest`（reliability）、`--v2_rel_gate_deg 30`、`--v2_apply_selection`（DA3）/ `v2_materialize_selected.py`（VGGT）。基于今晚证据，建议：

1. **不要把 30° rel 门直接套到 VGGT lane**。terrains（54.2°）/courtyard（65.2°）/delivery_area（32.8°）三个最佳场景全部会被关掉 rel，预期损失 +50~90% 量级的涨幅。VGGT 的 pose_gate 语义（<3° 关）方向是对的（保住 relief 这类饱和场景），建议保留 3° 下沿 + 增加**上沿动态判断**：unmasked 分歧 ∈ (3°, 30°) 时不硬关，交给 AB q_rot 降权。
2. **DA3 lane 上 30° 门大概率也救不了 courtyard**：若 DA3 unmasked student≈teacher（饱和假说），门不触发；触发也只是关 rel，前 10 步无 cap 的毒害已经铸成。DA3 侧真正前置的是 **probe pair-split**：建议训练前算两个 probe pair 的 unmasked teacher 自洽度（或 A/B 分歧），pair-split >~20° 时 rel 全局降权（如 ×0.1）而非硬关——courtyard(51.5)/relief_2(29.7)/playground(22.9) 都能被压住，且不误伤 VGGT terrains（该规则只挂 DA3 lane）。
3. **selector 应用时修两个已知缺陷再上线**：(a) tau_qual=0.05 单分量一票否决太脆——terrains 因 couple +13% 全灭（误伤最佳）、playground 因 rkd +50% 全灭（正确救场），建议改分量级加权打分（rot 大幅收敛应能抵消 couple 小幅恶化）；(b) 记录并上报"selector  pick 与 step100 的 GT 差"，今晚数据表明二者相关性为负的场景真实存在（relief_2-DA3、terrains-VGGT），全量队列跑完要用这个做 selector 有效性审计。
4. **给 depth 侧补一个结构保真信号**：哪怕极简——probe 深度对 teacher 深度的边缘保持率/高频能量比，或融合前对预测深度做梯度幅值中位数的 scene-level 记录。delivery_area 型失败今晚零信号，全量队列（hiroom/scannetpp 无纹理区域更多）只会更常见。
5. **修 DA3 trace 落盘**：主 7 场景今晚没有 training_trace.csv，pair 级审计缺 DA3 半边。v3 起跑前先 smoke 验证落盘。
6. **grad cap 前 10 步保护**：C_R 未标定期对 rel 权重加一个临时上限（如 ramp 从 0.1×w 起），或在第 1 步就用瞬时 ||g_aux|| 做软 cap。courtyard step1 gn=133 vs 健康场景 0.01–1.3，量级差 100×，完全分得开。
7. **保留双 eval**：v3 已计划 selector-materialized ckpt 二次 eval（`eval32_metrics_V2selected.json`），务必保留并与 step100 并存对比；同时把 pose_gate/pair-split/mask-split 四个 scene 级量写进 analyze 输出，便于跑完后直接做覆盖审计。
8. **清理**：删除或重命名 workspace/protocol_v2/analysis/da3_eth3d.{json,md}（旧 mv13 run，数字会误导）。

## 6. 证据索引

- 逐场景 eval（DA3）：logs/v2_da3_eth3d.log:219-220, 432-433, 645-646, 858-859, 1071-1072, 1284-1285, 1497-1498；logs/v2_da3_eth3d_rest.log:219-220, 432-433, 645-646, 858-859
- selector 回放：logs/v2_da3_eth3d.log:210, 423, 636, 849, 1062, 1275, 1488；rest:210, 423；logs/v2_vggt_eth3d.log:71, 126, 181, 236；logs/v2_vggt_eth3d_rest.log:106, 161, 216, 271, 326, 381, 436
- VGGT eval JSON：workspace/protocol_v2/vggt_eth3d/eval32_metrics_V2_4scenes.json、vggt_eth3d_rest/eval32_metrics_V2.json
- VGGT pose_gate：logs/v2_vggt_eth3d.log:20, 78, 133, 188；rest:55, 113, 168, 223, 278, 333, 388（语义 = baseline student vs 8-view teacher unmasked 相对旋转中位，train_arms.py:3041, 3108-3114）
- probe traces：workspace/protocol_v2/{da3,vggt}_eth3d{,_rest}/probe_trace/*.jsonl（DA3 每行一步含 records 列表；VGGT 逐 record 一行）
- probe GT 探针（VGGT only）：workspace/protocol_v2/vggt_eth3d{,_rest}/probe_metrics.csv（probe_auc03/auc30、e_depth 逐步）
- training traces：workspace/protocol_v2/{vggt_eth3d, vggt_eth3d_rest, da3_eth3d_rest}/training_trace.csv（da3_eth3d 主 7 场景缺失）
- baseline：workspace/protocol_v2/baselines.json；DA3 baseline abs_rel/recon：artifacts/diagnostics/final_protocol/da3_baseline/eth3d_baseline.json、eth3d_recon_baseline.json
- 场景元信息：workspace/protocol_v2/ab_manifests/eth3d/*.json（N/tau/10 train + 2 probe pairs）
- 启动配置（等权版证据）：scripts/run_v2_lane_da3.sh、scripts/run_v2_lane_vggt.sh 的 V2 变量
- 机制代码：src/free_geometry/tta_v2/probe.py（probe 语义）、controller.py（tau_qual=0.05 单分量否决）、reliability.py（AB 权重，今晚未走）；src/depth_anything_3/test_time_adaption/protocol_v1.py:1397-1411（rel 门，今晚未激活）、1353-1355（grad cap 前 10 步未标定）、1096-1100（geo_w 路径，今晚未走）；diagnostics/free_geometry/train_arms.py:451-476（rel 门检查）、3088-3114（pose_gate）
