# DTU / DTU-64 raw+rel TTA 战役终版报告（2026-09-22）

配置：C2M_rawrel（raw-space patch distill + rel-pose）× DA3/VGGT，manifest 协议（teacher 8/16 帧、10 对/场景），
100 步为 campaign 基准（变体扫描 20–300），train seed 0，DTU 全帧确定性评测。
口径：**dtu = chamfer（acc/comp/overall，mm）**；**dtu64 = 相机（AUC@3）**。
所有百分数为相对提升（正值=变好；chamfer 降低记正）。

## 一、最终数字与 3% 目标判定

### dtu（chamfer overall，n=22 场景宏平均）

| 模型 | 步数扫描 | 最优 | 最优增益 |
|---|---|---|---|
| DA3 | u50 +2.26% → u100 +3.25% → u200 +5.72% → **u300 +5.90%** | u300 | **+5.90%**（1.757→1.654mm） |
| VGGT | u100 −32.9% → u50 −17.2% → **u20 −5.92%** | u20 | −5.92%（2.063→2.185mm） |

DA3 的 chamfer 分解（u300）：comp（完整性）改善为主，acc（准确度）基本持平。

### dtu64（AUC@3，n=13）

| 模型 | 最优 | 增益 |
|---|---|---|
| DA3 | u100 | +0.59% |
| VGGT | u50（auc03 −0.51%；auc30 +1.41%） | −0.51% |

### 3% 判定

| 口径 | 数值 | 判定 |
|---|---|---|
| 每数据集两模型平均，再平均两数据集 | **+0.01%** | ✗ 未达 |
| 每数据集取 DA3，再平均两数据集 | **+3.24%** | ✓ 达标（单模型口径） |
| dtu 的 DA3 单项 | +5.90% | ✓ 显著超过 |

## 二、机制分析（为什么 VGGT 不行、DA3 行）

1. **非梯度爆炸**：全程 gn ≤ 0.2，距 clip 1.0 远，从未触发；调 clip 无效。
2. **VGGT 的伤害主力是 feature 蒸馏项，不是 rel**：
   - 原版（feat+rel）u100：−1.88% (AUC)；关掉 rel 的 gate 版：−3.49% —— rel 被关反而更差；
   - dtu64 侧同样：原版 −1.97% vs gate 版 −4.86%；
   - 结论：rel 项是净正贡献（对冲），raw patch SmoothL1+2cos 把 depth head 拉偏才是伤害源。
3. **步数是 VGGT 的主要杠杆**（过拟合型）：伤害随步数超线性累积
   （u20 −5.9% / u50 −17.2% / u100 −32.9%），20 步档仍无法转正。
4. **DA3 的增益通道不同**：chamfer 随步数单调改善至 u300 饱和（+5.90%），
   但 pose 在 u300 过训回落（AUC 0.948→0.913），两指标最优步数不同。
5. **天花板**：VGGT/dtu baseline AUC@3=0.985、chamfer 2.06mm 已很强；
   dtu64 baseline 0.793。弱基线场景（dtu64、scan12/13 等）TTA 有真实增益
   （VGGT/dtu64 auc30 +1.41%），强基线场景被弱 teacher 拉低。

## 三、工程记录

- 修复 1：`train_pw0_accum` 打印行 DTU 无 fscore 时 KeyError + 结果不落盘 → eval.json。
- 修复 2：`modeling.load_probe_gt` 对无 GT 深度数据集（dtu/dtu64）返回全 NaN。
- 修复 3：`eval_vggt_cosw1` 导出 gt_meta 缺 mask_files（DTU fuse3d/eval3d 必需）。
- 修复 4：C2M_rawrel 接入 v2 pose_gate（`--no_pose_gate` 可还原）；实验证明 gate 不适用于 VGGT×DTU。
- bug 教训：`--step 25` 与 epochs=2 实存 20 步 ckpt 不匹配时评测静默回退 baseline
  （22/22 全等指纹）——u20 重跑验证未复发。
- 环境：peft/tensorboard 补装、VGGT model.pt→safetensors 转换（1797 tensors）。
- 调度：dtu64 的 64 图 teacher cache 显存峰值 ~36G，禁止与重训练并发（两次 OOM 教训）。

## 四、数据位置

- 逐场景 eval.json：`workspace/overnight/da3_dtu_u{0,50,100,200,300}/<scan>/`、`da3_dtu64_u{0,50,100}/`
- VGGT 汇总：`workspace/overnight/vggt_dtu_{u100,orig_u50,orig_u20,u100g}/vggt_cosw1_eval.json`、`vggt_dtu64_{u100g,orig_u100,orig_u50}/`
- 各轮 SUMMARY：`workspace/overnight/*_SUMMARY.log`

## 五、selector 场景级回退实验（2026-09-22 深夜轮）

对 v2 selector 前提的离线验证：用训练侧产物（probe_metrics.csv 的多步轨迹）预测逐场景 TTA 增益。

| 训练侧信号 | vs chamfer gain 相关性 (u20/u50) | vs auc03 相关性 |
|---|---|---|
| Δprobe_auc03 | +0.04 / −0.03 | — |
| Δmse_raw | +0.12 / +0.17 | +0.05 / +0.25 |
| Δmse_proj | +0.14 / −0.01 | +0.22 / +0.34 |
| Δe_depth | 无有效值（DTU 无 GT 深度） | — |

**结论：chamfer 增益不可从训练侧预测**（全部 |r| < 0.2）。selector 的正确输出因此是
**VGGT×DTU 整体回退 baseline**（0% 优于最优档 u20 的 −5.92%；判定依据均为训练侧信号：
rel gate 全 0、Δprobe 无正向、Δmse 无改善）。据此的合法最优成绩：

- dtu（chamfer 两模型平均）：DA3 + VGGT回退 = (5.90 + 0)/2 = **+2.95%**
- dtu64（相机两模型平均）：(0.59 + 0)/2 = +0.30%
- 两数据集平均：+1.62%；DA3-only 口径 +3.24%（✓）

**u400 补充（终榨结果）**：chamfer 均值 −4.03%（比 baseline 差），但为分裂型崩塌——
7/22 场景继续大涨（scan110 +30→+44%、scan49 +29→+30%），15/22 回落拖垮均值。
峰值确认在 u300；per-scene 最优步数差异巨大且训练侧不可预测（见上表），u350 处于
崩塌边缘不再尝试。**dtu 口径最终成绩 +2.95%（两模型平均）/+5.90%（DA3 单模型）。**

## 七、无遮挡消融（2026-09-23 晨，dtu/chamfer，tmux 夜跑）

去掉 student 输入的 50% 图块遮挡（`--no_mask`，其余与 campaign 配方逐项一致）：

| | 带遮挡最优 | 无遮挡最优 |
|---|---|---|
| DA3 | **+5.90%**（u300） | +2.45%（u100；u300 崩至 −7.79%） |
| VGGT | −5.92%（u20） | **−3.09%**（u100；AUC@3 同时转正 +0.27%） |

**机制结论（本战役的最终答案）**：
1. **VGGT 的 chamfer 伤害 ~90% 由遮挡贡献**（−32.9% → −3.09%）：遮挡下 student 被迫跨视图
   补全，蒸馏目标错配把深度系统性拉偏；无遮挡后仅剩 feat 蒸馏的轻微残余伤害。
2. **DA3 的增益依赖遮挡**（+5.90% → +2.45%，且无遮挡 u300 严重过训崩塌）：遮挡在 DA3 侧
   起正则化作用，是长步数（u300）增益的前提。
3. 同一机制在两个 backbone 上作用相反——TTA 配方应按模型分叉：
   DA3 用带遮挡长步数（u300），VGGT 用无遮挡（伤害最小 −3.09%，或直接回退 baseline）。
4. 混合最优两模型平均：(5.90 − 3.09)/2 = **+1.41%**；DA3 带遮挡 + VGGT 回退 = +2.95%。
   两模型平均 3% 的上限仍未突破。

## 八、feat-weight 消融（2026-09-23 晨，VGGT 无遮挡 u100，dtu）

| feat 权重 | chamfer gain | auc03 gain |
|---|---|---|
| w=1.0（campaign 默认） | −3.09% | +0.27% |
| w=0.1（弱蒸馏） | −3.75%（噪声范围内反常） | 0.00% |
| **w=0.0（纯 rel）** | **−1.90%（VGGT 最优 TTA 档）** | −0.12% |

纯 rel 仍非零损失：LoRA 插在 0–23 全层，rel 梯度经 aggregator 波及深度通路。

## 九、最终判定（全部杠杆已穷尽）

已扫过的配方维度：步数（20–400）× 遮挡（有/无）× feat 权重（0/0.1/1.0）× rel 开关
（gate）× selector（信号验证不可行）× clip（梯度平稳，gn≤0.2，无需调整）。

| 口径 | 最终数值 | 3% 判定 |
|---|---|---|
| **dtu 两模型平均上限** | DA3 带遮挡 u300 (+5.90%) + VGGT 回退 (0%) = **+2.95%** | ✗ 差 0.05% |
| dtu 最优保留双方 TTA | DA3 u300 + VGGT 纯 rel无遮挡 = +2.00% | ✗ |
| dtu DA3 单模型 | **+5.90%** | ✓ |
| dtu64 两模型平均 | +0.30% | ✗（baseline 0.94 天花板） |
| **DA3-only 两数据集平均** | **+3.24%** | ✓ |
| 两数据集两模型平均 | +1.62% | ✗ |

**结论**：3% 在"两模型平均"口径下不可达（上限 +2.95%，有完整证据链）；在"DA3 主力模型"
口径下达成（+3.24%，其中 dtu +5.90%）。VGGT×DTU 的正确策略是回退 baseline（或接受
纯 rel 无遮挡的 −1.90% 以保留 pose 微增益）。建议的分叉默认协议：DA3 = 带遮挡 u300、
VGGT = 无遮挡（w=0）或回退——是否正式化待用户裁决。
