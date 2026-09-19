# DA3 TTA 统一协议 v2.0（2026-09-16，深夜实验闭环版）

> 适用模型：DepthAnything3-GIANT-1.1（DA3）。与 VGGT 侧 `TTA_PROTOCOL_v3` 同族的
> per-scene test-time adaptation 协议。**本协议所有分支只依赖可测量量（N、τ、
> step-0 残差、收敛曲线），不出现数据集名/模型名。** 评测指标口径：AUC@3 与 F1
> 为主指标（论文口径），CD/AbsRel 附带。
>
> v2.0 相对 v1.0 的变化：two_stage 顺序课程成为稠密长序列主轴；dw5（depth
> distill ×5）永久删除；CTK（camera-token KD）双模型证伪并关闭；三族 GT-free
> 门控信号全部证伪，协议改走"构造上安全"路线。

## 0. 一屏总览（最终配方）

```
if N >= 500:   # 稠密连续视频（7scenes 型）
    采样 = 20 对 combo（固定槽位 + 端点锚定各半，kind 标记）
    课程 = two_stage 0.7（前 70 步只用固定对，后 30 步只用端点对）
    比率 = ratio_mix {8:4, 16:4, 24:8}（每对随机一比率）
    步数 = 100；early_stop 关闭（相位结构自带边界）
else:          # 稀疏宽基线（hiroom/eth3d 型）与一切 N<500
    采样 = 10-20 对固定槽位（N≥100→20 对，N<100→10 对）+ 2 对 probe
    比率 = 纯 8:4（16:4 在稀疏场景 teacher 自退化，已证伪）
    步数 = 上限 100；early_stop 开启（tail-10 vs prev-10 <2%，
           step ≥ max(30, warmup+10)）
Loss（全场景统一）= maskdistill(1.0) + 1.5·rkd_huber(δ=0.2) + 1.0·couple
训练 = LoRA r32/α32（40 层）+ camera token 可训练 + AdamW 3e-5 cosine
评测 = N≥100 → benchmark-100（seed 42）；N<100 → allv；AUC@3 + F1 一体同出
```

## 1. Loss 细节（rkdc1h，全场景统一）

- **maskdistill(1.0)**：student 输入遮 50% 的 14×14 patch（seeded），在被遮位置
  对 teacher（无遮）的 tap[19,27,33,39] 3072 维 token（过 head.norm，与 DualDPT
  首个算子相同）做 Huber(β=1)+2·(1−cos)，teacher conf 加权。
- **1.5·rkd_huber(δ=0.2)**：共享帧相机中心的轨迹形状（均值归一化成对距离 +
  三角角，anchor-free，量规无关），w2c 外参经冻结 cam_dec 解码。
- **1.0·couple**：深度-位姿量规耦合标量（log RMS_centers − log mean_depth，
  teacher conf 5% 分位门控），一个前向一个标量。
- **已删除**：dw5（depth_distill ×5，在两个数据集上验证为负贡献，永久禁用）；
  rel 项（DA3 饱和，有害）；ctk（见 §4 判决）。

## 2. two_stage 顺序课程（仅 N≥500）

固定槽位对建位姿结构（利 AUC），端点锚定对补长程一致性（利 F1）。
**混合（同 epoch 混训）会产生梯度干扰，混及格落在两个纯格之间**（7scenes 实测：
0%SE→+5.63/+3.08，25%SE→+2.87/+2.82，50%SE→+3.58/+3.94，100%SE→+0.37/+5.33）。
顺序执行（先固定后端点）隔离干扰，是当前唯一双过线配置：
**7scenes +7.62% AUC / +5.50% F1（均值，paired vs baseline）。**

适用边界（实验闭环，勿越界）：
| 场景类型 | 结果 | 判定 |
|---|---|---|
| 7scenes（N=1000 连续视频） | +7.62/+5.50 | ✓ 唯一赢家 |
| eth3d（N=14-76 稀疏摆拍）8:4 / 16:4 | +0.12/−6.01；−2.51/−6.31 | ✗ 两比率均失败 |
| hiroom（N=10-23）16:4 全帧 teacher | −1.37/−3.87 | ✗ |
| VGGT scannetpp（N=100-534）16:4 | −2.76/−4.84 | ✗（SE 相位单独 −4.5 AUC） |

机理：端点锚定蒸馏只在"端点帧仍共享大量场景内容"的稠密长序列成立；
稀疏宽基线场景端点对近乎不重叠，对 depth/pose 两个 head 都是腐蚀。

## 3. 选帧规则（零 GT）

- τ 调度：τ > 0.55 → dense_equidistant_sift，否则纯随机；seed=43 冻结。
- combo 对生成（N≥500）：每对以 0.5 概率取固定槽位（shared 槽位 [0,2,4,6]）
  或端点锚定（首+尾+随机中间帧，L≥4），kind 标记写入 pair。
- ratio_mix（N≥500）：每对从 {8:4,16:4,24:8} 随机取一比率（SelfEvo 式混合
  上下文非对称）；ns ≤ tn//2 槽位约束已内置。
- 稀疏场景（N<500）：纯 8:4 固定槽位；16:4 被否（teacher 窗口掺入低重叠帧，
  teacher 位姿自退化并传给学生）。
- teacher/student 集合互异（禁区去重）；评测帧 N≥100 → benchmark-100，否则 allv。

## 4. 已关闭路线（全部实测证伪，勿重启）

1. **CTK（camera-token 直接蒸馏）**：DA3 w=3.0 → 7scenes +2.05/+2.89（全面退化，
   尾巴 redkitchen −7.6/stairs −7.9）；w=1.0 → 无效。VGGT 7scenes w=1.0 →
   AUC +0.58（不动）且 F1 +14.10→+5.25（砸穿）。VGGT scannetpp 历史
   （B5_CTK）：AUC +4.47 / F1 −3.41。B1 空间问题（head.norm vs token_norm）
   已修复后结论不变 → **机制本身与融合目标冲突，双模型关闭**。
2. **GT-free 门控回退**（三族信号全部证伪，18 场景证据链）：
   - 训练轨迹（loss 走势/grad 尖刺）：office gn=16.7 却 F1+21.8、redkitchen
     平静却受损 → 不可分。
   - 4 帧 probe 对残差：relief_2 残差改善最多（d_cp −3.29）受损最狠
     （−15.6/−26.4）→ 与训练对同分布，看不见 100v 漂移。
   - 全序列漂移统计（couple/中心散布/深度比）：符号规则跨数据集翻转
     （7scenes 负漂移=受损 vs eth3d 正漂移=受损），eth3d 内部 playground
     （最好）与 facade（最惨）共享负漂移签名 → 无阈值可分。
3. **混合采样任何比例**（3:1 / 50:50）：梯度干扰，双指标均不超过纯格。
4. **dw5 / rel 项 / 16:4（稀疏场景）/ EMA teacher / SelfEvo 完整版**。

## 5. 诚实边界（写进论文）

- **DA3-scannetpp（天花板格）**：baseline AUC 0.8467，student(4v)≈teacher(16v)
  （rel 残差 0.0075、rkd 0.002、深度差 ~4% 均匀）。teacher 没有比 student
  多知道任何东西，任何 loss 都榨不出 +5%（全部配置 ±0.5% 内）。本协议在该
  格保持双微正、不伤害，并如实标注为不可达。
- **DA3 稀疏数据集（eth3d/hiroom）**：最优分别为 +2.95/−1.21 与 +3.33/+0.62。
  【勘误 2026-09-16】eth3d 数字来自 `workspace/da3_protocol_eth3d_t8s4`
  （纯 8:4，**无 early_stop**——es 变体全部 SIGTERM 崩溃、输出目录为空，
  见 `da3_baseline/e3_es*.log`）；hiroom 数字来自 es 变体，但同配置不开 es
  的 `da3_protocol_hiroom_t8s4` 更高（+3.74/+1.57）。
  双 +5 不可达的机制证据完整（two_stage 失败 + 门控证伪 +
  饱和模型 headroom 分析）。
- **场景级 F1 +5% 不可能全覆盖**（kicker F1=0.9983、office 0.9965）；指标按
  数据集均值判定。

## 6. 工程纪律

- baseline 永不重跑、永不删除（`artifacts/diagnostics/final_protocol/da3_baseline/`）。
- 串行 1 路训练；`PYTORCH_ALLOC_CONF=expandable_segments:True`；启动前 pgrep。
- 训练进程存活期间不改 `protocol_v1.py`；改动必 `py_compile`。
- 接力链用受跟踪后台任务 + 锚定 pgrep（`^python3 scripts/...`），nohup bash -c
  形式会被静默回收（已踩坑）。
