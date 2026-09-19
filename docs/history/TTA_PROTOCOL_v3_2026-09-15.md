# Free-Geometry TTA 协议 v3.2（2026-09-16，深夜实验闭环版）

**版本**：v3.2　**日期**：2026-09-16　**状态**：VGGT 四数据集定稿 + CTK/two_stage 证伪闭环
**取代**：v3.0（2026-09-15）的结果部分；span 门控 loss 定义沿用。
**DA3 侧对应文档**：`TTA_PROTOCOL_DA3_v1_2026-09-15.md`（v2.0，同晚重写）。

---

## 0. v3.2 一屏总览

```
门控（v3.1 起，只看在线量）：
    random 且 N ≥ 100  → RKDC1H = maskdistill + 1.5·rkd_huber(δ=0.2) + 1.0·couple
    否则               → maskrel  = maskdistill + 1.0·rel
训练：LoRA r32/α32（layers 0-23），heads 冻结，lr 3e-5 cosine，100 步
      （10 对 × 10 epochs）；early_stop 已实现（tail-10 vs prev-10 <2%，
      step ≥ max(30, warmup+10)），7scenes 实测全线 36-80 步收敛，省 ~50% 时间
评测：benchmark-100（seed42）/ allv（N<100）；paired vs 冻结 baseline
已关闭：CTK（camera-token KD，双模型证伪）、two_stage（scannetpp 证伪）、
        三族 GT-free 门控信号（证据见 DA3 v2.0 文档 §4）
```

## 1. 当前冠军成绩（口径 A：逐场景相对提升均值，paired vs 冻结 baseline）

| 数据集 | 冠军臂 | dAUC@3 | dF1 | 双 +5%？ |
|---|---|---|---|---|
| HiRoom (30) | maskrel @allv | **+18.54%**† | **+18.52%** | ✓ |
| ETH3D (11) | maskrel @allv | **+32.24%** | **+25.46%** | ✓ |
| ScanNet++ (20) | RKDC1H @100v | **+5.50%** | +2.22% | AUC✓ F1✗ |
| 7Scenes (7) | maskrel @100v | +0.62% | **+14.10%** | AUC✗ F1✓ |

（存档：`artifacts/diagnostics/final_protocol/<ds>/eval32_metrics_C2M.json` 与
`scannetpp_v3/eval32_metrics_RKDC1H.json`；mean 键不计入均值。
†HiRoom 勘误 2026-09-16：dAUC 剔除了 baseline AUC=0 的场景
`20241230/828760/cam_sampled_04`（29 场景均值），dF1 为 30 场景均值——
两个数字的场景集不一致；统一剔除时 dF1=+21.87%，统一保留（0/0 记 0）时
dAUC=+17.93%。详见 `docs/RESULTS_VERIFICATION_2026-09-16.md`。）

## 2. 缺口格的今晚攻关结论（全部实测，勿重跑）

### 2.1 ScanNet++ F1（+2.22 → 目标 +5）：two_stage 已证伪
- two_stage 0.7（前 140 步固定对、后 60 步端点锚定对，20 对 kind-tagged
  manifest，`final_protocol/scannetpp_2stage/`）：**step200 = −2.76/−4.84**；
  受控对比 step100→200：SE 相位单独造成 AUC −4.5、F1 −3.1。
- 结论：scannetpp（N=100-534）对端点锚定蒸馏仍太稀疏；two_stage 的适用域
  收紧为"仅 7scenes 型 N≈1000 连续视频"（DA3 侧同样结论，见其 §2）。
- RKDC1H 守擂（+5.50/+2.22）。F1 +5 在该格当前无已验证路径。

### 2.2 7Scenes AUC（+0.62 → 目标 +5）：无臂可用
- CTK（maskrel_CTK，early_stop）：AUC +0.58（不动）且 F1 +14.10→+5.25
  （砸穿）→ CTK 线双模型正式关闭（DA3 侧 w=3 退化/w=1 无效，同晚）。
- 历史臂全表（存档）：C2M_MC +3.60/+18.34（双优但已被作者否决）、
  REL10 +1.51/+14.45、REL2 +1.33/+14.96、maskrel +0.62/+14.10、
  RKDC1 系 AUC 全负（−3.1~−7.3）。
- baseline AUC 均值 0.2383（stairs 0.053）——**不是天花板，headroom 真实
  存在但当前臂家族够不到**，诚实标注为开放问题。
- maskrel 守擂（F1 +14.10）。

## 3. Loss 定义（沿用 v3.0 span 门控，未变）

```
L = L_maskdistill + 1.0·L_couple
    + 1[span > 150] · L_rel                    # 长跨距分支
    + 1[span ≤ 150] · 1.5·L_rkd_huber(δ=0.2)   # 短跨距分支
```
span = student 4 共享帧的帧下标跨距（max−min），纯数据集不可知。
（v3.1 实际执行：scannetpp 走 RKDC1H，其余走 maskrel；与 span 门控同族。）

## 4. 选帧 / 训练 / 评测

- 选帧：16:4（池≥16）/ 8:4（8≤N<16）；τ>0.55 稠密等距否则纯随机；10 训练对
  + 2 探针对；two_stage manifest（kind-tagged）仅用于已证伪的 scannetpp 实验。
- 训练：LoRA r32/α32（layers 0-23），heads 冻结，100 步；early_stop 已实现
  （`--early_stop`），two_stage（`--two_stage 0.7`，kind-tagged manifest 必需，
  与 early_stop 互斥）。
- 评测：benchmark-100（seed42）/ allv（N<100）；paired；baseline 永不重跑。
- early_stop 的 per-scene 停步落盘为 `C2M_*_step{K}` 多目录，评测前用
  `*_ES` 合并目录取各场景最终步（符号链接，见 scannetpp_2stage/7scenes 实例）。

## 5. 已知失败模式（沿用 v3.0 §5，新增两条）

1. 难场景型（容量瓶颈）、上下文迁移病、长跨距漂移、HiRoom CD/AbsRel 微涨
   ——见 v3.0 §5，结论不变。
2. **CTK 融合毒**（新）：camera-token 直接蒸馏在两种模型上都以 F1 为代价
   换不到 AUC；B1 空间问题（norm 选择）修复后结论不变，机制层面与融合
   目标冲突。
3. **SE 相位稀疏毒**（新）：端点锚定蒸馏在 N<~500 的一切场景上腐蚀
   depth/pose 双 head（eth3d/hiroom/scannetpp 四组对照实验一致）。

## 6. 被证伪的路线（勿再走，v3.2 增补）

v3.0 原列表（rel/abs 加回、couple 加重、RKLH、XRKD/XAC/XAP/TGM、归一化绝对
位姿靶、24:8+重位姿项、fixfree 作为主结果）**加上**：
CTK（一切 camera-token 直接蒸馏）、two_stage（N<500 的任何场景）、
混合采样任何比例（DA3 侧证据：梯度干扰，混及格不超纯格）、
三族 GT-free 门控信号（轨迹/probe 残差/全序列漂移，18 场景证据链）。
