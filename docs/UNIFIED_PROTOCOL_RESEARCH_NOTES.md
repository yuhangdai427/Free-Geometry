# 统一 TTA Protocol 研究笔记（2026-09-16）

目标：把当前"每个 (模型 × 数据集) 格各自调最优配置"的现状，收敛为一个
**dataset-agnostic 且 model-agnostic 的统一 protocol**——所有分支只依赖
**在线可观测、GT-free 的量**（场景帧数 N、几何重叠度 τ、step-0 残差、
训练 loss 曲线），杜绝任何"某数据集/某模型专用"的超参或 loss 组合。
判定标准沿用 runbook 的提法：同一 loss、同一超参下，四数据集 × 双模型
的 8 格 agreement ≥90%（每格两指标与该格历史冠军差 <1% 视为 agreement）。

这不是超参微调：分支条件必须是**测量量**，不是数据集名或模型名。

## 1. 现有可观测条件机制（全部已实现，可直接复用）

| 观测量 | 定义与实现 | 已承担的决策 |
|---|---|---|
| τ（重叠度） | 相邻帧 SIFT 匹配率中位数（≤40 等距对，320px/400 特征/0.75 ratio），`protocol_v1.py:compute_tau` | τ>0.55 → dense_equidistant_sift 选帧；否则 random 窗口。四数据集实测分布：7scenes 0.67–0.73，scannetpp 0.34–0.55，hiroom 0.04–0.20，eth3d 0.06–0.24（分离干净） |
| N（帧数） | 场景帧数 | `teacher_N = 16 if N≥16 else 8`；N<8 弃场景；**评测** N≥100 → benchmark-100 否则全帧。文档级（未进代码）：N≥100→n_train=20；N≥500 → combo+two_stage 分支 |
| step-0 残差 | LoRA 重置后（student≡baseline）teacher-student 各 loss 分量值，`scripts/gate_da3.py:scene_residuals` | 原设计：rel 残差超阈值则加回 rel@1.0（DA3 v1.0 草案，后被 v2.0 删除——rel 在 DA3 判定饱和有害）；残差**族**作为门控信号已被证伪（§4） |
| loss 曲线 | tail-10 vs prev-10 改善率、梯度范数 | early-stop（2%/30 步/warmup+10 下限），全协议默认开启 |
| probe 门控 | 2 个 held-out probe 对上的指标变化 | `deploy_accept`：τ≤0.55 ∧ N≥100 时 Δprobe_auc03>−0.01 才接受 TTA。**注意 Δprobe_auc03 含 GT**，不可作最终 protocol 的门控；真 GT-free 替代是 mse_raw/loss_slope/gnorm 族（但也已证伪，见 §4） |

## 2. 当前每格冠军配置的分歧维度（要统一的对象）

| 维度 | DA3×7s | DA3×eth3d | DA3×hiroom | DA3×spp | VGGT×7s | VGGT×eth3d | VGGT×hiroom | VGGT×spp |
|---|---|---|---|---|---|---|---|---|
| loss arm | rkdc1h | rkdc1h | rkdc1h | rkdc1h | maskrel | maskrel | maskrel | **RKDC1H** |
| teacher:student | ratio_mix 8:4/16:4/24:8 | 8:4 | 8:4 | 8:4 | 16:4 | 8:4 | 8:4 | 16:4 |
| 帧策略 | combo 0.5 + **two_stage 0.7** | fixed | fixed | fixed | fixed | fixed | fixed | fixed |
| early-stop | ✗（two_stage 禁用） | ✗（es 崩溃未复现） | ✓ | ✓ | ✓(v3.2) | ✓ | — | ✗ |
| n_train | 20 | 10 | 10 | 10 | 10 | 10 | 10 | 10 |
| LoRA 深度/特殊 | 40 层 + camera token 可训练 | 同左 | 同左 | 同左 | 24 层 + token 冻结 | 同左 | 同左 | 同左 |
| tap 层 | [19,27,33,39] | 同左 | 同左 | 同左 | [4,11,17,23] | 同左 | 同左 | 同左 |

观察：**loss arm 在 DA3 上已统一（恒 rkdc1h）**；VGGT 上仍按数据集切换
（maskrel vs RKDC1H）。帧策略只在一个格（DA3×7scenes，N≈1000 连续视频）
用了 two_stage。ratio 的分歧大致跟随 N。

## 3. 已有的两个"统一草稿"（互相独立，尚未合并）

- **VGGT v3（`TTA_PROTOCOL_v3_2026-09-15.md` §0）**：
  `random(τ≤0.55) ∧ N≥100 → RKDC1H；否则 → maskrel`。
  等价的 span-gate 版（§3：`span>150 → rel；span≤150 → 1.5·rkd_huber`，
  span = 学生 4 共享帧的帧号跨度）**只写在文档，代码未实现**——span 是比
  (τ,N) 更细的可观测量，值得实现后对比。
- **DA3 v2.0（`TTA_PROTOCOL_DA3_v1_2026-09-15.md`）**：
  `N≥500 → 20 对 combo + two_stage 0.7 + ratio_mix{8:4,16:4,24:8}，无 es`；
  `否则 → 10–20 对 fixed 8:4 + es`。loss 全场景统一 rkdc1h（rel/ctk/dw5 已删）。

两份草稿的共同点：分支条件只用 (τ, N)；分歧点：**跨模型的 loss 选择规则
不存在**——同一格（scannetpp）VGGT 用 RKDC1H 而 DA3 用"任意（饱和）"，
7scenes 型数据 VGGT 选 maskrel 而 DA3 选 rkdc1h+课程。这是统一 protocol
的核心空白。

## 4. 迁移性证据库（设计统一规则时的硬约束）

**跨格成立（可进统一协议）：**
- 固定槽位选帧→利 AUC；端点锚定→利 F1（4 组受控对照，3 数据集方向一致）。
- 端点锚定蒸馏在 N≲500 的稀疏场景**双杀**（SE 相位单独 −4.5 AUC）；
  仅 N≈1000 连续视频受益 → two_stage 的适用域 = N≥500~1000。
- couple 项在双模型上都治"深度-位姿量规脱钩"型 F1 伤（1ada F1 −31.6→+24.1）。
- maskdistill 主干在全部 8 格非负。
- early-stop 在小场景无损省时（7scenes 36–80 步停）。
- 混合采样任何比例都产生梯度干扰，混合格永远介于两个纯格之间。

**不迁移（统一规则必须绕开的坑）：**
- rel-pose：dense 视频/小场景有益，scannetpp(random 支路)上毒（VGGT F1 −2.0），
  DA3 上饱和有害。
- ctk（camera-token KD）：双模型上都拿 F1 换 AUC 不划算（已双模型关闭）。
  注意 DA3 版 ctk 实现用了深度头 norm 而非相机路径真实读出空间（raw token），
  "方向冲突"结论部分是伪影的可能未排除。
- probe 门控：τ>0.55 反相关（7scenes 6/7 场景符号错）；24:8 上完全失效。
- 三族 GT-free 门控信号（loss 轨迹/probe 残差/全序列漂移）：18 场景证伪，
  符号跨数据集翻转（7scenes 负漂移=受损 vs eth3d 正漂移=受损）。
- DA3-scannetpp：teacher(16v)≈student(4v)（rel 残差 0.0075），任何 loss
  榨不出 +5%，统一协议在该格只需"不伤害"（回退 baseline）。
- 单一 loss 打全场不存在（C2M_MC/SCL 在 hiroom 各丢一项）。

## 5. 空白点（后续研究的机会）

1. **跨模型 loss 选择规则**：VGGT 按 (τ,N) 切 maskrel/RKDC1H，DA3 恒 rkdc1h。
   候选统一观测量：**step-0 rel 残差**（原本就是为这个设计的：残差大 =
   teacher 在相对位姿上有信息增量 → rel/rkdc1h 有东西可学；残差≈0 = 饱和
   → 只留 maskdistill+couple）。它作为"门控信号"被证伪，但作为"**loss-arm
   选择器**"是另一个问题（选择器只需在 arm 间排序，不需要预测逐场景涨跌），
   未被实验过。
2. **span 门控未实现**：v3 文档的 span（学生共享帧帧号跨度）是比 τ 更直接
   的"相机运动幅度"代理，代码里没有。实现成本≈0（manifest 里就有帧号）。
3. **SFT 不存在**：全仓库无任何 supervised fine-tuning 机制（检索零匹配）；
   若指"用 SfM/自监督信号微调"，需新建；若指"TTA LoRA 本身"，即现有机制。
4. **teacher 质量的在线估计**：τ 只度量相邻帧重叠，不度量 teacher 本身
   在该场景的可靠性（scannetpp 饱和格的本质）。step-0 深度残差的分布
   （是否 ~4% 均匀）是候选的"饱和检测器"，未系统化。
5. **两套管线的模型抽象层**：DA3 的 `student_forward_c2m` 返回
   (tap_feats, ext_w2c, depth[, cam])，已经是模型中立的接口；VGGT 侧分散在
   modeling.py。统一 protocol 落地时先抽 teacher-cache / student-forward /
   head-norm 三个接口，rkdc1h/maskrel 全部写在矩阵层面（已 gauge-free，
   可直接复用）。

## 6. 下一步实验设计（候选）

**E1（统一规则 v0，纯文档可验证）**：
```
观测 (N, τ, step-0 rel 残差 r₀，深度残差分布)：
  r₀ < ε（饱和，如 DA3-scannetpp）      → maskdistill + couple（回退保护 + es）
  τ ≤ 0.55 且 N < 500（稀疏摆拍）        → rkdc1h, 8:4 fixed, es
  其余（N ≥ 500 连续视频或 τ > 0.55）    → rkdc1h, combo + two_stage 0.7
                                          + ratio_mix, n_train=20
```
在现有 8 格存档上离线核对：该规则每格选出的配置 vs 该格历史冠军的差距
（全部 ckpt/metrics 都在，无需 GPU）。
**E2（LODO 验证，防"换汤超参"）**：把 8 格当 8 个留一验证折——规则超参
（τ 阈值、N 阈值、ε、two_stage 比例）只在 7 格上定，锁死后在第 8 格冷启动
验证；8 折全过才算"统一"而非"过拟合"。
**E3（span 门控 A/B）**：实现 span 分支，与 (τ,N) 分支在 eth3d/hiroom/
scannetpp 上对比切换一致率。
**E4（r₀ 作为 loss-arm 选择器）**：在 VGGT-scannetpp（r₀ 大）与
DA3-scannetpp（r₀≈0.0075）两端验证：r₀ 大 → RKDC1H 优于 maskrel，
r₀ 小 → 两者无差 → 用 r₀ 切换无损。

## 7. 相关文件索引

- 协议实现：`src/depth_anything_3/test_time_adaption/protocol_v1.py`、
  `diagnostics/free_geometry/{train_arms,abs_pose_loss,modeling,build_final_manifest}.py`
- 门控分析：`scripts/gate_analysis_gt_free.py`、`scripts/gate_analysis_s8t24.py`、
  `scripts/gate_da3.py`、`artifacts/diagnostics/final_protocol/GATE_ANALYSIS*.md`
- 证据文档：`docs/TTA_PROTOCOL_v1/v3/DA3_v1`、`docs/EVAL_AUDIT.md`、
  `docs/RESULTS_VERIFICATION_2026-09-16.md`
- 结果存档：`artifacts/diagnostics/final_protocol/`（baseline 与 metrics 永不删除）、
  `workspace/da3_*/smoke_summary.json`
