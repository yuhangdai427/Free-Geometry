# Protocol v2 — Relative Rotation Loss 设计说明（2026-09-19 02:45）

## 0. 证据基础（今晚 eth3d 实测 probe 轨迹）

逐 pair 的 probe rot_deg（teacher-student 相对旋转角距离，gauge-free）：

| 场景 | step0 (p0/p1) | step100 | 结果 |
|---|---|---|---|
| courtyard | 84.7°/33.2° | 84.2°/31.6° | AUC −18.6% |
| delivery_area | 2.5°/1.5° | 平 | AUC +27.5% / F1 −19.1% |
| pipes | 0.83°/2.39° | 缓降 | AUC +9.9% / F1 +1.3% |

三个事实：
1. **双峰分布**：≤几度可学；≥30° 学不动（100 步不动）。Huber（knee 20°）只能限幅，不能识别。
2. courtyard 的 couple 同步漂移（0.074→0.108）——位姿崩与尺度崩同发。
3. diag2 既有证据：坏情形下 ‖g_R‖=152 vs ‖g_base‖=6.15、cos(g_R, g_base)=−0.78。

## 1. 稳健性设计（三层过滤，按成本排序）

**(a) 逐边 Huber（已实现）**：ℓ_ij = 16·H_δ(d_ij)，d=sin(θ/2)，δ=sin(10°)（knee=20°）。零成本，限制极端边幅度。

**(b) ~~cycle-consistency~~（已废弃——数学上恒空）**：绝对位姿导出的相对旋转的三角闭合是恒等式（R_iR_kᵀ·R_kR_jᵀ·R_jR_iᵀ ≡ I 对任意旋转成立），不携带任何 teacher 错误信息。cycle 检验只对独立两两估计有意义，对本场景（绝对位姿 teacher）无效。已写代码验证后删除（identity rig 与 90° 破坏 rig 的闭合误差恒为 0，实证了这一点）。

**(c) A/B 上下文一致性（已接线）**：q^R_AB = 1/(1+(u^R/τ_R)²)，u^R 为两个不同 extras 上下文下同一真实边的相对旋转角差，τ_R 为场景中位数。这是可靠性的主信号。

**(d) 场景级 rel 门（新增，回答"提前检测"）**：step 0 时先做一次**无 mask 的** student 前向（2 个 probe pair 各一次，廉价），算 teacher-student rot_deg：
- masked probe 高而 unmasked 低 → 差异来自遮挡（可学），放行；
- **unmasked 也高（场景中位 > 30°）→ 教师目标对该场景不可信/不可学 → 该场景关闭 rel 分支**（v2_rel_weight 强制为 0），JSONL/日志显式记 `rel_gate=off`。
- courtyard 预期触发；pipes/delivery_area 预期放行。

**(e) ramp（新增）**：λ_R(t) = w_R · min(1, t/20)。早期特征尚未稳定时不给旋转压力（diag2 的爆炸都在早期）。

## 2. 融合设计（梯度层，与既有 loss 共存）

最终更新规则（每个 optimizer step）：

```
g = g_base + λ_R(t) · min(1, C_R/‖g_R‖) · g_R
```

- **两管线统一为全局范数 cap**（保方向；VGGT 逐参数版废弃）。
- **R 与 T 拆分**：`--v2_rot_weight` 与 `--v2_tdir_weight` 独立；梯度范数分开记录（g_rot/g_tdir/g_base 每步入 trace）。tdir 保留近零基线跳过 + q^T。
- 每步记录 cos(g_R, g_base)：持续 <0 是场景级"梯度冲突"证据，进分析报告（不在线动权重——用户的计划明确禁止动态调权）。
- 固定 λ_R=1.0（与臂约定一致），不强制 1:1、不放大小梯度。
- 多对累积（accum=2）：今晚不启用（每场景成本翻倍），列为下一步旋钮。

## 3. 与回退机制的衔接

- probe 分量里 rot_deg 已存在；selector 的资格规则会自动惩罚旋转恶化的 ckpt。
- rel_gate=off 的场景：rel 不参与训练，probe 仍记录 rot_deg（观察用）。
- 若某场景 rel_gate=off 且 probe 各分量无改善 → selector 自然回退 baseline。

## 4. 验收标准（实现后）

- courtyard 类场景：rel_gate=off 触发，训练全程无 rel 分支。
- 合成三角闭合测试：故意给 teacher 一条 90° 错边 → 该边 q_cyc 显著低于一致边。
- ramp：step<20 时 λ_R 线性增长，trace 可见。
- 回归：flag 全关 = 位级旧行为。
