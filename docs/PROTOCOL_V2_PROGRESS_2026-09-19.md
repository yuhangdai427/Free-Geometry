# Protocol v2 隔夜实验 · 阶段汇总（2026-09-19 11:05 更新）

最终统一 config（冻结）：A/B 双上下文 manifest（VGGT 16:4 / DA3 8:4）+ 几何侧可靠性（q_rot/q_tdir/geo_w，q_feat 关闭）+ 旧 couple（couple_fix 关闭）+ 稳健 R（逐边 Huber + ramp + 全局 cap）+ tau 门（rel，τ>0.55 关闭，仅 DA3）+ probe 每 10 步 + selector 执行链。

## 已完成数据集（截至 09:45）

| 模型 | 数据集 | 场景数 | ΔAUC3 | ΔF1 | ΔChamfer | selector（回退/改善） |
|---|---|---|---|---|---|---|
| DA3 | eth3d | 11 | +3.3% | −4.4% | +7.6% | 1 / 10 |
| DA3 | 7scenes | 7 | +3.8% | −0.7% | +0.4% | 1 / 6 |
| DA3 | scannetpp | 20 | −0.6% | −0.7% | +0.6% | 11 / 9 |
| DA3 | hiroom | 30 | **+4.1%** | **+1.6%** | +16.8%⚠ | 9 / 21 |
| DA3 | dtu | 22 | +0.4% | n/a（用 CD） | final +63% → sel **+16%**⚠ | 8 / 14 |
| VGGT | eth3d | 12 | +18.6% | +0.2% | +2.7% | 1 / 10 |
| VGGT | 7scenes | 8 | −9.4% | +6.3% | +0.9% | 0 / 7 |
| VGGT | scannetpp | 21 | +3.2% | −1.4% | +0.4% | 2 / 18 |
| VGGT | hiroom | 31 | **+7.7%** | **+12.0%** | +8.3% | 5 / 25 |

进行中：DA3 dtu64（11/13，~11:10 完）。
scannetpp 已 20/20 全最终 config（fix6 重跑完成，日志重建合并）。
⚠ hiroom chamfer 与 F1 背离（+16.8% 变差），报告中讨论。

## Selector 质量（final vs selected，跨 70 场景统计）

| run | dAUC final→sel | dF1 final→sel | 救回灾难 | 误伤 |
|---|---|---|---|---|
| da3/eth3d | +3.33→+1.59% | −4.37→−1.20%（+3.17pp） | 1 | 0 |
| da3/7scenes | +3.81→+2.45% | −0.74→−1.65%（−0.90pp） | 0 | 0 |
| da3/scannetpp | −0.93→−0.29% | −0.92→−0.15%（+0.76pp） | 0 | 0 |
| vggt/eth3d | +18.58→+16.74% | +0.24→+0.42% | 0 | 0 |
| vggt/7scenes | −9.45→−9.45% | +6.30→+6.30% | 0 | 0 |
| vggt/scannetpp | +3.21→+3.56% | −1.42→−0.60%（+0.81pp） | 0 | 0 |

**结论：selector 是尾部风险削减器**——削掉 F1 灾难尾部（eth3d +3.17pp），代价是 AUC 均值稀释 1-2pp；70 场景 0 误伤。它不是收益放大器。

## 改进实验（消融，冻结副本 fg_abl，5 场景 eth3d 子集，selected vs selected）

1. **masked 位置加权（w×=(1+corr_mask)）**：**dF1 +1.61pp / dAUC −0.29pp**。方向一致（4/5 场景 F1 改善），n=5 统计弱。机制：改变轨迹使 selector 能挑到更健康的中间步。不能修复 relief 类融合级崩溃。→ 报告为候选增量，待全量验证。
2. **quantile-couple**（逐帧 log 深度分位匹配替换标量 couple）：已实现+合成验证，车道排队中。
3. rkd 漂移当 selector 信号：统计不显著（ρ=−0.24, p=0.31），已否决。

## 今晚已修代码问题（全部验证）

| 问题 | 处置 |
|---|---|
| rotation-Huber 零残差反向 NaN | 分支精确写法（小残差直接 z）+ gradcheck |
| 失效 student 被 selector 误报改善 | probe 记录 validity，selector 先查有效性 |
| selector 资格规则误杀（小分母分量） | 分量自适应 abs_floor + 重设计主规则 |
| v2 路径 rot/tdir 日志恒 0（显示 bug） | 读 v2_rel_rot 键；训练数值本不受影响（trace 实证） |
| analyzer probe_trace 非递归 glob（hiroom 场景名含/） | 递归 + relpath |
| analyze --cleanup 对 peft 目录失效（0 MiB） | 新 v2_prune_running.py：selector 真回放保 step0/100/selected |
| 磁盘爆雷（521G 满→torchvision 崩） | 清理纪律 + 40G 巡检门 + 周期 pruner |

## 已知能力边界（诚实记录）

- **selector 盲区**：probe 度量"多像 teacher"，看不见融合级崩溃（relief F1 −72%、1ada7a0617 F1 −47.6% 均被 probe 判改善）。probe 分量漂移与 GT 变化统计不相关。
- **+5% 双指标目标未达成**：当前最强 VGGT/eth3d（AUC +18.6%）与 DA3/hiroom（双正）；F1 普遍偏弱是系统性问题，masked-boost 是首个正向杠杆（+1.6pp）但不够。
- hiroom chamfer +16.8% 与 F1 +1.6% 背离：重建的离群点尾部变差，需逐场景排查（报告 B 部分）。

## 11:05 追加

### Selector 变体离线研究（n=100，GT 仅离线评估用）
| 策略 | dAUC | dF1 |
|---|---|---|
| always_final | +3.75% | −0.01% |
| 当前 selector | +2.73% | +0.23% |
| oracle（base/final 逐场景取优） | +6.32% | +4.06% |

selector ≈ 尾部保险（AUC −1pp 换 F1 +0.24pp + 灾难拦截），但**距 oracle 天花板很远**。两处系统盲区：①融合级崩溃（relief）；②烂 teacher 场景（hiroom 828815：final 真实 +75%AUC 被回退，因 probe 量的是"离 teacher 距离"）。我的 trustgate 修补变体离线不优于现 selector，已否决。

**VGGT hiroom 警示**：final +7.7%/+12.0% → selected +0.4%/+14.3%——selector 系统性沿 AUC↔F1 前沿偏向 F1。报告需并列 final 与 selected 两列，由使用者按指标偏好选择策略。

### 消融对照（DA3 eth3d 5 场景，selected vs selected）
| 臂 | dAUC uplift | dF1 uplift | 注 |
|---|---|---|---|
| masked-boost α=1 | −0.29pp | **+1.61pp** | 稳定小正，4/5 场景 F1≥主线 |
| quantile-couple | **+2.27pp** | **+1.88pp** | 高方差：relief +9.4/+13.9pp 大救，courtyard −7.2pp 被伤 |

两者互补不互斥，均为 protocol 候选增量（需更大 n 验证）。
