# Protocol v2 隔夜实验报告（2026-09-19 最终版）

> 12:30 验收版。GT 只用于最终评测与离线分析，从不进入在线选择。
> 全部原始数据：`workspace/protocol_v2/analysis/*.json`、`final_report_table.json`、`NIGHT_OPS_STATE.md`（时间线）。

## 0. 一句话结论

**统一 protocol（同一 config 跑两模型×全数据集）在 AUC 上 macro +3.6%（final）/+2.0%（selected）、F1 上 +1.9%/+2.6%（macro，DTU 系无 F1 用 chamfer）；"每数据集双指标 +5%"的目标未达成**——但 VGGT/eth3d（AUC +18.6%）、VGGT/hiroom（final +7.7%/+12.0% 双超 5%）、DA3/hiroom（+4.1%/+1.6%）证明 protocol 能产生真实大收益；**GT-free checkpoint 选择（selector）是现阶段的最大瓶颈与最大改进空间**（oracle 天花板 +6.3%/+4.1%）。

## A. 最终统一 protocol（两模型全数据集同一套，已冻结）

```
选帧:    A/B 双 teacher 上下文 manifest（VGGT 16:4 / DA3 8:4，SIFT overlap 排序 extras）
可靠性:  冻结权重，仅几何侧（q_rot/q_tdir/geo_w = 1/(1+(u/Q80)²)）；q_feat 关闭（吃 VGGT F1）
feature: 全位置 Huber(β=1)+2(1−cos)，teacher-conf 权重 mean-1，masked 输入 50%
几何:    逐边 rotation Huber（ℓ=16·H_δ(d)，δ=sin10°，等效 knee 20°）+ tdir 1−cos（近零基线剔除）
         + RKD distance/angle Huber（既有）+ 旧 couple（非对称 mask）
更新:    ramp λ=min(1,step/20) + aux 分支全局范数上限（前10步校准 ×4）
验证:    2 probe 对 × 2 固定 mask，每 10 步；ckpt 每 10 步（含 θ₀）
选择:    selector = 最优改善分 + 50% 灾变网 + 无改善回退 baseline；tau 门（τ>0.55 的 dense 场景关闭 rel，仅 DA3）
```

## B. 全量结果表（Δ 为相对 baseline 的相对变化）

| 模型 | 数据集 | n | final dAUC | final dF1 | selected dAUC | selected dF1 | chamfer(final→sel) | selector 回退/改善 |
|---|---|---|---|---|---|---|---|---|
| DA3 | eth3d | 11 | +3.33% | −4.37% | +1.59% | −1.20% | +7.6%→+6.6% | 1/10 |
| DA3 | 7scenes | 7 | +3.81% | −0.74% | +2.45% | −1.65% | +0.4%→+0.6% | 1/6 |
| DA3 | scannetpp | 20 | −0.55%* | −0.66%* | −0.30%* | −0.13%* | +0.6% | 11/9 |
| DA3 | hiroom | 30 | +4.08% | +1.55% | +1.78% | +0.45% | +16.8%→+7.6%⚠ | 9/21 |
| DA3 | dtu | 22 | +0.45% | n/a | +0.17% | n/a | **+63.4%→+16.1%** | 8/14 |
| DA3 | dtu64 | 13 | +0.94% | n/a | +0.86% | n/a | n/a | 2/11 |
| VGGT | eth3d | 11 | +18.58% | +0.24% | +16.74% | +0.42% | +2.7%→+4.1% | 1/10 |
| VGGT | 7scenes | 7 | −9.45% | +6.30% | −9.45% | +6.30% | +0.9%→+0.9% | 0/7 |
| VGGT | scannetpp | 20 | +3.21% | −1.42% | +3.56% | −0.60% | +0.4%→−0.2% | 2/18 |
| VGGT | hiroom | 30 | **+7.66%** | **+12.01%** | +0.38% | +14.32% | +8.3% | 5/25 |

\* scannetpp 为 14+6 合并（fix6 重跑后全 20 场景最终 config，final 列从日志重建，selected 列为 14+6 加权合并；明细见 analysis/da3_scannetpp_merged.json）。
macro 均值：AUC final +3.62% / sel +2.01%；F1 final +1.94% / sel +2.58%。

## C. Selector 统计与效用（离线研究，n=100+）

| 策略 | dAUC | dF1 | 说明 |
|---|---|---|---|
| always_baseline | 0 | 0 | 零风险零收益 |
| always_final | +3.75% | −0.01% | 均值最高但尾部裸奔 |
| **当前 selector** | +2.73% | +0.23% | 尾部保险：70 场景 0 误伤 |
| oracle(base/final) | +6.32% | +4.06% | 天花板 |

- **拦截成功案例**：DA3 eth3d delivery_area F1 灾难（−34%→回退）；DTU chamfer 爆雷 scan114/23/13（+180~312%→回退归零）。
- **漏检案例（盲区）**：①融合级崩溃（relief F1 −15%、scannetpp 1ada7a0617 F1 −47.6%，probe 全分量改善）；②烂 teacher 场景（hiroom 828815：GT +75%AUC 真实改善被回退——probe 量"离 teacher 距离"，远离烂 teacher 被误判为退化）；③DTU scan10 chamfer +356% 漏网（DTU 全部 chamfer 回归=这一个场景）。
- **系统性偏好**：probe total 由 feature 主导 → selector 沿 AUC↔F1 前沿偏向 F1（vggt/hiroom 丢 7.3pp AUC 换 2.3pp F1）。
- probe 分量漂移（含 rkd）与 GT 变化统计不相关（Spearman ρ=−0.24, p=0.31）——不能当新信号用。
- 离线修补尝试（step0 信任门）不优于现 selector，已否决。**selector 改进需要新信息源（如固定对应点重投影误差），不是调阈值。**

## D. 消融实验（冻结副本 fg_abl，DA3 eth3d 5 场景子集，selected vs selected）

| 臂 | n=5（最差F1子集） | n=11（全部） | 判定 |
|---|---|---|---|
| masked 位置加权 w×=(1+corr) | −0.29pp / +1.61pp | **+0.19pp / +0.59pp** | 弱正但安全（11 场景双指标均非负），可作低风险候选；收益远小于子集暗示的水平 |
| quantile-couple | +2.27pp / +1.88pp | **−0.05pp / −0.45pp** | **否决**：前 5 场景的"胜率"是小样本运气（relief 大救被 playground −19pp AUC、relief_2 −14pp F1 抵消）。高方差杠杆，不可进主线 |

教训：5 场景子集上"显著"的消融结论在扩到 11 场景后翻案——本报告所有消融均以 n=11 为准。

## E. 今晚修复的代码问题（全部带验证）

1. rotation-Huber 零残差反向 NaN → 分支精确写法（小残差直接 z），gradcheck 通过。
2. 失效 student 被 selector 误报改善 → probe validity 链 + selector 先查有效性。
3. selector 资格规则误杀（小分母分量）→ abs_floor 校准 + 主规则重设计（回退率 26%→8.7%）。
4. couple 对称化"修复"是 regression（chess −22%）→ 回退旧版（消融梯子实证）。
5. τ=median 砍半可靠性权重 → Q80。
6. VGGT 8:4 manifest（上下文砍半）→ 16:4 重建（eth3d −4.2%→+15.2%，有 probe 轨迹机制证据）。
7. v2 路径 rot/tdir 日志恒 0（显示 bug，训练数值正常）→ 已修，车道新场景已显示真值。
8. analyzer/materialize 非递归 glob（hiroom 场景名含/）→ 递归修复×2。
9. analyze --cleanup 对 peft 目录失效 → v2_prune_running.py（selector 真回放保 step0/100/selected）。
10. 磁盘爆雷（torchvision .pyc 写不了→全线崩溃）→ 清理纪律+40G 巡检门+周期 pruner；3 进程并发 GPU 95.8G 险情 → 并发上限纪律。

## F. 诚实的能力边界与建议

- **+5% 双指标目标未达成。** 达到的是：VGGT/eth3d AUC、VGGT/hiroom 双指标（final）、VGGT/7scenes F1。DA3 的 F1 全线偏弱是系统性问题。
- saturated 数据集（scannetpp、DTU 的 AUC）上 TTA 空间本就很小；主要价值是不退化（selector 的回退正是为此）。
- **下一步最高价值方向**（按证据排序）：①给 selector 加独立于 teacher 的观测（固定对应点重投影误差）以补盲区；②quantile-couple 上门控后全量验证；③masked-boost 全量验证（F1 稳定小正）；④VGGT/7scenes AUC 回归的根因（AUC↔F1 前沿现象）需专门研究。
- VGGT/7scenes 的 AUC −9.4%：dense 视频场景 rel-on 对 VGGT 有利有弊（F1 +6.3%），呈现 AUC↔F1 互换，非 bug。

## G. 复现与审计入口

- 车道脚本：`scripts/run_v3_lane_{da3,vggt}.sh`；分析：`scripts/protocol_v2_analyze.py`、`v2_final_report.py`、`v2_selector_quality.py`、`v2_selector_variants.py`、`v2_prune_running.py`。
- 消融副本：`/root/autodl-tmp/fg_abl`（ABL_MASK_BOOST / ABL_COUPLE_Q 环境变量开关，合成测试验证 boost=0 与主线逐位一致）。
- 时间线与全部中间结论：`workspace/protocol_v2/NIGHT_OPS_STATE.md`（§1-31）。
