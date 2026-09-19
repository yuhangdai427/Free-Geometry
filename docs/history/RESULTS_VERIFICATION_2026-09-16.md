# 终表数据核实报告（2026-09-16）

对"双模型 × 4 数据集"TTA 冠军终表的每一格做了一次完整溯源审计：数字是否
真实存在于磁盘、能否从存档重算出来、以及 Evaluation 是否全部满足
**"N≥100 用 100 views，N<100 用该场景全部 views"** 的协议。

结论先行：**8 格数字全部可从磁盘存档精确复现，100-views-or-all 规则全部
PASS，baseline 与 TTA 的评测帧集全部一致**。发现 3 处标签/口径问题
（数字本身无误，但表格表述需要勘误），详见 §3。

## 1. 口径（复现终表必须遵守）

- **逐场景相对提升均值**：`mean_s[ (TTA_s − baseline_s) / baseline_s ] × 100%`，
  AUC@3 与 F1 各自独立计算。绝对分点（AUC 差 ×100）或先池化再取相对值都
  **不能**复现终表（例如 DA3×7scenes 绝对分点为 +1.78/+3.25，池化相对为
  +6.48/+6.48，均不等于 +7.62/+5.50）。
- VGGT 侧 `eval32_metrics_*.json` 中的 `mean` 汇总键**不计入**，逐场景键才算。
- 分母为 0 的场景剔除（见 §3 问题③）。
- baseline 永不重跑：DA3 用 `artifacts/diagnostics/final_protocol/da3_baseline/
  <ds>_baseline.json`（AUC）与 `<ds>_recon_baseline.json`（F1）；VGGT 用同一
  metrics 文件里的 `A0_baseline@<view>` 键。

## 2. 逐格溯源表

| 格 | 冠军配方（run args 原文） | 结果存档 | 重算值 | 表格值 |
|---|---|---|---|---|
| DA3×7scenes | `arm=rkdc1h, two_stage=0.7, ratio_mix="8:4,16:4,24:8", combo=True, combo_se_frac=0.5, n_train=20, steps=100, seed=0`（7 场景） | `workspace/da3_7scenes_2stage/smoke_summary.json` | **+7.62 / +5.50** | +7.62 / +5.50 ✓ |
| DA3×eth3d | `arm=rkdc1h, teacher_N=8, steps=100`（11 场景；**无 early_stop**，见 §3-①） | `workspace/da3_protocol_eth3d_t8s4/smoke_summary.json` | **+2.95 / −1.21** | +2.95 / −1.21 ✓ |
| DA3×hiroom | `arm=rkdc1h, teacher_N=8, early_stop=True, n_train=10`（30 场景） | `workspace/da3_protocol_hiroom_t8s4_es/smoke_summary.json` | **+3.33 / +0.62** | +3.33 / +0.62 ✓ |
| DA3×scannetpp | 天花板格（teacher≈student，所有配置 ±0.5% 内） | `workspace/da3_protocol_scannetpp_t8s4/smoke_summary.json` → −0.17/+0.36；`_mix` → −0.04/+0.11 | 落在 ±0.5/±0.4 | ±0.5 / ±0.4 ✓ |
| VGGT×7scenes | `C2M_maskrel`，16:4 manifest，`@100v` | `final_protocol/7scenes/eval32_metrics_C2M.json`（`A0_baseline@100v` vs `C2M_maskrel@100v`） | **+0.62 / +14.10** | +0.62 / +14.10 ✓ |
| VGGT×eth3d | `C2M_maskrel`，8:4 manifest，`@allv` | `final_protocol/eth3d/eval32_metrics_C2M.json`（`@allv`） | **+32.24 / +25.46** | +32.24 / +25.46 ✓ |
| VGGT×hiroom | `C2M_maskrel`，8:4 manifest，`@allv`（场景集见 §3-③） | `final_protocol/hiroom/eval32_metrics_C2M.json`（`@allv`） | **+18.54 / +18.52** | +18.54 / +18.52 ✓ |
| VGGT×scannetpp | `C2M_RKDC1H`，16:4 manifest，`@100v` | `final_protocol/scannetpp_v3/eval32_metrics_RKDC1H.json` + `final_protocol/scannetpp/eval32_metrics_C2M.json` 的 baseline 键 | **+5.50 / +2.22** | +5.50 / +2.22 ✓ |

scannetpp 跨目录配对合法性：`scannetpp/` 与 `scannetpp_v3/` 两份
`scene_manifest.json` 的 20 个场景 `eval32_frames` **逐字节一致**。

偏度提示（不影响真实性，写论文时建议报 per-scene 表）：
- DA3×7scenes 的 +7.62 高度偏斜：stairs +38.91（base AUC 0.205→0.285）、
  chess +16.77 贡献大头，redkitchen −6.05。
- VGGT×eth3d 的 +32.24 由 relief +113.6、terrains +86.3、courtyard +57.5
  主导，facade −12.1。

## 3. 发现的 3 个标签/口径问题（已勘误）

① **DA3×eth3d 的配方标签写了 "8:4 + early_stop"，但复现数字的 run 没有
开 early_stop**。所有真正的 es 运行都以 SIGTERM 崩溃收场（日志
`da3_baseline/e3_es.log`、`e3_es2.log`、`e3_final.log`，目标输出目录
`workspace/da3_protocol_eth3d_es*/` 为空）。正确标签是"纯 8:4"。
已在 `TTA_PROTOCOL_DA3_v1_2026-09-15.md` §5 勘误。

② **DA3×hiroom 的"冠军" es run 并不是磁盘上最好的 run**：同配置不开 es 的
`da3_protocol_hiroom_t8s4` 为 +3.74/+1.57，高于 es 的 +3.33/+0.62。
选 es 版作为冠军的唯一理由是协议统一性（小场景防过拟合）。已在 DA3 文档
§5 标注。

③ **VGGT×hiroom 一行内两个数字的场景集不同**：场景
`20241230/828760/cam_sampled_04` 的 baseline AUC=0（相对值分母为零），
dAUC 剔除它（29 场景均值），dF1 保留它（30 场景均值，其 F1 baseline
0.0499 非零）。统一剔除 → dF1=+21.87；统一保留（0/0 记 0）→ dAUC=+17.93。
已在 `TTA_PROTOCOL_v3_2026-09-15.md` §1 加脚注。

## 4. 100-views-or-all 评测审计（全部 PASS）

规则实现（两侧同一逻辑）：`N ≥ 100 → random.seed(42) shuffle 后取前 100 帧
（benchmark-100）`；`N < 100 → 全部帧`。DA3 侧
`src/depth_anything_3/test_time_adaption/protocol_v1.py:262-268`，VGGT 侧
`diagnostics/free_geometry/build_final_manifest.py:198-204`。

| 数据集 | 场景数 | 每场景帧数 N | eval 帧数 | 规则 |
|---|---|---|---|---|
| 7scenes | 7（DA3）/ 7（VGGT） | 500–1000 | 100（`@100v`） | PASS |
| scannetpp | 20 / 20 | 100–534 | 100（`@100v`；N=100 的场景取全部 100） | PASS |
| eth3d | 11 / 11 | 14–76 | N（`@allv`） | PASS |
| hiroom | 30 / 30 | 10–23 | N（`@allv`） | PASS |

- 四个 DA3 冠军 run 的 `eval_max_frames` 均为 0（无截断）。
- baseline 与 TTA 帧集匹配：DA3 四个 run 逐场景 `n_eval_frames` 与 baseline
  JSON 完全一致；`da3_7scenes_2stage` 还保存了 `protocol.json`，其
  `eval_frames` 与 manifest 的 `eval32_frames` **逐帧索引一致**。VGGT 的
  baseline 与 TTA 在同一文件、同一 `@<view>` 键下。
- 注意不要误用 `@4v`/`@8v` 键（7scenes/scannetpp 文件里存在，是嵌套跨度
  子集消融，不是终表数字）。

## 5. 复现脚本示意

```python
import json, numpy as np
s    = json.load(open('workspace/da3_7scenes_2stage/smoke_summary.json'))['scenes']
base = json.load(open('artifacts/diagnostics/final_protocol/da3_baseline/7scenes_baseline.json'))['scenes']
rb   = json.load(open('artifacts/diagnostics/final_protocol/da3_baseline/7scenes_recon_baseline.json'))['scenes']
da = [(r['eval']['auc03']-base[sc]['auc03'])/base[sc]['auc03']*100 for sc,r in s.items()]
df = [(r['eval']['recon_fscore']-rb[sc]['fscore'])/rb[sc]['fscore']*100 for sc,r in s.items()]
print(np.mean(da), np.mean(df))   # +7.62 +5.50
```
