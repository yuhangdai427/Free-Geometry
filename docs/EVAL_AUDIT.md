# 评测管线审计说明（VGGT / DA3-giant-1.1 baseline）

**日期**：2026-09-13　**范围**：`artifacts/diagnostics/final_protocol/` 全部结果 + DA3 smoke

## 1. 调用链（脚本 → 库函数）

| 环节 | VGGT（我们的所有 baseline/TTA 数字） | DA3-giant-1.1（smoke） |
|---|---|---|
| 推理 | `diagnostics/free_geometry/eval_viewcounts.py` → `train_arms.py::infer_eval32`（模型=VGGT-1B，LoRA 重置为基线） | `scripts/train_da3_protocol.py`（内置 eval） |
| 图像预处理 | `diagnostics/free_geometry/common.py::load_image_model`：长边→504、14 取整、/255 无增广（与 `scripts/benchmark_vggt.py` 的 504 预处理一致） | DA3 `InputProcessor`：504、ImageNet 归一化 |
| 位姿指标 | `src/vggt/vggt/bench/evaluator.py::VGGTEvaluator` → `src/depth_anything_3/bench/utils.py::compute_pose` | 同左（`compute_pose`，w2c） |
| 重建指标 | 数据集类自己的 `fuse3d`/`eval3d`（`src/depth_anything_3/bench/datasets/*.py`） | 未跑（CPU 重，baseline 在另一台 server） |
| 深度指标 | `diagnostics/free_geometry/depth_metrics.py`（自定义） | smoke 内置（单尺度最小二乘） |
| 汇总 | `run_eval.py`（调 VGGTEvaluator 写 eval32_metrics.json） | smoke_summary.json |

## 2. 指标的精确定义（以代码为准）

- **Pose AUC@3**（`compute_pose`，bench/utils.py:307-330）：预测与 GT 轨迹**各自先刚性对齐到首个相机**（`align_to_first_camera`，SE3 逆，无尺度），再算**所有视角对**的相对旋转角误差 + 相对平移**方向**角误差（度），对阈值曲线积分得 AUC@3/5/15/30。主指标 = auc03。**没有 Umeyama 尺度对齐，不是 VGGT 论文的 per-frame max(rot,trans) 协议**。
- **Recon（recon_unposed）**：`fuse3d` 用 GT 内参去畸变+ROI+缩放彩色图（ScanNet++ 到 768×1024），**预测轨迹先与 GT 做 Umeyama Sim(3) 尺度对齐**（`_prep_unposed`，scannetpp.py:431-438），TSDF 融合（ScanNet++：voxel 0.02m/trunc 0.15m/max_depth 5m；7Scenes/HiRoom：4/512≈0.0078m/0.04m；ETH3D ×5），`eval3d`：GT mesh 采样点、预测云裁到 GT AABB+0.1m，阈值 5cm 算 acc/comp/overall/P/R/**F1**。**全程无 confidence 过滤**。
- **深度**（depth_metrics.py）：GT=各数据集原始深度（scannetpp uint16 mm PNG resize-only；hiroom uint16/65535×100m；eth3d float32 raw），**场景共享 log 尺度**校正后算 AbsRel/δ1.25（另有 per-frame median 列在 CSV 里）。
- **评测帧**：benchmark-100 = evaluator.py `_sample_frames` else 分支逐行复刻（`random.seed(42)`→shuffle range(N)→取前 100→升序）；N<100 → 全部帧。GT 只在评测侧使用。

## 3. 场景清单

- ScanNet++：20 场景 = DA3 benchmark 官方 `SCANNETPP_SCENES`（constants.py:210-231），逐一核对相同
- 7Scenes：7 场景全量；HiRoom：val 列表 30 场景；ETH3D：官方 11（剔除已知问题场景）

## 4. 与"官方数字"最可能的差异点（请审查者优先核对）

1. **Pose 协议不同源**：我们的 AUC 是 DA3 benchmark 的"首相机刚性对齐 + 成对相对角误差"口径；VGGT 论文表是 Sim(3) Umeyama + per-frame 最大角误差口径。**两者不可直接比**。我们可比的对象是本 repo（DA3 论文 benchmark 套件）的数字。
2. **分辨率 504**：与 `scripts/benchmark_vggt.py`（本 repo 官方 VGGT baseline 脚本）一致；VGGT 官方 repo 默认 518。
3. **帧子集**：benchmark-100（seed42）是本 repo 评测器的行为；VGGT/DA3 论文部分数据集可能用全帧或别的子集。
4. **两套对齐并存**：pose 指标用首相机刚性对齐，recon 用 Umeyama 带尺度对齐——这是 repo 评测器的原设计，不是我们改的。
5. **HiRoom** 不是 VGGT 论文数据集，是本 repo 自有 benchmark。
6. 我们的 baseline（A0）走的是诊断 harness 的 npz 导出路径（预处理等价于 benchmark_vggt.py，但推理入口不同）。**建议的交叉验证**：用 `scripts/benchmark_vggt.py` 在单场景（如 chess/09c1414f1b）跑一遍，与我们 A0_baseline 的 AUC/F1 对比，应一致。

## 5. 快速复核入口

- 单场景交叉验证：`python scripts/benchmark_vggt.py --datas 7scenes --scenes chess`（输出与本 repo `A0_baseline@100v` 的 chess 行对比）
- 指标计算源码：`src/depth_anything_3/bench/utils.py::compute_pose` / `evaluate_3d_reconstruction`；`src/depth_anything_3/bench/datasets/scannetpp.py::fuse3d/eval3d`
- 我们的评测帧集合：各 `scene_manifest.json` 的 `eval32_frames` 字段（可复算 seed42 流程）
