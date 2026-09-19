# TTA 协议 v1 → DA3-giant-1.1 移植笔记

**日期**：2026-09-13　**协议源**：docs/TTA_PROTOCOL_v1_2026-09-13.md（VGGT 侧 v1.0）
**实现**：`src/depth_anything_3/test_time_adaption/protocol_v1.py` + `scripts/train_da3_protocol.py`
**模型**：`model_weights/DA3-GIANT-1.1`（本地 safetensors，无下载）

## 1. DA3 与 VGGT 的接口差异（代码审计结论）

| 项 | VGGT-1B | DA3-giant-1.1 | 对移植的影响 |
|---|---|---|---|
| aggregator 层数 | 24（每层 frame+global 交替） | **40** 个 DinoV2 block；`alt_start=13`，i≥13 且奇数层做跨视图 global attention，其余 local | tap 层按总层数均布重取（见 §2） |
| embed / token 维度 | 1024；aggregator 输出 2048（拼接） | 1536；`cat_token=True` → 输出 **3072** = [raw local track | backbone 最终 LN(global track)] | loss 在 3072 维拼接 token 上做，与 VGGT 同构 |
| 特殊 token | 1 camera + 4 register，patch_start_idx=5 | 1 个 cls/camera token（`patch_start_idx=1`），**无 register token**；`get_intermediate_layers` 返回前已剥掉 → patch token 下标从 0 开始 | mask/特征切片不需要 psi 偏移 |
| camera token 注入 | aggregator 起始层 | `i==alt_start`(=13) 处把 token0 替换为 [ref,src] 相机 token | 冻结该参数（协议规定） |
| camera/pose head | 4-block trunk，输出 pose_enc 9 维 (T+quat+FoV)，解码为 **w2c** | `CameraDec`（2 层 MLP + 3 个 fc），输入=最后 tap 层 raw camera token(3072)，输出 pose_enc 9 维 (T+quat_xyzw+fov_h/fov_w)，`pose_encoding_to_extri_intri` 解码为 **c2w**，API 再 `affine_inverse` 得 w2c | rel-pose loss 先 c2w→w2c 再套 VGGT 原公式（旋转 chordal + 平移方向 1−cos，尺度无关） |
| DPT 共享 LayerNorm | `depth_head.norm`（2048） | **存在**：`DualDPT.norm = LayerNorm(3072)`，对 4 个 stage 输入共用 | post-LN 空间直接可用，与 VGGT 完全对应 |
| head tap 层 | DPT 用 [4,11,17,23]，与 loss tap 相同 | DualDPT 固定吃 `out_layers=[19,27,33,39]`（config 定死，顺序敏感） | teacher 一次前传取并集 [6,19,26,27,33,39]，head 切位置 [1,3,4,5]、loss 切 {0,1,2,5}；student 不跑 DPT head，只 tap [6,19,26,39] |
| 深度 conf | `depth_conf`（DPT 输出） | `depth_conf`（DualDPT 主头 conf 通道） | teacher conf 加权照旧 |
| 输入预处理 | 长边 504、14 取整、[0,1]、**无 ImageNet norm** | `InputProcessor`：upper_bound_resize 到 504、14 取整、/255、**ImageNet 归一化** | 掩码"置零"在归一化之后做（=像素均值灰），保证网络看到严格的 0 |
| 推理 dtype | fp32（诊断侧）/fp16 AMP（训练侧） | API 原生 **bf16** autocast | 训练/缓存统一 bf16 autocast，loss 内 fp32；不用 GradScaler（bf16 不需要） |
| 参考视角选择 | 无 | S≥3 时在 layer 12 重排视图，`ref_view_strategy="first"` 时为恒等（b_idx=0），收集时已还原顺序 | 全程 `"first"` 保证确定性 |

## 2. 适配决定（逐条对应协议）

1. **选帧**：完全复用 VGGT 侧规则（`build_final_manifest.py` 的逐行移植）：N<8 剔除；teacher_N=16（N≥16）否则 8；τ=相邻帧 SIFT 匹配率中位数（≤40 对、320px、400 特征、ratio 0.75）；τ>0.55 等距铺满+共享帧 SIFT 重叠率∈[0.1,0.5] 过滤（不足回退），否则纯随机；共享 4 帧固定在 teacher 槽位 [0,2,4,6]；10 训练对+2 探针对；种子 `stable_seed("final", tag, dataset, scene)`（与 VGGT manifest 同方案）。评测帧：N≥100 → seed42 shuffle 取前 100 排序（benchmark-100），否则 allv。
2. **maskdistill tap 层**：40 层取 L≈1/6,1/2,2/3，末层 → **[6, 19, 26, 39]**（对 VGGT [4,11,17,23]/24 的等比映射）。特征 = 3072 维拼接 token，过 `DualDPT.norm`（共享 LN）→ post-LN 空间，Huber(β=1)+2(1−cos)，teacher 深度 conf 逐 patch 加权，只在掩码位置计入——公式与 `loss_b5_maskdistill` 逐项一致。
3. **掩码**：输入图像 14×14 块（=patch 对齐）50% 置零（归一化后张量置零），`mask_image_blocks` 逐行移植；种子 `stable_seed("mask", scene, epoch, pi, seed)`。
4. **rel-pose**：DA3 `CameraDec` 直接输出 9 维 pose 编码（与 VGGT 同为 T+quat+FoV 布局），解码 c2w→w2c 后套 `loss_pose_rel` 原公式（共享 4 视角 6 个视角对：相对旋转 chordal + 相对平移方向 1−cos），1:1 与 maskdistill 相加；梯度穿过冻结 CameraDec 回传 LoRA。
5. **训练**：LoRA r=32/α=32/dropout=0 于 **全部 40 层**（attn.qkv/proj + SwiGLU w12/w3，SwiGLUFFNFused 解融合以让 LoRA 拿梯度，沿用现有脚手架 `patch_swiglu_for_lora`）；heads（DualDPT、CameraDec）与 camera token 冻结；AdamW lr 3e-5、wd 1e-5、clip 1.0；线性 warmup 15 步→cosine（eta_min 1e-8）；固定 100 步 = 10 对 × 10 epoch，batch=1 对/步；teacher 冻结、每场景预缓存 12 对（训练循环零 teacher 前传）。
6. **与 VGGT 侧的刻意差异**（记录备查）：
   - teacher 缓存放 **bf16**（VGGT 侧 fp32）：DA3 原生推理即 bf16 autocast，缓存 bf16 与部署一致且省显存（12 对 ≈1.2GB）；loss 内统一升 fp32。
   - teacher 缓存只存共享 4 视角的 tap 特征（VGGT 存全部 8 视角；DA3 teacher_N=16 时全存太贵，且 loss 只用共享 4 视角）。
   - 训练精度用 bf16 autocast 无 GradScaler（VGGT 侧 fp16+scaler 为对齐其部署；DA3 部署 dtype 即 bf16）。
   - student 每步只跑 backbone + CameraDec（不跑 DualDPT 深度头；C2M 两项 loss 都不需要 student 的深度输出），比 VGGT 侧省一次 DPT 前传。
   - 每场景 LoRA 重 init 用显式种子 `stable_seed("lora_init", scene, seed)`（VGGT 侧依赖全局种子顺序）。

## 3. Smoke 结果（ScanNet++，2 场景，100 步）

机器：单卡 97GB（与并行 VGGT 任务共享，错峰执行）；产物在 `workspace/da3_protocol_smoke/`（C2M）与 `workspace/da3_protocol_smoke_maskonly/`（纯 maskdistill 对照），逐 step 曲线见各自 `training_trace.csv`。
训练峰值显存 **20772MiB**、评测峰值 **18422MiB**（均 <25GB）；全程无 NaN；每场景训练 ~102s（teacher 缓存 ~35s + 100 步 ~65s）、评测 ~21s。评测帧集 = benchmark-100（两场景 N≥100 均触发）。AbsRel 用全场景单一最小二乘尺度（ls_scale 见 summary json）。

| 场景 | N | τ | 策略 | 配方 | loss 前10→后10 | maskdistill 分段均值（1-30/31-70/71-100） | rel_rot 分段均值 | AUC@3 | AUC@30 | AbsRel(LS) | δ<1.25 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 09c1414f1b | 397 | 0.478 | random | C2M | 4.317→4.716 | 0.792 / 0.786 / 0.781 | 2.79 / 2.86 / 3.03 | **0.8131** | 0.9790 | 0.0259 | 0.9860 |
| 〃 | 〃 | 〃 | 〃 | mask-only（对照） | 0.794→0.740 | 0.791 / 0.764 / 0.741（单调降） | （未加权，监测）2.79→3.06 | 0.8030 | 0.9779 | 0.0259 | 0.9860 |
| 1ada7a0617 | 100 | 0.481 | random | C2M | 1.996→1.041 | 0.629 / 0.630 / 0.621 | 0.87 / 0.55 / 0.39（降） | **0.5127** | 0.9426 | 0.0256 | 0.9785 |

读数要点：
- **loss 下降 / 无 NaN / 显存达标**：1ada7a0617 的 C2M 总 loss 与两项分量均下降；09c1414f1b 的 maskdistill 在对照组单调下降。两场景三次运行无 NaN、无 OOM。
- **09c1414f1b 的 rel-pose 项在 step 1 就 ≈2.16**（此时 LoRA B=0，student=baseline）——是"16 视角 teacher vs 宽基线随机 4 视角 student"在该场景上的固有不一致，不是训练发散；训练只让它轻微变大（2.8→3.0），对照组（无 pose 梯度）同样漂移，说明漂移是特征适配的副产物而非 pose loss 所致。
- 1ada7a0617 在 step 70-80 出现一次 rel-pose 瞬态尖峰（rot 1.2-1.4，grad_norm 48/23），随 lr 衰减自行恢复，终值正常。
- AUC@3 两场景一升一稳（无 baseline 对照，baseline 在另一台 server）；AbsRel ≈0.026、δ1.25 ≈0.98 为合理量级。
- ckpt 兼容性已验证：`c2m_final_lora_peft/` 可被 `StudentModel.load_lora_weights`（即 `scripts/benchmark_da3.py` 的 LoRADepthAnything3 路径）正常加载，lora_B 全部非零。

## 4. 已知风险 / 待办

- **LoRA 注入面**：DA3 为 40 层 ×4 模块（attn.qkv/proj + SwiGLU w12/w3），可训练参数 31.5M（PEFT 实测）；VGGT 侧结构类似（24 frame + 24 global blocks × qkv/proj/fc1/fc2），维度 1024 vs DA3 1536。r/α 沿用协议值，lr=3e-5 在 DA3 上的最优性未验证——smoke 只看收敛/无 NaN。
- **rel-pose 目标在宽基线随机 pair 上噪声大**：09c1414f1b 在 step 1（纯 baseline）rel_rot 就 ≈2.2 chordal，100 步内不收敛（2.8→3.0）；1ada7a0617 则正常收敛（0.87→0.39）。对照实验证明漂移与 pose 梯度无关。若全量跑 ScanNet++，建议对照协议 §4.3 精细版（τ≤0.55 → 纯 maskdistill，本 smoke 已验证该路径单调收敛且 AUC@3 与 C2M 相当：0.803 vs 0.813）或给 rel-pose 加逐对权重/置信门控。
- **DA3 全局注意力显存随视角数增长（SDPA flash）**，benchmark-100（100 帧）推理在单模型 bf16 下实测峰值 18.4GB；更长帧集需重测。
- **τ 分派在 ScanNet++ 上走 random 分支**（两场景 τ≈0.48<0.55），dense/SIFT 过滤路径本 smoke 未覆盖（7Scenes 等稠密视频才会触发；代码逐行移植自 VGGT 侧 build_final_manifest.py）。
- **AbsRel 用全场景单一最小二乘尺度**（GT 有效像素上拟合）——DA3-giant 非严格 metric，与 VGGT 侧 scale-invariant 口径一致地去除尺度；summary json 同时记录未缩放 AbsRel 备查（~0.51-0.68，尺度差 ~2-3 倍）。
- teacher 16 视角前传在 S≥3 时触发参考视角选择；已固定 `ref_view_strategy="first"`（恒等重排）。若未来改策略，teacher/student 需保持一致。
- probe 对（2 对）目前只进缓存未做探针评测（VGGT 侧用于 headroom/feature-MSE 诊断）；如需诊断可在 `protocol_v1.cache_teacher_pair` 之上加探针。
- 重建 F1（TSDF）未在 smoke 跑（CPU 重，且用户另一台 server 有 baseline）；评测入口 `Evaluator` + `recon_unposed` 与本训练产物兼容（`scripts/benchmark_da3.py` 的 LoRADepthAnything3 可加载本 ckpt，已验证）。
- 首版实现（双模型常驻 GPU + GPU 缓存）单进程 23.6GB，与并行任务叠加 OOM 过一次；现版（缓存上 CPU + teacher/student 分阶段驻留）把训练峰值压到 20.8GB。共享 GPU 时建议 `PYTORCH_ALLOC_CONF=expandable_segments:True` 并错峰。
