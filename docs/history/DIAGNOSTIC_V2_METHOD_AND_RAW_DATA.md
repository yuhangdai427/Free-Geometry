# 诊断 v2：方法、命令与原始数据索引（2026-09-18）

本文档回答"每个实验具体怎么做的、原始数据在哪"。所有结论的数字都能在
`workspace/fg_diag_v2_raw/*.csv` 或 `workspace/fg_diag_v2_*.json` 里逐行核对。

## 0. 时间线与确切命令

```
20:26  tmux diag2_lane1: train_da3_protocol.py --dataset scannetpp
       --scenes 09c1414f1b 1ada7a0617 21d970d8de 7bc286c1b6 9071e139d9
       --arm rkdc1h  --loss_all_pos --skip_eval --teacher_N 8 --swanlab
       --output_root workspace/diag2_spp_rkdc1h
       之后: 同参数 --dataset 7scenes --scenes office chess → workspace/diag2_7s_rkdc1h
20:26  tmux diag2_lane2: 同上但 --arm rkdc1hr --rel_weight 1.0
       → workspace/diag2_spp_rkdc1hr, workspace/diag2_7s_rkdc1hr
       （两 lane 并行共享 GPU0，每 lane 峰值 22.8 GiB；每场景约 100 步 ≈ 100–160 s）
20:44  tmux diag2_run: fg_diagnostic_v2.py --dataset scannetpp --scenes <5 场景>
       --base_root workspace/diag2_spp_rkdc1h --rel_root workspace/diag2_spp_rkdc1hr
       --check_c_on probe0 --out workspace/fg_diag_v2_scannetpp.json
       之后: --dataset 7scenes --scenes office chess --check_a_pairs 5 8 9
       --check_c_on pair:5 --out workspace/fg_diag_v2_7scenes.json
21:10  python3 scripts/fg_diag_v2_dump.py   （JSON → 人可读 CSV，见 §5）
```

注意：早上磁盘清理误删了原 checkpoint，以上训练是**用与历史 run 完全相同的参数重训**
（参数逐项记录在各 output_root/smoke_summary.json 的 args 字段；seed=0，
协议构建/LoRA 初始化/mask/训练顺序全部由 `stable_seed` 决定，可复现）。
swanlab 项目 `free-geometry-tta`（逐步）与 `free-geometry-tta-pairs`（按 pair）含全部训练曲线。

## 1. 三个状态的确切构造（`scripts/fg_diagnostic_v2.py` main, ~L300–350）

| 状态 | 构造 | 验证 |
|---|---|---|
| θ₀ | `torch.manual_seed(stable_seed("lora_init", scene, 0))` 后 `P.reset_lora_(student)` — 与训练 step-0 前的初始化 RNG 流完全一致（训练代码 `protocol_v1.py:882` 同 seed 同顺序） | B=0 ⇒ 前向=frozen；A 由 seed 决定 ⇒ ∇_B L ∝ G·Aᵀ 也一致 |
| θ_base | `load_lora_weights(workspace/diag2_spp_rkdc1h/ckpts/{scene}/c2m_final_lora.pt)` | **os.path.isfile 断言** + sha256 记入 JSON（v1 的静默回退已删除） |
| θ_rel | 同上，`diag2_spp_rkdc1hr`（rkdc1h + 1.0×修正 rel） | 同上 |

sha256 前 16 位（全值在 `*_meta.csv` / JSON meta.states）：

| 场景 | θ_base | θ_rel |
|---|---|---|
| 09c1414f1b | c4b418784895bd42 | b86aa000a157fd75 |
| 1ada7a0617 | d23be22c69205783 | e428cc06eab2731a |
| 21d970d8de | b030bdb06610befa | da999e144aec3951 |
| 7bc286c1b6 | 4b8efce27ea5acaf | 6600b5e685c3dc99 |
| 9071e139d9 | ef33cfb6af45e316 | cca4a9db33b36485 |
| office | 81210768d3b03e76 | 384adbd184958ec1 |
| chess | eeb873b453a56eba | 50b976a8d38a7c52 |

可训练参数 = 31,460,352：lora_early 15,728,640（160 个张量，block 0–19）
+ lora_late 15,728,640（block 20–39）+ camera_token 3,072。测量时 `student.eval()`
（LoRA dropout=0，train/eval 无随机差；确定性）。

## 2. 输入（probe / 训练对）的构造

- 协议：`P.build_scene_protocol(files, scene, dataset=…, n_train=10, n_shared=4, teacher_N=8)`
  — 与训练完全相同 ⇒ probe 对训练从未见过。
- 每个输入 = 8 帧 teacher 上下文；student 只看 slots [0,2,4,6] 的 4 帧。
  teacher cache 由一次冻结前向产生（`cache_teacher_pair`，含 4 层 token、conf、ext4、depth4）。
- **每个输入的 8 帧确切编号**（`*_meta.csv` 的 inputs_frames 列 / JSON meta.inputs）：

| 输入 | 帧 |
|---|---|
| 09c probe0 | 26, 93, 120, 130, 153, 242, 277, 322 |
| 09c probe1 | 23, 59, 68, 228, 239, 336, 346, 367 |
| office probe0 | 31, 114, 281, 198, 531, 364, 781, 448 |
| office pair5 | 68, 151, 318, 235, 568, 401, 818, 485 |
| office pair8 | 80, 163, 330, 246, 580, 413, 830, 496 |
| office pair9 | 28, 111, 278, 195, 528, 361, 778, 445 |

（7 个场景 × 全部输入的完整帧表在 `*_meta.csv`；pair5/8/9 = `protocol["train_pairs"][5/8/9]`，
即训练时编号为 5/8/9 的那 10 对中的 3 对。）

## 3. 五种输入条件

- `clean`：原图。
- `m0–m3`：`mask_image_blocks(imgs4, 0.5, patch_hw, gen)`，14×14 图像块遮挡 50%，
  `gen = torch.Generator().manual_seed(stable_seed("diag2_mask", scene, input_name, k))`，
  k∈{0,1,2,3} — **与训练 mask 不同的命名空间，四个固定 mask，跨状态可复现**。
  （训练用 `stable_seed("mask", scene, epoch, pair_idx, seed)`，随 epoch 换。）

## 4. 每个测量的定义（代码位置）

| 量 | 定义 | 代码 |
|---|---|---|
| maskdistill | 4 个 tap 层(19/27/33/39) post-LN token：Huber(β=1)+2(1−cos)，teacher-conf 权重，**全位置**（loss_all_pos，ones[1,4,P]） | `fg_diagnostic_v2.py: forward_all` 调 `P.loss_maskdistill` |
| rkd_sh_d / rkd_sh_a | 相机中心 mean-normalized 6 对距离 / 12 三角角的 Huber(δ=0.2)，**拆成两个标量**用于独立反传 | `rkd_parts()`，逐行复刻 `protocol_v1.loss_rkd_shared_pose_huber_w2c` |
| couple | log(RMS_centers/mean_depth) 标量对齐（teacher 5% 分位 conf 门控） | `P.loss_couple_w2c` |
| rel_rot | Σ_ij ‖R_i R_jᵀ(s) − R_i R_jᵀ(t)‖²_F / 6（chordal） | `forward_all`（修正公式） |
| rel_tdir | Σ_ij [1−cos(t̂_ij(s), t̂_ij(t))] / 6，t_ij = t_i − R_i R_jᵀ t_j | 同上 |
| 梯度 | 同一次前向（retain_graph）对 6 个标量分别 backward；每参数拍平快照；cos = F.cosine_similarity；g_base := g_md + 1.5(g_d+g_a) + 1.0·g_cp（线性组合，不再反传） | `grad_matrix()` |
| 参数组 | 名字含 blocks.0–19→lora_early，20–39→lora_late，camera_token→cam_token；ALL=全部拼接 | `classify_param_groups()` |
| 逐相机对 | rot_geodesic_deg = arccos((tr(R̂ₛᵀR̂_t)−1)/2)（度）；tdir_angle_deg 同理；‖t_rel‖=教师/学生各自；orth_err=‖RᵀR−I‖_F；det；depth 有效像素比 | `decompose_pairs()` |
| Check C | 快照可训练参数 → 每分支恢复 θ、**新建** AdamW(lr=3e-5, wd=1e-5)、对应加权 loss backward → clip(1.0) → step → 在更新输入与 probe0/clean 上重测 6 分量 → 恢复快照 | main, M3 段 |

**Check C 的诚实声明**：无历史优化器矩（checkpoint 未保存 optimizer state），是"新初始化
AdamW 的局部实验"，不是原训练步的重放；每分支从同一 θ 快照出发，分支间严格可比。

## 5. 原始数据文件

全精度 JSON（机器格式，含本文件 §1–2 的全部元数据）：
- `workspace/fg_diag_v2_scannetpp.json`（829 KB）
- `workspace/fg_diag_v2_7scenes.json`（586 KB）

人可读 CSV（`scripts/fg_diag_v2_dump.py` 生成，一行一次测量，repr 全精度）：

| 文件 | 行数 | 内容 |
|---|---|---|
| `{ds}_components.csv` | 150 | 场景×状态×输入×条件 → 6 个 loss 分量值 |
| `{ds}_camera_pairs.csv` | 900 | 同坐标 → 每个相机对(01..23)的 chordal/角度/‖t‖/orth/det/深度有效率 |
| `{ds}_grad_norms.csv` | 240/96 | 场景×状态×probe×条件×参数组 → 7 个梯度范数 |
| `{ds}_grad_cosines.csv` | 4440/1776 | 同坐标 → 6×6 余弦全矩阵 + 对 g_base + 该次前向的 loss 值 |
| `{ds}_check_c.csv` | 480/192 | 场景×状态×分支×评估点 → 每分量 before/after/Δ + 预剪梯度范数 |
| `{ds}_meta.csv` | 5/2 | ckpt 路径+sha256、参数组、每个输入的帧列表 |

训练侧原始数据：
- `workspace/diag2_{spp,7s}_{rkdc1h,rkdc1hr}/training_trace.csv` — 每步：scene/step/epoch/
  pair_idx/loss/lr/grad_norm/peak_mem + 各分量（swanlab `free-geometry-tta-pairs` 由此上传）
- `workspace/diag2_*/ckpts/{scene}/protocol.json` — 每场景完整协议（所有 pair 的帧列表）

## 6. 与报告结论的对应（结论 → 数据位置）

| 结论 | 原始数据坐标 |
|---|---|
| 4/5 场景 θ₀ rel_rot≈0 | `scannetpp_components.csv` filter state=theta0, cond=clean, 列 rel_rot |
| probe1 = 相机 0/3 旋转错 87°/127° | `scannetpp_camera_pairs.csv` filter scene=09c1414f1b, input=probe1, cond=clean, 列 rot_geodesic_deg |
| ‖g_R‖ 152 vs ‖g_base‖ 6 | `scannetpp_grad_norms.csv` filter scene=09c1414f1b, state=theta_rel, probe=probe1, cond=m0, group=ALL |
| cos(g_R,g_base)=+1.00 (p1/m0) | `scannetpp_grad_cosines.csv` 同坐标, cosine=cos(g_rel_rot,g_base) |
| g_t ∥ g_rkA (0.96–0.99) | `scannetpp_grad_cosines.csv`, cosine=cos(g_rkd_sh_a,g_rel_tdir), 各坐标 |
| md vs R = −0.78 | 同文件 filter probe=probe1, cond=clean, cosine=cos(g_maskdistill,g_rel_rot), group=ALL |
| office m2 mask 灾难 | `7scenes_components.csv` filter input∈{probe0,pair5}, cond=m2, 列 rel_rot, 三状态对比 |
| pair9: θ₀ 2.03 / θ_b 0.004 / θ_r 1.88 | `7scenes_components.csv` filter input=pair9, cond=clean |
| Check C 数字 | `*_check_c.csv` |

## 7. 已知局限（不隐藏）

1. 每格是**单次前向**测量（无重复/无误差棒）；确定性（eval 模式+固定 seed），但换 GPU/
   cudnn 版本可能有 1e-3 级浮点差。
2. probe 只有 2 个/场景（协议固定划分），pair5/8/9 只在 office/chess 展开。
3. Check C 用新初始化 AdamW（见 §4 声明）。
4. m0–m3 是诊断命名空间的 4 个 mask，不代表训练时实际抽到的 mask 分布；office pair8
   在 m0–m3 下良性 ⇒ 训练日志中它的尖峰来自训练时其他 mask 抽签（不可由本数据复现）。
5. 重训的 θ_base/θ_rel 与被删除的原 checkpoint 不是同一文件（参数与 seed 相同，浮点非必
   同）；sha256 已记录，本报告所有数字都来自重训后的这两个状态。
