# TCO 显存诊断与调度约束

2026-09-20：实验已按用户要求停止；此记录不授权自动恢复。

## 协议来源更正

不能将下面记录的“100 帧训练 OOM”表述成论文已确认的 TCO 复现要求。

- Self-Geometry IV-A 写明沿用 DA3 最多 100 帧的评测协议，并采用对比方法的
  官方训练配置；TCO 相机先验改用冻结模型预测。它没有明确披露 TCO 每步
  训练帧数、是否另采训练子集以及 DA3 移植的显存处理。
  来源：https://arxiv.org/html/2608.10708v2#S4.SS1
- TCO 原文 4.1.1 明确因显存限制在 7Scenes/NRGBD 采用稀疏输入：步长分别
  200/500；ETH3D/DTU 步长为 5。附录 Table 9 的耗时实验采用 10 视图。
  原文没有要求统一使用 100 帧训练。
  来源：https://arxiv.org/html/2604.03878v1#S4.SS1.SSS1
- 固定提交 65387333877723cf99086f7f29089379dd50c59a 的官方采样清单示例：
  `7scenes_mv-recon_seq-id-map-kf200.json` 中 chess/seq-03 为 5 帧；
  `ETH3D_mv-recon_seq-id-map-kf5.json` 中 courtyard 为 8 帧；
  `DTU_mv-recon_seq-id-map-kf5.json` 中 scan1 为 10 帧。
- 官方 `tco_vggt_lora.py` 每轮处理传入的全部 S 帧；传入数量由数据采样清单
  决定。`num_view_groups=100` 是渲染损失的分组参数，不是训练帧数设置。

当前移植把 DA3 benchmark 的最多 100 帧直接作为 TCO 每步输入，是本仓库
将两种协议结合后的实现选择，有官方训练循环依据，但没有证据证明完全等同
于 Self-Geometry 作者的 TCO 设置。也不能反向断言作者一定保留了原 TCO 的
稀疏采样。协议待厘清，不能只把问题作为显存优化处理后启动全量。

## 已确认的 OOM

`artifacts/paper_protocol_tco_ram_limited` 中 DA3-Giant + TCO 的两个
100 帧场景（7Scenes/chess、ScanNet++/09c1414f1b）均在第一次训练前向的
DualDPT 深度 head 插值操作中 OOM，还未完成一次参数更新。

两份 worker.log 均记录：

- GPU 总容量：94.97 GiB。
- TCO 本进程显存占用：94.48 GiB，其中 PyTorch 已分配 93.59 GiB。
- 剩余显存：478.31 MiB；本次申请：486.00 MiB。
- GPU 准入：90 GiB，即调度器全部容量；suite 的 gpu_workers=1。

因此，这两次失败不能归因于其他实验大量占用 GPU：TCO 本进程已经占满
几乎整卡。90 GiB 是调度器的互斥预算，不是 CUDA 分配硬上限。
这只证明当时实现与配置的 100 帧训练超出显存，不能推广为所有 TCO 场景
或经过内存优化的实现都无法在 96 GB 卡上运行。较小场景已有成功记录。
后续加入的 depth-head chunk checkpoint 尚未完成 100 帧训练验证。

原始证据：

- `artifacts/paper_protocol_tco_ram_limited/tco/da3/workers/7scenes/chess/worker.log`
- `artifacts/paper_protocol_tco_ram_limited/tco/da3/workers/scannetpp/09c1414f1b/worker.log`
- `artifacts/paper_protocol_tco_ram_limited/tco/suite.json`

## 调度约束

TCO 必须独占 GPU，不与其他模型训练、GPU 推理或 DTU GPU 融合并行。
现有 `gpu_queue.reservation()` 已让所有 TCO 工作申请全部 GPU 调度预算，
`scene_worker.py` 持有该准入锁至整个场景任务结束。纯 CPU 评测可以重叠，
但必须遵守共享主机 RAM 限额。所有 GPU 任务必须走同一准入器；手动启动的
外部 GPU 进程不受该锁控制，恢复前仍需检查实际 GPU 占用。
