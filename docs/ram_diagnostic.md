# 2026-09-20 主机 RAM 诊断与处置

baseline 和 Self-Geometry 的双模型、五个首场景、seed 0 已完成训练和完整评测。
Test3R 的十个训练任务都成功完成 50 次更新。VGGT/ETH3D/courtyard 的 CPU
重建评测运行约 2249 秒后收到 SIGTERM（exit -15），尚未导出第一个融合点云。
这与用户终止进程一致，不能仅据该退出码宣称内核 OOM。没有保存被杀进程的
RSS 峰值或 C++ 调用栈，因此不声称已精确量出 TSDF 分配了多少 RAM。

现有导出文件的轻量诊断显示：38 张原始 RGB 均为 6048×4032；单份 float32
全分辨率深度栈约 3.45 GiB，RGB 栈约 2.59 GiB。保留的官方 ETH3D evaluator
一次加载所有原图并放大深度，随后执行 Open3D TSDF 融合。
VGGT baseline 的 Umeyama 对齐 scale 为 10.873，中位对齐深度约 7.96 米；
Test3R 的 scale 为 143.766，中位深度约 133.42 米，最大约 222.61 米。
Self-Geometry 对应中位深度约 7.88 米。官方融合 voxel 为 0.0390625 米，
深度截断为 100000 米；因此退化的远距离几何不会被常规深度截断排除，极可能
扩大 TSDF 稀疏体素分配。此为有数据支持的诊断推断，不是完成过的高内存重演。

此前调度只控制 GPU 显存，CPU eval 没有主机 RAM 上限；两个评测可与训练
重叠。它放大了个别场景退化的影响。这是资源调度缺陷，不是所有数据加载或
模型评测接口不可用。此前按用户要求停止，最新指令已恢复缺失评测；保留原训练与结果。

## 已实施的限制

- 全部实验共享 systemd user slice `self-geometry.slice`：MemoryMax=72 GiB、
  MemoryHigh=64 GiB、MemorySwapMax=0；不同矩阵 runner 也共享总上限。
- 每个训练、融合、评测子任务置于独立 scope，默认 MemoryMax=32 GiB、无 swap。
  cgroup 的 memory.oom.group=1 保证超限时一起终止该任务的子进程。
- 默认 CPU eval 并发从 2 降为 1；超限按失败记录，不改变帧数、融合或指标数学。
- 不能建立硬限制时直接报错，不静默降级为无限制运行。
- 在独立 64 MiB scope 中分配 160 MiB 的实际探针已被内核 SIGKILL；
  日志证明生效上限和 group OOM 标志，见本地 artifacts/ram_limit_probe.json。

当前训练 RNG 仅 0；完整实验的评测抽帧为 42/43/44，同图片只评一次。
修正后的 TCO 在 artifacts/paper_protocol_tco_sparse_train 补跑，训练采用独立稀疏清单；
Test3R 复用既有 checkpoint 补评测，单评测 scope 上限 48 GiB，全局仍为 72 GiB。
ETH3D 已改为逐帧加载原分辨率 RGBD，减少全场景数组副本，未修改 TSDF 指标参数。
