# 从分叉实验迁移到统一协议

基线：`b647686481c6846ad2bd2a3d0d69dd7c5eb82c23`。

## 替换关系

| 旧路径/功能 | 当前入口 |
|---|---|
| DA3 protocol_v1 + protocol2 flags | `fg train --model da3` |
| VGGT diagnostics/train_arms | `fg train --model vggt` |
| 三模型 migration trainer | 同一 `fg train`，选择 omega/pi3/dvlt |
| 多种 manifest builder、硬编码 slots | `fg prepare` / `fg inspect` |
| arm 字符串、v2_rel_weight、多个 mask flag | 严格 YAML 中独立 loss 权重与 input_mask_ratio |
| q_feat / geo_w 耦合、梯度 cap / ramp / gate | 对应监督量独立 q，单次 backward |
| v2 selector 和离线 selector 变体 | 默认监测；显式统一 probe policy |
| 一次性实验排队、清盘、训练启动脚本 | 删除；历史实现可在基线提交复现 |

`diagnostics/` 的实验代码整体删除；`bench/experiments/`、token 实验和一次性 LoRA 比较脚本一并移除。`scripts/` 保留标准 benchmark、可视化和权重下载工具。旧 DA3/VGGT 实验训练 dataset/loss/config、test3r 适配、未被当前推理使用的训练栈删除；必要的模型 checkpoint 包装保留，避免破坏标准 benchmark。

`docs/history/` 保存原研究报告，`workspace/` 中原始结果保留。历史文档中的路径是当时提交的路径；对应旧入口不会继续存在于当前工作树。没有在新目录再复制一套可运行 legacy trainer。

## 兼容边界

- 旧 manifest 不自动转换：旧排序/stride 与重复补齐已改变实验含义，必须重新 `fg prepare`。
- 新 checkpoint 是统一可恢复格式，使用 `fg evaluate`。旧格式 LoRA 仍由原标准 benchmark 的模型包装读取。
- 不自动把旧最佳配方的结果当作新配方的结果。比较必须同 manifest、初始化、mask 序列和更新预算重新运行。
- 训练与 GT 指标读取分离。标准 benchmark 通过真实图像路径匹配 GT，不能用另一份索引顺序直接对齐。
- 模型/source/数据根目录通过参数指定；不需要某台机器的 `/root/autodl-tmp` 布局。

## 回退

本改动推送独立分支；主分支保持不变。需要旧实验时从基线提交建立独立 checkout，不恢复旧分支到当前统一入口。
