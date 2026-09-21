# 实验口径记录

## Teacher 帧数规则（2026-09-20 确认）

**所有实验统一使用 auto 模式（不传 `--teacher_N`），让代码自动分派：**

```python
if teacher_N is None:
    teacher_N = 16 if N >= 16 else 8   # N < 16 → 8 帧 teacher
teacher_N = min(teacher_N, N)           # 不超过场景总帧数
```

**禁止**显式传 `--teacher_N 16` — 会在 N<16 的场景上导致 teacher 拿走全部帧（如 pipes N=14 → teacher=14），上下文差距过大，无论 loss 怎么调都无法修复。

**VGGT/Ω 的 manifest 已按此规则构建（auto），DA3 的 CLI 需要不传此参数。**
