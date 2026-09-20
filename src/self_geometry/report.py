"""Per-scene, per-dataset reporting without dropping numerical regressions."""
import csv
import json
from pathlib import Path
import numpy as np
from .common import write_json, config
from .benchmark import DATASETS, PAPER_DATASETS, POSE_METRICS, F1_METRICS, DTU_METRICS, metrics_for

# Retained for optional four-dataset diagnostic utilities.
METRICS = [*POSE_METRICS, *F1_METRICS]
PAPER = {
    '7scenes': {'baseline': [None, .25, .86, .43, .37], 'adapted': [None, .28, .87, .44, .38]},
    'eth3d': {'baseline': [.03, .20, .76, .52, .43], 'adapted': [.07, .27, .83, .60, .47]},
    'scannetpp': {'baseline': [None, .56, .94, .60, .40], 'adapted': [None, .54, .94, .54, .41]},
    'hiroom': {'baseline': [None, .51, .88, .60, .68], 'adapted': [None, .47, .85, .55, .64]},
}


def collect(root, c, names=DATASETS):
    from .data import dataset
    root = Path(root)
    rows, summary, missing = [], {}, []
    for name in names:
        expected = dataset(name, c).SCENES
        stages = {}
        for stage in ('baseline', 'adapted'):
            valid = []
            for scene in expected:
                path = root / name / scene / stage / 'metrics.json'
                reason = 'missing'
                if path.exists():
                    try:
                        metrics = json.loads(path.read_text())
                        values = {k: float(metrics[k]) for k in metrics_for(name)}
                        if not all(np.isfinite(v) for v in values.values()):
                            raise ValueError('nonfinite metrics')
                        valid.append(values)
                        rows.append(dict(dataset=name, scene=scene, stage=stage, **values))
                        continue
                    except (ValueError, TypeError, KeyError) as exc:
                        reason = str(exc)
                missing.append(dict(dataset=name, scene=scene, stage=stage, reason=reason))
            stages[stage] = dict(
                completed=len(valid), expected=len(expected), complete=len(valid) == len(expected),
                means={k: float(np.mean([r[k] for r in valid])) for k in metrics_for(name)} if valid else {},
            )
        summary[name] = stages
    return dict(datasets=summary, missing=missing), rows


def report(root, c=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if c is None:
        status = root / 'campaign.json'
        c = json.loads(status.read_text()).get('config') if status.exists() else None
    c = c or config()
    result, rows = collect(root, c)
    summary = result['datasets']
    write_json(root / 'summary.json', result)
    with (root / 'scenes.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'scene', 'stage', *METRICS, *DTU_METRICS])
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Self-Geometry / ' + c.get('model', 'vggt').upper(), '',
             '固定主配置；所有有限数值结果均保留，包括退化场景。部分覆盖的均值仅供进度查看，不能直接对比论文完整数据集。', '',
             '| 数据集 | 方法 | 完成 | AUC@1 ↑ | AUC@3 ↑ | AUC@30 ↑ | F1 unposed ↑ | F1 posed ↑ |',
             '|---|---|---|---|---|---|---|---|']
    for name in PAPER_DATASETS:
        for stage in ('baseline', 'adapted'):
            row = summary[name][stage]
            values = [f'{row["means"][k]:.4f}' if k in row['means'] else '—' for k in METRICS]
            lines.append('| ' + ' | '.join([name, stage, f'{row["completed"]}/{row["expected"]}', *values]) + ' |')
            paper = ['—' if v is None else f'{v:.2f}' for v in PAPER[name][stage]]
            if c.get('model', 'vggt') == 'vggt':
                lines.append('| ' + ' | '.join([name, '论文 VGGT ' + stage, '—', *paper]) + ' |')
    lines += ['', '## DTU（额外数据集，非论文结果）', '',
              '沿用 DA3 的 DTU 协议；距离单位 mm，越小越好。overall = (acc + comp) / 2。acc/comp 字段名按 DA3 返回值原样保留，详见 docs/method.md。', '',
              '| 方法 | 完成 | AUC@1 ↑ | AUC@3 ↑ | AUC@30 ↑ | unposed acc ↓ | unposed comp ↓ | unposed overall ↓ | posed acc ↓ | posed comp ↓ | posed overall ↓ |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
    for stage in ('baseline', 'adapted'):
        row = summary['dtu'][stage]
        values = [f'{row["means"][k]:.4f}' if k in row['means'] else '—' for k in metrics_for('dtu')]
        lines.append('| ' + ' | '.join([stage, f'{row["completed"]}/{row["expected"]}', *values]) + ' |')
    lines += ['', f'未完成/无效评测：{len(result["missing"])} 项；详细原因见 summary.json，进程失败日志见 campaign.json。']
    for stage in ('baseline', 'adapted'):
        if all(summary[n][stage]['complete'] for n in PAPER_DATASETS):
            avg = {k: float(np.mean([summary[n][stage]['means'][k] for n in PAPER_DATASETS])) for k in METRICS}
            lines += ['', stage + ' 论文四数据集等权均值：' + json.dumps(avg)]
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return result
