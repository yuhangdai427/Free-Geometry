#!/usr/bin/env python3
"""Independent model/dataset/seed statistics for the dual-model suite."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from self_geometry import ROOT
from self_geometry.aggregate import aggregate_seeds
from self_geometry.benchmark import metrics_for
from self_geometry.common import write_json
from self_geometry.report import report


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT / 'artifacts/full_dual')
    a = p.parse_args(argv)
    root = a.root.resolve()
    plan = json.loads((root / 'plan.json').read_text())
    models, seeds, names = plan['models'], plan['seeds'], plan['datasets']
    result = dict(models={}, seeds=seeds, adaptations=plan['adaptations'], complete=True, failures=[])
    lines = ['# Self-Geometry: DA3-Giant + VGGT', '',
             '每模型、每数据集先取所有场景的等权均值，再计算跨 seed 均值 ± 样本标准差（ddof=1）。缺场景时不输出完整跨 seed 统计。', '',
             '| 模型 | 数据集 | 阶段 | 各 seed 覆盖 | 指标 | mean ± std |',
             '|---|---|---|---|---|---|']
    for model in models:
        summaries = {str(s): report(root / model / f'seed_{s}', dict(plan['configs'][model], seed=s)) for s in seeds}
        stats = aggregate_seeds(summaries, names, seeds)
        result['models'][model] = stats
        for name in names:
            for stage in (('baseline',) if plan['configs'][model].get('method') == 'baseline' else ('baseline', 'adapted')):
                row = stats[name][stage]
                result['complete'] &= row['complete']
                coverage = ', '.join(f'{s}: {v["completed"]}/{v["expected"]}' for s, v in row['per_seed'].items())
                for key in metrics_for(name):
                    value = row['metrics'].get(key)
                    if value:
                        sd = f'{value["sample_std"]:.4f}' if value['sample_std'] is not None else 'n/a'
                        display = f'{value["mean"]:.4f} ± {sd}'
                    else:
                        display = '待完整覆盖'
                    lines.append(f'| {model} | {name} | {stage} | {coverage} | {key} | {display} |')
    state = root / 'suite.json'
    if state.exists():
        result['failures'] = [dict(job=k, **v) for k, v in json.loads(state.read_text())['jobs'].items() if v['exit_code']]
    lines += ['', f'完整评测：{result["complete"]}；失败任务：{len(result["failures"])}。',
              'DTU overall 是 mm 距离，越小越好；不与其他数据集的 F1 混算。有限的退化指标原样保留。']
    write_json(root / 'summary.json', result)
    write_json(root / 'failures.json', result['failures'])
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(root / 'REPORT.md')


if __name__ == '__main__':
    main()
