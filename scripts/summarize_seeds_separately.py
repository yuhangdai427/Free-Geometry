#!/usr/bin/env python3
"""Summarize the full suite, retaining incomplete coverage and failed stages."""
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT / 'artifacts/separate_seeds')
    a = p.parse_args(argv)
    root = a.root.resolve()
    plan = json.loads((root / 'plan.json').read_text())
    seeds, names = plan['seeds'], plan['datasets']
    summaries, failures = {}, []
    for seed in seeds:
        folder = root / f'seed_{seed}'
        summaries[str(seed)] = report(folder, dict(plan['config'], seed=seed))
        status = folder / 'campaign.json'
        if status.exists():
            for job, record in json.loads(status.read_text())['jobs'].items():
                if record['exit_code']:
                    failures.append(dict(seed=seed, job=job, **record))
    datasets = aggregate_seeds(summaries, names, seeds)
    complete = all(datasets[n][s]['complete'] for n in names for s in ('baseline', 'adapted'))
    result = dict(complete=complete, seeds=seeds, datasets=datasets, failures=failures,
                  planned_adaptations=plan['adaptations'], first_only=plan['first_only'],
                  aggregation='scene macro per dataset per seed; mean and sample std (ddof=1) across seeds')
    write_json(root / 'summary.json', result)
    write_json(root / 'failures.json', failures)
    lines = ['# 五数据集、多 seed 全量报告', '',
             f'Seeds: {seeds}；计划适配 {plan["adaptations"]} 次；完整评测：{complete}。', '',
             '先在数据集内对所有官方场景取等权均值，再对 seed 取均值和样本标准差（ddof=1）。',
             '只有所有请求的 seed 都覆盖完整数据集时才输出跨 seed 统计；部分覆盖数值见各 seed/REPORT.md。',
             'DTU 为额外实验，距离单位 mm、越低越好；其他四数据集 F1 与 AUC 越高越好。两类重建指标不混算。', '',
             '| 数据集 | 阶段 | seed 覆盖（完成/全部） | 指标 | mean ± std |',
             '|---|---|---|---|---|']
    for name in names:
        for stage in ('baseline', 'adapted'):
            row = datasets[name][stage]
            coverage = ', '.join(f'{s}: {v["completed"]}/{v["expected"]}' for s, v in row['per_seed'].items())
            for key in metrics_for(name):
                stat = row['metrics'].get(key)
                if stat:
                    sd = f'{stat["sample_std"]:.4f}' if stat['sample_std'] is not None else 'n/a'
                    value = f'{stat["mean"]:.4f} ± {sd}'
                else:
                    value = '待完整覆盖'
                lines.append(f'| {name} | {stage} | {coverage} | {key} | {value} |')
    lines += ['', f'失败阶段：{len(failures)}。失败详情及日志路径见 failures.json。数值下降不标为运行失败。',
              'paired_adapted_minus_baseline 给出相同 seed 的增量及其跨 seed 标准差，见 summary.json。']
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(root / 'REPORT.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
