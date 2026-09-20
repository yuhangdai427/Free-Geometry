#!/usr/bin/env python3
"""Method-separated scene means, seed statistics and explicit failure coverage."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from self_geometry.common import write_json
from self_geometry.benchmark import metrics_for, DATASETS
from self_geometry.data import dataset


def summarize(root):
    root = Path(root)
    plan = json.loads((root/'matrix_plan.json').read_text())
    result = dict(cells=plan['cells'], results={}, missing=[], failures=[], complete=True,
                  test3r_schedule='exhaustive' if plan['config'].get('test3r_max_updates') is None else 'budget_variant')
    lines = ['# DA3 / VGGT method comparison', '',
             'Scene macro means, then seed mean ± sample standard deviation. Regressions are retained.', '',
             '| Model | Method | Dataset | Seed coverage | Metric | Mean ± std |',
             '|---|---|---|---|---|---|']
    for model in plan['models']:
        result['results'][model] = {}
        for method in plan['methods']:
            rows = result['results'][model][method] = {}
            stage = 'baseline' if method == 'baseline' else 'adapted'
            for name in plan['datasets']:
                expected = list(dataset(name,plan['config']).SCENES)
                keys = metrics_for(name)
                seeds = {}
                for seed in plan['seeds']:
                    values = []
                    for scene in expected:
                        path = root/method/model/f'seed_{seed}'/name/scene/stage/'metrics.json'
                        try:
                            record = json.loads(path.read_text())
                            record = {k:float(record[k]) for k in keys}
                            if not all(np.isfinite(v) for v in record.values()):
                                raise ValueError('nonfinite metric')
                            values.append(record)
                        except (OSError, ValueError, KeyError, TypeError) as e:
                            result['missing'].append(dict(model=model,method=method,dataset=name,scene=scene,seed=seed,error=str(e)))
                    seeds[str(seed)] = dict(completed=len(values),expected=len(expected),
                        means={k:float(np.mean([v[k] for v in values])) for k in keys} if values else {})
                complete = all(v['completed'] == v['expected'] for v in seeds.values())
                row = rows[name] = dict(complete=complete,per_seed=seeds,metrics={})
                result['complete'] &= complete
                coverage = ', '.join(f'{s}: {v["completed"]}/{v["expected"]}' for s,v in seeds.items())
                for key in keys:
                    display = 'incomplete'
                    if complete:
                        nums = [v['means'][key] for v in seeds.values()]
                        mean = float(np.mean(nums))
                        sd = float(np.std(nums,ddof=1)) if len(nums)>1 else None
                        row['metrics'][key] = dict(mean=mean,sample_std=sd)
                        display = f'{mean:.4f} ± {sd:.4f}' if sd is not None else f'{mean:.4f} ± n/a'
                    lines.append(f'| {model} | {method} | {name} | {coverage} | {key} | {display} |')
    for method in plan['methods']:
        state = root/method/'suite.json'
        if state.exists():
            for job,v in json.loads(state.read_text())['jobs'].items():
                if v['exit_code']:
                    result['failures'].append(dict(method=method,job=job,**v))
    result['full_matrix'] = (result['complete'] and set(plan['models'])=={'da3','vggt'}
                            and set(plan['methods'])=={'baseline','self_geometry','test3r','tco'}
                            and set(plan['datasets'])==set(DATASETS) and len(plan['seeds'])==3
                            and not plan['first_only'] and plan['config']['max_frames']==100)
    lines += ['',f'Coverage complete: {result["complete"]}; full matrix: {result["full_matrix"]}; failures: {len(result["failures"])}.',
              'DTU distances are mm (lower is better); other datasets use F1 (higher is better).',
              f'Test3R schedule: {result["test3r_schedule"]}. See matrix_plan.json for overrides.']
    write_json(root/'matrix_summary.json',result)
    write_json(root/'matrix_failures.json',result['failures'])
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();r=summarize(a.root);print(a.root/'REPORT.md')
