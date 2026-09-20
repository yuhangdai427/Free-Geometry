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


def selected_results(root, plan):
    """Show actual planned-scene metrics without pretending they cover a dataset."""
    rows = []
    lines = ['# Selected-scene evaluations', '',
             'Raw [0,1] AUC/F1; DTU distances in mm. Missing evaluation is pending, not a score of zero.', '',
             '| Model | Method | Dataset / scene | Seed | Frames | AUC@1 | AUC@3 | AUC@30 | Unposed F1 / DTU distance | Posed F1 / DTU distance | Status |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    schedule = {k: plan.get('config', {}).get(k) for k in
                ('test3r_max_triplets', 'test3r_epochs', 'test3r_accum', 'test3r_max_updates')}
    lines[2:2] = [f'Test3R settings: `{json.dumps(schedule)}`', '']
    for job in plan['jobs']:
        directory = root/job['method']/job['model']/f'seed_{job["seed"]}'/job['dataset']/job['scene']
        stage = 'baseline' if job['method'] == 'baseline' else 'adapted'
        row = dict(job, status='pending', metrics={})
        manifest_path = directory/'manifest.json'
        row['frames'] = len(json.loads(manifest_path.read_text())['image_files']) if manifest_path.exists() else None
        path = directory/stage/'metrics.json'
        if path.exists():
            try:
                metrics = json.loads(path.read_text())
                row['metrics'] = {key: float(metrics[key]) for key in metrics_for(job['dataset'])}
                if not all(np.isfinite(v) for v in row['metrics'].values()):
                    raise ValueError('Nonfinite metric')
                row['status'] = 'evaluated'
            except (ValueError, KeyError, TypeError) as exc:
                row.update(status='invalid', error=str(exc), metrics={})
        geom = 'overall' if job['dataset'] == 'dtu' else 'fscore'
        keys = ['auc01', 'auc03', 'auc30', f'recon_unposed_{geom}', f'recon_posed_{geom}']
        values = [f'{row["metrics"][k]:.6f}' if k in row['metrics'] else '—' for k in keys]
        lines.append(f'| {job["model"]} | {job["method"]} | {job["dataset"]}/{job["scene"]} | {job["seed"]} | {row["frames"] or "—"} | '+ ' | '.join(values)+f' | {row["status"]} |')
        rows.append(row)
    result = dict(scope='selected scenes only; not dataset averages', test3r_settings=schedule,
                  complete=sum(r['status']=='evaluated' for r in rows), total=len(rows), rows=rows)
    write_json(root/'selected_results.json', result)
    (root/'SELECTED_RESULTS.md').write_text('\n'.join(lines)+'\n')
    return result


def summarize(root):
    root = Path(root)
    plan = json.loads((root/'matrix_plan.json').read_text())
    selected_results(root, plan)
    result = dict(cells=plan['cells'], results={}, missing=[], failures=[], complete=True,
                  test3r_schedule=('budget_variant' if plan['config'].get('test3r_max_updates') is not None
                                   else f"triplet_cap_{plan['config']['test3r_max_triplets']}_per_epoch"
                                   if plan['config'].get('test3r_max_triplets') is not None else 'exhaustive'))
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
