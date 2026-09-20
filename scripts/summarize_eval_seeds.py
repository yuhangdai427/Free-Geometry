#!/usr/bin/env python3
"""Scene macro means followed by evaluation-seed means, with alias counts."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from self_geometry.common import write_json
from self_geometry.benchmark import metrics_for


def summarize(root):
    root = Path(root)
    evaluation = json.loads((root/'plan.json').read_text())
    plan = json.loads((Path(evaluation['source'])/'matrix_plan.json').read_text())
    rows, groups = [], []
    for method in plan['methods']:
        for model in plan['models']:
            for name, scenes in plan['scenes'].items():
                means, unique = [], set()
                completed = 0
                for seed in evaluation['eval_seeds']:
                    values = []
                    for scene in scenes:
                        directory = root/method/model/f'eval_seed_{seed}'/name/scene
                        row = dict(method=method, model=model, dataset=name, scene=scene, eval_seed=seed, status='missing')
                        try:
                            ref = json.loads((directory/'evaluation_ref.json').read_text())
                            canonical = Path(ref['canonical_directory'])
                            metrics = json.loads((canonical/ref['stage']/'metrics.json').read_text())
                            if not all(np.isfinite(metrics[k]) for k in metrics_for(name)):
                                raise ValueError('Incomplete or nonfinite metrics')
                            unique.add((str(canonical), ref['checkpoint_sha256']))
                            row.update(status='evaluated', metrics=metrics, canonical=str(canonical))
                            values.append(metrics); completed += 1
                        except (OSError, ValueError, KeyError, TypeError):
                            pass
                        rows.append(row)
                    if len(values) == len(scenes):
                        means.append({k: float(np.mean([v[k] for v in values])) for k in metrics_for(name)})
                complete = len(means) == len(evaluation['eval_seeds'])
                metrics = ({k: dict(mean=float(np.mean([m[k] for m in means])),
                                   sample_std=float(np.std([m[k] for m in means], ddof=1)) if len(means)>1 else 0.)
                            for k in metrics_for(name)} if complete else {})
                groups.append(dict(method=method, model=model, dataset=name, complete=complete,
                                   completed=completed, expected=len(scenes)*len(evaluation['eval_seeds']),
                                   unique_evaluations=len(unique), metrics=metrics))
    result = dict(protocol=evaluation, groups=groups, rows=rows,
                  note='Sampling seeds share one trained checkpoint; aliases are reused results, not independent repeats.')
    write_json(root/'summary.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    summarize(p.parse_args().root)
