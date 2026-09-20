#!/usr/bin/env python3
"""Show training/evaluation coverage and active tasks for the method matrix."""
import argparse
import json
from pathlib import Path
from collections import defaultdict
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
plan=json.loads((a.root/'matrix_plan.json').read_text())
counts=defaultdict(lambda:[0,0,0])
for job in plan['jobs']:
    method,model=job['method'],job['model']
    stage='baseline' if method=='baseline' else 'adapted'
    path=a.root/method/model/f'seed_{job["seed"]}'/job['dataset']/job['scene']/stage
    row=counts[(model,method)]
    row[0]+=int((path/'exports/mini_npz/results.npz').exists() and (path/('protocol.json' if stage=='baseline' else 'complete.json')).exists())
    row[1]+=int((path/'metrics.json').exists());row[2]+=1
for (model,method),(trained,evaluated,total) in counts.items():
    print(f'{model:5s} {method:14s} predictions {trained:4d}/{total:<4d} evaluations {evaluated:4d}/{total:<4d}')
for method in plan['methods']:
    state=a.root/method/'suite.json'
    if state.exists():
        s=json.loads(state.read_text())
        if s.get('active'):
            print(method,'active:',', '.join(s['active']))
        failures=[k for k,v in s['jobs'].items() if v['exit_code']]
        if failures:
            print(method,'failed tasks:',len(failures))
