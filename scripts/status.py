#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from self_geometry import ROOT

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, default=ROOT / 'artifacts/full_dual')
a = p.parse_args()
root = a.root.resolve()
for name in ('plan.json', 'experiments.json', 'suite.json'):
    path = root / name
    if path.exists():
        data = json.loads(path.read_text())
        print(name, json.dumps({k: v for k, v in data.items() if k not in ('jobs', 'config', 'configs', 'scenes')}, ensure_ascii=False))
for folder in sorted([*root.glob('seed_*'), *root.glob('*/seed_*')]):
    if not folder.is_dir():
        continue
    complete = list(folder.rglob('adapted/complete.json'))
    metrics = list(folder.rglob('metrics.json'))
    print(f'{folder.relative_to(root)}: adapted={len(complete)}, evaluations={len(metrics)}')
    status = folder / 'campaign.json'
    if status.exists():
        data = json.loads(status.read_text())
        print(' active:', data.get('active', []))
        for job, value in data['jobs'].items():
            if value['exit_code']:
                print(' FAILED:', job, 'log:', value['log'])
