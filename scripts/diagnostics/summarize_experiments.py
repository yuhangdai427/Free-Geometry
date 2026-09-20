#!/usr/bin/env python3
"""Aggregate finite diagnostic experiments without GT-based recipe selection."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import numpy as np
from self_geometry import ROOT
from self_geometry.common import config,write_json
from self_geometry.data import dataset
from self_geometry.report import METRICS,report

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'artifacts');a=p.parse_args();root=a.root.resolve()
lines=['# 有限歧义排查与重复性实验','','主配置预先固定；下列诊断不用于逐场景选模型或选择主配置。诊断均值仅覆盖每个数据集的第一个场景，不等同于论文的数据集均值。','',
'| 配置 | 诊断完成 | AUC@1 | AUC@3 | AUC@30 | F1 unposed | F1 posed |','|---|---|---|---|---|---|---|']
records={}
for path in [root/'main']+sorted((root/'variants').glob('*')):
    if not path.is_dir():continue
    values=[]
    for name in ['eth3d','7scenes','scannetpp','hiroom']:
        scene=dataset(name,config()).SCENES[0];metric=path/name/scene/'adapted/metrics.json'
        if metric.exists():values.append(json.loads(metric.read_text()))
    means={k:float(np.mean([v[k] for v in values])) for k in METRICS} if values else {}
    records[str(path.relative_to(root))]=dict(completed=len(values),means=means)
    lines.append('| '+' | '.join([str(path.relative_to(root)),f'{len(values)}/4']+[f'{means[k]:.4f}' if k in means else '—' for k in METRICS])+' |')
lines+=['','## 重复运行', 'seed 0、1、2 的各指标标准差仅在同一场景三次均完成时计算。']
repeats={}
for name in ['eth3d','7scenes','scannetpp','hiroom']:
    scene=dataset(name,config()).SCENES[0]
    paths=[root/folder/name/scene/'adapted/metrics.json' for folder in ['main','variants/seed1','variants/seed2']]
    if all(p.exists() for p in paths):
        m=[json.loads(p.read_text()) for p in paths]
        repeats[name]={k:{'mean':float(np.mean([v[k] for v in m])),'sample_std':float(np.std([v[k] for v in m],ddof=1))} for k in METRICS}
        lines+=['',name+': '+json.dumps(repeats[name])]
write_json(root/'diagnostics.json',dict(variants=records,seeds=repeats))
(root/'DIAGNOSTICS.md').write_text('\n'.join(lines)+'\n')
report(root/'main')
print(root/'DIAGNOSTICS.md')
