#!/usr/bin/env python3
"""Fork the SAME step-5 checkpoint, compare uninterrupted and restored step 6.

Independent CUDA training prefixes need not be bitwise identical (FlashAttention /
grid_sample backward). Sharing the prefix isolates actual restore correctness.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from self_geometry import ROOT
from self_geometry.common import config,digest,write_json
from self_geometry import training

p=argparse.ArgumentParser();p.add_argument('--worker',choices=['reference','resume'])
p.add_argument('--model',choices=['vggt','da3'],default='vggt')
p.add_argument('--root',type=Path)
p.add_argument('--source',type=Path)
a=p.parse_args()
root=(a.root or ROOT/f'artifacts/resume_v2/{a.model}').resolve()
source=(a.source or ROOT/'artifacts'/('da3_smoke' if a.model=='da3' else 'smoke_dtu_v2')/'dtu/scan1').resolve()
manifest=json.loads((source/'manifest.json').read_text())
c=config(overrides=[f"max_frames={manifest['max_frames']}",'iterations=6'],model=a.model)
if a.worker:
    directory=root/('reference' if a.worker=='reference' else 'resumed')
    if a.worker=='reference':
        original=training.save_torch
        def fork_after_five(path,value):
            original(path,value)
            if Path(path).name=='last.pt' and value.get('next_iteration')==5:
                dst=root/'resumed/adapted';dst.mkdir(exist_ok=True)
                shutil.copy2(path,dst/'last.pt')
                shutil.copy2(Path(path).parent/'steps.jsonl',dst/'steps.jsonl')
        training.save_torch=fork_after_five
    training.adapt(c,directory,resume=a.worker=='resume')
    sys.exit(0)
manifest=json.loads((source/'manifest.json').read_text())
for name in ['reference','resumed']:
    directory=root/name;directory.mkdir(parents=True,exist_ok=True)
    for file in ['manifest.json','matches.pt','baseline/exports/mini_npz/results.npz']:
        dst=directory/file;dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():os.link(source/file,dst)
    protocol=json.loads((source/'baseline/protocol.json').read_text())
    protocol['identity']=digest(training.identity(c,manifest));protocol['config']=c
    write_json(directory/'baseline/protocol.json',protocol)
for worker in ['reference','resume']:
    with (root/(worker+'.log')).open('w') as f:
        subprocess.run([sys.executable,__file__,'--worker',worker,'--model',a.model,'--root',str(root),'--source',str(source)],stdout=f,stderr=subprocess.STDOUT,check=True)
x=torch.load(root/'reference/adapted/last.pt',map_location='cpu',weights_only=False)
y=torch.load(root/'resumed/adapted/last.pt',map_location='cpu',weights_only=False)
assert x['history']==y['history'], 'Losses from restored RNG/state differ'
assert x['best']==y['best']
maximum=0.0
for k in x['parameters']:
    maximum=max(maximum,float((x['parameters'][k]-y['parameters'][k]).abs().max()))
    torch.testing.assert_close(x['parameters'][k],y['parameters'][k],rtol=0,atol=1e-7)
write_json(root/'result.json',dict(passed=True,updates=6,interruption_after=5,parameter_tensors=len(x['parameters']),
    max_parameter_difference=maximum,absolute_tolerance=1e-7,loss_history_equal=True))
print('Resume equivalence passed:',maximum)
