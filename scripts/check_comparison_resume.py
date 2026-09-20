#!/usr/bin/env python3
"""Fork one durable checkpoint and verify resumed Test3R/TCO on real models."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from self_geometry import ROOT
from self_geometry import comparisons as methods
from self_geometry.common import config,digest,write_json
from self_geometry.training import identity

p=argparse.ArgumentParser();p.add_argument('--model',choices=['da3','vggt'],required=True)
p.add_argument('--method',choices=['tco','test3r'],required=True)
p.add_argument('--root',type=Path)
p.add_argument('--source',type=Path)
a=p.parse_args()
source=a.source or ROOT/f'artifacts/comparison_smoke/{a.method}/{a.model}/seed_0/eth3d/courtyard'
root=a.root or ROOT/f'artifacts/comparison_resume/{a.model}/{a.method}'
if root.exists():
    p.error('Verification root already exists; choose a fresh --root')
c=config(source.parents[3]/f'config_{a.model}.json')
c['checkpoint_every']=1
manifest=json.loads((source/'manifest.json').read_text())
for name in ('reference','resumed'):
    directory=root/name
    for file in ('manifest.json','baseline/exports/mini_npz/results.npz'):
        dst=directory/file;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/file,dst)
    protocol=json.loads((source/'baseline/protocol.json').read_text())
    protocol.update(identity=digest(identity(c,manifest)),config=c)
    write_json(directory/'baseline/protocol.json',protocol)
original=methods.save_torch
forked=False

def save_and_fork(path,value):
    global forked
    original(path,value)
    if Path(path).name=='last.pt' and value.get('updates')==1 and not forked:
        dst=root/'resumed/adapted/last.pt';dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dst);forked=True
methods.save_torch=save_and_fork
methods.adapt_comparison(c,root/'reference','eth3d',resume=False)
assert forked
methods.save_torch=original
original_load=methods.load_trainable
restored_exactly=[]
def verified_load(model,state):
    original_load(model,state)
    params=dict(model.named_parameters())
    assert all(torch.equal(params[k].detach().cpu(),v) for k,v in state.items())
    restored_exactly.append(True)
methods.load_trainable=verified_load
original_optimizer_load=torch.optim.Optimizer.load_state_dict
optimizer_verified=[]
def verified_optimizer_load(optimizer,state):
    result=original_optimizer_load(optimizer,state)
    restored=optimizer.state_dict()
    assert restored['param_groups']==state['param_groups']
    for key,values in state['state'].items():
        for field,value in values.items():
            actual=restored['state'][key][field]
            assert torch.equal(actual.cpu(),value.cpu()) if torch.is_tensor(value) else actual==value
    optimizer_verified.append(True)
    return result
torch.optim.Optimizer.load_state_dict=verified_optimizer_load
methods.adapt_comparison(c,root/'resumed','eth3d',resume=True)
assert restored_exactly and optimizer_verified
x=torch.load(root/'reference/adapted/last.pt',map_location='cpu',weights_only=False)
y=torch.load(root/'resumed/adapted/last.pt',map_location='cpu',weights_only=False)
maximum=0.
for name in x['parameters']:
    maximum=max(maximum,float((x['parameters'][name]-y['parameters'][name]).abs().max()))
    assert torch.isfinite(y['parameters'][name]).all()
assert len(x['history'])==len(y['history']) and x['updates']==y['updates']
loss_error=max(abs(i['loss']-j['loss']) for i,j in zip(x['history'],y['history']))
assert loss_error < 1e-6,loss_error
assert torch.equal(x['rng']['cuda'][0],y['rng']['cuda'][0])
result=dict(passed=True,model=a.model,method=a.method,restored_parameters_exact=True,
            restored_optimizer_exact=True,
            rng_after_step_equal=True,max_parameter_difference=maximum,max_loss_difference=loss_error,
            parameter_equivalence=maximum<=1e-6,
            note='CUDA backward is not bitwise deterministic; restored parameters, optimizer, RNG and next forward loss checked separately from subsequent updates.')
write_json(root/'result.json',result);print(result)
