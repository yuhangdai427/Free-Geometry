#!/usr/bin/env python3
"""Does the selected checkpoint improve correspondence losses at full context?"""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import torch
from self_geometry import ROOT
from self_geometry.common import config,write_json
from self_geometry.geometry import residuals,filter_mask,huber
from self_geometry.matching import get_pair
from self_geometry.training import load_prediction
c=config();torch.set_num_threads(4)
d=ROOT/'artifacts/main/hiroom/20241230/828738/cam_sampled_08'
init=json.loads((d/'adapted/initialization.json').read_text());target=init['target'];delta=init['thresholds']
pairs=torch.load(d/'matches.pt',map_location='cpu',weights_only=False)['pairs']
base=load_prediction(d/'baseline/exports/mini_npz/results.npz','cpu');adapt=load_prediction(d/'adapted/exports/mini_npz/results.npz','cpu')
results={}
for scope in ['target','all_pairs']:
    keys=[(target,j) for j in range(len(base['depth'])) if j!=target] if scope=='target' else list(pairs)
    for name,pred in [('baseline',base),('adapted',adapt)]:
        for mask_name in ['baseline','current']:
            ecs=[];mvcs=[]
            for i,j in keys:
                xi,xj=get_pair(pairs,i,j,'cpu')
                if not len(xi):continue
                ec,mvc,valid=residuals(pred,i,j,xi,xj)
                if mask_name=='baseline':
                    a,b,v=residuals(base,i,j,xi,xj);mask=filter_mask(a,b,v,c['filter_keep'])&valid
                else:mask=filter_mask(ec,mvc,valid,c['filter_keep'])
                if mask.any():ecs.append(ec[mask]);mvcs.append(mvc[mask])
            ec=torch.cat(ecs);mvc=torch.cat(mvcs)
            results[f'{scope}/{name}/{mask_name}_mask']={'matches':len(ec),'ec_mean':float(ec.mean()),'mvc_mean':float(mvc.mean()),'ec_huber':float(huber(ec,delta['ec']).mean()),'mvc_huber':float(huber(mvc,delta['mvc']).mean())}
write_json(ROOT/'artifacts/hiroom_diagnosis/full_context_losses.json',results)
print(json.dumps(results,indent=2))
