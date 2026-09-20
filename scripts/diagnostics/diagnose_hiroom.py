#!/usr/bin/env python3
"""Diagnostic ONLY: test reference-order sensitivity and GT correspondence quality.
GT is used for post-hoc diagnostics, never for training or checkpoint selection.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import cv2
import numpy as np
import torch
from self_geometry import ROOT
from self_geometry.common import config,seed_all,write_json
from self_geometry.data import dataset
from self_geometry.model import load_images,load_model,predict,inject_lora,load_trainable
from self_geometry.geometry import residuals,filter_mask
from self_geometry.training import load_prediction
from self_geometry.matching import get_pair
from depth_anything_3.bench.utils import compute_pose
from depth_anything_3.utils.geometry import as_homogeneous

c=config();seed_all(0);scene='20241230/828738/cam_sampled_08'
d=ROOT/'artifacts/main/hiroom'/scene
out=ROOT/'artifacts/hiroom_diagnosis';out.mkdir(exist_ok=True)
manifest=json.loads((d/'manifest.json').read_text());initial=json.loads((d/'adapted/initialization.json').read_text())
images=load_images(manifest['image_files'],c['image_size']).cuda();target=initial['target']
order=[target]+[i for i in range(len(images)) if i!=target];undo=np.argsort(order)
with np.load(d/'gt_meta.npz') as z:gt_ext=torch.from_numpy(as_homogeneous(z['extrinsics']))
base=load_prediction(d/'baseline/exports/mini_npz/results.npz','cuda')
adapted=load_prediction(d/'adapted/exports/mini_npz/results.npz','cuda')
model=load_model(c)
with torch.no_grad():base_reorder=predict(model,images[order],c)
base_reorder={k:v[undo] for k,v in base_reorder.items()}
inject_lora(model,c)
load_trainable(model,torch.load(d/'adapted/best.pt',map_location='cpu',weights_only=False)['parameters']);model.eval()
with torch.no_grad():adapt_reorder=predict(model,images[order],c)
adapt_reorder={k:v[undo] for k,v in adapt_reorder.items()}
result={'reference_target':target,'ordering':{}}
for name,pred in [('base_original',base),('base_target_first',base_reorder),('adapt_original',adapted),('adapt_target_first',adapt_reorder)]:
    pose=compute_pose(torch.from_numpy(as_homogeneous(pred['extrinsics'].cpu().numpy())),gt_ext)
    result['ordering'][name]={'pose':{k:float(v) for k,v in pose.items()},'depth_median':float(pred['depth'].median()),'fx_median':float(pred['intrinsics'][:,0,0].median())}
# Precision proxy: GT epipolar distance (does not require GT depth or visibility).
gt=dataset('hiroom',c).get_data(scene);ks=[]
for i,path in enumerate(gt.image_files):
    h,w=cv2.imread(path).shape[:2];k=gt.intrinsics[i].copy();k[0]*=images.shape[-1]/w;k[1]*=images.shape[-2]/h;ks.append(k)
gt_pred=dict(depth=base['depth'],intrinsics=torch.tensor(np.stack(ks),device='cuda'),extrinsics=torch.tensor(gt.extrinsics[:,:3],device='cuda'))
pairs=torch.load(d/'matches.pt',map_location='cpu',weights_only=False)['pairs']
raw=[];filtered=[]
with torch.no_grad():
    for (i,j) in pairs:
        xi,xj=get_pair(pairs,i,j,'cuda')
        if not len(xi):continue
        ec,mvc,valid=residuals(base,i,j,xi,xj);mask=filter_mask(ec,mvc,valid,c['filter_keep'])
        gt_ec,_,_=residuals(gt_pred,i,j,xi,xj)
        raw.append(gt_ec);filtered.append(gt_ec[mask])
for name,values in [('raw',raw),('baseline_filtered',filtered)]:
    x=torch.cat(values)
    result[name]={'matches':len(x),'gt_sampson_root_median':float(x.median()),'fraction_under_1px':float((x<1).float().mean()),'fraction_under_3px':float((x<3).float().mean())}
write_json(out/'diagnosis.json',result)
print(json.dumps(result,indent=2))
