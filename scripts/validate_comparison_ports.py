#!/usr/bin/env python3
"""Real-weight cache equivalence, prompt gradient and reset checks (two models)."""
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from self_geometry import ROOT
from self_geometry.common import config,write_json,seed_all
from self_geometry.model import load_model,load_images,predict,training_mode
from self_geometry.training import load_prediction
from self_geometry.comparisons import cache_tco_encoder,remove_adapters,install_test3r,test3r_loss

results={}
for name in ('da3','vggt'):
    c=config(ROOT/'configs/comparison_fast.yaml',model=name)
    c['method']='test3r'
    seed_all(0,4)
    manifest=json.loads((ROOT/f'artifacts/comparison_smoke/baseline/{name}/seed_0/eth3d/courtyard/manifest.json').read_text())
    images=load_images(manifest['image_files'],504,name).cuda()
    model=load_model(c)
    with torch.no_grad(): original=predict(model,images,c)
    golden=load_prediction(ROOT/f'artifacts/comparison_smoke/baseline/{name}/seed_0/eth3d/courtyard/baseline/exports/mini_npz/results.npz','cuda')
    golden_error={k:float((original[k]-golden[k]).abs().max()) for k in original}
    assert all(v==0 for v in golden_error.values()),golden_error
    cache={};cache_tco_encoder(model,c,cache)
    with torch.no_grad():
        first=predict(model,images,c);second=predict(model,images,c)
    errors={stage:{k:float((p[k]-original[k]).abs().max()) for k in original} for stage,p in [('first',first),('cached',second)]}
    assert all(v==0 for row in errors.values() for v in row.values()),errors
    remove_adapters(model)
    install_test3r(model,c);training_mode(model,c)
    # Distinct source views produce a nonzero consistency gradient.
    loss=test3r_loss(model,images,[1*3+2,2*9+1],c)
    loss.backward()
    grads=[p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    norm=float(torch.stack([g.square().sum() for g in grads]).sum().sqrt())
    assert norm>0
    parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    remove_adapters(model)
    with torch.no_grad(): restored=predict(model,images,c)
    reset={k:float((restored[k]-original[k]).abs().max()) for k in original}
    assert all(v==0 for v in reset.values()),reset
    results[name]=dict(golden_baseline_errors=golden_error,cache_max_errors=errors,reset_max_errors=reset,prompt_parameters=parameters,gradient_norm=norm,loss=float(loss.detach()),point_source=c.get('test3r_vggt_points') if name=='vggt' else 'depth')
    print(name,results[name],flush=True)
    del model,images,original,first,second,restored,loss,grads,cache,golden
    torch.cuda.empty_cache()
write_json(ROOT/'artifacts/comparison_port_validation.json',results)
