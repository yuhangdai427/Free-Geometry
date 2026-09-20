#!/usr/bin/env python3
"""Integration checks on real VGGT, including zero-LoRA equivalence and FP32 sensitivity."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from self_geometry import ROOT
from self_geometry.common import config,seed_all,write_json
from self_geometry.model import load_images,load_model,predict,inject_lora,trainable_state,training_mode
from self_geometry.data import dataset

c=config();seed_all(0);ds=dataset('eth3d',c);files=ds.get_data('courtyard').image_files[:3]
images=load_images(files,c['image_size']).cuda();model=load_model(c)
with torch.no_grad():
    original=predict(model,images,c)
    fp32=predict(model,images,dict(c,precision='fp32'))
modules=inject_lora(model,c);model.eval()
with torch.no_grad():zero=predict(model,images,c)
checks={}
for k in original:
    torch.testing.assert_close(zero[k],original[k],rtol=0,atol=0)
    checks[k]={'zero_lora_max_abs':float((zero[k]-original[k]).abs().max()),
               'bf16_fp32_mean_abs':float((fp32[k]-original[k]).abs().mean()),
               'bf16_fp32_max_abs':float((fp32[k]-original[k]).abs().max())}
assert all(not p.requires_grad for n,p in model.named_parameters() if not n.endswith(('.a','.b')))
assert len(modules)==72
write_json(ROOT/'artifacts/runtime_check.json',dict(modules=len(modules),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),checks=checks))
print(json.dumps(checks,indent=2))
