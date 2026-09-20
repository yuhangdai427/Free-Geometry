#!/usr/bin/env python3
"""Compare our preprocessing to the actual local baseline method, without importing its training framework."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import cv2
import numpy as np
import torch
from self_geometry import ROOT
from self_geometry.common import config,write_json
from self_geometry.data import dataset
from self_geometry.model import load_images
from depth_anything_3.bench.evaluator import Evaluator

c=config();original=ROOT.parent/'free_geometry/pr2/scripts/benchmark_vggt.py'
tree=ast.parse(original.read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BaseVGGT')
fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_load_images')
ns={'cv2':cv2,'np':np,'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(original),'exec'),ns)
records=[]
for name in ['eth3d','7scenes','scannetpp','hiroom']:
    ds=dataset(name,c);scene=ds.SCENES[0];data=ds.get_data(scene)
    sampled=Evaluator._sample_frames(SimpleNamespace(max_frames=100),data,scene)
    files=list(sampled.image_files)
    reference=ns['_load_images'](SimpleNamespace(image_size=504),files[:2])[0]
    actual=load_images(files[:2],504)
    torch.testing.assert_close(actual,reference,rtol=0,atol=0)
    records.append(dict(dataset=name,scene=scene,frames=len(files),shape=list(actual.shape),preprocessing_max_difference=0))
write_json(ROOT/'artifacts/protocol_check.json',{'reference':str(original),'checks':records})
print(json.dumps(records,indent=2))
