#!/usr/bin/env python3
"""Verify optimized loading and zero-LoRA output against real saved baselines."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from self_geometry import ROOT
from self_geometry.common import config, seed_all, write_json
from self_geometry.model import load_model, load_images, predict, inject_lora, training_mode

results = {}
for name, folder in [('da3', 'da3_smoke'), ('vggt', 'smoke_dtu_v2')]:
    scene = ROOT / 'artifacts' / folder / 'dtu/scan1'
    manifest = json.loads((scene / 'manifest.json').read_text())
    c = config(model=name)
    seed_all(0)
    images = load_images(manifest['image_files'], c['image_size'], name).cuda()
    tick = time.monotonic()
    model = load_model(c)
    load_seconds = time.monotonic() - tick
    with torch.no_grad():
        pred = predict(model, images, c)
    with np.load(scene / 'baseline/exports/mini_npz/results.npz') as saved:
        errors = {key: float(np.max(np.abs(saved[key] - value.cpu().numpy()))) for key, value in pred.items()}
    assert all(v == 0 for v in errors.values()), (name, errors)
    modules = inject_lora(model, c)
    model.eval()
    with torch.no_grad():
        zero = predict(model, images, c)
    for key in pred:
        torch.testing.assert_close(pred[key], zero[key], rtol=0, atol=0)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert parameters == (15728640 if name == 'da3' else 18874368)
    training_mode(model, c)
    small = predict(model, images[:2], c)
    loss = small['depth'].mean() + small['extrinsics'].square().mean()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)
    results[name] = dict(passed=True, baseline_max_abs_error=errors, zero_lora_exact=True,
                         modules=len(modules), parameters=parameters, finite_nonzero_gradients=True,
                         optimized_load_seconds=load_seconds)
    write_json(ROOT / 'artifacts/model_validation.json', results)
    print(json.dumps({name: results[name]}), flush=True)
    del model, images, pred, zero, small, loss, gradients
    torch.cuda.empty_cache()
