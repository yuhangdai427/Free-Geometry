#!/usr/bin/env python3
"""Measure checkpointing/fused-AdamW tradeoff on real weights and 13 views.

This is a forward/backward microbenchmark, not a benchmark accuracy result.
13 is the maximum default FAN batch (target plus twelve angle bins).
"""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from self_geometry import ROOT
from self_geometry.common import config, seed_all, write_json
from self_geometry.data import dataset
from self_geometry.model import load_model, load_images, predict, inject_lora, remove_lora, training_mode
from self_geometry.optimization import combine_gradients


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', nargs='+', default=['da3', 'vggt'], choices=['da3', 'vggt'])
    p.add_argument('--views', type=int, default=13)
    p.add_argument('--steps', type=int, default=4)
    a = p.parse_args()
    results = {}
    for name in a.models:
        c = config(model=name)
        seed_all(0)
        files = dataset('dtu', c).get_data('scan1').image_files[:a.views]
        images = load_images(files, c['image_size'], name).cuda()
        model = load_model(c)
        rows = []
        for checkpointing, fused in [(True, False), (False, True)]:
            c.update(checkpointing=checkpointing, fused_optimizer=fused)
            remove_lora(model)
            seed_all(0)
            inject_lora(model, c)
            training_mode(model, c)
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=c['lr'], weight_decay=c['weight_decay'], fused=fused)
            torch.cuda.reset_peak_memory_stats()
            times = []
            for i in range(a.steps + 1):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                start = time.monotonic()
                pred = predict(model, images, c)
                # Exercise all depth/camera paths and three backwards, as in GD.
                mvc = pred['depth'].mean() + pred['intrinsics'].mean() * .001
                ec = pred['extrinsics'].square().mean()
                aux = pred['depth'].square().mean() * .01
                zero = aux * 0
                combine_gradients([mvc, ec, aux, zero, zero], params, [1.] * 5, c['gd'])
                torch.nn.utils.clip_grad_norm_(params, c['clip'])
                optimizer.step()
                torch.cuda.synchronize()
                if i:
                    times.append(time.monotonic() - start)
                del pred, mvc, ec, aux, zero
            rows.append(dict(checkpointing=checkpointing, fused_optimizer=fused,
                             step_seconds=times, mean_seconds=sum(times) / len(times),
                             peak_gib=torch.cuda.max_memory_allocated() / 1024**3))
            del optimizer, params
            remove_lora(model)
            torch.cuda.empty_cache()
        results[name] = dict(views=len(files), input_shape=list(images.shape), profiles=rows,
                             measured_speedup=rows[0]['mean_seconds'] / rows[1]['mean_seconds'])
        write_json(ROOT / 'artifacts/speed_benchmark.json', results)
        print(json.dumps({name: results[name]}), flush=True)
        del model, images
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
