"""Test3R encoder-prompt port and official TCO loss bridge for DA3/VGGT.

Training inputs are RGB and frozen predictions only. See docs/comparisons.md
for architecture mappings, upstream pins and explicit schedule variants.
"""
import json
import math
import random
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
import torch
from torch import nn
from . import ROOT
from .common import seed_all, digest, save_torch, write_json, rng_state, restore_rng
from .model import (LoRALinear, predict, training_mode, trainable_state, load_trainable,
                    remove_lora, load_model, load_images)
from .training import stage_lock, identity, prediction_path, export, load_prediction

METHODS = ('baseline', 'test3r', 'self_geometry', 'tco')
TCO_SHA = '65387333877723cf99086f7f29089379dd50c59a'
TEST3R_SHA = 'a2eb94bc716df27521f053d417fcf6afa9870ff5'


def tco_settings(c, dataset):
    # ScanNet++/HiRoom were not evaluated in upstream: fixed indoor transfer.
    steps, lr, photo = {'eth3d': (40, 5e-4, .2), '7scenes': (40, 1e-3, .2),
                       'dtu': (50, 2e-4, 1.), 'scannetpp': (40, 1e-3, .2),
                       'hiroom': (40, 1e-3, .2)}[dataset]
    return dict(steps=int(steps if c.get('tco_steps') is None else c['tco_steps']),
                lr=float(lr if c.get('tco_lr') is None else c['tco_lr']),
                photo=float(photo if c.get('tco_photo_weight') is None else c['tco_photo_weight']))


class PromptBlock(nn.Module):
    """Keep deep visual prompts inside the frozen image encoder only.

    The author's first prompt passes through blocks 0 and 1. After block i
    (1 <= i < last), replace it with prompt i. Strip after the final block.
    Register prompts inside each block so activation recomputation is safe.
    """
    def __init__(self, base, dim, count, index, last):
        super().__init__()
        self.base, self.count, self.index, self.last = base, count, index, last
        if index != 1:
            self.prompt = nn.Parameter(next(base.parameters()).new_zeros(1, count, dim))

    def forward(self, x, *args, **kwargs):
        if self.index == 0:
            x = torch.cat([self.prompt.expand(x.shape[0], -1, -1), x], dim=1)
        elif self.index != 1:
            x = torch.cat([self.prompt.expand(x.shape[0], -1, -1), x[:, self.count:]], dim=1)
        # Prefix encoder blocks in these two models have no RoPE; callers may
        # pass pos=None. Prompts are stripped before the multi-view decoder.
        if kwargs.get('pos') is not None:
            raise ValueError('Encoder prompt port requires pre-RoPE blocks')
        x = self.base(x, *args, **kwargs)
        return x[:, self.count:] if self.last else x


def install_test3r(model, c):
    model.requires_grad_(False)
    if c['model'] == 'vggt':
        blocks = model.aggregator.patch_embed.blocks
    else:
        net = model.backbone.pretrained
        blocks = net.blocks
    count = len(blocks) if c['model'] == 'vggt' else net.alt_start
    if count < 2:
        raise ValueError('No image-only encoder prefix for Test3R')
    for i in range(count):
        block = blocks[i]
        dim = block.attn.qkv.in_features
        blocks[i] = PromptBlock(block, dim, c['test3r_prompt_size'], i, i == count - 1)
    return count


def remove_adapters(model):
    for name, module in list(model.named_modules()):
        if isinstance(module, (PromptBlock, CachedEncoderBlock)):
            parent, attr = name.rsplit('.', 1)
            setattr(model.get_submodule(parent), attr, module.base)
    remove_lora(model)
    model.eval()
    if hasattr(model, '_sg_checkpointing'):
        model._sg_checkpointing = False


class CachedEncoderBlock(nn.Module):
    """Cache a fixed RGB sequence's frozen encoder, before decoder adaptation."""
    def __init__(self, base, cache, last=False, clone=False):
        super().__init__()
        self.base, self.cache, self.last, self.clone = base, cache, last, clone

    def forward(self, x, *args, **kwargs):
        if 'output' in self.cache:
            result = self.cache['output'] if self.last else x
        else:
            with torch.no_grad():
                result = self.base(x, *args, **kwargs)
            if self.last:
                self.cache['output'] = result
        # DA3 inserts its camera tokens in-place after the encoder prefix.
        return result.clone() if self.last and self.clone else result


def cache_tco_encoder(model, c, scene_cache):
    frozen = scene_cache.setdefault('tco_frozen_encoder', {})
    if c['model'] == 'vggt':
        model.aggregator.patch_embed = CachedEncoderBlock(model.aggregator.patch_embed, frozen, last=True)
    else:
        net = model.backbone.pretrained
        for i in range(net.alt_start):
            net.blocks[i] = CachedEncoderBlock(net.blocks[i], frozen, last=i==net.alt_start-1, clone=True)


def uncache_tco_encoder(model):
    """A training-set encoder cache must never be used on evaluation images."""
    for name, module in list(model.named_modules()):
        if isinstance(module, CachedEncoderBlock):
            parent, _, attr = name.rpartition('.')
            setattr(model.get_submodule(parent) if parent else model, attr, module.base)


def adapt_tco_sparse(c, directory, name, model=None, scene_cache=None, resume=True):
    from .data import prepare_tco_training
    from .training import baseline
    from .cache import copy_cached
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    train_dir = directory/'tco_training'
    train_manifest = prepare_tco_training(name, manifest['scene'], c, train_dir)
    train_c = dict(c, image_size=c['tco_train_image_size'], max_frames=-1)
    sig = digest(identity(c, manifest))
    with stage_lock(directory, 'adapted'):
        out = directory/'adapted'
        if (out/'complete.json').exists():
            done = json.loads((out/'complete.json').read_text())
            if done['identity'] != sig or done['training_manifest'] != train_manifest['fingerprint']:
                raise ValueError('Sparse TCO identity mismatch')
            if prediction_path(directory, 'adapted').exists():
                return
        model = load_model(c) if model is None else model
        remove_adapters(model)
        images = load_images(train_manifest['image_files'], train_c['image_size'], c['model']).cuda()
        baseline(train_c, train_dir, model=model, images=images)
        cache = dict(images=images)
        print(json.dumps(dict(event='tco_train_eval_split', train_frames=len(images),
              eval_frames=len(manifest['image_files']), sampling=train_manifest['sampling'],
              train_image_size=train_c['image_size'], eval_image_size=c['image_size'])), flush=True)
        adapt_comparison(train_c, train_dir, name, model=model, scene_cache=cache, resume=resume)
        # Restore saved adapters even when the inner adaptation returned from cache.
        uncache_tco_encoder(model)
        remove_adapters(model)
        install_tco(model, c)
        load_trainable(model, torch.load(train_dir/'adapted/final.pt', map_location='cpu', weights_only=True))
        cache.clear()
        del images
        model.eval()
        model._sg_checkpointing = False
        torch.cuda.empty_cache()
        eval_images = (scene_cache or {}).get('images')
        if eval_images is None:
            eval_images = load_images(manifest['image_files'], c['image_size'], c['model']).cuda()
        with torch.no_grad():
            result = predict(model, eval_images, c)
        export(result, prediction_path(directory, 'adapted'))
        copy_cached(train_dir/'adapted/final.pt', out/'final.pt')
        info = json.loads((train_dir/'adapted/complete.json').read_text())
        info.update(identity=sig, config=c, training_config=train_c,
                    training_manifest=train_manifest['fingerprint'], training_frames=len(train_manifest['image_files']),
                    evaluation_frames=len(manifest['image_files']), training_sampling=train_manifest['sampling'],
                    implementation='tco_sparse_train_full_eval_v1')
        write_json(out/'complete.json', info)


def install_tco(model, c):
    model.requires_grad_(False)
    names = []
    for name, layer in list(model.named_modules()):
        if not isinstance(layer, nn.Linear):
            continue
        leaf = name.rsplit('.', 1)[-1]
        if c['model'] == 'vggt':
            selected = name.startswith(('aggregator.frame_blocks.', 'aggregator.global_blocks.'))
        else:
            prefix = 'backbone.pretrained.blocks.'
            selected = name.startswith(prefix) and int(name[len(prefix):].split('.')[0]) >= model.backbone.pretrained.alt_start
        # DA3 giant uses a fused SwiGLU feed-forward layer (w12/w3).
        if selected and leaf in ('qkv', 'proj', 'fc1', 'fc2', 'w12', 'w3'):
            parent, attr = name.rsplit('.', 1)
            setattr(model.get_submodule(parent), attr, LoRALinear(layer, 4, 16., 0.))
            names.append(name)
    if not names:
        raise ValueError('No TCO decoder adapters')
    return names


def pair_points(model, pairs, c):
    """Batch independent two-view predictions; return first-camera points."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    if c['model'] == 'vggt' and c.get('test3r_vggt_points', 'depth') == 'native':
        if model.point_head is None:
            raise ValueError('Native Test3R point loss requires the pretrained VGGT point head')
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=c['precision'] == 'bf16'):
            tokens, patch_start = model.aggregator(pairs)
        # VGGT point maps are expressed in the first camera's coordinate frame.
        # The common image i is first in both pairs, as in the author's DUSt3R.
        with torch.autocast('cuda', enabled=False):
            points, _ = model.point_head(tokens, images=pairs, patch_start_idx=patch_start)
        return points[:, 0].float()
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=c['precision'] == 'bf16'):
        if c['model'] == 'da3':
            mean = pairs.new_tensor([.485, .456, .406])[None, None, :, None, None]
            std = pairs.new_tensor([.229, .224, .225])[None, None, :, None, None]
            p = model((pairs - mean) / std, ref_view_strategy='first')
            depth = p['depth'][:, 0].float()
            intr = p['intrinsics'][:, 0].float()
        else:
            p = model(pairs)
            depth = p['depth'][:, 0, ..., 0].float()
            _, intr = pose_encoding_to_extri_intri(p['pose_enc'].float(), pairs.shape[-2:])
            intr = intr[:, 0]
    h, w = depth.shape[-2:]
    y, x = torch.meshgrid(torch.arange(h, device=depth.device), torch.arange(w, device=depth.device), indexing='ij')
    xyz = torch.stack([x, y, torch.ones_like(x)], dim=-1).float()
    rays = torch.einsum('bij,hwj->bhwi', torch.linalg.inv(intr), xyz)
    return rays * depth[..., None]


def triplet_order(n, seed, limit=None):
    """Same shuffled ordered N^3 population, without allocating image tuples.

    Full mode matches random.shuffle(range(N^3)). Budget mode samples without
    replacement and is explicitly a different schedule in the saved config.
    """
    rng = random.Random(seed)
    if limit is not None and limit < n**3:
        return rng.sample(range(n**3), limit)
    order = list(range(n**3))
    rng.shuffle(order)
    return order


def triplet_schedule(n, c):
    """Deterministic per-epoch sampling; a triplet cap is distinct from updates."""
    accum, epochs = c['test3r_accum'], c['test3r_epochs']
    cap, budget = c.get('test3r_max_triplets'), c.get('test3r_max_updates')
    # A shorter update budget consumes a prefix of the same capped sample.
    # Preserve the legacy uncapped budget sampler when no triplet cap is set.
    limit = cap if cap is not None else (None if budget is None else budget * accum)
    order = triplet_order(n, c['seed'], limit)
    total = epochs * len(order)
    if budget is not None:
        total = min(total, budget * accum)
    full_epochs, tail = divmod(total, len(order))
    settings = dict(population=n**3, triplets_cap=cap, epochs=epochs,
                    triplets_per_epoch=len(order), microsteps=total, updates_budget=budget,
                    expected_updates=full_epochs * (len(order) // accum) + tail // accum)
    return order, settings


def test3r_loss(model, images, encoded, c):
    n = len(images)
    triples = [(q // (n*n), q // n % n, q % n) for q in encoded]
    batch = torch.stack([images[[i,j]] for i,j,k in triples] + [images[[i,k]] for i,j,k in triples])
    points = pair_points(model, batch, c)
    a, b = points.chunk(2)
    return (a - b).abs().mean()


@lru_cache(maxsize=1)
def official_tco():
    path = ROOT / 'external/TCO'
    if not (path / 'tco_loss.py').exists():
        raise RuntimeError('Run bash scripts/bootstrap_comparison.sh to fetch pinned TCO source')
    revision = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    modified = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True).strip()
    if revision != TCO_SHA or modified:
        raise RuntimeError(f'TCO source differs from the clean pinned revision {TCO_SHA}')
    # Upstream uses absolute imports. This directory is only added in TCO workers.
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    from tco_loss import compute_tco_losses
    from utils.geometry import to_homogeneous, invert_se3
    return compute_tco_losses, to_homogeneous, invert_se3


def tco_loss(model, images, base, c, settings):
    compute, homogeneous, inverse = official_tco()
    p = predict(model, images, c)
    ext = homogeneous(p['extrinsics'][None])
    pred = dict(depth=p['depth'][None], depth_conf=p['conf'][None],
                camera_poses=inverse(ext), intrinsics=p['intrinsics'][None], pose_type='rel')
    # These are frozen predictions, despite upstream's misleading gt_* names.
    return compute(predictions=pred, images=images[None],
                   gt_extrinsics=homogeneous(base['extrinsics'][None]),
                   gt_intrinsics=base['intrinsics'][None],
                   lambda_pose=1., pose_translation_weight=2.,
                   lambda_intrinsics=c['tco_intrinsics_weight'],
                   lambda_mv_consistency=settings['photo'], num_view_groups=100,
                   pose_rot_loss_type=c.get('tco_pose_rot_loss', 'cosine'), pose_trans_loss_type='normed_l1')


def adapt_comparison(c, directory, dataset, model=None, scene_cache=None, resume=True):
    method = c['method']
    if method not in ('test3r', 'tco'):
        raise ValueError(method)
    directory = Path(directory)
    if method == 'tco' and c.get('tco_training_sampling') == 'official_sparse_v1':
        manifest = json.loads((directory/'manifest.json').read_text())
        if manifest.get('role') != 'tco_training':
            return adapt_tco_sparse(c, directory, dataset, model, scene_cache, resume)
    with stage_lock(directory, 'adapted'):
        out = directory / 'adapted'
        manifest = json.loads((directory / 'manifest.json').read_text())
        sig = digest(identity(c, manifest))
        done = out / 'complete.json'
        if done.exists():
            if json.loads(done.read_text())['identity'] != sig:
                raise ValueError('Adaptation identity mismatch')
            if prediction_path(directory, 'adapted').exists():
                return
        if json.loads((directory/'baseline/protocol.json').read_text())['identity'] != sig:
            raise ValueError('Baseline identity mismatch')
        cache = {} if scene_cache is None else scene_cache
        images = cache.get('images')
        if images is None:
            images = load_images(manifest['image_files'], c['image_size'], c['model']).cuda()
            cache['images'] = images
        if len(images) < 2:
            raise ValueError('Adaptation requires >= 2 views')
        model = load_model(c) if model is None else model
        remove_adapters(model)
        seed_all(c['seed'], c['threads'])
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        if method == 'tco':
            modules = install_tco(model, c)
            cache_tco_encoder(model, c, cache)
            settings = tco_settings(c, dataset)
            total = int(settings['steps'])
            train_c = dict(c, checkpointing=c['tco_checkpointing'])
            optimizer_type, betas, lr = torch.optim.Adam, (.9, .999), settings['lr']
            base = load_prediction(prediction_path(directory, 'baseline'), 'cuda')
        else:
            modules = install_test3r(model, c)
            accum = c['test3r_accum']
            order, settings = triplet_schedule(len(images), c)
            # Match upstream's epoch boundary update test, including carried
            # leftover gradients. Checkpoints below save those gradients too.
            total = settings['microsteps']
            train_c = c
            optimizer_type, betas, lr = torch.optim.AdamW, (.9, .95), float(c['test3r_lr'])
            print(json.dumps(dict(event='test3r_schedule', seed=c['seed'], **settings)), flush=True)
            if c.get('test3r_max_triplets') is not None:
                write_json(out/'triplets.json', dict(seed=c['seed'], frames=len(images),
                           encoding='q = i*N*N + j*N + k', order=order, **settings))
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optimizer_type(params, lr=lr, betas=betas, weight_decay=0., fused=c['fused_optimizer'])
        start, history, updates = 0, [], 0
        checkpoint = out / 'last.pt'
        if resume and checkpoint.exists():
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if saved['identity'] != sig:
                raise ValueError('Resume identity mismatch')
            load_trainable(model, saved['parameters'])
            optimizer.load_state_dict(saved['optimizer'])
            start, history, updates = saved['step'], saved['history'], saved['updates']
            for name, p in model.named_parameters():
                if name in saved.get('gradients', {}):
                    p.grad = saved['gradients'][name].to(p.device)
            restore_rng(saved['rng'])
        training_mode(model, train_c)
        if method == 'tco' and c['model'] == 'vggt':
            model.aggregator.patch_embed.eval()
        # DINO's activation checkpointing must be non-reentrant for prompts.
        if c['model'] == 'vggt' and method == 'test3r':
            model.aggregator.patch_embed.use_reentrant = False
        def save(step):
            save_torch(checkpoint, dict(identity=sig, step=step, updates=updates,
                parameters=trainable_state(model), optimizer=optimizer.state_dict(),
                gradients={n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None},
                rng=rng_state(), history=history))
        cursor = start
        while cursor < total:
            if method == 'tco':
                optimizer.zero_grad(set_to_none=True)
                loss, detail = tco_loss(model, images, base, train_c, settings)
                chunk, do_update = 1, True
                scaled = loss
            else:
                local = cursor % len(order)
                chunk = min(c['test3r_pair_batch'], accum - local % accum, len(order)-local, total-cursor)
                loss = test3r_loss(model, images, order[local:local+chunk], train_c)
                scaled = loss * chunk / accum
                detail = {'consistency': loss.detach()}
                do_update = (local + chunk) % accum == 0
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{method}: nonfinite loss at {cursor}')
            scaled.backward()
            if do_update:
                norm = torch.nn.utils.clip_grad_norm_(params, 1. if method == 'tco' else math.inf,
                                                     error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
            cursor += chunk
            record = dict(step=cursor, updates=updates, loss=float(loss.detach()),
                          terms={k:float(v) for k,v in detail.items()})
            if do_update:
                record['gradient_norm'] = float(norm)
            history.append(record)
            print(json.dumps(record), flush=True)
            if (do_update and updates % c['checkpoint_every'] == 0) or cursor == total:
                save(cursor)
        model.eval()
        with torch.no_grad():
            result = predict(model, images, c)
        export(result, prediction_path(directory, 'adapted'))
        # Both originals evaluate the final adapted parameters, not GT-selected best.
        save_torch(out/'final.pt', trainable_state(model))
        write_json(out/'history.json', history)
        write_json(done, dict(identity=sig, method=method, config=c, settings=settings,
                             steps=cursor, updates=updates, trainable_parameters=sum(p.numel() for p in params),
                             modules=modules, seconds=time.monotonic()-tick,
                             peak_gib=torch.cuda.max_memory_allocated()/1024**3,
                             prior='frozen_prediction' if method == 'tco' else None,
                             upstream_commit=TCO_SHA if method == 'tco' else TEST3R_SHA,
                             point_source=c.get('test3r_vggt_points', 'depth') if c['model']=='vggt' and method=='test3r' else 'depth',
                             implementation='da3_vggt_port_v2'))
