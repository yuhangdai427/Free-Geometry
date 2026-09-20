import math
import json
from pathlib import Path
from functools import wraps
from contextlib import nullcontext, ExitStack
from unittest.mock import patch
import cv2
import numpy as np
import torch
from torch import nn
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
    @property
    def weight(self): return self.base.weight
    @property
    def bias(self): return self.base.bias
    def forward(self, x):
        return self.base(x) + nn.functional.linear(nn.functional.linear(self.dropout(x), self.a), self.b) * self.scale


def inject_lora(model, c):
    model.requires_grad_(False)
    names = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name.endswith('.qkv') and (
            name.startswith('aggregator.') or name.startswith('backbone.pretrained.') or
            c['camera_lora'] and name.startswith('camera_head.')):
            parent, attr = name.rsplit('.', 1)
            setattr(model.get_submodule(parent), attr, LoRALinear(module, c['rank'], c['alpha'], c['dropout']))
            names.append(name)
    if not names: raise ValueError('No QKV LoRA modules selected')
    return names


def trainable_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}


def load_trainable(model, state):
    params = dict(model.named_parameters())
    expected = {k for k, v in params.items() if v.requires_grad}
    if set(state) != expected: raise ValueError('LoRA parameter set differs')
    with torch.no_grad():
        for k, v in state.items(): params[k].copy_(v)


def load_model(c, device='cuda'):
    from .common import isolated_rng
    # Every parameter is replaced by a strictly checked checkpoint. Avoid filling
    # billions of parameters with random values that are immediately discarded.
    # Keep allocations and constant non-parameter buffers intact.
    with isolated_rng(), ExitStack() as stack:
        for name in ('uniform_', 'normal_', 'trunc_normal_', 'kaiming_uniform_',
                     'kaiming_normal_', 'xavier_uniform_', 'xavier_normal_',
                     'zeros_', 'ones_', 'constant_'):
            stack.enter_context(patch.object(nn.init, name, lambda tensor, *a, **kw: tensor))
        return _load_model(c, device)


def _load_model(c, device='cuda'):
    if c.get('model', 'vggt') == 'da3':
        return load_da3(c, device)
    use_point = c.get('method') == 'test3r' and c.get('test3r_vggt_points', 'depth') == 'native'
    model = VGGT(enable_point=use_point, enable_track=False)
    state = torch.load(c['weights'], map_location='cpu', weights_only=True, mmap=True)
    # Only native Test3R requires the point head; tracking is never used here.
    excluded = ('track_head.',) if use_point else ('point_head.', 'track_head.')
    state = {k:v for k,v in state.items() if not k.startswith(excluded)}
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.aggregator.use_reentrant = False
    return model.to(device).eval()


def load_images(files, size, model='vggt'):
    if model == 'da3':
        from depth_anything_3.utils.io.input_processor import InputProcessor
        processor = InputProcessor()
        # Geometry, photometry and LightGlue consume RGB in [0, 1]. Normalize
        # only at the DA3 forward boundary; resizing is the official processor.
        processor.NORMALIZE = nn.Identity()
        images, _, _ = processor(list(files), process_res=size,
                                  process_res_method='upper_bound_resize', sequential=True)
        return images.contiguous()
    result = []
    for f in files:
        im = cv2.imread(str(f))
        if im is None: raise FileNotFoundError(f)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h,w = im.shape[:2]; scale = size / max(h,w)
        nh = max(14, round(round(h*scale)/14)*14)
        nw = max(14, round(round(w*scale)/14)*14)
        im = cv2.resize(im,(nw,nh),interpolation=cv2.INTER_AREA if scale <= 1 else cv2.INTER_CUBIC)
        result.append(im.astype(np.float32)/255)
    if len({x.shape for x in result}) != 1: raise ValueError('Mixed image dimensions in scene')
    return torch.from_numpy(np.stack(result)).permute(0,3,1,2).contiguous()


def predict(model, images, c):
    ctx = torch.autocast('cuda', dtype=torch.bfloat16) if images.is_cuda and c['precision']=='bf16' else nullcontext()
    if c.get('model', 'vggt') == 'da3':
        mean = images.new_tensor([.485, .456, .406])[None, :, None, None]
        std = images.new_tensor([.229, .224, .225])[None, :, None, None]
        with ctx:
            p = model(((images - mean) / std)[None], ref_view_strategy=c['ref_view_strategy'])
        return dict(depth=p['depth'][0].float(), conf=p['depth_conf'][0].float(),
                    extrinsics=p['extrinsics'][0, :, :3].float(), intrinsics=p['intrinsics'][0].float())
    # The DA3 benchmark exports depth + pose. A Test3R point head is used only
    # by its training loss; avoid computing it during baseline/final export.
    point_head = model.point_head
    model.point_head = None
    try:
        with ctx:
            p = model(images)
    finally:
        model.point_head = point_head
    ext, intr = pose_encoding_to_extri_intri(p['pose_enc'].float(), image_size_hw=images.shape[-2:])
    return dict(depth=p['depth'][0,...,0].float(), conf=p['depth_conf'][0].float(),
                extrinsics=ext[0].float(), intrinsics=intr[0].float())


def training_mode(model, c):
    model.eval()
    if c.get('model', 'vggt') == 'vggt' and c['checkpointing']: model.aggregator.train()
    if c.get('model', 'vggt') == 'da3': model._sg_checkpointing = c['checkpointing']
    for m in model.modules():
        if isinstance(m, LoRALinear): m.dropout.train()


def remove_lora(model):
    """Restore frozen modules before starting another independent seed."""
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            parent, attr = name.rsplit('.', 1)
            setattr(model.get_submodule(parent), attr, module.base)
    model.requires_grad_(False)
    model.zero_grad(set_to_none=True)


def load_da3(c, device):
    from omegaconf import OmegaConf
    from safetensors import safe_open
    from depth_anything_3.cfg import create_object
    from torch.utils.checkpoint import checkpoint
    cfg = json.loads(Path(c['weights']).with_name('config.json').read_text())['config']
    # The Gaussian branch is not used by depth/pose evaluation.
    cfg = {k: v for k, v in cfg.items() if k not in ('gs_head', 'gs_adapter')}
    model = create_object(OmegaConf.create(cfg))
    with safe_open(c['weights'], framework='pt', device='cpu') as f:
        state = {k.removeprefix('model.'): f.get_tensor(k) for k in f.keys()
                 if not k.startswith(('model.gs_head.', 'model.gs_adapter.'))}
        # safetensors omits tied tensors and records their canonical names.
        for alias, source in (f.metadata() or {}).items():
            if source.removeprefix('model.') in state:
                state[alias.removeprefix('model.')] = state[source.removeprefix('model.')]
    model.load_state_dict(state, strict=True, assign=True)
    model.requires_grad_(False)
    model._sg_checkpointing = False
    # Wrap forward methods, not modules: preserve exact official state_dict names.
    for block in model.backbone.pretrained.blocks:
        original = block.forward
        def wrap(fn):
            @wraps(fn)
            def forward(*args, **kwargs):
                if model._sg_checkpointing and torch.is_grad_enabled():
                    return checkpoint(fn, *args, use_reentrant=False, **kwargs)
                return fn(*args, **kwargs)
            return forward
        block.forward = wrap(original)
    return model.to(device).eval()
