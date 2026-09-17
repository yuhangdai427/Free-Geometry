"""Model-agnostic per-scene TTA trainer for the migration (omega/pi3/dvlt).

Reads pre-built protocol JSONs (see scripts/fgmig_build_protocols.py) so the
trainer runs in ANY conda env without importing the DA3 code chain.
100 steps, AdamW, warmup 15% + cosine, clip 1.0, wd 1e-5; lr 3e-5 (LoRA models)
or adapter-specified small lr for full-FT. Input masking is normalized-zero,
loss is ALL-position (2026-09-16 ablation winner).
"""
import json
import os
from typing import Dict, List

import numpy as np
import torch

from .losses import compute_arm_loss, valid_mask_from_conf
from .masking import mask_image_blocks

WD, CLIP, WARMUP_RATIO = 1e-5, 1.0, 0.15
STUDENT_SLOTS = [0, 2, 4, 6]


def load_images(model_key: str, paths: List[str]) -> torch.Tensor:
    """[N,3,H,W] float in [0,1]; multiples of the model patch size."""
    import cv2
    if model_key == "omega":
        res, patch = 416, 16
    else:
        res, patch = 504, 14
    out = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = res / max(h, w)
        nh = max(patch, int(round(h * scale / patch)) * patch)
        nw = max(patch, int(round(w * scale / patch)) * patch)
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        img = cv2.resize(img, (nw, nh), interpolation=interp)
        out.append(torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1))
    # pad to the common grid (mixed aspect ratios within a scene)
    H = max(t.shape[1] for t in out)
    W = max(t.shape[2] for t in out)
    H, W = int(np.ceil(H / patch)) * patch, int(np.ceil(W / patch)) * patch
    batch = torch.ones(len(out), 3, H, W)  # pad value 1.0 (matches omega loader)
    for i, t in enumerate(out):
        batch[i, :, :t.shape[1], :t.shape[2]] = t
    return batch


def _conf_patch(conf: torch.Tensor, patch_hw) -> torch.Tensor:
    """[S,H,W] teacher conf -> [1,S,P] mean-normalized patch weights."""
    import torch.nn.functional as F
    S, H, W = conf.shape
    ph, pw = patch_hw
    c = F.avg_pool2d(conf.reshape(S, 1, H, W), kernel_size=(H // ph, W // pw))
    c = c.reshape(1, S, ph * pw)
    return (c / c.mean().clamp_min(1e-8)).detach()


def _stable_seed(*parts) -> int:
    import hashlib
    key = "::".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def train_scene(adapter, proto: Dict, scene: str, device: str = "cuda",
                arm: str = "rkdc_allpos", steps: int = 100, epochs: int = 10,
                lr: float = None, seed: int = 0, mask_ratio: float = 0.5,
                log_fn=print, trace: List = None):
    imgs = load_images(adapter.model_key, proto["image_files"])
    train_pairs = proto["train_pairs"]
    n_train = len(train_pairs)

    adapter.load(device)
    # ---- teacher caches (frozen, once per scene) ----
    caches, pair_imgs = [], []
    for pi, pair in enumerate(train_pairs):
        t = torch.stack([imgs[i] for i in pair["teacher_frames"]]) \
            .unsqueeze(0).to(device)
        cache = adapter.forward_teacher(t, STUDENT_SLOTS[:len(pair["student_frames"])])
        caches.append(cache)
        pair_imgs.append(t[0, STUDENT_SLOTS[:len(pair["student_frames"])]].cpu())
        del t
    patch_hw = adapter.patch_hw
    log_fn(f"[{scene}] cached {n_train} teacher pairs; patch_hw={patch_hw}")

    # ---- student ----
    adapter.reset_student(device)
    params = adapter.student_params()
    lr = lr if lr is not None else (1e-5 if not adapter.uses_lora else 3e-5)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=WD)
    n_steps = min(steps, n_train * epochs)
    warm = max(1, int(n_steps * WARMUP_RATIO))
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    sched = SequentialLR(opt, [
        LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=warm),
        CosineAnnealingLR(opt, T_max=n_steps - warm, eta_min=1e-8)], [warm])

    step, losses = 0, []
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        if step >= n_steps:
            break
        order = [int(i) for i in torch.randperm(
            n_train, generator=torch.Generator().manual_seed(
                _stable_seed("order", scene, epoch, seed))).tolist()]
        for pi in order:
            cache = caches[pi]
            prep = {**cache,
                    "conf_patch": _conf_patch(cache["conf"], patch_hw),
                    "valid": valid_mask_from_conf(cache["conf"])}
            images4 = pair_imgs[pi].unsqueeze(0).to(device)
            images4_in, _pmask = mask_image_blocks(
                images4, mask_ratio, patch_hw,
                torch.Generator(device=images4.device).manual_seed(
                    _stable_seed("mask", scene, epoch, pi, seed)))
            student = adapter.forward_student(images4_in)
            loss = compute_arm_loss(arm, student, prep)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{scene} step {step + 1}: loss={float(loss)}")
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, CLIP)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            losses.append(float(loss))
            if trace is not None:
                trace.append({"scene": scene, "step": step, "epoch": epoch,
                              "pair": pi, "loss": float(loss),
                              "lr": sched.get_last_lr()[0], "grad_norm": float(gn),
                              "peak_mib": torch.cuda.max_memory_allocated() / 2 ** 20})
            if step >= n_steps:
                break
    log_fn(f"[{scene}] done steps={step} loss {np.mean(losses[:10]):.4f}->"
           f"{np.mean(losses[-10:]):.4f} lr={lr} params={adapter.student_label()} "
           f"peak={torch.cuda.max_memory_allocated() / 2 ** 20:.0f}MiB")
    return {"steps": step, "loss_first10": float(np.mean(losses[:10])),
            "loss_last10": float(np.mean(losses[-10:]))}


def export_scene_eval(adapter, proto: Dict, scene: str, out_dir: str,
                      device: str = "cuda"):
    """Export mini_npz results for the metric chain (da3 env, fg_eval_from_npz)."""
    if not hasattr(adapter, "teacher"):
        adapter.load(device)
    imgs = load_images(adapter.model_key, proto["image_files"])
    frames = proto["eval_frames"]
    ev = torch.stack([imgs[i] for i in frames]).unsqueeze(0).to(device)
    out = adapter.export_eval(ev)
    assert out["intr"] is not None, "adapter must provide model-resolution K"
    exp = os.path.join(out_dir, "exports", "mini_npz")
    os.makedirs(exp, exist_ok=True)
    np.savez_compressed(os.path.join(exp, "results.npz"),
                        depth=np.asarray(out["depth"], np.float32),
                        extrinsics=np.asarray(out["extr_w2c"], np.float32),
                        intrinsics=np.asarray(out["intr"], np.float32),
                        conf=np.asarray(out["conf"], np.float32))
    del ev
    torch.cuda.empty_cache()
