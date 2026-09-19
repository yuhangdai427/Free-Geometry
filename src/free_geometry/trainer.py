"""The only optimizer loop: adapters own models, this module owns the protocol."""

import json
import math
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

from .checkpoint import (
    CheckpointPolicy,
    atomic_save,
    config_fingerprint,
    load_checkpoint,
    model_identity,
    restore_rng,
    rng_state,
)
from .losses import compute_losses, prepare_supervision, supervision_summary
from .masking import mask_images
from .sampling import inspect_manifest, stable_seed, validate_manifest
from .types import tree_to


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_trace(path, record):
    with Path(path).open("a") as f:
        f.write(json.dumps(record, allow_nan=False) + "\n")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def learning_rate(step, config):
    """LR for the next successful update, step is zero-based completed updates."""
    warm = (
        min(config.max_steps - 1, max(1, int(config.max_steps * config.warmup_ratio)))
        if config.warmup_ratio
        else 0
    )
    if step < warm:
        return config.lr * (0.01 + 0.99 * step / max(warm - 1, 1))
    progress = (step - warm) / max(config.max_steps - warm - 1, 1)
    return 1e-8 + 0.5 * (config.lr - 1e-8) * (1 + math.cos(math.pi * progress))


def _task_images(images, valid, manifest, indices, device):
    return (
        images[indices][None].to(device),
        valid[indices].to(device),
        tuple(manifest["frame_ids"][i] for i in indices),
    )


def _cache(adapter, images, valid, manifest, task, config):
    def teacher(side):
        data, support, ids = _task_images(
            images, valid, manifest, task[f"teacher_{side}"], config.train.device
        )
        return adapter.predict(
            data, ids, support, teacher=True, slots=task[f"slots_{side}"]
        ).to("cpu", detach=True)

    a = teacher("a")
    b = None
    comparison_error = None
    if "teacher_b" in task and config.reliability.enabled:
        try:
            b = teacher("b") if task["ab_informative"] else a
        except (ValueError, FloatingPointError) as exc:
            comparison_error = str(exc)
    supervision = prepare_supervision(
        a, b, config.loss, config.reliability, task.get("ab_informative", False)
    )
    supervision["comparison_error"] = comparison_error
    return a, supervision


def evaluate_probes(adapter, images, valid, manifest, contexts, config, step):
    state = rng_state()
    model = adapter.trainable_model()
    was_training = model.training
    records = []
    try:
        model.eval()
        with torch.no_grad():
            for task, teacher, cache, error in contexts:
                for mask_id in range(2):
                    row = {"task": task["id"], "mask": mask_id, "valid": False}
                    try:
                        if error:
                            raise ValueError(error)
                        data, support, ids = _task_images(
                            images, valid, manifest, task["shared"], config.train.device
                        )
                        data, mask = mask_images(
                            data,
                            teacher.patch_hw,
                            config.train.input_mask_ratio,
                            stable_seed(
                                "probe_mask",
                                manifest["fingerprint"],
                                task["id"],
                                mask_id,
                                config.train.seed,
                            ),
                            adapter.mask_fill,
                        )
                        out = adapter.predict(data, ids, support)
                        result = compute_losses(
                            out,
                            teacher.to(config.train.device),
                            tree_to(cache, config.train.device),
                            config.loss,
                        )
                        row.update(
                            valid=True,
                            **result.record(),
                            input_mask_ratio=float(mask.float().mean()),
                        )
                    except (ValueError, FloatingPointError) as exc:
                        row["error"] = str(exc)
                    records.append(row)
    finally:
        model.train(was_training)
        restore_rng(state)
    valid_score = all(r["valid"] for r in records) and bool(records)
    score = sum(r["total"] for r in records) / len(records) if valid_score else None
    return {"step": step, "score": score, "records": records}


def train_scene(
    adapter,
    manifest,
    config,
    output_dir,
    resume=None,
    log_fn=print,
    interrupt_after=None,
):
    """interrupt_after is a test/debug hook; max_steps remains the planned LR horizon."""
    config.validate()
    validate_manifest(manifest)
    if manifest["status"] != "ready":
        raise ValueError(manifest["reason"])
    if manifest["sampling"] != config.to_dict()["sampling"]:
        raise ValueError(
            "configuration sampling differs from manifest; use matching config or regenerate"
        )
    root = Path(output_dir)
    if resume and Path(resume).resolve().parent.parent != root.resolve():
        raise ValueError("resume must use its original run directory")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists() and resume is None:
        raise FileExistsError(
            "run already exists; use --resume or a new output directory"
        )
    log_fn(
        json.dumps(
            {
                "phase": "prepare",
                "scene": manifest["scene"],
                "config": config.to_dict(),
                "strategy": manifest["strategy"],
                "N_valid": manifest["N_valid"],
                "sift_called": manifest["sift_called"],
                "tau": manifest["tau"],
            },
            allow_nan=False,
        )
    )
    seed_all(stable_seed("initialization", manifest["scene"], config.train.seed))
    device = torch.device(config.train.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable; use a CUDA host for real models"
        )
    adapter.amp_enabled = config.train.amp
    images, valid, preprocess = adapter.prepare_images(manifest["image_files"])
    identity = model_identity(config, adapter.resolve_weights())
    adapter.load(config.train.device)
    caches = [
        _cache(adapter, images, valid, manifest, task, config)
        for task in manifest["train"]
    ]
    probes = []
    for task in manifest["probe"]:
        try:
            a, cache = _cache(adapter, images, valid, manifest, task, config)
            probes.append((task, a, cache, None))
        except (ValueError, FloatingPointError) as exc:
            probes.append((task, None, None, str(exc)))
    # Teacher has no further forward role during optimization; free accelerator memory.
    adapter.teacher.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    seed_all(stable_seed("student", manifest["scene"], config.train.seed))
    adapter.reset_student(config.train.device)
    params = adapter.student_params()
    if not params:
        raise ValueError("no trainable parameters")
    opt = torch.optim.AdamW(
        params, lr=config.train.lr, weight_decay=config.train.weight_decay
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=config.train.amp and device.type == "cuda"
    )
    policy = CheckpointPolicy()
    step = cursor = epoch = 0
    failures = 0
    if resume:
        payload = load_checkpoint(resume, config, manifest, identity)
        adapter.load_trainable_state(payload["parameters"])
        opt.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        restore_rng(payload["rng"])
        step, epoch, cursor = payload["step"], payload["epoch"], payload["cursor"]
        policy = CheckpointPolicy(**payload["policy"])
        # Resuming an older checkpoint replaces the superseded trace tail.
        for name in ("training.jsonl", "probe.jsonl"):
            path = root / name
            if path.exists():
                rows = [json.loads(x) for x in path.read_text().splitlines()]
                path.write_text(
                    "".join(json.dumps(r) + "\n" for r in rows if r["step"] <= step)
                )
    write_json(root / "config.json", config.to_dict())
    write_json(root / "manifest.json", manifest)
    write_json(root / "preprocessing.json", preprocess)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    startup = {
        "config": config.to_dict(),
        "manifest": inspect_manifest(manifest),
        "model": adapter.describe(),
        "model_identity": identity,
        "loss_definitions": {
            "feature": "SmoothL1(beta=1) + 2*(1-cosine); confidence * patch q",
            "rotation": "relative_rotation_huber; squared Frobenius residual",
            "translation": "relative_translation_direction; 1-cosine",
            "rkd": "normalized_center_distance + triangle_cosine; Huber",
            "couple": "squared log(spread/mean_depth) difference on fixed A support",
        },
        "seeds": {
            "frames": config.sampling.seed,
            "initialization": stable_seed(
                "initialization", manifest["scene"], config.train.seed
            ),
            "student": stable_seed("student", manifest["scene"], config.train.seed),
            "order_and_mask_root": config.train.seed,
        },
        "revision": revision,
        "supervision": [supervision_summary(c) for _, c in caches],
    }
    write_json(root / "startup.json", startup)
    log_fn(
        json.dumps(
            {
                "phase": "ready",
                "scene": manifest["scene"],
                "model": startup["model"],
                "loss_definitions": startup["loss_definitions"],
                "seeds": startup["seeds"],
                "teacher_cache_count": len(caches),
                "supervision": startup["supervision"],
            },
            allow_nan=False,
        )
    )

    def save(label):
        atomic_save(
            {
                "version": "unified-1",
                "step": step,
                "epoch": epoch,
                "cursor": cursor,
                "parameters": adapter.trainable_state(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": rng_state(),
                "policy": policy.state_dict(),
                "config_hash": config_fingerprint(config),
                "config": config.to_dict(),
                "manifest": manifest["fingerprint"],
                "model_identity": identity,
                "schedule": {"completed_updates": step},
            },
            root / "checkpoints" / f"{label}.pt",
        )

    def probe():
        row = evaluate_probes(adapter, images, valid, manifest, probes, config, step)
        append_trace(root / "probe.jsonl", row)
        stop = policy.observe(step, row["score"], config.probe)
        save(f"step{step:06d}")
        return stop

    if not resume:
        probe()
    n = len(manifest["train"])
    last_probe = step if not resume else -1
    if resume and (root / "probe.jsonl").exists():
        rows = [json.loads(x) for x in (root / "probe.jsonl").read_text().splitlines()]
        last_probe = max((r["step"] for r in rows), default=-1)
    stopped = (
        config.probe.decision
        and step >= config.probe.min_stop_step
        and policy.stale >= config.probe.patience
    )
    try:
        while step < config.train.max_steps and not stopped:
            order = list(range(n))
            random.Random(
                stable_seed("order", manifest["scene"], epoch, config.train.seed)
            ).shuffle(order)
            index = order[cursor]
            task = manifest["train"][index]
            a, cache = caches[index]
            data, support, ids = _task_images(
                images, valid, manifest, task["shared"], config.train.device
            )
            data, mask = mask_images(
                data,
                a.patch_hw,
                config.train.input_mask_ratio,
                stable_seed(
                    "train_mask",
                    manifest["scene"],
                    epoch,
                    task["id"],
                    config.train.seed,
                ),
                adapter.mask_fill,
            )
            opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:
                group["lr"] = learning_rate(step, config.train)
            student = adapter.predict(data, ids, support)
            loss = compute_losses(
                student, a.to(device), tree_to(cache, device), config.loss
            )
            scaler.scale(loss.total).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, config.train.clip_grad)
            finite = bool(torch.isfinite(grad_norm))
            if not scaler.is_enabled() and not finite:
                raise FloatingPointError("nonfinite parameter gradient")
            old_scale = scaler.get_scale()
            if finite:
                scaler.step(opt)
                scaler.update()
            else:
                # Even finite per-parameter gradients can have an overflowing norm.
                # Never let AdamW weight decay mutate state on a rejected update.
                scaler.update(new_scale=old_scale / 2)
            updated = finite and scaler.get_scale() >= old_scale
            if not updated:
                failures += 1
                append_trace(
                    root / "events.jsonl",
                    {
                        "step": step,
                        "task": task["id"],
                        "event": "amp_overflow",
                        "consecutive_failures": failures,
                        "scale": scaler.get_scale(),
                    },
                )
                if failures >= 3:
                    raise FloatingPointError("three consecutive AMP overflow updates")
                continue
            failures = 0
            step += 1
            cursor += 1
            row = {
                "step": step,
                "epoch": epoch,
                "task": task["id"],
                "input_mask_ratio": float(mask.float().mean()),
                "grad_norm": float(grad_norm),
                "lr": opt.param_groups[0]["lr"],
                **loss.record(),
            }
            row["supervised_patch_fraction"] = sum(
                int(v.sum()) for v in cache["masks"].get("feature", {}).values()
            ) / max(
                1, sum(v.numel() for v in cache["masks"].get("feature", {}).values())
            )
            append_trace(root / "training.jsonl", row)
            if step == 1 or step % 10 == 0:
                log_fn(json.dumps(row, allow_nan=False))
            if cursor == n:
                epoch += 1
                cursor = 0
            # Do not retain a complete student graph while running probes.
            del student, loss, data
            if step % config.probe.every == 0 or step == config.train.max_steps:
                stopped = probe()
                last_probe = step
            if interrupt_after is not None and step >= interrupt_after:
                save("resume")
                return {"step": step, "interrupted": True}
    except Exception as exc:
        write_json(
            root / "failure.json",
            {"step": step, "epoch": epoch, "cursor": cursor, "error": str(exc)},
        )
        raise
    if last_probe != step:
        probe()
    save("final")
    if config.probe.decision and not policy.best_step:
        raise ValueError("no valid selected checkpoint; baseline fallback is disabled")
    summary = {
        "steps": step,
        "stopped_early": stopped,
        "final": str(root / "checkpoints/final.pt"),
        "selected": str(root / "checkpoints" / f"step{policy.best_step:06d}.pt")
        if config.probe.decision
        else None,
        "primary": "selected" if config.probe.decision else "final",
    }
    write_json(root / "summary.json", summary)
    return summary
