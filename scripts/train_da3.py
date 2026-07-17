#!/usr/bin/env python3
"""DA3 benchmark-dataset trainer for the active Free-Geometry LoRA/finetune workflow."""

import argparse
import os
import random
import time
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from depth_anything_3.test_time_adaption.benchmark_dataset import BenchmarkFreeGeometryDataset
from depth_anything_3.test_time_adaption.config import FreeGeometryConfig, get_default_config
from depth_anything_3.test_time_adaption.losses import (
    DA3CrossFrameCFAngleLoss,
    DA3CrossFrameCFDistanceLoss,
    PatchHuberCosineLoss,
)
from depth_anything_3.test_time_adaption.models import DA3StudentFinetune, StudentModel, TeacherModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    num_training_steps: int,
    warmup_steps: int,
    eta_min: float = 1e-6,
    steps_per_epoch: int | None = None,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    if scheduler_type in (None, "none"):
        return None

    if scheduler_type == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup_scheduler = None
        if warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, num_training_steps - warmup_steps),
            eta_min=eta_min,
        )
        if warmup_scheduler is None:
            return cosine_scheduler
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

    if scheduler_type == "step":
        from torch.optim.lr_scheduler import StepLR

        step_size = max(1, steps_per_epoch or num_training_steps)
        return StepLR(optimizer, step_size=step_size, gamma=0.7)

    if scheduler_type == "linear":
        from torch.optim.lr_scheduler import LinearLR

        return LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=max(1, num_training_steps),
        )

    if scheduler_type == "constant":
        from torch.optim.lr_scheduler import ConstantLR

        return ConstantLR(optimizer, factor=1.0, total_iters=max(1, num_training_steps))

    raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def save_checkpoint(
    student: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    loss: float,
    output_dir: str,
    filename: str = "checkpoint.pt",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, filename)

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "loss": loss,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict(),
    }

    weights_path = os.path.join(output_dir, filename.replace(".pt", "_lora.pt"))
    if isinstance(student, DA3StudentFinetune):
        student.save_finetune_weights(weights_path)
    else:
        student.save_lora_weights(weights_path)
    checkpoint["lora_path"] = weights_path

    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    latest_path = os.path.join(output_dir, "latest.pt")
    latest_lora_path = os.path.join(output_dir, "latest_lora.pt")
    torch.save(checkpoint, latest_path)
    if isinstance(student, DA3StudentFinetune):
        student.save_finetune_weights(latest_lora_path)
    else:
        student.save_lora_weights(latest_lora_path)

    return checkpoint_path


def _to_float(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)


def train_epoch(
    teacher: TeacherModel,
    student: nn.Module,
    train_loader: DataLoader,
    feat_criterion: nn.Module,
    cf_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    config: FreeGeometryConfig,
    args,
    writer: Optional[SummaryWriter] = None,
    cf_distance_criterion: Optional[nn.Module] = None,
) -> int:
    student.train()
    teacher.eval()

    device = config.training.device
    log_interval = config.training.log_interval
    save_interval = config.training.save_interval
    grad_accum_steps = config.training.gradient_accumulation_steps

    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
        teacher_images = batch["teacher_images"].to(device)
        student_images = batch["student_images"].to(device)

        with autocast(enabled=config.training.use_amp):
            with torch.no_grad():
                teacher_output = teacher.forward_features_only(teacher_images)
            student_output = student.forward_features_only(student_images)

            feat_loss, feat_details = feat_criterion(teacher_output, student_output)
            cf_loss, cf_details = cf_criterion(teacher_output, student_output)
            combined_loss = feat_loss + args.cf_weight * cf_loss

            cf_dist_loss = torch.tensor(0.0, device=device)
            cf_dist_details = {}
            if cf_distance_criterion is not None:
                cf_dist_loss, cf_dist_details = cf_distance_criterion(teacher_output, student_output)
                combined_loss = combined_loss + args.cf_distance_weight * cf_dist_loss

            loss = combined_loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                student.get_trainable_params(),
                config.training.max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

        total_loss += _to_float(combined_loss)
        num_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches
            lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * config.data.batch_size / max(elapsed, 1e-6)

            detail_parts = [
                f"feat: {_to_float(feat_loss):.4f}",
                f"cf: {_to_float(cf_loss):.4f}",
                f"cf_dist: {_to_float(cf_dist_loss):.4f}",
                f"combined: {_to_float(combined_loss):.4f}",
            ]
            for key in ("patch_huber", "patch_huber_cos", "patch_huber_total"):
                if key in feat_details:
                    detail_parts.append(f"{key}: {_to_float(feat_details[key]):.4f}")
            for key in ("cf_angle1_loss", "cf_angle2_loss", "cf_angle3_loss"):
                if key in cf_details:
                    detail_parts.append(f"{key}: {_to_float(cf_details[key]):.4f}")
            for key in ("cf_dist_d1_loss", "cf_dist_d2_loss", "cf_dist_d3_loss"):
                if key in cf_dist_details:
                    detail_parts.append(f"{key}: {_to_float(cf_dist_details[key]):.4f}")

            print(
                f"Epoch {epoch} | Step {global_step} | "
                f"Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                f"Speed: {samples_per_sec:.1f} samples/s"
            )
            print(f"  {' | '.join(detail_parts)}")

            if writer is not None:
                writer.add_scalar("train/loss", avg_loss, global_step)
                writer.add_scalar("train/lr", lr, global_step)
                writer.add_scalar("train/feat_loss", _to_float(feat_loss), global_step)
                writer.add_scalar("train/cf_loss", _to_float(cf_loss), global_step)
                writer.add_scalar("train/cf_dist_loss", _to_float(cf_dist_loss), global_step)
                writer.add_scalar("train/combined_loss", _to_float(combined_loss), global_step)
                for key, value in feat_details.items():
                    writer.add_scalar(f"train/{key}", _to_float(value), global_step)
                for key, value in cf_details.items():
                    writer.add_scalar(f"train/{key}", _to_float(value), global_step)
                for key, value in cf_dist_details.items():
                    writer.add_scalar(f"train/{key}", _to_float(value), global_step)

        if save_interval > 0 and global_step % save_interval == 0:
            save_checkpoint(
                student,
                optimizer,
                scheduler,
                scaler,
                epoch,
                global_step,
                total_loss / num_batches,
                config.training.output_dir,
                f"checkpoint_step{global_step}.pt",
            )

    return global_step


def main() -> None:
    parser = argparse.ArgumentParser(description="DA3 Free-Geometry training on benchmark datasets")

    parser.add_argument(
        "--dataset",
        type=str,
        default="scannetpp",
        choices=["scannetpp", "eth3d", "7scenes", "hiroom", "dtu"],
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        choices=["scannetpp", "eth3d", "7scenes", "hiroom", "dtu"],
        help="Train on multiple benchmark datasets with one shared adapter. Overrides --dataset.",
    )
    parser.add_argument("--samples_per_scene", type=int, default=4)
    parser.add_argument("--seeds_list", type=int, nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_views", type=int, default=8)

    parser.add_argument("--model_name", type=str, default="depth-anything/DA3-GIANT-1.1")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--finetune", action="store_true")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default=None,
        choices=["cosine", "linear", "constant", "step", "none"],
    )
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/tta")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)

    parser.add_argument("--patch_huber_weight", type=float, default=1.0)
    parser.add_argument("--patch_huber_cos_weight", type=float, default=2.0)
    parser.add_argument("--patch_huber_delta", type=float, default=1.0)

    parser.add_argument("--cf_weight", type=float, default=2.0)
    parser.add_argument("--cf_topk", type=int, default=4)
    parser.add_argument("--cf_num_ref_samples", type=int, default=256)
    parser.add_argument("--cf_num_shared_samples", type=int, default=256)
    parser.add_argument("--cf_angle1_weight", type=float, default=1.0)
    parser.add_argument("--cf_angle2_weight", type=float, default=1.0)
    parser.add_argument("--cf_angle3_weight", type=float, default=1.0)
    parser.add_argument("--cf_shared_chunk_size", type=int, default=64)
    parser.add_argument(
        "--cf_selection_mode",
        type=str,
        default="topk",
        choices=["topk", "random", "mixed"],
    )
    parser.add_argument("--use_cf_distance", action="store_true")
    parser.add_argument("--cf_distance_weight", type=float, default=1.0)
    parser.add_argument("--cf_distance_chunk_size", type=int, default=16)
    parser.add_argument("--cf_distance_type", type=str, default="l2", choices=["l2", "cosine"])
    parser.add_argument("--cf_normalize_distance", action="store_true", default=True)
    parser.add_argument("--cf_no_normalize_distance", action="store_true")
    parser.add_argument("--cf_d1_weight", type=float, default=1.0)
    parser.add_argument("--cf_d2_weight", type=float, default=1.0)
    parser.add_argument("--cf_d3_weight", type=float, default=1.0)
    parser.add_argument("--cf_distance_temperature", type=float, default=1.0)
    parser.add_argument("--cf_distance_mode", type=str, default="kl", choices=["kl", "huber"])
    parser.add_argument("--cf_distance_huber_beta", type=float, default=0.5)

    args = parser.parse_args()

    if args.cf_no_normalize_distance:
        args.cf_normalize_distance = False

    config = get_default_config()
    config.data.batch_size = args.batch_size
    config.data.num_workers = args.num_workers
    config.data.num_views = args.num_views
    config.data.student_indices = [0, 4, 8, 12] if args.num_views == 16 else [0, 2, 4, 6]
    config.model.model_name = args.model_name
    config.model.lora_rank = args.lora_rank
    config.model.lora_alpha = args.lora_alpha
    config.training.epochs = args.epochs
    config.training.lr = args.lr
    if args.warmup_steps is not None:
        config.training.warmup_steps = args.warmup_steps
    if args.lr_scheduler is not None:
        config.training.lr_scheduler = args.lr_scheduler
    if args.weight_decay is not None:
        config.training.weight_decay = args.weight_decay
    config.training.output_dir = args.output_dir
    config.training.seed = args.seed
    if args.log_interval is not None:
        config.training.log_interval = args.log_interval
    if args.save_interval is not None:
        config.training.save_interval = args.save_interval

    set_seed(config.training.seed)
    os.makedirs(config.training.output_dir, exist_ok=True)

    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    config.training.device = str(device)
    print(f"Using device: {device}")
    print("\nConfiguration:")
    train_dataset_names = args.datasets if args.datasets is not None else [args.dataset]
    print(f"  Dataset(s): {' '.join(train_dataset_names)}")
    print(f"  Teacher views: {config.data.num_views}")
    print(f"  Student views: 4 (indices: {config.data.student_indices})")
    print(f"  Output layers: {config.model.output_layers}")

    samples_per_scene = len(args.seeds_list) if args.seeds_list is not None else args.samples_per_scene
    print("\nCreating dataset...")
    train_datasets = [
        BenchmarkFreeGeometryDataset(
            dataset_name=dataset_name,
            num_views=config.data.num_views,
            image_size=config.data.image_size,
            student_indices=config.data.student_indices,
            augment=config.data.augment,
            samples_per_scene=samples_per_scene,
            seed=config.training.seed,
            seeds_list=args.seeds_list,
        )
        for dataset_name in train_dataset_names
    ]
    train_dataset = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Train samples: {len(train_dataset)}")

    print("\nCreating models...")
    teacher = TeacherModel(
        model_name=config.model.model_name,
        output_layers=config.model.output_layers,
        embed_dim=config.model.embed_dim,
    ).to(device)

    if args.finetune:
        student: nn.Module = DA3StudentFinetune(
            model_name=config.model.model_name,
            output_layers=config.model.output_layers,
            embed_dim=config.model.embed_dim,
            train_camera_token=config.model.train_camera_token,
        ).to(device)
    else:
        student = StudentModel(
            model_name=config.model.model_name,
            output_layers=config.model.output_layers,
            embed_dim=config.model.embed_dim,
            lora_rank=config.model.lora_rank,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            train_camera_token=config.model.train_camera_token,
        ).to(device)

    feat_criterion = PatchHuberCosineLoss(
        student_frame_indices=config.data.student_indices,
        huber_weight=args.patch_huber_weight,
        cos_weight=args.patch_huber_cos_weight,
        target_layers=config.model.output_layers,
        delta=args.patch_huber_delta,
    )
    cf_criterion = DA3CrossFrameCFAngleLoss(
        student_frame_indices=config.data.student_indices,
        num_teacher_views=config.data.num_views,
        target_layer=config.model.output_layers[-1],
        topk=args.cf_topk,
        num_ref_samples=args.cf_num_ref_samples,
        num_shared_samples=args.cf_num_shared_samples,
        angle1_weight=args.cf_angle1_weight,
        angle2_weight=args.cf_angle2_weight,
        angle3_weight=args.cf_angle3_weight,
        shared_chunk_size=args.cf_shared_chunk_size,
        selection_mode=args.cf_selection_mode,
    )

    cf_distance_criterion = None
    if args.use_cf_distance:
        cf_distance_criterion = DA3CrossFrameCFDistanceLoss(
            student_frame_indices=config.data.student_indices,
            num_teacher_views=config.data.num_views,
            target_layer=config.model.output_layers[-1],
            topk=args.cf_topk,
            num_ref_samples=args.cf_num_ref_samples,
            num_shared_samples=args.cf_num_shared_samples,
            d1_weight=args.cf_d1_weight,
            d2_weight=args.cf_d2_weight,
            d3_weight=args.cf_d3_weight,
            shared_chunk_size=args.cf_shared_chunk_size,
            distance_chunk_size=args.cf_distance_chunk_size,
            distance_type=args.cf_distance_type,
            normalize_distance=args.cf_normalize_distance,
            temperature=args.cf_distance_temperature,
            distance_mode=args.cf_distance_mode,
            huber_beta=args.cf_distance_huber_beta,
            selection_mode=args.cf_selection_mode,
        )

    optimizer = torch.optim.AdamW(
        student.get_trainable_params(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )

    num_training_steps = len(train_loader) * config.training.epochs
    if args.warmup_ratio is not None:
        config.training.warmup_steps = int(num_training_steps * args.warmup_ratio)
        print(
            f"Warmup ratio {args.warmup_ratio} -> {config.training.warmup_steps} steps "
            f"(of {num_training_steps} total)"
        )
    scheduler = get_lr_scheduler(
        optimizer,
        config.training.lr_scheduler,
        num_training_steps,
        config.training.warmup_steps,
        args.eta_min,
        len(train_loader),
    )
    scaler = GradScaler(enabled=config.training.use_amp)

    log_dir = os.path.join(
        config.training.output_dir,
        "logs",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    writer = SummaryWriter(log_dir)

    print("\nStarting training...")
    global_step = 0
    for epoch in range(config.training.epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{config.training.epochs}")
        print(f"{'=' * 60}")

        global_step = train_epoch(
            teacher=teacher,
            student=student,
            train_loader=train_loader,
            feat_criterion=feat_criterion,
            cf_criterion=cf_criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            config=config,
            args=args,
            writer=writer,
            cf_distance_criterion=cf_distance_criterion,
        )

        save_checkpoint(
            student,
            optimizer,
            scheduler,
            scaler,
            epoch,
            global_step,
            0.0,
            config.training.output_dir,
            f"epoch_{epoch}.pt",
        )

    print("\nTraining complete!")
    writer.close()


if __name__ == "__main__":
    main()
