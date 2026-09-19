"""Opt-in real-weight acceptance. No synthetic result is reported as a GPU pass.

FG_GPU_SPECS=/absolute/specs.json pytest tests/unified/test_gpu_models.py
File shape: {"da3": {"config": "...yaml", "manifest": "...json"}, ...}.
All five keys are required when opting in, so missing coverage fails visibly.
"""

import json
import os
from pathlib import Path

import pytest
import torch

from free_geometry.adapters import get_adapter
from free_geometry.config import load_config
from free_geometry.losses import compute_losses, prepare_supervision
from free_geometry.masking import mask_images
from free_geometry.sampling import validate_manifest
from free_geometry.trainer import train_scene

SPEC_FILE = os.environ.get("FG_GPU_SPECS")


@pytest.mark.skipif(
    not SPEC_FILE,
    reason="requires CUDA, five model weights and real scene manifests; FG_GPU_SPECS unset",
)
@pytest.mark.parametrize("name", ["da3", "vggt", "omega", "pi3", "dvlt"])
def test_real_model_losses_and_checkpoint(name, tmp_path):
    assert torch.cuda.is_available(), "GPU acceptance must run on CUDA"
    specs = json.loads(Path(SPEC_FILE).read_text())
    assert set(specs) == {"da3", "vggt", "omega", "pi3", "dvlt"}
    config = load_config(specs[name]["config"])
    assert config.model.name == name
    manifest = validate_manifest(json.loads(Path(specs[name]["manifest"]).read_text()))
    adapter = get_adapter(config.model)
    adapter.amp_enabled = config.train.amp
    images, valid, _ = adapter.prepare_images(manifest["image_files"])
    adapter.resolve_weights()
    adapter.load(config.train.device)
    adapter.reset_student(config.train.device)
    task = manifest["train"][0]
    frames = task["teacher_a"]
    shared = task["shared"]
    teacher = adapter.predict(
        images[frames][None].to("cuda"),
        [manifest["frame_ids"][i] for i in frames],
        valid[frames].to("cuda"),
        teacher=True,
        slots=task["slots_a"],
    )
    x, _ = mask_images(
        images[shared][None].to("cuda"), teacher.patch_hw, 0.5, 42, adapter.mask_fill
    )
    student = adapter.predict(
        x, [manifest["frame_ids"][i] for i in shared], valid[shared].to("cuda")
    )
    bundle = compute_losses(
        student, teacher, prepare_supervision(teacher, loss=config.loss), config.loss
    )
    assert set(bundle.contributions) == {
        "feature",
        "rotation",
        "translation",
        "rkd",
        "couple",
    }
    params = adapter.student_params()
    for loss_name, value in bundle.contributions.items():
        grads = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
        present = [g for g in grads if g is not None]
        assert present and all(torch.isfinite(g).all() for g in present), loss_name
        assert sum(float(g.abs().sum()) for g in present) > 0, loss_name
    teacher_ptrs = {p.data_ptr() for p in adapter.teacher.parameters()}
    assert not teacher_ptrs.intersection(p.data_ptr() for p in params)
    del grads, present, bundle, student, teacher, x, params, adapter
    torch.cuda.empty_cache()
    config.train.max_steps = 2
    config.probe.decision = False
    adapter = get_adapter(config.model)
    result = train_scene(adapter, manifest, config, tmp_path / name)
    assert result["steps"] == 2 and Path(result["final"]).exists()
    from free_geometry.evaluation import export_scene

    del adapter
    torch.cuda.empty_cache()
    export_scene(
        get_adapter(config.model),
        manifest,
        config,
        tmp_path / (name + "_eval"),
        result["final"],
    )
