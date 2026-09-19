import copy

import pytest
import torch
from PIL import Image

from free_geometry.adapters.base import BaseAdapter
from free_geometry.checkpoint import CheckpointPolicy, load_checkpoint
from free_geometry.config import ProtocolConfig
from free_geometry.sampling import SceneSource, build_manifest
from free_geometry.trainer import learning_rate, train_scene


class TinyLoRA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(2, 3) * 0.1)
        self.b = torch.nn.Parameter(torch.zeros(8, 2))

    def forward(self, x):
        return x @ self.a.T @ self.b.T


class TinyAdapter(BaseAdapter):
    model_key = "tiny"
    patch_size = 2
    default_resolution = 8

    def __init__(self):
        self.resolution = 8
        self.config = ProtocolConfig().model

    def load(self, device="cpu"):
        self.teacher = TinyLoRA().to(device).requires_grad_(False).eval()

    def reset_student(self, device="cpu"):
        self.net = TinyLoRA().to(device).eval()

    def _forward(self, model, images):
        s = images.shape[1]
        pooled = torch.nn.functional.adaptive_avg_pool2d(images[0], (4, 4))
        x = images[0].mean((-2, -1))
        change = model(x)
        scalar = x[:, 0] + 2 * x[:, 1] + 3 * x[:, 2]
        # Differentiable native outputs; every active branch reaches LoRA.
        angle = scalar * 0.1 + change[:, 0]
        c = angle.cos()
        sn = angle.sin()
        zero = angle * 0
        one = zero + 1
        r = torch.stack((c, -sn, zero, sn, c, zero, zero, zero, one), -1).reshape(
            s, 3, 3
        )
        centers = (
            torch.stack((scalar, scalar.square(), scalar.sin()), -1) + change[:, 1:4]
        )
        t = -(r @ centers[:, :, None])
        top = torch.cat((r, t), -1)
        bottom = torch.tensor([0, 0, 0, 1.0], device=images.device).expand(s, 1, 4)
        ext = torch.cat((top, bottom), 1)
        # Teacher long-context introduces a fixed contextual readout/depth effect.
        depth = (2 + scalar * 0.1 + change[:, 4] + scalar.mean() * 0.02)[
            :, None, None
        ].expand(s, 8, 8)
        features = (
            pooled.flatten(2).transpose(1, 2)[None]
            + change[:, 5:8][None, :, None, :]
            + 0.2
        )
        return {
            "readouts": {"tap": features},
            "depth": depth,
            "conf": torch.ones_like(depth),
            "centers": centers,
            "ext_w2c": ext,
        }

    def forward_teacher(self, images, slots):
        o = self._forward(self.teacher, images)
        return {
            k: (
                {n: v[:, slots] for n, v in value.items()}
                if k == "readouts"
                else value[slots]
            )
            for k, value in o.items()
        }

    def forward_student(self, images):
        return self._forward(self.net, images)

    @property
    def patch_hw(self):
        return (4, 4)


def fixture(tmp_path):
    files = []
    for i in range(12):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (8, 8), (20 + i * 17, 180 - i * 9, 30 + i * 12)).save(path)
        files.append(str(path))
    c = ProtocolConfig()
    c.train.device = "cpu"
    c.train.max_steps = 6
    c.probe.every = 2
    c.train.lr = 0.01
    m = build_manifest(
        SceneSource("tiny", files, [str(i) for i in range(12)]), c.sampling
    )
    return c, m


def test_resume_is_exact_and_probe_does_not_change_rng(tmp_path):
    c, m = fixture(tmp_path)
    a = TinyAdapter()
    train_scene(a, m, c, tmp_path / "full", log_fn=lambda _: None)
    full = a.trainable_state()
    b = TinyAdapter()
    train_scene(b, m, c, tmp_path / "resumed", log_fn=lambda _: None, interrupt_after=3)
    resume = tmp_path / "resumed/checkpoints/resume.pt"
    b = TinyAdapter()
    train_scene(b, m, c, tmp_path / "resumed", resume=resume, log_fn=lambda _: None)
    for name, value in full.items():
        torch.testing.assert_close(value, b.trainable_state()[name], rtol=0, atol=0)
    assert (tmp_path / "full/training.jsonl").read_text() == (
        tmp_path / "resumed/training.jsonl"
    ).read_text()
    changed = copy.deepcopy(c)
    changed.probe.every = 3
    d = TinyAdapter()
    train_scene(d, m, changed, tmp_path / "different_probe", log_fn=lambda _: None)
    for name, value in full.items():
        torch.testing.assert_close(value, d.trainable_state()[name], rtol=0, atol=0)
    changed.train.lr = 0.002
    with pytest.raises(ValueError, match="configuration"):
        load_checkpoint(resume, changed, m)


def test_policy_and_scheduler():
    c = ProtocolConfig()
    c.probe.decision = True
    policy = CheckpointPolicy()
    assert not policy.observe(0, 0.0, c.probe)
    assert not policy.observe(20, 2.0, c.probe)
    assert not policy.observe(40, 2.0, c.probe)
    assert not policy.observe(60, None, c.probe)
    assert policy.observe(80, 2.0, c.probe)
    assert policy.best_step == 20
    assert learning_rate(99, c.train) == pytest.approx(1e-8)
    c.train.max_steps = 1
    assert learning_rate(0, c.train) > 0


def test_teacher_immutable_and_student_reset(tmp_path):
    _c, _m = fixture(tmp_path)
    a = TinyAdapter()
    a.load("cpu")
    before = copy.deepcopy(a.teacher.state_dict())
    a.reset_student("cpu")
    for param in a.student_params():
        with torch.no_grad():
            param.add_(1)
    for k, v in before.items():
        torch.testing.assert_close(v, a.teacher.state_dict()[k])
    a.reset_student("cpu")
    assert torch.count_nonzero(a.net.b) == 0


def test_resume_stopped_run_does_not_repeat_updates_or_probes(tmp_path, monkeypatch):
    from free_geometry import trainer

    c, m = fixture(tmp_path)
    c.probe.decision = True
    c.probe.min_candidate_step = 2
    c.probe.min_stop_step = 4
    c.probe.patience = 1
    monkeypatch.setattr(
        trainer,
        "evaluate_probes",
        lambda *args: {"step": args[-1], "score": 1.0, "records": []},
    )
    root = tmp_path / "stopped"
    result = train_scene(TinyAdapter(), m, c, root, log_fn=lambda _: None)
    assert result["steps"] == 4 and result["stopped_early"]
    before = (root / "probe.jsonl").read_text()
    result = train_scene(
        TinyAdapter(), m, c, root, resume=result["final"], log_fn=lambda _: None
    )
    assert result["steps"] == 4 and result["stopped_early"]
    assert (root / "probe.jsonl").read_text() == before
    with pytest.raises(ValueError, match="original run directory"):
        train_scene(TinyAdapter(), m, c, tmp_path / "other", resume=result["final"])


def test_weight_identity_detects_content_change(tmp_path):
    from free_geometry.checkpoint import model_identity

    c = ProtocolConfig()
    c.model.weights = "organization/model"
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"original")
    first = model_identity(c, str(tmp_path))
    weight.write_bytes(b"replaced")
    assert model_identity(c, str(tmp_path)) != first


def test_invalid_b_preserves_a_supervision(tmp_path):
    from free_geometry.trainer import _cache

    c, m = fixture(tmp_path)
    adapter = TinyAdapter()
    adapter.load("cpu")
    images, valid, _ = adapter.prepare_images(m["image_files"])
    original = adapter.forward_teacher
    calls = 0

    def predict(images, slots):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("B ray fit rank deficient")
        return original(images, slots)

    adapter.forward_teacher = predict
    _, cache = _cache(adapter, images, valid, m, m["train"][0], c)
    assert cache["comparison_error"] == "B ray fit rank deficient"
    assert not any(cache["availability"].values())
    for value in cache["q"].values():
        for weight in value.values() if isinstance(value, dict) else [value]:
            assert (weight == 1).all()
