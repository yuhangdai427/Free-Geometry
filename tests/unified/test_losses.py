import math
from dataclasses import replace

import pytest
import torch

from free_geometry.config import LossConfig, ReliabilityConfig
from free_geometry.geometry import centers_from_w2c, rkd_statistics
from free_geometry.losses import compute_losses, prepare_supervision
from free_geometry.types import ModelOutput


def output(n=4):
    gen = torch.Generator().manual_seed(44)
    ext = torch.eye(4).repeat(n, 1, 1)
    ext[:, :3, 3] = torch.randn(n, 3, generator=gen)
    return ModelOutput(
        tuple(map(str, range(n))),
        {"tap": torch.randn(1, n, 4, 6, generator=gen)},
        torch.rand(n, 4, 4, generator=gen) + 1,
        torch.ones(n, 4, 4),
        ext,
        centers_from_w2c(ext),
        torch.ones(n, 4, 4, dtype=torch.bool),
        (2, 2),
    )


def trainable(a):
    ext = a.ext_w2c.clone().requires_grad_()
    return replace(
        a,
        readouts={k: v.clone().requires_grad_() for k, v in a.readouts.items()},
        depth=a.depth.clone().requires_grad_(),
        ext_w2c=ext,
        centers=centers_from_w2c(ext),
    )


def test_identity_and_all_terms_backward():
    a = output()
    s = trainable(a)
    cache = prepare_supervision(a, a)
    assert all(
        torch.equal(q, torch.ones_like(q))
        for k, q in cache["q"].items()
        if k != "feature"
    )
    assert torch.equal(cache["q"]["feature"]["tap"], torch.ones(1, 4, 4))
    result = compute_losses(s, a, cache)
    for key in ("rotation", "translation", "rkd", "couple"):
        assert abs(float(result.raw[key].detach())) < 1e-6
    result.total.backward()
    assert torch.isfinite(s.ext_w2c.grad).all()
    assert torch.isfinite(s.depth.grad).all()
    assert not a.ext_w2c.requires_grad


@pytest.mark.parametrize(
    "name", ["feature", "rotation", "translation", "rkd", "couple"]
)
def test_each_term_reaches_student(name):
    a = output()
    s = trainable(a)
    with torch.no_grad():
        s.readouts["tap"].add_(0.1)
        s.ext_w2c[0, 0, 0] += 0.1
        s.ext_w2c[0, 1, 3] += 0.3
        s.depth.mul_(1.2)
    s = replace(s, centers=centers_from_w2c(s.ext_w2c))
    cfg = LossConfig(
        **{
            k: float(k == name)
            for k in ("feature", "rotation", "translation", "rkd", "couple")
        }
    )
    result = compute_losses(s, a, prepare_supervision(a, loss=cfg), cfg)
    result.total.backward()
    relevant = s.readouts["tap"].grad if name == "feature" else s.ext_w2c.grad
    assert (
        relevant is not None
        and torch.isfinite(relevant).all()
        and relevant.abs().sum() > 0
    )


def test_sum_gradient_equals_parts():
    a = output()
    s = trainable(a)
    with torch.no_grad():
        s.ext_w2c[0, 0, 3] += 0.4
    s = replace(s, centers=centers_from_w2c(s.ext_w2c))
    result = compute_losses(s, a, prepare_supervision(a))
    params = (s.ext_w2c, s.depth, s.readouts["tap"])
    terms = [
        torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
        for v in result.contributions.values()
    ]
    total = torch.autograd.grad(result.total, params)
    for i, g in enumerate(total):
        expected = sum(t[i] for t in terms if t[i] is not None)
        torch.testing.assert_close(g, expected)


def test_couple_fixed_support_invalid_student():
    a = output()
    cfg = LossConfig(feature=0, rotation=0, translation=0, rkd=0)
    cache = prepare_supervision(a, loss=cfg)
    s = replace(a, depth=a.depth.clone())
    s.depth[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="fixed teacher support"):
        compute_losses(s, a, cache, cfg)
    a.valid[0, 0, 0] = False
    cache = prepare_supervision(a, loss=cfg)
    assert compute_losses(s, a, cache, cfg).total == 0


def test_general_rkd_and_similarity_invariance():
    a = output(6)
    d, angle, _, _, idx = rkd_statistics(a.centers)
    assert len(idx) == 6 * math.comb(5, 2)
    rotation = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    other = a.centers @ rotation * 3 + torch.tensor([4.0, 5, 6])
    d2, angle2, *_ = rkd_statistics(other)
    torch.testing.assert_close(d, d2)
    torch.testing.assert_close(angle, angle2)


def test_ab_fixed_threshold_monotone_and_independent():
    a = output()
    b = output()
    b.depth = b.depth * 1.25
    cache = prepare_supervision(a, b)
    assert abs(float(cache["q"]["couple"]) - 0.5) < 1e-5
    assert torch.equal(cache["q"]["rotation"], torch.ones(6))
    b.depth = b.depth * 1.25
    assert prepare_supervision(a, b)["q"]["couple"] < cache["q"]["couple"]
    b.depth[0, 0, 0] = float("nan")
    cache = prepare_supervision(a, b)
    assert cache["q"]["couple"] == 1 and cache["availability"]["couple"] == 0
    cache = prepare_supervision(a, b, reliability=ReliabilityConfig(enabled=False))
    assert cache["q"]["couple"] == 1


def test_teacher_nan_only_excludes_affected_edges():
    a = output()
    a.ext_w2c[0, 0, 0] = float("nan")
    cfg = LossConfig(feature=0, translation=0, rkd=0, couple=0)
    cache = prepare_supervision(a, loss=cfg)
    assert cache["masks"]["rotation"].sum() == 3
    s = output()
    s.ext_w2c.requires_grad_()
    result = compute_losses(s, a, cache, cfg)
    result.total.backward()
    assert torch.isfinite(s.ext_w2c.grad).all()


def test_zero_teacher_baselines_fail_explicitly():
    a = output()
    a.ext_w2c[:, :3, 3] = 0
    with pytest.raises(ValueError, match="translation"):
        prepare_supervision(a)
