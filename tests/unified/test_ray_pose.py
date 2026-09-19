import pytest
import torch

from free_geometry.ray_pose import rays_to_pose_differentiable


def synthetic(h=28, w=35):
    dtype = torch.float64
    y, x = torch.meshgrid(
        torch.arange(h, dtype=dtype), torch.arange(w, dtype=dtype), indexing="ij"
    )
    k = torch.tensor([[30.0, 0, 17], [0, 32, 13], [0, 0, 1]], dtype=dtype)
    angle = torch.tensor(0.4, dtype=dtype)
    r = torch.stack(
        [
            torch.stack([angle.cos(), -angle.sin(), angle * 0]),
            torch.stack([angle.sin(), angle.cos(), angle * 0]),
            torch.tensor([0, 0, 1], dtype=dtype),
        ]
    )
    c = torch.tensor([2.0, -1, 3], dtype=dtype)
    pixel = torch.stack((x, y, torch.ones_like(x)), -1)
    dirs = pixel @ torch.linalg.inv(k).T @ r.T
    rays = torch.cat((dirs, c.expand(h, w, 3)), -1)[None, None]
    return rays, r, c, k


def test_known_camera_and_gradient():
    rays, r, c, k = synthetic()
    rays.requires_grad_()
    ext, intr = rays_to_pose_differentiable(rays, 28, 35, 7)
    torch.testing.assert_close(ext[0, 0, :3, :3], r.T, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(ext[0, 0, :3, 3], -r.T @ c, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(intr[0, 0], k, atol=1e-7, rtol=1e-7)
    (ext.square().sum() + intr.square().sum() * 0.001).backward()
    assert torch.isfinite(rays.grad).all() and rays.grad.abs().sum() > 0


def test_directional_finite_difference():
    rays, _, _, _ = synthetic()
    direction = (
        torch.randn(
            rays.shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64
        )
        * 0.01
    )
    alpha = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)

    def f(a):
        ext, intr = rays_to_pose_differentiable(rays + a * direction, 28, 35, 7)
        return ext.square().sum() + 0.001 * intr.square().sum()

    analytic = torch.autograd.grad(f(alpha), alpha)[0]
    numeric = (f(alpha.detach() + 1e-5) - f(alpha.detach() - 1e-5)) / 2e-5
    torch.testing.assert_close(analytic, numeric, rtol=1e-4, atol=1e-5)


def test_rank_deficiency_rejected():
    rays, _, _, _ = synthetic()
    rays[..., :3] = torch.tensor([0.0, 0, 1.0])
    with pytest.raises(ValueError, match="rank deficient|singular"):
        rays_to_pose_differentiable(rays, 28, 35, 7)
