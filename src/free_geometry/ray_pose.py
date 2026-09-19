"""Differentiable DVLT ray fitting; deliberately distinct from official RANSAC.

Normalized homogeneous DLT, three angular IRLS rounds, and QR factorization.
The minimum-eigenvector backward uses only the null-space spectral gap, avoiding
SVD backward singularities when *other* singular values happen to repeat.
"""

import torch
import torch.nn.functional as F


class _NullVector(torch.autograd.Function):
    @staticmethod
    def forward(ctx, normal):
        values, vectors = torch.linalg.eigh(normal)
        gap = values[..., 1] - values[..., 0]
        if not bool(torch.isfinite(values).all()) or bool(
            (gap <= 1e-10 * values[..., -1].clamp_min(1)).any()
        ):
            raise ValueError("ray pose: rank deficient or ambiguous homography")
        ctx.save_for_backward(values, vectors)
        return vectors[..., 0]

    @staticmethod
    def backward(ctx, grad):
        values, vectors = ctx.saved_tensors
        v, rest = vectors[..., 0], vectors[..., 1:]
        coefficients = (rest.transpose(-1, -2) @ grad[..., None]).squeeze(-1) / (
            values[..., :1] - values[..., 1:]
        )
        z = (rest @ coefficients[..., None]).squeeze(-1)
        return 0.5 * (
            z[..., :, None] * v[..., None, :] + v[..., :, None] * z[..., None, :]
        )


def rays_to_pose_differentiable(rays, height, width, patch_size, angular_threshold=0.2):
    """rays [B,S,H,W,6] -> differentiable (w2c [B,S,4,4], K [B,S,3,3])."""
    if rays.ndim != 5 or rays.shape[-1] != 6 or not bool(torch.isfinite(rays).all()):
        raise ValueError("ray pose: malformed/nonfinite rays")
    b, s, h, w, _ = rays.shape
    ph, pw = height // patch_size, width // patch_size
    if min(ph, pw) < 2:
        raise ValueError("ray pose requires a two-dimensional patch grid")
    sampled = (
        F.interpolate(
            rays.reshape(b * s, h, w, 6).permute(0, 3, 1, 2),
            size=(ph, pw),
            mode="bilinear",
            align_corners=True,
        )
        .permute(0, 2, 3, 1)
        .reshape(b * s, -1, 6)
    )
    # Small 9x9 solves in double precision; upstream gradient returns in model dtype.
    origin = sampled[..., 3:].double().mean(1)
    directions = sampled[..., :3].double()
    if bool((directions.norm(dim=-1) < 1e-8).any()):
        raise ValueError("ray pose: zero direction")
    directions = F.normalize(directions, dim=-1)
    scale = float(max(height, width))
    y, x = torch.meshgrid(
        torch.linspace(0, height - 1, ph, device=rays.device, dtype=torch.float64),
        torch.linspace(0, width - 1, pw, device=rays.device, dtype=torch.float64),
        indexing="ij",
    )
    points = torch.stack(
        (
            (x - (width - 1) / 2) / scale,
            (y - (height - 1) / 2) / scale,
            torch.ones_like(x),
        ),
        -1,
    ).reshape(-1, 3)
    dx, dy, dz = directions.unbind(-1)
    zero = torch.zeros_like(dx)
    skew = torch.stack((zero, -dz, dy, dz, zero, -dx, -dy, dx, zero), -1).reshape(
        b * s, -1, 3, 3
    )
    design = torch.einsum("bnij,nk->bnijk", skew, points).reshape(b * s, -1, 3, 9)
    weights = torch.ones_like(dx)
    for iteration in range(3):
        a = (design * weights.sqrt()[..., None, None]).reshape(b * s, -1, 9)
        normal = a.transpose(-1, -2) @ a / a.shape[1]
        homography = _NullVector.apply(normal).reshape(b * s, 3, 3)
        det = torch.linalg.det(homography)
        if bool((det.abs() < 1e-10).any()):
            raise ValueError("ray pose: singular camera")
        homography = homography * torch.where(det < 0, -1.0, 1.0)[:, None, None]
        prediction = F.normalize(
            torch.einsum("bij,nj->bni", homography, points), dim=-1
        )
        dot = (prediction * directions).sum(-1).clamp(-1, 1)
        # Piecewise smooth at exact agreement; avoid sqrt(0) in the gradient.
        cross2 = torch.cross(prediction, directions, dim=-1).square().sum(-1)
        angle = torch.atan2(cross2.clamp_min(1e-24).sqrt(), dot)
        weights = 1 / (1 + (angle / angular_threshold).square())
    rotation, inverse_k = torch.linalg.qr(homography)
    diagonal = inverse_k.diagonal(dim1=-2, dim2=-1)
    signs = torch.where(diagonal < 0, -1.0, 1.0)
    rotation = rotation * signs[:, None, :]
    inverse_k = inverse_k * signs[:, :, None]
    if bool((torch.linalg.det(rotation) < 0.99).any()):
        raise ValueError("ray pose: improper rotation")
    k = torch.linalg.inv(inverse_k)
    k = k / k[:, 2:3, 2:3]
    pixel_transform = rays.new_tensor(
        [[scale, 0, (width - 1) / 2], [0, scale, (height - 1) / 2], [0, 0, 1]],
        dtype=torch.float64,
    )
    intrinsics = pixel_transform @ k
    rw = rotation.transpose(-1, -2)
    translation = -(rw @ origin[..., None])
    top = torch.cat((rw, translation), -1)
    bottom = torch.zeros(b * s, 1, 4, device=rays.device, dtype=torch.float64)
    bottom[..., 3] = 1
    ext = torch.cat((top, bottom), -2).reshape(b, s, 4, 4)
    dtype = torch.float64 if rays.dtype == torch.float64 else torch.float32
    return ext.to(dtype), intrinsics.reshape(b, s, 3, 3).to(dtype)
