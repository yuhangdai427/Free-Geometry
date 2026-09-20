"""Pixel-coordinate geometry. Extrinsics are world-to-camera [R|t]."""
import torch
import torch.nn.functional as F


def relative(source, target):
    r = target[:3,:3] @ source[:3,:3].T
    t = target[:3,3] - r @ source[:3,3]
    return r, t


def skew(t):
    z = t[0]*0
    return torch.stack([z,-t[2],t[1],t[2],z,-t[0],-t[1],t[0],z]).reshape(3,3)


def homogeneous(x): return torch.cat([x, torch.ones_like(x[...,:1])], -1)


def pixel_grid(h, w, device):
    y,x = torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
    return torch.stack([x,y],-1).float().reshape(-1,2)


def sample(image, xy):
    # image C,H,W; xy M,2. Pixel centers align exactly, including boundaries.
    h,w = image.shape[-2:]
    grid = xy * xy.new_tensor([2/max(w-1,1), 2/max(h-1,1)])-1
    return F.grid_sample(image[None],grid[None,None],align_corners=True,padding_mode='zeros')[0,:,0].T


def project(xy, depth, ks, kt, source, target):
    r,t = relative(source,target)
    xyz = (homogeneous(xy) @ torch.linalg.inv(ks).T) * depth[:,None]
    xyz = xyz @ r.T + t
    px = xyz @ kt.T
    uv = px[:,:2] / px[:,2:].clamp_min(1e-6)
    return uv, xyz[:,2]


def residuals(pred, i, j, xi, xj):
    ki,kj = pred['intrinsics'][i],pred['intrinsics'][j]
    ei,ej = pred['extrinsics'][i],pred['extrinsics'][j]
    r,t = relative(ej,ei)
    f = torch.linalg.inv(ki).T @ skew(t) @ r @ torch.linalg.inv(kj)
    # Scale normalization preserves Sampson distance and avoids epsilon dominating tiny baselines.
    f = f / torch.linalg.vector_norm(f).clamp_min(1e-12)
    hi,hj = homogeneous(xi),homogeneous(xj)
    fi,fj = hj @ f.T, hi @ f
    denom = (fi[:,:2].square().sum(-1)+fj[:,:2].square().sum(-1)).clamp_min(1e-12)
    ec = (hi*fi).sum(-1).abs() / denom.sqrt()
    d = sample(pred['depth'][j,None],xj)[:,0]
    projected,z = project(xj,d,kj,ki,ej,ei)
    mvc = torch.linalg.vector_norm(projected-xi,dim=-1)
    h,w = pred['depth'].shape[-2:]
    valid = (d>1e-6)&(z>1e-6)&torch.isfinite(ec)&torch.isfinite(mvc)
    valid &= (projected[:,0]>=0)&(projected[:,0]<=w-1)&(projected[:,1]>=0)&(projected[:,1]<=h-1)
    valid &= torch.linalg.vector_norm(t).detach()>1e-8
    return ec,mvc,valid


def filter_mask(ec,mvc,valid,keep):
    mask = valid.detach().clone()
    for values in (ec,mvc):
        if mask.any():
            threshold = torch.quantile(values.detach()[mask],keep)
            mask &= values.detach()<=threshold
    return mask


def threshold(r):
    if not r.numel(): return 1e-6
    r = r.detach().abs()
    return max(float(1.345*1.4826*r.median()),1e-6)


def huber(r, delta):
    a = r.abs(); d = torch.as_tensor(delta,device=r.device,dtype=r.dtype)
    return torch.where(a<=d,0.5*a.square(),d*(a-0.5*d))


def photo_residual(images, pred, i, j):
    h,w = images.shape[-2:]
    xy = pixel_grid(h,w,images.device)
    # Inverse warp: target depth sends target pixels into source image.
    uv,z = project(xy,pred['depth'][i].reshape(-1),pred['intrinsics'][i],pred['intrinsics'][j],
                   pred['extrinsics'][i],pred['extrinsics'][j])
    valid = (z>1e-6)&(uv[:,0]>=0)&(uv[:,0]<=w-1)&(uv[:,1]>=0)&(uv[:,1]<=h-1)
    valid &= torch.isfinite(uv).all(-1)&(pred['depth'][i].reshape(-1)>1e-6)
    warped = sample(images[j],uv).T.reshape(1,3,h,w)
    target = images[i:i+1]
    pool = lambda x: F.avg_pool2d(F.pad(x,(1,1,1,1),mode='reflect'),3,1)
    a,b = pool(target),pool(warped)
    va,vb,cov = pool(target*target)-a*a,pool(warped*warped)-b*b,pool(target*warped)-a*b
    ssim = ((2*a*b+0.01**2)*(2*cov+0.03**2))/((a*a+b*b+0.01**2)*(va+vb+0.03**2))
    residual = (0.85*((1-ssim)/2).clamp(0,1)+0.15*(target-warped).abs()).mean(1).reshape(-1)
    return residual[valid.detach()]


def smoothness(depth, image):
    dx = (depth[:,1:]-depth[:,:-1]).abs()
    dy = (depth[1:]-depth[:-1]).abs()
    ix = (image[:,:,1:]-image[:,:,:-1]).abs().mean(0)
    iy = (image[:,1:]-image[:,:-1]).abs().mean(0)
    return (dx*torch.exp(-ix)).mean()+(dy*torch.exp(-iy)).mean()
