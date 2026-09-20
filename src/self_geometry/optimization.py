import math
import torch


def fan_setup(rotations, width=15, compat9=False):
    rotations = rotations.detach().double().cpu()
    traces = torch.einsum('aij,bij->ab',rotations,rotations)
    angles = torch.acos(((traces-1)/2).clamp(-1,1))*180/math.pi
    bins = torch.floor(angles/width).long().clamp(max=(8 if compat9 else math.ceil(180/width)-1))
    candidates=[]
    for i,row in enumerate(bins):
        groups = [[j for j in range(len(row)) if j!=i and row[j]==b] for b in sorted(set(row.tolist()))]
        groups = [g for g in groups if g]
        p = torch.tensor([len(g)/(len(row)-1) for g in groups],dtype=torch.float64)
        entropy = float(-(p*p.log()).sum())
        candidates.append((len(groups),entropy,-i,groups))
    _,_,negative_index,groups = max(candidates,key=lambda x:x[:3])
    return -negative_index,groups


def fan_sample(target, groups, step):
    return [target]+[g[step%len(g)] for g in groups]


def project_gradients(primary, reference, eps=1e-20):
    dot = sum((a.float()*b.float()).sum() for a,b in zip(primary,reference))
    norm = sum(b.float().square().sum() for b in reference)
    coefficient = torch.where(norm>eps,dot/norm.clamp_min(eps),torch.zeros_like(dot))
    return [a-coefficient*b for a,b in zip(primary,reference)],float(dot.detach()),float(norm.detach())


def dwa(history, eps=1e-12):
    if len(history)<2: return [1.0]*5
    a,b = torch.tensor(history[-1]),torch.tensor(history[-2])
    ratios = (a/b.clamp_min(eps)).clamp(0.5,2.0)
    return (5*ratios.softmax(0)).tolist()


def learning_rate(step,c):
    # Discrete interpretation of 5% warmup: ceil(50*.05)=3 updates.
    warm = max(1,math.ceil(c['iterations']*c['warmup_fraction']))
    if step<warm: return c['lr']*(step+1)/warm
    frac = 1.0 if c['iterations']-1<=warm else (step-warm)/(c['iterations']-1-warm)
    return c['final_lr']+(c['lr']-c['final_lr'])*(1+math.cos(math.pi*frac))/2


def combine_gradients(losses, params, weights, mode):
    def grads(loss,retain):
        values = torch.autograd.grad(loss,params,retain_graph=retain,allow_unused=True)
        return [torch.zeros_like(p) if g is None else g for p,g in zip(params,values)]
    mvc = grads(losses[0],True); ec = grads(losses[1],True)
    dot = sum((a.float()*b.float()).sum() for a,b in zip(mvc,ec))
    den = torch.sqrt(sum(a.float().square().sum() for a in mvc)*sum(b.float().square().sum() for b in ec))
    cosine = float((dot/den.clamp_min(1e-20)).detach())
    if mode=='mvc_orthogonal_ec' or mode=='mvc_conflict_only' and float(dot.detach())<0:
        mvc,_,_ = project_gradients(mvc,ec)
    elif mode=='ec_orthogonal_mvc': ec,_,_ = project_gradients(ec,mvc)
    elif mode not in ('none','mvc_conflict_only'): raise ValueError(mode)
    auxiliary = grads(sum(w*l for w,l in zip(weights[2:],losses[2:])),False)
    for p,a,b,g in zip(params,mvc,ec,auxiliary): p.grad = weights[0]*a+weights[1]*b+g
    return cosine
