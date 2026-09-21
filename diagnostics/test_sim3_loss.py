"""Unit test for loss_pose_sim3 (2026-09-20).
Synthetic checks, CPU-only:
  T1 gauge invariance: teacher = Sim(3)-transformed student poses -> loss ~ 0
  T2 sensitivity:      perturbed student center -> loss > 0, grows with perturbation
  T3 scale recovery:   teacher world scaled by k -> sim3_s ~ 1/k (student/teacher gauge)
  T4 rotation action:  random Sim(3) incl. rotation -> residual rotation ~ 0
  T5 gradient:         loss backprops to student ext only
"""
import sys, torch
sys.path.insert(0, "src")
from depth_anything_3.test_time_adaption.protocol_v1 import loss_pose_sim3

torch.manual_seed(0)

def rand_w2c(S):
    """Random w2c poses from random centers + random rotations."""
    c = torch.randn(S, 3) * 2.0
    R = []
    for _ in range(S):
        A = torch.randn(3, 3)
        Q, _ = torch.linalg.qr(A)
        if torch.det(Q) < 0:
            Q[:, -1] *= -1
        R.append(Q)
    R = torch.stack(R)
    t = -(R @ c.unsqueeze(-1)).squeeze(-1)
    return torch.cat([R, t.unsqueeze(-1)], dim=-1)  # [S,3,4]

def sim3_transform(ext, s, Ra, ta):
    """World transform x' = s*Ra@x + ta applied to w2c poses:
    c' = s*Ra@c + ta, R' = R @ Ra^T."""
    R = ext[..., :3, :3]
    c = -(R.transpose(-1, -2) @ ext[..., :3, 3].unsqueeze(-1)).squeeze(-1)
    c2 = s * (Ra @ c.T).T + ta
    R2 = R @ Ra.T
    t2 = -(R2 @ c2.unsqueeze(-1)).squeeze(-1)
    return torch.cat([R2, t2.unsqueeze(-1)], dim=-1)

S = 4
E_s = rand_w2c(S)

# --- T1 + T4: random Sim(3) with rotation ---
A = torch.randn(3, 3); Ra, _ = torch.linalg.qr(A)
if torch.det(Ra) < 0: Ra[:, -1] *= -1
s_true, ta = 3.7, torch.tensor([1.5, -2.0, 0.7])
E_t = sim3_transform(E_s, s_true, Ra, ta)
loss, ex = loss_pose_sim3(E_s, E_t)
print(f"T1/T4 random Sim3 (s={s_true}): loss={loss.item():.2e} "
      f"(c={ex['sim3_c']:.2e} r={ex['sim3_r']:.2e} s_recovered={ex['sim3_s']:.4f})")
assert loss.item() < 1e-8, "gauge invariance FAILED"
assert abs(ex["sim3_s"] - 1.0 / s_true) < 1e-3, "scale direction: sim3_s should be student/teacher spread"

# --- T3: pure scale k -> s ~ 1/k ---
for k in (0.1, 10.0):
    E_t = sim3_transform(E_s, k, torch.eye(3), torch.zeros(3))
    _, ex = loss_pose_sim3(E_s, E_t)
    print(f"T3 k={k}: sim3_s={ex['sim3_s']:.4f} (expect {1/k:.2f})")
    assert abs(ex["sim3_s"] - 1.0 / k) < 1e-3

# --- T2: sensitivity (monotone growth; Huber is quadratic at small residuals) ---
base = None
for eps in (0.0, 0.01, 0.1, 0.5):
    E_p = E_s.clone(); E_p[2, :3, 3] += E_s[2, :3, :3] @ torch.tensor([0.0, 0.0, eps])
    # (shift along camera z = move center by eps)
    l, _ = loss_pose_sim3(E_p, sim3_transform(E_s, 2.0, torch.eye(3), torch.tensor([1., 2., 3.])))
    print(f"T2 eps={eps}: loss={l.item():.4e}")
    if eps == 0.0: base = l.item()
    else: assert l.item() > prev, "sensitivity not monotone"
    prev = l.item()
assert prev > base * 10

# --- T5: gradient flows to student, not needed for teacher (detached by contract) ---
E_t = sim3_transform(E_s, 5.0, torch.eye(3), torch.tensor([0., 0., 0.]))
E_s_g = E_s.clone().requires_grad_(True)
l, _ = loss_pose_sim3(E_s_g, E_t)
l.backward()
assert E_s_g.grad is not None and E_s_g.grad.abs().sum() > 0
print(f"T5 grad |g|={E_s_g.grad.abs().sum().item():.4e} OK")

# --- T1b: batched [1,S,3,4] input shape also works ---
l_b, _ = loss_pose_sim3(E_s.unsqueeze(0), sim3_transform(E_s, s_true, Ra, ta).unsqueeze(0))
print(f"T1b batched: loss={l_b.item():.2e}")
assert l_b.item() < 1e-8

# --- T6: collinear centers (forward-motion degeneracy, delivery_area case) ---
# Centers along a line -> center-only Umeyama R_a is 180°-ambiguous about the
# motion axis; the orientation-based solve must still recover the true gauge.
line_dir = torch.nn.functional.normalize(torch.tensor([1.0, 0.05, 0.02]), dim=0)
c_lin = torch.stack([line_dir * t for t in (-1.0, 0.3, 0.9, 2.1)]) + torch.tensor([0.5, 0.5, 0.5])
E_lin = rand_w2c(1)  # placeholder to reuse rand rotations shape
R4 = []
for _ in range(4):
    A = torch.randn(3, 3); Q, _ = torch.linalg.qr(A)
    if torch.det(Q) < 0: Q[:, -1] *= -1
    R4.append(Q)
R4 = torch.stack(R4)
t4 = -(R4 @ c_lin.unsqueeze(-1)).squeeze(-1)
E_lin = torch.cat([R4, t4.unsqueeze(-1)], dim=-1)
# random Sim(3) world transform
A = torch.randn(3, 3); Ra2, _ = torch.linalg.qr(A)
if torch.det(Ra2) < 0: Ra2[:, -1] *= -1
E_lin_t = sim3_transform(E_lin, 2.3, Ra2, torch.tensor([0.3, -0.8, 1.1]))
l_lin, ex_lin = loss_pose_sim3(E_lin, E_lin_t)
print(f"T6 collinear centers: loss={l_lin.item():.2e} (expect ~0; "
      f"center-only Umeyama flips 180deg here)")
assert l_lin.item() < 1e-6, "collinear-center gauge recovery FAILED"

print("ALL SIM3 TESTS PASSED")
