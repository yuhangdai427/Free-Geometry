"""Diagnostic (2026-09-20): why is sim3_r ~ 7.4 (worse than random) on
delivery_area while pairwise rel_rot is small?

Hypothesis: teacher/student w2c orientations differ by a LEFT factor
R_s,i = D * R_t,i (camera-frame convention drift), which:
  - cancels in pairwise relative rotations R_i R_j^T  -> old rel loss blind
  - NOT removable by world Sim(3) (right action R' = R R_a^T) -> sim3_r huge

Measures per pair (zero-LoRA student = frozen baseline, unmasked input):
  A rel_rot      pairwise relative rotation chordal (old loss quantity)
  B sim3_r       center-Umeyama aligned absolute rotation residual (new loss)
  C left_res     residual after best-fit LEFT rotation D (Procrustes on R_t->R_s)
  D angle(D)     the left factor's rotation angle in degrees
  E per-cam angle between R_s,i and R_t,i R_a^T (median, deg)
"""
import sys, torch
sys.path.insert(0, "src")
sys.path.insert(0, "diagnostics/free_geometry")
sys.path.insert(0, "scripts")
import numpy as np
from depth_anything_3.test_time_adaption import protocol_v1 as P

ds_name, scene = "eth3d", "delivery_area"
import common as fg_common
fg_common.set_dataset(ds_name)
from train_da3_protocol import make_dataset, get_scene_data

device = "cuda"
dataset_obj = make_dataset(ds_name)
scene_data = get_scene_data(scene)
image_files = list(scene_data.image_files)

print(f"[{scene}] N={len(image_files)} loading models...")
teacher = P.create_teacher("model_weights/DA3-GIANT-1.1")
student = P.create_student("model_weights/DA3-GIANT-1.1")

proto = P.build_scene_protocol(image_files, scene, dataset=ds_name,
                               n_train=10, n_shared=4, teacher_N=16)
print(f"tau={proto['tau']:.3f} teacher_N={proto['teacher_N']}")

student.to("cpu"); teacher.to(device)
torch.manual_seed(P.stable_seed("lora_init", scene, 0))
P.reset_lora_(student)
student.to(device); student.eval()

def centers(ext):
    R, t = ext[..., :3, :3].float(), ext[..., :3, 3].float()
    return -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)

def rot_angle_deg(R):
    return torch.rad2deg(torch.acos(((R.trace() - 1) / 2).clamp(-1, 1)))

results = []
for pi, pair in enumerate(proto["train_pairs"][:5]):
    imgs = P.load_images_da3([image_files[i] for i in pair["teacher_frames"]])
    imgs = imgs.unsqueeze(0).to(device)
    ph, pw = imgs.shape[-2] // P.PATCH_SIZE, imgs.shape[-1] // P.PATCH_SIZE
    slots = list(range(0, 2 * len(pair["student_frames"]), 2))
    cache = P.cache_teacher_pair(teacher, imgs, slots, (ph, pw))
    imgs4 = imgs[0, slots].unsqueeze(0)
    with torch.no_grad():
        _, ext_s, _ = P.student_forward_c2m(student, imgs4)
    ext_t = cache["ext4"].to(device)
    if ext_s.dim() == 4: ext_s = ext_s[0]
    R_s, R_t = ext_s[..., :3, :3].float(), ext_t[..., :3, :3]

    # A: pairwise relative chordal (old rel loss)
    rel_rot = 0.0; n = 0
    for i in range(4):
        for j in range(i + 1, 4):
            rel_rot += ((R_s[i] @ R_s[j].T - R_t[i] @ R_t[j].T) ** 2).sum(); n += 1
    rel_rot /= n

    # B: center-Umeyama aligned absolute residual (new sim3 loss)
    _, sim3x = P.loss_pose_sim3(ext_s, ext_t)
    sim3_r = sim3x["sim3_r"]

    # C: best-fit LEFT factor D: R_s,i ~= D @ R_t,i
    M = sum(R_t[i] @ R_s[i].T for i in range(4))   # maximize tr(D M)
    U, S, Vt = torch.linalg.svd(M)
    D = Vt.T @ U.T
    if torch.det(D) < 0:
        Vt2 = Vt.clone(); Vt2[-1] *= -1; D = Vt2.T @ U.T
    left_res = (((D @ R_t - R_s) ** 2).sum(dim=(-2, -1))).mean()

    # E: median per-camera angle between R_s,i and R_t,i @ R_a^T
    cs, ct = centers(ext_s), centers(ext_t)
    with torch.no_grad():
        mu_t, mu_s = ct.mean(0), cs.mean(0)
        tc, sc = ct - mu_t, cs - mu_s
        H = tc.T @ sc
        U2, S2, Vt2 = torch.linalg.svd(H)
        Sc = torch.ones(3, device=device)
        if torch.det(U2 @ Vt2) < 0: Sc[-1] = -1
        R_a = (Vt2.T * Sc) @ U2.T
        var_t = (tc ** 2).sum()
        s_a = ((S2 * Sc).sum() / var_t).clamp_min(1e-8)
    ang_pair = [rot_angle_deg(R_s[i].T @ (R_t[i] @ R_a.T)) for i in range(4)]
    ang_left = rot_angle_deg(D)

    results.append((rel_rot.item(), sim3_r, left_res.item(),
                    ang_left.item(), torch.stack(ang_pair).median().item(),
                    s_a.item()))
    print(f"pair{pi}: rel_rot={rel_rot:.3f}  sim3_r={sim3_r:.3f}  "
          f"left_res={left_res:.3f}  angle(D)={np.degrees(1):.0f}"
          .replace("angle(D)=0", "") if False else
          f"pair{pi}: rel_rot={rel_rot:.3f}  sim3_r={sim3_r:.3f}  "
          f"left_res={left_res:.3f}  angle(D)={ang_left.item():.1f}deg  "
          f"median_cam_angle={torch.stack(ang_pair).median().item():.1f}deg  "
          f"scale={s_a.item():.2f}")

r = np.array(results)
print("\n== SUMMARY (5 pairs) ==")
print(f"rel_rot  (pairwise, old loss)     : {r[:,0].mean():.3f}  <- small?")
print(f"sim3_r   (center-aligned abs)     : {r[:,1].mean():.3f}  <- huge")
print(f"left_res (best LEFT D removed)    : {r[:,2].mean():.3f}")
print(f"angle(D) left factor              : {r[:,3].mean():.1f} deg")
print(f"median per-cam angle after Sim3   : {r[:,4].mean():.1f} deg")
print(f"scale s (student/teacher spread)  : {r[:,5].mean():.2f}")
print("\nInterpretation: if left_res << sim3_r and angle(D) consistent across pairs"
      " -> systematic camera-convention left factor; if left_res ~ sim3_r ~ 6+"
      " -> teacher orientations are genuinely noisy on this scene.")
