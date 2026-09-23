"""Model handling for the diagnostic bake-off: teacher caching, student with
resettable LoRA, exact head replay, NORM/PROJECTED taps, probe evaluation.

Audited invariants this module relies on:
- DPTHead indexes aggregated_tokens_list by ABSOLUTE layer id (dpt_head.py:206),
  so a replay list must have 24 slots with tensors at [4,11,17,23].
- DPTHead forward: strip tokens -> norm -> reshape -> projects[j] ->
  _apply_pos_embed -> resize_layers -> scratch_forward (incl. layer_rn) ->
  custom_interpolate -> SECOND _apply_pos_embed -> output_conv2 ->
  activate_head("exp")  (dpt_head.py:205-247). Replay = call the unmodified head.
- CameraHead uses tokens[:, :, 0] of the LAST layer and a 4-block cross-view
  trunk (camera_head.py:86-141).
- Heads must run under autocast(enabled=False) with fp32 inputs.
- PEFT LoRA init: A ~ kaiming_uniform(a=sqrt5), B = 0 -> student == baseline.
"""

import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common import (
    MODEL_PATH, PATCH_START_IDX, STUDENT_INDICES, TAP_LAYERS,
    e_depth, load_gt_depth, load_image_model, stable_seed,
)

_PATCH_PROJECT_J = {layer: j for j, layer in enumerate(TAP_LAYERS)}


def load_teacher(device="cuda"):
    from vggt.models.vggt import VGGT

    vggt = VGGT.from_pretrained(MODEL_PATH)
    vggt.eval()
    for p in vggt.parameters():
        p.requires_grad = False
    return vggt.to(device)


def load_student(device="cuda", lora_rank=32, lora_alpha=32.0, train_camera_token=False):
    """Per-scene student: LoRA r=32/a=32/dropout=0 on ALL 24 layers (matches the
    deployed runs' lora_layers_start=0), heads frozen."""
    from vggt.vggt.test_time_adaption.models import VGGTStudentModel

    student = VGGTStudentModel(
        model_name=MODEL_PATH,
        output_layers=TAP_LAYERS,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        train_camera_token=train_camera_token,
        lora_layers=list(range(24)),
    )
    return student.to(device)


def reset_lora_(student) -> None:
    """In-place PEFT-faithful LoRA re-init: A ~ kaiming_uniform(sqrt5), B = 0."""
    import math
    import torch.nn as nn

    base = student._get_vggt_model()
    count = 0
    for m in base.modules():
        if hasattr(m, "lora_A") and hasattr(m, "lora_B"):
            for a in m.lora_A.values():
                nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
                count += 1
            for b in m.lora_B.values():
                nn.init.zeros_(b.weight)
    assert count > 0, "no LoRA modules found"


def get_base_vggt(student):
    return student._get_vggt_model()


# --------------------------------------------------------------------------
# forwards
# --------------------------------------------------------------------------

@torch.no_grad()
def aggregator_all(vggt, images):
    """Full 24-layer aggregator forward. Returns (list[24], patch_start_idx)."""
    out, psi = vggt.aggregator(images)
    assert len(out) == 24 and psi == PATCH_START_IDX
    return out, psi


def build_replay_list(feats_by_layer: Dict[int, torch.Tensor]) -> List[Optional[torch.Tensor]]:
    slots = [None] * 24
    for layer, t in feats_by_layer.items():
        slots[layer] = t
    return slots


@torch.no_grad()
def replay_depth_nograd(vggt, feats24, images, psi=PATCH_START_IDX):
    with torch.autocast(device_type="cuda", enabled=False):
        depth, conf = vggt.depth_head(feats24, images=images, patch_start_idx=psi)
    return depth, conf


@torch.no_grad()
def replay_camera_nograd(vggt, feats24):
    with torch.autocast(device_type="cuda", enabled=False):
        pose_list = vggt.camera_head(feats24)
    return pose_list[-1]


def slice_views(feats24, view_idx) -> List[Optional[torch.Tensor]]:
    out = []
    for t in feats24:
        out.append(None if t is None else t[:, view_idx].contiguous())
    return out


# --------------------------------------------------------------------------
# feature-space projections (patch tokens only)
# --------------------------------------------------------------------------

def to_patch(feats: torch.Tensor, psi: int = PATCH_START_IDX) -> torch.Tensor:
    return feats[:, :, psi:, :]


def to_norm(depth_head, h: torch.Tensor) -> torch.Tensor:
    """h: [B,S,P,C] patch-only -> LayerNorm (shared depth_head.norm), same shape."""
    B, S, P, C = h.shape
    return depth_head.norm(h.reshape(B * S, P, C)).reshape(B, S, P, C)


def to_projected(depth_head, h: torch.Tensor, layer: int, patch_hw: Tuple[int, int]) -> torch.Tensor:
    """h: [B,S,P,C] patch-only -> LN -> reshape -> projects[j]. Returns [B,S,Cj,ph,pw]."""
    B, S, P, C = h.shape
    ph, pw = patch_hw
    assert P == ph * pw, f"P={P} != {ph}x{pw}"
    n = depth_head.norm(h.reshape(B * S, P, C))
    n_map = n.transpose(1, 2).reshape(B * S, C, ph, pw)
    z = depth_head.projects[_PATCH_PROJECT_J[layer]](n_map)
    return z.reshape(B, S, z.shape[1], ph, pw)


# --------------------------------------------------------------------------
# teacher cache
# --------------------------------------------------------------------------

@torch.no_grad()
def cache_teacher_pair_ctx(vggt, imagesN: torch.Tensor, shared_slots: List[int],
                           patch_hw: Tuple[int, int]) -> Dict:
    """One N-view teacher forward (N = imagesN.shape[1]). Same as
    cache_teacher_pair but the 4 shared views sit at positions shared_slots
    of the N-view list instead of the hardcoded STUDENT_INDICES. Caches (fp32, GPU):
    - feats: {layer: [1,N,P_total,2048]} full tokens (incl. special)
    - depth4: teacher N->4 depth on shared views [1,4,H,W] (frozen DPT replay)
    - conf4:  [1,4,H,W]
    - pose_encN: [1,N,9] teacher camera (N-view head context)
    - patch_hw, shared_slots
    """
    feats24, psi = aggregator_all(vggt, imagesN)
    feats = {layer: feats24[layer].float() for layer in TAP_LAYERS}

    slots4 = [None] * 24
    for layer in TAP_LAYERS:
        slots4[layer] = feats[layer][:, shared_slots].contiguous()
    depth4, conf4 = replay_depth_nograd(vggt, slots4, imagesN[:, shared_slots], psi)
    pose_encN = replay_camera_nograd(vggt, feats24)

    return {
        "feats": feats,
        "depth4": depth4.float(),
        "conf4": conf4.float(),
        "pose_encN": pose_encN.float(),
        "patch_hw": patch_hw,
        "shared_slots": list(shared_slots),
    }


@torch.no_grad()
def cache_teacher_pair(vggt, images8: torch.Tensor, patch_hw: Tuple[int, int]) -> Dict:
    """One 8-view teacher forward. Caches (fp32, GPU):
    - feats: {layer: [1,8,P_total,2048]} full tokens (incl. special)
    - depth4: teacher 8->4 depth on shared views [1,4,H,W] (frozen DPT replay)
    - conf4:  [1,4,H,W]
    - pose_enc8: [1,8,9] teacher camera (8-view head context)
    """
    cache = cache_teacher_pair_ctx(vggt, images8, STUDENT_INDICES, patch_hw)
    return {
        "feats": cache["feats"],
        "depth4": cache["depth4"],
        "conf4": cache["conf4"],
        "pose_enc8": cache["pose_encN"],
        "patch_hw": cache["patch_hw"],
    }


@torch.no_grad()
def cache_teacher_pair_consensus(vggt, scene_data, teacher_frames8: List[int],
                                 shared_frames: List[int], pool: List[int],
                                 patch_hw: Tuple[int, int], K: int = 2,
                                 seed_tag=None, device="cuda") -> Dict:
    """Consensus teacher: average K 8-view contexts that keep the same 4 shared
    frames (slots [0,2,4,6]) but vary the 4 extra frames (slots [1,3,5,7]).
    k=0 uses the original pair's 8 frames (comparable with single-teacher
    caches); k>=1 samples extras from `pool` with
    stable_seed(52_000+k, *seed_tag). Averaging:
    - feats: per-layer arithmetic mean of the K contexts' RAW aggregated
      tokens on the shared slots; extra slots are filled with the k=0
      context's tokens so the cache stays isomorphic to cache_teacher_pair.
    - depth4: exp(mean of the K contexts' shared-4 log-depth) (frozen DPT
      replay per context); conf4: arithmetic mean of the K confs.
    - pose_enc8: k=0 context's camera replay (poses are NOT averaged -
      coordinate systems differ across contexts).
    Returns the same dict shape as cache_teacher_pair.
    """
    base_extras = [f for i, f in enumerate(teacher_frames8) if i not in STUDENT_INDICES]
    tag = tuple(seed_tag) if isinstance(seed_tag, (list, tuple)) else (seed_tag,)

    feat_acc = {layer: None for layer in TAP_LAYERS}
    extra0 = {layer: None for layer in TAP_LAYERS}
    logd_acc, conf_acc, pose_enc8 = None, None, None
    for k in range(K):
        if k == 0:
            extras = base_extras
        else:
            rng = random.Random(stable_seed(52_000 + k, *tag))
            extras = rng.sample(pool, 4)
        frames = [None] * 8
        for si, s in zip(STUDENT_INDICES, shared_frames):
            frames[si] = s
        for ei, e in zip([1, 3, 5, 7], extras):
            frames[ei] = e
        images8, _ = load_pair_images(scene_data, frames, device)
        feats24, psi = aggregator_all(vggt, images8)

        slots4 = [None] * 24
        for layer in TAP_LAYERS:
            shared_t = feats24[layer][:, STUDENT_INDICES].float()
            feat_acc[layer] = shared_t if feat_acc[layer] is None else feat_acc[layer] + shared_t
            if k == 0:
                extra_idx = [i for i in range(8) if i not in STUDENT_INDICES]
                extra0[layer] = feats24[layer][:, extra_idx].float().contiguous()
            slots4[layer] = shared_t.contiguous()
        d4, c4 = replay_depth_nograd(vggt, slots4, images8[:, STUDENT_INDICES], psi)
        ld = torch.log(d4.float().clamp_min(1e-6))
        logd_acc = ld if logd_acc is None else logd_acc + ld
        conf_acc = c4.float() if conf_acc is None else conf_acc + c4.float()
        if k == 0:
            pose_enc8 = replay_camera_nograd(vggt, feats24).float()
        del feats24, images8

    feats = {}
    extra_idx = [i for i in range(8) if i not in STUDENT_INDICES]
    for layer in TAP_LAYERS:
        full = feat_acc[layer].new_zeros(
            1, 8, feat_acc[layer].shape[2], feat_acc[layer].shape[3])
        full[:, STUDENT_INDICES] = feat_acc[layer] / K
        full[:, extra_idx] = extra0[layer]
        feats[layer] = full

    return {
        "feats": feats,
        "depth4": torch.exp(logd_acc / K),
        "conf4": conf_acc / K,
        "pose_enc8": pose_enc8,
        "patch_hw": patch_hw,
    }


# --------------------------------------------------------------------------
# pair loading
# --------------------------------------------------------------------------

def load_pair_images(scene_data, teacher_frames: List[int], device="cuda"):
    """Returns (images8 [1,8,3,H,W], images4 [1,4,3,H,W]) pixel-aligned."""
    imgs = [load_image_model(scene_data.image_files[i]) for i in teacher_frames]
    arr = np.stack(imgs, axis=0)  # [8,H,W,3]
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).float().unsqueeze(0)
    images8 = t.to(device)
    images4 = images8[:, STUDENT_INDICES].contiguous()
    return images8, images4


def load_probe_gt(scene_data, student_frames: List[int], out_hw: Tuple[int, int]) -> np.ndarray:
    """[4,H,W] GT depth (NaN=invalid) for the shared/student views."""
    gtf = scene_data.aux.gt_depth_files
    # DTU/dtu64 carry no GT depth files (pose+recon protocol): all-invalid GT,
    # e_depth() handles the empty omega by returning (nan, 0).
    if not (gtf and isinstance(gtf, (list, tuple)) and isinstance(gtf[0], str)):
        return np.full((len(student_frames), out_hw[0], out_hw[1]),
                       np.nan, dtype=np.float32)
    return np.stack([
        load_gt_depth(gtf[i], out_hw) for i in student_frames
    ], axis=0)


# --------------------------------------------------------------------------
# probe evaluation
# --------------------------------------------------------------------------

def student_preds(student, images4: torch.Tensor):
    """Full forward WITH grad path. Returns (feats24, psi, preds dict)."""
    base = get_base_vggt(student)
    aggregator = student._get_aggregator()
    feats24, psi = aggregator(images4)
    with torch.autocast(device_type="cuda", enabled=False):
        feats24_f = [t.float() for t in feats24]
        depth, conf = base.depth_head(feats24_f, images=images4, patch_start_idx=psi)
        pose_list = base.camera_head(feats24_f)
    preds = {"depth": depth, "conf": conf, "pose_enc": pose_list[-1]}
    return feats24, psi, preds


def feature_mses(depth_head, student_feats24, teacher_cache, patch_hw) -> Dict[str, float]:
    """RAW/NORM/PROJECTED MSE between student 4-view and teacher shared-4 feats."""
    out = {}
    raw_sum, norm_sum, proj_sum = 0.0, 0.0, 0.0
    for layer in TAP_LAYERS:
        hs = to_patch(student_feats24[layer].float())
        ht = to_patch(teacher_cache["feats"][layer][:, STUDENT_INDICES].float())
        raw_sum += torch.mean((hs - ht) ** 2).item()
        ns = to_norm(depth_head, hs)
        nt = to_norm(depth_head, ht)
        norm_sum += torch.mean((ns - nt) ** 2).item()
        zs = to_projected(depth_head, hs, layer, patch_hw)
        zt = to_projected(depth_head, ht, layer, patch_hw)
        proj_sum += torch.mean((zs - zt) ** 2).item()
    n = len(TAP_LAYERS)
    out["mse_raw"] = raw_sum / n
    out["mse_norm"] = norm_sum / n
    out["mse_proj"] = proj_sum / n
    return out


def pose_auc(pose_enc: torch.Tensor, gt_extrinsics: np.ndarray) -> Dict[str, float]:
    """pose_enc [1,S,9] + GT w2c [S,4,4] -> auc dict. Uses repo compute_pose."""
    from depth_anything_3.bench.utils import compute_pose
    from depth_anything_3.utils.geometry import as_homogeneous
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    S = pose_enc.shape[1]
    H, W = 378, 504
    ext, _ = pose_encoding_to_extri_intri(
        pose_enc, image_size_hw=(H, W), pose_encoding_type="absT_quaR_FoV"
    )
    pred_se3 = as_homogeneous(ext.squeeze(0).detach().float().cpu())
    gt_se3 = as_homogeneous(torch.from_numpy(np.asarray(gt_extrinsics)).float())
    res = compute_pose(pred_se3, gt_se3)
    return {k: float(v) for k, v in res.items()}


def evaluate_probes(student, scene_data, probe_pairs, teacher_caches, device="cuda") -> Dict:
    """Probe metrics for the CURRENT student state (no grad).
    Returns dict of scalars averaged over the scene's probe pairs."""
    base = get_base_vggt(student)
    records = []
    with torch.no_grad():
        for pair, cache in zip(probe_pairs, teacher_caches):
            images8, images4 = load_pair_images(scene_data, pair["teacher_frames"], device)
            feats24, psi, preds = student_preds(student, images4)
            depth = preds["depth"].squeeze(0).squeeze(-1).float().cpu().numpy()  # [4,H,W]
            gt = load_probe_gt(scene_data, pair["student_frames"], depth.shape[-2:])
            e, n = e_depth(depth, gt)
            mses = feature_mses(base.depth_head, feats24, cache, cache["patch_hw"])
            gt_ext = np.asarray(scene_data.extrinsics)[pair["student_frames"]]
            auc = pose_auc(preds["pose_enc"].float(), gt_ext)
            records.append({"e_depth": e, "n_valid": n, **mses,
                            **{f"probe_{k}": v for k, v in auc.items()}})
    keys = records[0].keys()
    return {k: float(np.nanmean([r[k] for r in records])) for k in keys}
