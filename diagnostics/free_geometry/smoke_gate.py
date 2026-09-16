#!/usr/bin/env python3
"""Minimal smoke gate (the compressed D0): run on ONE scene / ONE probe pair.

Checks, in order (any failure aborts the whole run):
1. 24-slot depth replay from cached full forward == direct model forward.
2. 8->4 teacher replay == direct teacher forward on the 4-view subset path
   is NOT asserted (camera head differs); instead assert depth-head per-frame
   consistency: teacher 8v sliced feats -> depth == teacher 4v-only depth.
3. Fresh student (LoRA B=0) == frozen teacher on the same 4-view input.
4. PROJECTED-MSE loss backprop reaches LoRA params (some grad nonzero).
5. AMP-vs-FP32 feature noise floor printed for reference.
"""

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common import STUDENT_INDICES, TAP_LAYERS, load_manifest
import modeling as M


def main():
    run_root = sys.argv[1] if len(sys.argv) > 1 else "artifacts/diagnostics/bakeoff_v1"
    manifest = load_manifest(os.path.join(run_root, "scene_manifest.json"))
    scene = sorted(manifest["scenes"])[0]
    sc = manifest["scenes"][scene]
    pair = sc["probe_pairs"][0]

    from common import get_scene_data
    scene_data = get_scene_data(scene)

    teacher = M.load_teacher()
    images8, images4 = M.load_pair_images(scene_data, pair["teacher_frames"])
    ph, pw = images8.shape[-2] // 14, images8.shape[-1] // 14
    print(f"scene={scene} images8={tuple(images8.shape)} patch_grid={ph}x{pw}")

    # 1. replay == direct forward (8 views, fp32, no autocast)
    feats24, psi = M.aggregator_all(teacher, images8)
    depth_replay, _ = M.replay_depth_nograd(teacher, feats24, images8, psi)
    with torch.autocast(device_type="cuda", enabled=False):
        direct = teacher(images8)
    depth_direct = direct["depth"]
    diff = (depth_replay - depth_direct).abs().max().item()
    rel = diff / depth_direct.abs().max().item()
    print(f"[1] replay vs direct depth: max_abs={diff:.3e} rel={rel:.3e}")
    assert rel < 1e-4, "depth replay mismatch"

    # 2. depth head is per-frame: decode-then-slice == slice-then-decode
    #    (8v-context features deliberately differ from a 4v-only forward;
    #    that difference IS the teacher signal, not a bug.)
    slots4 = [None] * 24
    for layer in TAP_LAYERS:
        slots4[layer] = feats24[layer][:, STUDENT_INDICES].contiguous()
    depth_sliced, _ = M.replay_depth_nograd(teacher, slots4, images4, psi)
    depth8_all, _ = M.replay_depth_nograd(teacher, feats24, images8, psi)
    depth8_sliced = depth8_all[:, STUDENT_INDICES]
    diff2 = (depth_sliced - depth8_sliced).abs().max().item()
    rel2 = diff2 / depth8_sliced.abs().max().item()
    # NOTE: rel ~1e-3 comes from conv algorithms chosen per batch size (B*S=8 vs
    # 4); in log-residual space this is a ~1e-6 E_depth noise floor, ~4 orders
    # below typical E_depth (~0.01-0.1). All diagnostics replay with S=4
    # consistently, so comparisons are unaffected.
    print(f"[2] slice-then-decode vs decode-then-slice: max_abs={diff2:.3e} rel={rel2:.3e}")
    assert rel2 < 5e-3, "depth head is not per-frame consistent"

    # 2b. teacher signal magnitude: 8v-context depth vs 4v-context depth
    with torch.autocast(device_type="cuda", enabled=False):
        direct4 = teacher(images4)
    rel2b = (depth_sliced - direct4["depth"]).abs().max().item() / direct4["depth"].abs().max().item()
    print(f"[2b] teacher 8v-context vs 4v-context depth rel max diff: {rel2b:.3e} (expected NONZERO)")

    # 3. fresh student == teacher on 4-view input
    student = M.load_student()
    with torch.no_grad():
        feats24_s, psi_s, preds_s = M.student_preds(student, images4)
    diff3 = (preds_s["depth"] - direct4["depth"]).abs().max().item()
    rel3 = diff3 / direct4["depth"].abs().max().item()
    print(f"[3] fresh student vs teacher depth: max_abs={diff3:.3e} rel={rel3:.3e}")
    assert rel3 < 1e-3, "student init != frozen baseline"

    # 4. PROJECTED MSE grad reaches LoRA
    cache = M.cache_teacher_pair(teacher, images8, (ph, pw))
    student.train()
    feats24_s, psi_s, _ = M.student_preds(student, images4)
    base = M.get_base_vggt(student)
    loss = 0.0
    for layer in TAP_LAYERS:
        hs = M.to_patch(feats24_s[layer].float())
        ht = M.to_patch(cache["feats"][layer][:, STUDENT_INDICES].float())
        zs = M.to_projected(base.depth_head, hs, layer, (ph, pw))
        zt = M.to_projected(base.depth_head, ht, layer, (ph, pw))
        loss = loss + torch.mean((zs - zt) ** 2)
    loss = loss / len(TAP_LAYERS)
    loss.backward()
    grads = [p.grad for p in student.get_trainable_params()]
    nz = sum(1 for g in grads if g is not None and g.abs().sum() > 0)
    total = len(grads)
    frac = nz / total
    print(f"[4] PROJECTED loss={loss.item():.4f}, LoRA params with nonzero grad: {nz}/{total} ({frac:.1%})")
    assert frac > 0.4, "gradient does not reach enough LoRA params"
    student.zero_grad()

    # 5. AMP vs FP32 feature noise floor
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
        out_amp, _ = teacher.aggregator(images8)
    diff5 = (out_amp[23][:, :, 5:, :].float() - feats24[23][:, :, 5:, :].float())
    denom = feats24[23][:, :, 5:, :].float().std().item()
    print(f"[5] AMP vs FP32 layer23 patch feats: std(diff)={diff5.std().item():.4e} vs feat std={denom:.4e} (ratio={diff5.std().item()/denom:.2e})")

    print("\nSMOKE GATE PASSED")


if __name__ == "__main__":
    main()
