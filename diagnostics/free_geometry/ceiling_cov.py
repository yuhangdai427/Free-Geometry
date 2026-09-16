#!/usr/bin/env python3
"""Coverage + longer-context ceiling on the same 72 8->4 pairs (protocol of
patch_interp_f1).

New depth variants per pair (a0 / a100=teacher-8v already exist):
- c2:   student + C2_b5_rel step100 LoRA retrained on THIS manifest's 10 train
        pairs (in-sample on 10, held-out on the 2 probe pairs)
- c2p:  student + C2P_permgate step100 (permuted-conf gate control)
- c3r:  student + C3r step100 (trained on the disjoint 20-scene manifest pairs
        -> fully held-out on all 12 pairs here)
- t16:  teacher direct depth on shared 4 views with 16-view context
        (pair's 8 frames + 8 seeded pool extras)
- t32:  same with 32-view context (+24 extras; beyond the <=16v training
        protocol, trend point only)
Also caches teacher 8v conf4 per pair for the conf-gate mechanism analysis
(does teacher conf know which patches the teacher improves?).

Coverage = (trained-student gain vs a0) / (substitution ceiling a100-a0),
reported on all 12 pairs and on the subset where the ceiling is positive.

Outputs: artifacts/diagnostics/patch_interp_f1/ceiling_rows.jsonl,
conf/{scene}__p{pi}.npz, depths/*__{c2,c2p,c3r,t16,t32}.npz
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common import (  # noqa: E402
    STUDENT_INDICES, TAP_LAYERS, get_scene_data, load_image_model, load_manifest,
    stable_seed,
)
import modeling as M  # noqa: E402
from train_arms import perpatch_logres2  # noqa: E402
from depth_metrics import frame_metrics  # noqa: E402
import patch_interp_f1 as P  # noqa: E402

OUT = P.OUT
CK_V2 = "artifacts/diagnostics/bakeoff_v2_transductive/ckpts"
CK_C3R = "artifacts/diagnostics/final_c3r/ckpts"


def load_peft_into(student, peft_dir):
    """Overwrite LoRA A/B weights in-place from adapter_model.safetensors
    (avoids PEFT adapter-name clashes on repeated loads)."""
    from safetensors.torch import load_file
    sd = load_file(os.path.join(peft_dir, "adapter_model.safetensors"))
    base = M.get_base_vggt(student)
    mods = {}
    for name, m in base.named_modules():
        if hasattr(m, "lora_A") and hasattr(m, "lora_B"):
            mods[name] = m
    n = 0
    for k, v in sd.items():
        assert k.startswith("base_model.model.")
        rest = k[len("base_model.model."):]
        mod_path, kind = rest.rsplit(".lora_", 1)
        kind = "lora_" + kind.rsplit(".weight", 1)[0]
        hits = [m for nm, m in mods.items() if nm.endswith(mod_path)]
        assert len(hits) == 1, f"{k}: {len(hits)} module hits"
        ad = getattr(hits[0], kind)
        key = "default" if "default" in ad else list(ad.keys())[0]
        ad[key].weight.data.copy_(v.to(ad[key].weight.device))
        n += 1
    assert n == len(sd), (n, len(sd))
    return n


@torch.no_grad()
def student_depth(student, base, images4):
    feats24, psi = student._get_aggregator()(images4)
    feats24 = [f.float() for f in feats24]
    d, _ = M.replay_depth_nograd(base, feats24, images4, psi)
    return d.squeeze(0).squeeze(-1).float().cpu().numpy()


@torch.no_grad()
def teacher_ctx_depth(teacher, imagesN, shared_slots, images4):
    feats24, psi = M.aggregator_all(teacher, imagesN)
    slots = [None] * 24
    for l in TAP_LAYERS:
        slots[l] = feats24[l][:, shared_slots].float().contiguous()
    d, _ = M.replay_depth_nograd(teacher, slots, images4, psi)
    return d.squeeze(0).squeeze(-1).float().cpu().numpy()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="artifacts/diagnostics/bakeoff_v2_transductive/scene_manifest.json")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--max_pairs", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--variants", nargs="*", default=["c2", "c2p", "c3r", "t16", "t32"])
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    scenes = args.scenes or sorted(manifest["scenes"])
    teacher = M.load_teacher(args.device)
    student = M.load_student(args.device)
    base = M.get_base_vggt(student)
    os.makedirs(os.path.join(OUT, "conf"), exist_ok=True)
    tag = "ceiling" if set(args.variants) == {"c2", "c2p", "c3r", "t16", "t32"} else "gate"
    rows_path = os.path.join(OUT, f"{tag}_rows.jsonl")

    from common import frames_with_gt_depth  # local import ok

    with open(rows_path, "w") as rows_f:
        for scene in scenes:
            sc = manifest["scenes"][scene]
            scene_data = get_scene_data(scene)
            ok, _ = frames_with_gt_depth(scene)
            pool = [i for i in ok if i not in set(sc["eval32_frames"])]
            pairs = sc["train_pairs"] + sc["probe_pairs"]
            if args.max_pairs:
                pairs = pairs[: args.max_pairs]

            arm_ckpts = {}
            if "c2" in args.variants:
                arm_ckpts["c2"] = os.path.join(CK_V2, scene, "C2_b5_rel", "step100_lora_peft")
            if "c2p" in args.variants:
                arm_ckpts["c2p"] = os.path.join(CK_V2, scene, "C2P_permgate", "step100_lora_peft")
            if "c3r" in args.variants:
                arm_ckpts["c3r"] = os.path.join(CK_C3R, scene, "C3r", "step100_lora_peft")
            if "sc" in args.variants:
                arm_ckpts["sc"] = os.path.join(CK_V2, scene, "C2S_studconf", "step100_lora_peft")
            if "sci" in args.variants:
                arm_ckpts["sci"] = os.path.join(CK_V2, scene, "C2SI_invconf", "step100_lora_peft")
            if "ohem" in args.variants:
                arm_ckpts["ohem"] = os.path.join(CK_V2, scene, "C2O_ohem", "step100_lora_peft")

            for pi, pair in enumerate(pairs):
                images8, images4 = M.load_pair_images(scene_data, pair["teacher_frames"])
                H, W = images8.shape[-2:]
                ph, pw = H // 14, W // 14
                gt4 = M.load_probe_gt(scene_data, pair["student_frames"], (H, W))
                valid = np.isfinite(gt4) & (gt4 > 0)
                variant_depths = {}

                # teacher 8v conf (gate analysis)
                tcache = M.cache_teacher_pair(teacher, images8, (ph, pw))
                conf4 = tcache["conf4"].float().cpu().numpy()
                if conf4.ndim == 5:
                    conf4 = conf4.squeeze(2)
                np.savez_compressed(os.path.join(OUT, "conf", f"{scene}__p{pi:02d}.npz"),
                                    conf=conf4.astype(np.float16))
                del tcache

                # trained-student arms
                for name, ck in arm_ckpts.items():
                    M.reset_lora_(student)
                    load_peft_into(student, ck)
                    student.eval()
                    variant_depths[name] = student_depth(student, base, images4)

                # longer-context teacher
                t8 = pair["teacher_frames"]
                rest = [i for i in pool if i not in set(t8)]
                if "t16" in args.variants or "t32" in args.variants:
                    import random
                    rng = random.Random(stable_seed("ctxx", scene, pi))
                    extra24 = rng.sample(rest, 24)
                for tag, n_extra in (("t16", 8), ("t32", 24)):
                    if tag not in args.variants:
                        continue
                    framesN = list(t8) + extra24[:n_extra]
                    imgs = [load_image_model(scene_data.image_files[i]) for i in framesN]
                    arr = np.stack(imgs, 0)
                    imagesN = torch.from_numpy(arr).permute(0, 3, 1, 2).float().unsqueeze(0).to(args.device)
                    variant_depths[tag] = teacher_ctx_depth(
                        teacher, imagesN, STUDENT_INDICES, images4)
                    del imagesN
                    torch.cuda.empty_cache()

                for name, d in variant_depths.items():
                    r = np.log(np.clip(d, 1e-6, None)) - np.log(np.clip(gt4, 1e-6, None))
                    c = float(r[valid].mean())
                    ds = (d * np.exp(-c)).astype(np.float32)
                    absrels, d125s = [], []
                    for k in range(4):
                        fm = frame_metrics(ds[k], gt4[k])
                        if fm:
                            absrels.append(fm["absrel"])
                            d125s.append(fm["d125"])
                    pp = perpatch_logres2(d, gt4, (ph, pw))
                    np.savez_compressed(
                        os.path.join(OUT, "depths", f"{scene}__p{pi:02d}__{name}.npz"),
                        depth=ds.astype(np.float16))
                    rows_f.write(json.dumps({
                        "scene": scene, "pair": pi, "variant": name,
                        "absrel": float(np.mean(absrels)), "d125": float(np.mean(d125s)),
                        "logres2": float(np.nanmean(pp)), "scale": float(np.exp(-c))}) + "\n")
                    rows_f.flush()
                print(f"[{scene} p{pi}] {sorted(variant_depths)} done", flush=True)
                del variant_depths
                torch.cuda.empty_cache()
    print("[gpu] done", flush=True)


if __name__ == "__main__":
    main()
