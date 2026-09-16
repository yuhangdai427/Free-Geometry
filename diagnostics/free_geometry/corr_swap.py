#!/usr/bin/env python3
"""P1 correspondence-swap probe: is cross-frame info useful, and which
selection mechanism (same-position / similarity-topK / attention-top)?

Per shared-view sampled patch: replace the student's features (all 4 tap
layers) at that position with teacher features chosen by three mechanisms,
replay the frozen depth head, measure per-patch depth error vs GT change.
  (a) same-position teacher features (D1 baseline)
  (b) similarity top-K matched teacher EXTRA-view features (deployed CF's mechanism)
  (c) attention-top matched teacher extra-view features (teacher's own global
      attention logits via the block's qkv projection - the model's QK mechanism)
Output: artifacts/diagnostics/corr_swap/summary.md
"""

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common import STUDENT_INDICES, TAP_LAYERS, get_scene_data, load_manifest  # noqa: E402
import modeling as M  # noqa: E402
from train_arms import perpatch_logres2  # noqa: E402

EXTRA_INDICES = [i for i in range(8) if i not in STUDENT_INDICES]
OUT = "artifacts/diagnostics/corr_swap"
N_PATCH = 64
K_TOP = 4
PAIRS_PER_SCENE = 8


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = load_manifest("artifacts/diagnostics/bakeoff_v2_transductive/scene_manifest.json")
    teacher = M.load_teacher("cuda")
    student = M.load_student("cuda")
    base = M.get_base_vggt(student)
    gblk = teacher.aggregator.global_blocks[-1]
    rows = []
    for scene in sorted(manifest["scenes"]):
        sc = manifest["scenes"][scene]
        scene_data = get_scene_data(scene)
        M.reset_lora_(student); student.eval()
        for pi, pair in enumerate(sc["train_pairs"][:PAIRS_PER_SCENE]):
            images8, images4 = M.load_pair_images(scene_data, pair["teacher_frames"])
            ph, pw = images8.shape[-2] // 14, images8.shape[-1] // 14
            P = ph * pw
            tfeats, tpsi = M.aggregator_all(teacher, images8)
            # (c) 用 forward hook 抓 teacher 全局注意力块的真实输入（1024 维），
            # 经其 qkv 计算注意力 logits —— 模型自己的 QK 对应机制
            captured = {}
            def _pre_hook(module, args):
                captured["x"] = args[0].detach()
            hk = gblk.attn.register_forward_pre_hook(_pre_hook)
            tfeats, tpsi = M.aggregator_all(teacher, images8)
            hk.remove()
            sfeats, spsi = student._get_aggregator()(images4)
            sfeats = [f.float() for f in sfeats]
            sfeats = [f.float() for f in sfeats]
            gt4 = M.load_probe_gt(scene_data, pair["student_frames"], images8.shape[-2:])
            rng = torch.Generator(device="cuda").manual_seed(hash((scene, pi)) % 2**31)
            patch_ids = torch.randperm(P, generator=rng, device="cuda")[:N_PATCH]

            def replay_with(mod_fn):
                slots = [None] * 24
                for l in TAP_LAYERS:
                    f = sfeats[l].float().clone()
                    mod_fn(f, l)
                    slots[l] = f.contiguous()
                d4, _ = M.replay_depth_nograd(base, slots, images4, spsi)
                return d4.squeeze(0).squeeze(-1).float().cpu().numpy()

            # post-norm teacher tokens for selection (L23)
            ht8 = M.to_norm(base.depth_head, M.to_patch(tfeats[23][:, :, :, :].float()))
            ht_ex = ht8[:, EXTRA_INDICES].reshape(1, -1, ht8.shape[-1])
            # (c) teacher 全局注意力 logits（真实块输入，[B,SN,1024]）
            with torch.autocast(device_type="cuda", enabled=False):
                x = captured["x"].float()
                if x.dim() == 4:
                    x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
                qkv = gblk.attn.qkv(x).reshape(x.shape[0], x.shape[1], 3,
                                               gblk.attn.num_heads, -1).permute(2, 0, 3, 1, 4)
                q_a, k_a = qkv.unbind(0)[0], qkv.unbind(0)[1]       # [B,h,SN,d]
                hd = q_a.shape[-1]
                att = (q_a @ k_a.transpose(-1, -2)) / (hd ** 0.5)
                att = att.mean(1)[0]                            # [SN,SN]
            Nv = att.shape[0] // 8
            ex_mask = torch.zeros(att.shape[0], dtype=torch.bool, device=att.device)
            for e in EXTRA_INDICES:
                ex_mask[e * Nv + 5:(e + 1) * Nv] = True
            att_ex = att[:, ex_mask]                            # [SN, n_extra]
            ex_token_ids = torch.nonzero(ex_mask).squeeze(-1)   # [n_extra]
            top_c_view, top_c_tok = [], []
            pid_cpu = patch_ids.cpu()
            for s in STUDENT_INDICES:
                q_idx = s * Nv + 5 + pid_cpu
                top1 = att_ex[q_idx].topk(1, dim=-1).indices.squeeze(-1)
                gtok = ex_token_ids[top1]
                top_c_view.append(gtok // Nv)
                top_c_tok.append(gtok % Nv)
            top_c_view = torch.stack(top_c_view)                 # [4,N]
            top_c_tok = torch.stack(top_c_tok)                   # [4,N] (已含 patch_start)

            def swap_none(f, l):
                pass

            def swap_same(f, l):
                for vi, si in enumerate(STUDENT_INDICES):
                    f[0, vi, M.PATCH_START_IDX + patch_ids, :] = \
                        tfeats[l][0, si, M.PATCH_START_IDX + patch_ids, :].float()

            # (b) similarity top-K targets per patch (view-major flattened)
            hs_sel = M.to_norm(base.depth_head, M.to_patch(sfeats[23].float()))
            q = torch.nn.functional.normalize(hs_sel[0, :, patch_ids, :].reshape(-1, hs_sel.shape[-1]), dim=-1)
            bank = torch.nn.functional.normalize(ht_ex[0], dim=-1)
            simb = q @ bank.T                                        # [4N, 4P]
            top_b = simb.topk(K_TOP, dim=-1).indices                 # [4N,K]
            ex_flat = EXTRA_INDICES

            def swap_sim(f, l):
                tex = tfeats[l][:, ex_flat][:, :, M.PATCH_START_IDX:, :].float().reshape(-1, tfeats[l].shape[-1])
                tgt_all = tex[top_b.reshape(-1)].reshape(4 * N_PATCH, K_TOP, -1).mean(1)
                for vi in range(4):
                    f[0, vi, M.PATCH_START_IDX + patch_ids, :] = tgt_all[vi * N_PATCH:(vi + 1) * N_PATCH]

            # (c) attention top-1 targets per patch — 用上一步算好的 token 索引
            def swap_attn(f, l):
                for vi in range(4):
                    for k in range(N_PATCH):
                        ev = int(top_c_view[vi, k])
                        tk = int(top_c_tok[vi, k])
                        f[0, vi, M.PATCH_START_IDX + patch_ids[k], :] = \
                            tfeats[l][0, ev, tk, :].float()

            d0 = replay_with(swap_none)
            res = {}
            for name, fn in (("same", swap_same), ("sim", swap_sim), ("attn", swap_attn)):
                res[name] = replay_with(fn)
            pp0 = perpatch_logres2(d0, gt4, (ph, pw))
            pid = patch_ids.cpu().numpy()
            for name in ("same", "sim", "attn"):
                pp = perpatch_logres2(res[name], gt4, (ph, pw))
                e0 = pp0[:, pid][np.isfinite(pp0[:, pid])]
                e1 = pp[:, pid][np.isfinite(pp[:, pid])]
                m = np.isfinite(pp0[:, pid]) & np.isfinite(pp[:, pid])
                dlt = (pp0[:, pid][m] - pp[:, pid][m])
                rows.append(dict(scene=scene, pair=pi, mech=name,
                                 better=float((dlt > 0).mean()), dmean=float(dlt.mean())))
            print(f"[{scene} p{pi}] " + " ".join(
                f"{n}: better {rows[-3+k]['better']:.2f} d {rows[-3+k]['dmean']:+.5f}"
                for k, n in enumerate(("same", "sim", "attn"))), flush=True)
            del tfeats, sfeats
            torch.cuda.empty_cache()
    import collections, statistics
    agg = collections.defaultdict(lambda: [[], []])
    for r in rows:
        agg[r["mech"]][0].append(r["better"]); agg[r["mech"]][1].append(r["dmean"])
    lines = ["# P1 correspondence swap (6 scenes x 8 pairs x 64 patches)", "",
             "| mechanism | frac patches improved | mean Δerr |", "|---|---|---|"]
    for name in ("same", "sim", "attn"):
        lines.append(f"| {name} | {statistics.mean(agg[name][0]):.3f} | {statistics.mean(agg[name][1]):+.6f} |")
    lines += ["", "if sim/attn >> same: cross-frame correspondence info is useful and selection matters"]
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
