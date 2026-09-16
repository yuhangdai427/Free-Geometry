#!/usr/bin/env python3
"""P3: patch-token alpha interpolation restricted by layer set.

cam_interp (camera token, all 24 layers) was monotone +0.112. D1 (patch
tokens, all 4 tap layers) was mixed (5/12). feature_traceback says rescued
patches are rewritten at L17+L23. Test whether patch interpolation becomes
monotone when restricted to {17,23} vs {4,11,17,23} vs {23}, with sham
(teacher patch tokens from a different pair; must degrade).

Output: artifacts/diagnostics/patch_interp/summary.md
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

OUT = "artifacts/diagnostics/patch_interp"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
LAYER_SETS = {"L1723": [17, 23], "all4": [4, 11, 17, 23], "L23": [23]}


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = load_manifest("artifacts/diagnostics/bakeoff_v2_transductive/scene_manifest.json")
    teacher = M.load_teacher("cuda")
    student = M.load_student("cuda")
    base = M.get_base_vggt(student)
    rows = []
    for scene in sorted(manifest["scenes"]):
        sc = manifest["scenes"][scene]
        scene_data = get_scene_data(scene)
        M.reset_lora_(student); student.eval()
        pairs = sc["train_pairs"] + sc["probe_pairs"]
        for pi, pair in enumerate(pairs):
            images8, images4 = M.load_pair_images(scene_data, pair["teacher_frames"])
            ph, pw = images8.shape[-2] // 14, images8.shape[-1] // 14
            tfeats, tpsi = M.aggregator_all(teacher, images8)
            sfeats, spsi = student._get_aggregator()(images4)
            sfeats = [f.float() for f in sfeats]
            sfeats = [f.float() for f in sfeats]
            pair_sham = pairs[(pi + 3) % len(pairs)]
            images8_sh, _ = M.load_pair_images(scene_data, pair_sham["teacher_frames"])
            tfeats_sh, _ = M.aggregator_all(teacher, images8_sh)
            gt4 = M.load_probe_gt(scene_data, pair["student_frames"], images8.shape[-2:])

            def err_with(t_src, alpha, layers):
                slots = [None] * 24
                for l in TAP_LAYERS:
                    f = sfeats[l].float().clone()
                    if l in layers:
                        f[:, :, M.PATCH_START_IDX:, :] = (
                            (1 - alpha) * f[:, :, M.PATCH_START_IDX:, :]
                            + alpha * t_src[l][:, STUDENT_INDICES, M.PATCH_START_IDX:, :].float())
                    slots[l] = f.contiguous()
                d4, _ = M.replay_depth_nograd(base, slots, images4, spsi)
                pp = perpatch_logres2(d4.squeeze(0).squeeze(-1).float().cpu().numpy(), gt4, (ph, pw))
                return float(np.nanmean(pp))

            for lsname, layers in LAYER_SETS.items():
                for sham, tsrc in ((0, tfeats), (1, tfeats_sh)):
                    errs = [err_with(tsrc, a, layers) for a in ALPHAS]
                    rows.append(dict(scene=scene, pair=pi, ls=lsname, sham=sham, errs=errs))
                r0, r1 = rows[-2], rows[-1]
                print(f"[{scene} p{pi} {lsname}] real " +
                      " ".join(f"{e:.4f}" for e in r0["errs"]) + " | sham " +
                      " ".join(f"{e:.4f}" for e in r1["errs"]), flush=True)
            del tfeats, sfeats, tfeats_sh
            torch.cuda.empty_cache()
    import collections, statistics
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["ls"], r["sham"])].append(r["errs"])
    lines = ["# patch alpha interpolation by layer set (6 scenes x 12 pairs)", "",
             "| layers | sham | a0 | a.25 | a.5 | a.75 | a1 |", "|---|---|---|---|---|---|---|"]
    for lsname in LAYER_SETS:
        for sham in (0, 1):
            m = [statistics.mean(x[i] for x in agg[(lsname, sham)]) for i in range(len(ALPHAS))]
            lines.append(f"| {lsname} | {sham} | " + " | ".join(f"{v:.4f}" for v in m) + " |")
    lines += ["", "monotone decrease in real & increase in sham => path valid for that layer set"]
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
