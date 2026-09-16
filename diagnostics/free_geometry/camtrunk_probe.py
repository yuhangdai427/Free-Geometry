#!/usr/bin/env python3
"""CamTrunk substitution probe: does the camera-head trunk's INTERNAL iterative
feature carry the 8v-context pose information (the missing ~0.10 between
cam_interp's +0.112 ceiling and CamRel's +0.0134)?

Frozen-head replay only, no training. Per pair (6 scenes x 2 probe pairs):
- base:    student L23 camera tokens -> trunk_fn(4 iters) -> AUC0
- camsWP:  teacher shared L23 camera tokens as input -> AUC_cam
           (isolates "input swap at L23 only" - cam_interp swapped all 24
           layers in the aggregator, this is the head-input-only analogue)
- trNK1..4: student trunk run, but from iteration k onwards the shared-frame
           rows of `pose_tokens_modulated` (post-trunk) are replaced with the
           teacher's (teacher trunk runs on 8 views; shared rows sliced) ->
           AUC_trunk(k). Cumulative context injection into the iteration.
- sham:    same as trNK1 but with another pair's teacher features -> collapse

Also records per-round ||f_s - f_t|| MSE on shared rows and per-round pose MSE
(the "where does context enter" curve) and teacher-student final pose gap.

Output: artifacts/diagnostics/camtrunk_probe/summary.md
"""

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common import STUDENT_INDICES, get_scene_data, load_manifest  # noqa: E402
import modeling as M  # noqa: E402

OUT = "artifacts/diagnostics/camtrunk_probe"


@torch.no_grad()
def trunk_replay(head, pose_tokens, n_iter=4, swap_rows=None, swap_src=None, swap_from=99):
    """Replay camera_head.trunk_fn. pose_tokens [B,S,C] (post token_norm).

    swap_rows: row indices to replace; swap_src: generator yielding teacher's
    per-iteration post-trunk features [B,S,C] aligned to those rows;
    swap_from: first iteration (0-based) at which replacement applies.
    Returns (pose_list, per_iter_feats)."""
    B, S, C = pose_tokens.shape
    pred_pose_enc = None
    pose_list, feats = [], []
    for it in range(n_iter):
        if pred_pose_enc is None:
            module_input = head.embed_pose(head.empty_pose_tokens.expand(B, S, -1))
        else:
            pred_pose_enc = pred_pose_enc.detach()
            module_input = head.embed_pose(pred_pose_enc)
        shift, scale, gate = head.poseLN_modulation(module_input).chunk(3, dim=-1)
        from vggt.heads.camera_head import modulate
        ptm = gate * modulate(head.adaln_norm(pose_tokens), shift, scale) + pose_tokens
        ptm = head.trunk(ptm)
        if it >= swap_from and swap_src is not None:
            ptm = ptm.clone()
            ptm[:, swap_rows] = swap_src[it][:, swap_rows]
        feats.append(ptm)
        delta = head.pose_branch(head.trunk_norm(ptm))
        pred_pose_enc = delta if pred_pose_enc is None else pred_pose_enc + delta
        from vggt.heads.camera_head import activate_pose
        pose_list.append(activate_pose(pred_pose_enc, trans_act=head.trans_act,
                                       quat_act=head.quat_act, fl_act=head.fl_act))
    return pose_list, feats


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = load_manifest("artifacts/diagnostics/bakeoff_v2_transductive/scene_manifest.json")
    teacher = M.load_teacher("cuda")
    student = M.load_student("cuda")
    base = M.get_base_vggt(student)
    head = base.camera_head
    rows = []
    for scene in sorted(manifest["scenes"]):
        sc = manifest["scenes"][scene]
        scene_data = get_scene_data(scene)
        M.reset_lora_(student)
        student.eval()
        pairs = sc["probe_pairs"]
        all_pairs = sc["train_pairs"] + sc["probe_pairs"]
        for pi, pair in enumerate(pairs):
            images8, images4 = M.load_pair_images(scene_data, pair["teacher_frames"])
            gt_ext = np.asarray(scene_data.extrinsics)[pair["student_frames"]]
            tfeats, _ = M.aggregator_all(teacher, images8)
            sfeats, _ = student._get_aggregator()(images4)
            sham_pair = all_pairs[(all_pairs.index(pair) + 3) % len(all_pairs)]
            images8_sh, _ = M.load_pair_images(scene_data, sham_pair["teacher_frames"])
            tfeats_sh, _ = M.aggregator_all(teacher, images8_sh)

            ct = head.token_norm(tfeats[23][:, :, 0, :].float())          # [1,8,C]
            cs = head.token_norm(sfeats[23][:, :, 0, :].float())          # [1,4,C]
            ct_sh = head.token_norm(tfeats_sh[23][:, :, 0, :].float())    # [1,8,C]

            pose_t, feats_t = trunk_replay(head, ct)                       # teacher 8v trunk
            pose_s, feats_s = trunk_replay(head, cs)                       # student 4v trunk
            pose_sh, feats_sh = trunk_replay(head, ct_sh)
            DST_ROWS = [0, 1, 2, 3]
            feats_t_sh = [f[:, STUDENT_INDICES].contiguous() for f in feats_t]
            feats_sh_sh = [f[:, STUDENT_INDICES].contiguous() for f in feats_sh]

            # (a) input swap: teacher shared camera tokens as head input
            pose_cam, _ = trunk_replay(head, ct[:, STUDENT_INDICES].contiguous())

            res = {}
            res["base"] = M.pose_auc(pose_s[-1].float(), gt_ext)["auc03"]
            res["camsWP"] = M.pose_auc(pose_cam[-1].float(), gt_ext)["auc03"]
            res["teacher8v"] = M.pose_auc(
                pose_t[-1][:, STUDENT_INDICES].float(), gt_ext)["auc03"]

            # (b) cumulative trunk swap from iteration k
            for k in range(1, 5):
                pose_k, _ = trunk_replay(head, cs, swap_rows=DST_ROWS,
                                         swap_src=feats_t_sh, swap_from=k - 1)
                res[f"trNK{k}"] = M.pose_auc(pose_k[-1].float(), gt_ext)["auc03"]
            # (c) sham
            pose_shm, _ = trunk_replay(head, cs, swap_rows=DST_ROWS,
                                       swap_src=feats_sh_sh, swap_from=0)
            res["sham"] = M.pose_auc(pose_shm[-1].float(), gt_ext)["auc03"]

            # per-round feature + pose divergence (context entry curve)
            fmse, pmse = [], []
            for it in range(4):
                fmse.append(float(((feats_s[it] - feats_t[it][:, STUDENT_INDICES]) ** 2).mean()))
                pmse.append(float(((pose_s[it] - pose_t[it][:, STUDENT_INDICES]) ** 2).mean()))
            rows.append({"scene": scene, "pair": pi, **res, "fmse": fmse, "pmse": pmse})
            print(f"[{scene} pr{pi}] " + " ".join(f"{k}={v:.4f}" for k, v in res.items())
                  + f" fmse={[round(x, 5) for x in fmse]}", flush=True)
            del tfeats, sfeats, tfeats_sh
            torch.cuda.empty_cache()

    import statistics as st
    keys = ["base", "camsWP", "trNK1", "trNK2", "trNK3", "trNK4", "sham", "teacher8v"]
    lines = ["# CamTrunk substitution probe (6 scenes x 2 probe pairs)", "",
             "| cond | AUC@3 |", "|---|---|"]
    for k in keys:
        lines.append(f"| {k} | {st.mean(r[k] for r in rows):.4f} |")
    lines += ["", "## per-round feature MSE (student vs teacher, shared rows)", "",
              "| iter | mean fMSE |", "|---|---|"]
    for it in range(4):
        lines.append(f"| {it + 1} | {st.mean(r['fmse'][it] for r in rows):.5f} |")
    lines += ["", "reading: if trNK(k) - camsWP >= +0.02 for some k, trunk-internal features "
              "carry context info beyond the head input -> build CamTrunk loss arm; "
              "if trNK ~= camsWP, the trunk is input-determined (bracket redundancy) -> close.",
              "", "## per-pair detail", ""]
    for r in rows:
        lines.append(f"- {r['scene']} pr{r['pair']}: " +
                     " ".join(f"{k}={r[k]:.4f}" for k in keys))
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:20]), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
