#!/usr/bin/env python3
"""All-pairs step-0 rel-gap probe (2026-09-21): per-pair teacher-student
rel loss distribution on spp scenes, frozen model (no training).
Contrast set: 1ada7a0617 / 21d970d8de (TTA-damaged) vs bde1e479ad /
fb5a96b1a2 (TTA-benefited). Uses the ndispatch manifest pairs (16:4).
"""
import json, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "diagnostics", "free_geometry"))
os.chdir(os.path.join(ROOT, ".."))

import common
import modeling as M
import train_arms as T
from common import STUDENT_INDICES

common.set_dataset("scannetpp")
manifest = json.load(open("workspace/ndispatch/scannetpp/scene_manifest.json"))
SCENES = ["1ada7a0617", "21d970d8de", "bde1e479ad", "fb5a96b1a2"]

student = M.load_student()
teacher = M.load_teacher("cuda")

out = {}
for scene in SCENES:
    sc = manifest["scenes"][scene]
    scene_data = common.get_scene_data(scene)
    gaps = []
    with torch.no_grad():
        for pi, pair in enumerate(sc["train_pairs"]):
            images8, images4 = M.load_pair_images(scene_data, pair["teacher_frames"], "cuda")
            ph, pw = images8.shape[-2] // 14, images8.shape[-1] // 14
            cache = M.cache_teacher_pair(teacher, images8, (ph, pw))
            _, _, preds = M.student_preds(student, images4)
            gap, _ = T.loss_pose_rel(preds["pose_enc"],
                                      cache["pose_enc8"][:, STUDENT_INDICES].float())
            gaps.append(float(gap))
            del images8, images4, cache
    g = np.array(gaps)
    out[scene] = gaps
    print(f"{scene}: min={g.min():.5f} med={np.median(g):.5f} max={g.max():.5f} "
          f"p90={np.percentile(g,90):.5f} CV={g.std()/g.mean():.2f}")
    print("   " + " ".join(f"{v:.4f}" for v in gaps), flush=True)

json.dump(out, open("workspace/probe_spp2/pairgap_spp.json", "w"), indent=1)
print("-> workspace/probe_spp2/pairgap_spp.json")
