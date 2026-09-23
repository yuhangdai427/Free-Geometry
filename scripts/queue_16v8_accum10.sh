#!/bin/bash
# 16:8 (teacher 16, student 8) × 100 pairs × gradient accumulation (accum=10,
# equivalent to batch_size=10) × 100 updates = 10 epochs.
# SmoothL1+2cos+camrel, with mask. Scene: 09c1414f1b (scannetpp).
# Then evaluate: benchmark (100 manifest frames) + 10 training pairs AUC/F1.
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python
SC=09c1414f1b
MAN=artifacts/diagnostics/final_protocol/scannetpp/scene_manifest.json
OUT=workspace/spp_16v8_accum10

echo "[16v8] training start $(date '+%F %T')"
$PY scripts/train_pw0_accum.py \
    --dataset scannetpp --scene $SC \
    --vggt_sync --half_mode vggt --camrel \
    --n_train 100 --n_shared 8 \
    --updates 100 --accum 10 \
    --manifest $MAN \
    --out $OUT > logs/spp_16v8_train.log 2>&1
RC=$?
echo "[16v8] training done rc=$RC $(date '+%F %T')"

if [ $RC -ne 0 ]; then
    echo "[16v8] TRAINING FAILED"
    tail -5 logs/spp_16v8_train.log
    exit 1
fi

# benchmark eval (trained model already evaluated in training script via cam_dec)
grep "cam_dec" logs/spp_16v8_train.log
echo "[16v8] benchmark done"

# training-pair eval (10 pairs, 8-frame student)
echo "[16v8] pair eval start $(date '+%F %T')"
$PY - << 'PYEOF' > logs/spp_16v8_paieval.log 2>&1
import sys, os, json, shutil
sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "diagnostics/free_geometry")
import numpy as np, torch
import common as fg_common
from train_da3_protocol import make_dataset, get_scene_data, evaluate_scene
from depth_anything_3.test_time_adaption import protocol_v1 as P
from depth_anything_3.bench.utils import compute_pose
from depth_anything_3.utils.geometry import as_homogeneous

SC = "09c1414f1b"
fg_common.set_dataset("scannetpp"); ds = make_dataset("scannetpp")
sd = get_scene_data(SC)
files = list(sd.image_files)
gt_ext = np.asarray(sd.extrinsics)[:, :3, :].astype(np.float32)

proto = P.build_scene_protocol(files, SC, dataset="scannetpp",
                                n_train=100, n_shared=8, teacher_N=16)
pairs = proto["train_pairs"]
print(f"n_shared={proto['train_pairs'][0]['n_shared']}, "
      f"student_frames={pairs[0]['student_frames']}")

trained = P.create_student("model_weights/DA3-GIANT-1.1", train_camera_token=False)
P.reset_lora_(trained)
ckpt = "workspace/spp_16v8_accum10/ckpts/09c1414f1b/c2m_final_lora.pt"
trained.load_lora_weights(ckpt)
trained.to("cuda").eval()
untrained = P.create_student("model_weights/DA3-GIANT-1.1", train_camera_token=False)
P.reset_lora_(untrained); untrained.to("cuda").eval()

def fwd(model, im):
    with torch.no_grad():
        _, ext, _ = P.student_forward_c2m(model, im)
    return ext[0].detach().cpu().numpy()

print(f"\n{'pair':>5} {'model':>6} {'cond':>8} {'AUC':>8}")
for pi in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
    pair = pairs[pi]
    tf, sf = pair["teacher_frames"], pair["student_frames"]
    slots = [tf.index(f) for f in sf]
    gt8 = gt_ext[sf]  # 8 student frames
    gt16 = gt_ext[tf]  # all 16 teacher frames

    im16 = P.load_images_da3([files[i] for i in tf]).unsqueeze(0).to("cuda")
    im8 = im16[0, slots].unsqueeze(0)
    ph, pw = im8.shape[-2]//14, im8.shape[-1]//14
    gen = torch.Generator(device="cuda").manual_seed(
        P.stable_seed("mask", SC, 0, pi, 0))
    im8m, _ = P.mask_image_blocks(im8, 0.5, (ph,pw), gen)

    for mtag, model in [("tr", trained), ("un", untrained)]:
        # 8-frame masked student
        ext8 = fwd(model, im8m)
        a8 = float(compute_pose(
            as_homogeneous(torch.from_numpy(ext8).float()),
            as_homogeneous(torch.from_numpy(gt8).float())).auc03)
        print(f"{pi:>5} {mtag:>6} {'mask8':>8} {a8:>8.4f}", flush=True)

        # 16-frame teacher context (extract student slots)
        ext16 = fwd(model, im16)
        a16 = float(compute_pose(
            as_homogeneous(torch.from_numpy(ext16[slots]).float()),
            as_homogeneous(torch.from_numpy(gt8).float())).auc03)
        print(f"{pi:>5} {mtag:>6} {'teach16':>8} {a16:>8.4f}", flush=True)

    del im16, im8, im8m

# benchmark: untrained model on 100 manifest frames
man = json.load(open("artifacts/diagnostics/final_protocol/scannetpp/scene_manifest.json"))
bench = man["scenes"][SC]["eval32_frames"]
ev_un = evaluate_scene(untrained, sd, bench, scene=SC, dataset_obj=ds,
                        export_dir="/tmp/bench_un_16v8")
print(f"\nbenchmark untrained: AUC={ev_un['auc03']:.4f} F1={ev_un['recon_fscore']:.4f}", flush=True)
PYEOF
echo "[16v8] pair eval done rc=$? $(date '+%F %T')"
