# Scripts

This document contains the environment setup, training, and benchmarking commands for Free Geometry. The README stays focused on the paper, qualitative results, and demos.

## Environment Setup

```bash
conda create -n Free-Geo python=3.10 -y
conda activate Free-Geo

pip install xformers "torch>=2" torchvision
pip install -e .
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
pip install -e ".[app]"
pip install -e ".[all]"
```

## Training And Benchmarking

Only the maintained bash runners are documented here. The long direct `python scripts/train_*.py` and `python scripts/benchmark_*.py` commands are intentionally omitted.

### DA3

```bash
bash scripts/run_da3.sh train
bash scripts/run_da3.sh train_scannetpp
bash scripts/run_da3.sh train_hiroom
bash scripts/run_da3.sh train_7scenes
bash scripts/run_da3.sh train_eth3d

bash scripts/run_da3.sh benchmark_baseline
bash scripts/run_da3.sh benchmark_lora
bash scripts/run_da3.sh benchmark_all
bash scripts/run_da3.sh all
```

Default DA3 outputs:

- checkpoints: `checkpoints/all_da3_v2/{dataset}/`
- benchmark outputs: `workspace/all_da3_v2/`

### DA3 DTU-49

DTU-49 is a reconstruction-only benchmark. The dedicated runner compares the
DA3 baseline with a DTU-specific Free-Geometry LoRA using the DTU MVS protocol:
8 teacher views to 4 student views during adaptation, then all 49 views for
`recon_unposed` evaluation. It reports `acc`, `comp`, and `overall` in mm;
DTU-49 does not report pose AUC, pose F1, reconstruction F1, or CD.

```bash
bash scripts/run_da3_dtu49.sh check
bash scripts/run_da3_dtu49.sh baseline
bash scripts/run_da3_dtu49.sh train
bash scripts/run_da3_dtu49.sh free_geometry
# Or run baseline -> train -> Free-Geometry evaluation:
bash scripts/run_da3_dtu49.sh all
```

Default DTU-49 outputs:

- checkpoint: `checkpoints/da3_dtu49_free_geometry/epoch_2_lora.pt`
- baseline metrics: `workspace/da3_dtu49/baseline/seed43/metric_results/dtu_recon_unposed.json`
- Free-Geometry metrics: `workspace/da3_dtu49/free_geometry/epoch2_seed43/metric_results/dtu_recon_unposed.json`

### DA3 DTU Smoke On GPU 1

The ordered smoke runner validates both protocols on one native-resolution
scene each: DTU-64 pose (`scan105`, 64 views) first, then DTU-49 reconstruction
(`scan1`, 49 views). For each it runs a DA3 baseline, trains an independent
3-epoch Free-Geometry LoRA, and evaluates that LoRA. It writes checkpoints and
metrics under `checkpoints/da3_dtu_smoke_gpu1/` and `workspace/da3_dtu_smoke_gpu1/`.

```bash
CUDA_VISIBLE_DEVICES=1 GPU_ID=1 bash scripts/run_da3_dtu_smoke_gpu1.sh all
```

### DA3 Full DTU Evaluation On GPU 1

This runner evaluates all 13 DTU-64 pose scenes and all 22 DTU-49 reconstruction
scenes in order. Each protocol runs baseline, trains an independent 3-epoch
Free-Geometry LoRA over every scene, then evaluates the LoRA. Step-level losses
and epoch-average checkpoint losses are saved for auditability.

```bash
CUDA_VISIBLE_DEVICES=1 GPU_ID=1 bash scripts/run_da3_dtu_full_gpu1.sh all
```

### DA3 32-View Re-evaluation On GPU 1

After the full run has created `epoch_2_lora.pt` for both datasets, this
reuses those LoRA weights without training and caps every scene at 32 sampled
views for matched baseline and LoRA evaluation. The queue wrapper waits for the
active full tmux job to exit, so it does not interrupt or overlap GPU 1 work.

```bash
bash scripts/queue_da3_dtu_eval32_after_full_gpu1.sh
```

The full-run cleanup monitor removes each verified experiment's `model_results/`
as soon as its metric JSON is complete, retaining only metrics, checkpoints, and logs.

### VGGT

```bash
bash scripts/run_vggt.sh train
bash scripts/run_vggt.sh train_scannetpp
bash scripts/run_vggt.sh train_hiroom
bash scripts/run_vggt.sh train_7scenes
bash scripts/run_vggt.sh train_eth3d

bash scripts/run_vggt.sh benchmark_base
bash scripts/run_vggt.sh benchmark_lora
bash scripts/run_vggt.sh benchmark
bash scripts/run_vggt.sh all
```

Default VGGT outputs:

- checkpoints: `checkpoints/all_vggt_v3/{dataset}/`
- benchmark outputs: `workspace/all_vggt_v3/`

## GT-Free Test-Time Adaptation Protocol (2026-09)

Per-scene TTA used for the current champion results (see the README section
*Test-Time Adaptation Protocol* and `docs/TTA_PROTOCOL_*.md` for the method).
Everything runs from the repo root; the conda env is `da3` on the dev box.

### DA3 protocol (single entry point: train + eval in one process)

```bash
# champion recipes (see docs/RESULTS_VERIFICATION_2026-09-16.md for the archived runs)
python3 scripts/train_da3_protocol.py --dataset 7scenes \
  --scenes chess fire heads office pumpkin redkitchen stairs \
  --output_root workspace/da3_7scenes_2stage --steps 100 \
  --arm rkdc1h --n_train 20 --ratio_mix "8:4,16:4,24:8" \
  --combo --combo_se_frac 0.5 --two_stage 0.7

python3 scripts/train_da3_protocol.py --dataset eth3d \
  --scenes courtyard delivery_area electro facade kicker office pipes playground relief relief_2 terrains \
  --output_root workspace/da3_protocol_eth3d_t8s4 \
  --arm rkdc1h --teacher_N 8 --steps 100

python3 scripts/train_da3_protocol.py --dataset hiroom \
  --scenes <30 scenes from artifacts/diagnostics/final_protocol/hiroom/scene_manifest.json> \
  --output_root workspace/da3_protocol_hiroom_t8s4_es \
  --arm rkdc1h --teacher_N 8 --n_train 10 --early_stop --steps 100
```

Key flags: `--arm {c2m,pw0,rkdc1h,rkdc1hc}`, `--teacher_N` (8/16/32),
`--ratio_mix "8:4,16:4,24:8"` (per-pair mixed asymmetry), `--combo/--combo_se_frac`
(mix fixed-slot and endpoint-anchored pairs), `--two_stage 0.7` (curriculum;
requires `--combo`, disables early-stop), `--early_stop`, `--mask_ratio`,
`--eval_max_frames` (keep 0). Outputs: `smoke_summary.json` (AUC/AbsRel/F1/CD),
`training_trace.csv`, per-scene LoRA ckpts + `protocol.json` (frame lists).

Implementation: `src/depth_anything_3/test_time_adaption/protocol_v1.py`
(selection, losses, training loop) + `scripts/train_da3_protocol.py` (CLI/eval).

### VGGT protocol (three-step chain)

```bash
DG=diagnostics/free_geometry; RR=artifacts/diagnostics/final_protocol/<ds>
python3 $DG/train_arms.py   --run_root $RR --arms C2M_maskrel --epochs 10 --seed 0 --no_eval32
python3 $DG/eval_viewcounts.py --manifest $RR/scene_manifest.json --ckpt_root $RR/ckpts \
  --run_root $RR --step 100 --arms C2M_maskrel --view_subsets 100v   # SKIP_BASELINE=1 once A0 exists
python3 $DG/run_eval.py     --run_root $RR --datas <ds> --experiments "C2M_maskrel@100v"
```

Champion arms: `C2M_maskrel` (7scenes/eth3d/hiroom), `C2M_RKDC1H` (scannetpp).
Reference launchers: `diagnostics/free_geometry/run_<ds>_<arm>.sh`.
Frozen manifests are (re)generated by `diagnostics/free_geometry/build_final_manifest.py`.

### Supporting analyses

- `scripts/gate_analysis_gt_free.py`, `scripts/gate_analysis_s8t24.py`, `scripts/gate_da3.py` — acceptance-gate studies (note: `Δprobe_auc03` uses GT; only the mse/loss-slope signal families are GT-free).
- `scripts/eval_da3_baseline.py`, `scripts/eval_da3_recon.py` — frozen DA3 baselines (never rerun).
- `scripts/audit_da3_loss_paths.py`, `scripts/audit_mask_depth_signal.py` — loss-path audits.
- `scripts/build_hard_view_subsets.py`, `scripts/build_strict_covisibility_010_top5.py` — hard-view manifests (texture is RGB-only; the occlusion criterion uses GT geometry for *difficulty selection* only).
