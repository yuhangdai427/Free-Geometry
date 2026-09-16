<div align="center">

<h1 style="text-align:center;">
Free Geometry: Refining 3D Reconstruction from Longer Versions of Itself
</h1>

_Test-time self-evolution for feed-forward 3D reconstruction without 3D ground truth_

<p align="center">
  <a href="https://arxiv.org/abs/2604.14048" target="_blank">
    <img alt="Paper" src="https://img.shields.io/badge/arXiv-2604.14048-red?logo=arxiv" height="20" />
  </a>
  <a href="https://github.com/hiteacherIamhumble/Free-Geometry" target="_blank">
    <img alt="Code" src="https://img.shields.io/badge/GitHub-Free--Geometry-181717?logo=github" height="20" />
  </a>
  <a href="https://huggingface.co/PeterDAI/Free-Geometry" target="_blank">
    <img alt="Model Weights" src="https://img.shields.io/badge/HuggingFace-Model_Weights-yellow?logo=huggingface" height="20" />
  </a>
  <a href="./LICENSE" target="_blank">
    <img alt="License" src="https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg" height="20" />
  </a>
</p>

</div>

## 🔄 Updates
- **2026.04.15:** The Free Geometry paper and code were released.
- **2026.09.16:** Added the **GT-free test-time adaptation protocol** (per-scene LoRA, long-context teacher → short-context student distillation) with the new loss family — `maskdistill` + `rkd_huber` + `couple` (arm **RKDC1H**) and `maskdistill` + `rel` (arm **maskrel**) — verified champion results on 2 models × 4 datasets, and full protocol/verification docs under `docs/`. See the new [TTA Protocol](#-test-time-adaptation-protocol-gt-free) section.

## 📦 What This Repo Provides

| Component | Description |
| --------- | ----------- |
| **Free Geometry Training** | Test-time self-supervised adaptation pipelines for **Depth Anything 3 (DA3)** and **VGGT** via lightweight LoRA updates |
| **Benchmark Scripts** | Single-model evaluation entry points for camera pose and reconstruction benchmarking |
| **All-In-One Runners** | Reproducible shell runners for training, baseline benchmarking, LoRA benchmarking, and full end-to-end workflows |
| **Shared Multiview Evaluation** | A unified multiview benchmark pipeline across datasets and view counts |
| **Visualization Tools** | Result visualizer for comparing baseline and Free Geometry outputs from a shared `results/` root |
| **Inference / App Utilities** | CLI and Gradio-oriented utilities for image, video, and COLMAP-based 3D reconstruction workflows |

## 📝 Overview

Feed-forward 3D reconstruction models are efficient, but they are also rigid: once trained, they normally run in a zero-shot manner and cannot adapt to the scene they are currently reconstructing. In practice, that leaves visible errors under occlusion, specular surfaces, and other ambiguous visual conditions.

**Free Geometry** addresses this with a simple idea: reconstructions produced from **more views** are often more reliable than reconstructions produced from fewer views. The framework turns that property into a self-supervised test-time adaptation signal by masking part of an input sequence, enforcing cross-view feature consistency between full and partial observations, and preserving the pairwise geometric relations implied by held-out frames. This makes it possible to refine foundation reconstruction models with lightweight **LoRA** updates, without using any 3D ground truth at test time.

The current repository is the active Free Geometry workflow for **DA3** and **VGGT**, including training scripts, benchmarking scripts, multiview evaluation, and visualization utilities.

## 🌟 Why Free Geometry?

<p align="center">
  <img src="assets/teaser.png" width="100%" alt="Free Geometry teaser figure"/>
</p>

The core benefit of Free Geometry is that it improves scene-specific reconstructions using only the test sequence itself. Instead of relying on external labels or expensive optimization-heavy reconstruction pipelines, it uses longer-view consistency as supervision and adapts quickly through low-rank updates. According to the arXiv paper, this refinement takes **less than 2 minutes per dataset on a single GPU** and improves both pose accuracy and point-map quality across four benchmarks.

## 🧠 Free Geometry Pipeline

<p align="center">
  <img src="assets/arch.png" width="100%" alt="Free Geometry architecture figure"/>
</p>

Free Geometry builds a self-supervised task from a testing sequence by comparing reconstructions from **full observations** against reconstructions from **partial observations**. The training objective combines cross-view feature consistency with constraints that preserve the geometry implied by the held-out views. In this repository, that logic is exposed through dedicated DA3 and VGGT training scripts, LoRA checkpointing, and matching benchmark pipelines.

## 📊 Qualitative Results

<p align="center">
  <img src="assets/depth.png" width="100%" alt="Free Geometry depth refinement results"/>
</p>

<p align="center">
  <img src="assets/points.png" width="100%" alt="Free Geometry point-map refinement results"/>
</p>

<p align="center">
  <img src="assets/results.png" width="70%" alt="Free Geometry result summary"/>
</p>

The paper reports consistent gains on four benchmark datasets, with an average improvement of **3.73%** in camera pose accuracy and **2.88%** in point-map prediction. The qualitative figures above highlight the intended effect of Free Geometry: cleaner depth structure, more stable geometry, and improved multiview consistency after adaptation.

## 🧪 Test-Time Adaptation Protocol (GT-free)

This is the protocol used for the current champion results. Per scene, a **frozen long-context teacher** (8–24 views) supervises a **short-context student** (4–8 views) carrying LoRA adapters (r=32, all blocks, camera token trainable on DA3; ≤100 steps, cosine LR, optional early-stop). Frame selection is GT-free: a SIFT-overlap statistic τ (median adjacent-frame match fraction) dispatches between *dense-equidistant* (τ>0.55) and *random-window* sampling; shared student frames always sit at teacher slots `[0,2,4,6]`.

### Key losses

All terms are computed on the **shared frames**; teacher tensors are detached. Notation: `f_s/f_t` = student/teacher patch tokens at the frozen depth head's four input layers (DA3 `[19,27,33,39]`, VGGT `[4,11,17,23]`), compared **after the head's shared LayerNorm**; `c_s/c_t` = camera centers decoded from predicted w2c poses (`c = −Rᵀt`); `d_s/d_t` = predicted depth.

| Loss | Definition | Code |
|---|---|---|
| `maskdistill` | Student sees **50% block-masked** images, teacher sees clean images; `Huber(β=1) + 2·(1−cos)` on LN'd tokens, **only on masked patches**, weighted by teacher depth confidence, averaged over the 4 tap layers (MGD/iBOT-style) | `protocol_v1.py: loss_maskdistill` · `train_arms.py: loss_b5_maskdistill` |
| `rkd_huber` | Relational KD on camera centers, **gauge-free**: mean-normalized pairwise center distances + 12 triangle angles, per-residual `Huber(δ=0.2)` | `protocol_v1.py: loss_rkd_shared_pose_huber_w2c` · `abs_pose_loss.py: loss_rkd_shared_pose_huber` |
| `couple` | Cross-head **gauge coupling scalar**: `(log(RMS(c_s)/mean(d_s)) − log(RMS(c_t)/mean(d_t)))²` — keeps the pose scale and depth scale from decoupling | `protocol_v1.py: loss_couple_w2c` · `abs_pose_loss.py: loss_couple` |
| `rel` | Relative pose per view-pair: rotation chordal + translation-direction `1−cos` (scale-free; exactly the functional the AUC@3 metric measures) | `protocol_v1.py: loss_pose_rel` · `train_arms.py: loss_pose_rel` |

**Arms:** `RKDC1H = maskdistill + 1.5·rkd_huber + 1.0·couple` · `maskrel = maskdistill + 1.0·rel`.

### Verified champion results (paired vs frozen baseline)

Mean over scenes of the per-scene relative gain `(TTA − baseline)/baseline`. Eval on **100 views** (benchmark-100, seed 42) when the scene has ≥100 frames, else **all views**. Audit: [docs/RESULTS_VERIFICATION_2026-09-16.md](docs/RESULTS_VERIFICATION_2026-09-16.md) — every cell recomputed exactly from archived runs; footnotes below.

| Model × Dataset | Champion recipe | dAUC@3 | dF1 |
|---|---|---|---|
| DA3 × 7scenes | `rkdc1h` + two_stage 0.7 + ratio_mix `8:4,16:4,24:8` + 20 pairs | **+7.62%** | **+5.50%** |
| DA3 × eth3d | `rkdc1h`, pure 8:4 ‡ | +2.95% | −1.21% |
| DA3 × hiroom | `rkdc1h`, 8:4 + early_stop | +3.33% | +0.62% |
| DA3 × scannetpp | saturated ceiling (teacher≈student) | ±0.5% | ±0.4% |
| VGGT × 7scenes | `maskrel` @100v | +0.62% | **+14.10%** |
| VGGT × eth3d | `maskrel` @allv | **+32.24%** | **+25.46%** |
| VGGT × hiroom | `maskrel` @allv † | **+18.54%** | **+18.52%** |
| VGGT × scannetpp | `RKDC1H` @100v | **+5.50%** | +2.22% |

† AUC averaged over 29 scenes (one baseline-AUC=0 scene excluded), F1 over 30. ‡ The reproducing run has no early-stop (es variants crashed); DA3-scannetpp baseline AUC 0.8467 is already at ceiling. Metrics: AUC@3 = all-pairs relative-pose AUC (max of rotation / translation-direction error, 1° bins, first-camera aligned); F1 = TSDF-fusion point cloud (recon_unposed, RANSAC-Umeyama aligned), 5 cm threshold.

### Running it

**DA3** — one entry point trains and evaluates (AUC + AbsRel + F1 in one inference). Exact champion commands (scene lists are frozen in `artifacts/diagnostics/final_protocol/<ds>/scene_manifest.json`, regenerable via `diagnostics/free_geometry/build_final_manifest.py`):

```bash
# 7scenes champion: two-stage curriculum (fixed-slot pairs first, endpoint-anchored later)
python3 scripts/train_da3_protocol.py --dataset 7scenes \
  --scenes chess fire heads office pumpkin redkitchen stairs \
  --output_root workspace/da3_7scenes_2stage --steps 100 \
  --arm rkdc1h --n_train 20 --ratio_mix "8:4,16:4,24:8" \
  --combo --combo_se_frac 0.5 --two_stage 0.7

# eth3d / hiroom champion: plain 8:4 (teacher_N=8, 4 shared student frames)
python3 scripts/train_da3_protocol.py --dataset eth3d \
  --scenes courtyard delivery_area electro facade kicker office pipes playground relief relief_2 terrains \
  --output_root workspace/da3_protocol_eth3d_t8s4 \
  --arm rkdc1h --teacher_N 8 --steps 100          # add --early_stop for the hiroom recipe
```
Requires: model weights at `model_weights/DA3-GIANT-1.1`, dataset roots as in `docs/BENCHMARK.md`. Outputs: `<output_root>/smoke_summary.json` (all metrics), `training_trace.csv`, per-scene LoRA ckpts.

**VGGT** — three-step chain (train → view-count inference → metrics); example is the 7scenes champion:

```bash
DG=diagnostics/free_geometry; RR=artifacts/diagnostics/final_protocol/7scenes
python3 $DG/train_arms.py --run_root $RR --arms C2M_maskrel --epochs 10 --seed 0 --no_eval32
python3 $DG/eval_viewcounts.py --manifest $RR/scene_manifest.json --ckpt_root $RR/ckpts \
  --run_root $RR --step 100 --arms C2M_maskrel --view_subsets 100v   # set SKIP_BASELINE=1 once A0 exists
python3 $DG/run_eval.py --run_root $RR --datas 7scenes --experiments "C2M_maskrel@100v"
```
Swap `--arms C2M_RKDC1H` / `--view_subsets 100v|allv` for the scannetpp / small-scene recipes; add `--early_stop` for the v3.2 protocol. Reference launchers: `diagnostics/free_geometry/run_<ds>_<arm>.sh`.

**Protocol docs:** [docs/TTA_PROTOCOL_v3_2026-09-15.md](docs/TTA_PROTOCOL_v3_2026-09-15.md) (VGGT) · [docs/TTA_PROTOCOL_DA3_v1_2026-09-15.md](docs/TTA_PROTOCOL_DA3_v1_2026-09-15.md) (DA3) · [docs/EVAL_AUDIT.md](docs/EVAL_AUDIT.md) (metric chain) · [docs/UNIFIED_PROTOCOL_RESEARCH_NOTES.md](docs/UNIFIED_PROTOCOL_RESEARCH_NOTES.md) (toward a unified, dataset/model-agnostic protocol).

## 📚 Usage

Environment setup, training, and benchmarking commands are documented in [docs/scripts.md](docs/scripts.md).

## 🎬 Demo

### Free Geometry visualization

```bash
python scripts/visualize_free_geometry.py \
  --model_family da3 \
  --dataset hiroom \
  --frames 16 \
  --seed 43 \
  --results_root results \
  --host 127.0.0.1 \
  --port 7860
```

This opens the Gradio demo used for comparing baseline and Free Geometry outputs.

### DA3 Gradio app

```bash
python -m depth_anything_3.cli gradio \
  --model-dir <MODEL_DIR> \
  --workspace-dir workspace/gradio \
  --gallery-dir workspace/gallery \
  --host 127.0.0.1 \
  --port 7860
```

This launches the interactive DA3 demo.

## 🙏 Acknowledgement

Free Geometry is built around adapting strong feed-forward 3D reconstruction backbones, especially **Depth Anything 3** and **VGGT**. This repository also depends on the broader open-source 3D vision ecosystem, including packages such as **gsplat**.

## 📚 Citation

If you find Free Geometry useful, please consider citing the paper:

```bibtex
@misc{dai2026freegeometryrefining3d,
  title={Free Geometry: Refining 3D Reconstruction from Longer Versions of Itself},
  author={Dai, Yuhang and Yang, Xingyi},
  year={2026},
  eprint={2604.14048},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2604.14048}
}
```
