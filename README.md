# IMPASTO: Robotic Oil Painting Reproduction

An independent local simulation reproduction of [IMPASTO: Integrating Model-Based Planning with Learned Dynamics Models for Robotic Oil Painting Reproduction](https://arxiv.org/abs/2603.29315).

This project implements the paper's core pipeline in a fast, parallelizable MuJoCo environment:

- a Franka Emika Panda robot with a brush assembly, canvas, palette, washing station, and top-view camera;
- contact-driven paint deposition and closed-loop normal-force control;
- a learned action-conditioned dynamics model trained on simulator self-play data;
- receding-horizon planning and execution for single strokes and multi-stroke Flower/Fish paintings.

> This is an independent personal reproduction. It is not the authors' code, the official implementation, or an affiliated release. The paper's original robot data, per-stroke action annotations, complete CAD, and trained weights were not publicly available during this reproduction.

## Start Here

- [Three-stage reproduction report](reports/final_reproduction/index.html)
- [2× continuous painting showcase](reports/baseline_1107014_corrected_final/showcase/painting_process_2x.mp4)
- [Flower 2× replay](reports/baseline_1107014_corrected_final/showcase/flower_painting_2x.mp4)
- [Fish 2× replay](reports/baseline_1107014_corrected_final/showcase/fish_painting_2x.mp4)
- [Paper](https://arxiv.org/abs/2603.29315)

![Flower and Fish reproduction results](reports/baseline_1107014_corrected_final/audit/flower_comparison.png)

## Pipeline

### 1. Simulation environment

The scene contains a 7-DoF Panda, an 18 × 21 inch canvas, 24 pre-mixed paint slots, a washing station, a brush holder and a top-view camera. The simulator advances physics without wall-clock delays, updates the canvas from the brush's actual contact state, and supports independent parallel environments for data collection.

The local controller uses a position/orientation servo with a real-time normal-force feedback admittance loop. The brush is represented by an axial compliant model, and the painting process includes moving to the palette or washing station, loading or removing paint, approaching the canvas, making contact, painting, and retracting.

### 2. Learned dynamics

The dynamics model is an action-conditioned U-Net that predicts the next local canvas image from a two-channel 100 × 100 patch and a 6D stroke action. The released training run contains 1,000 simulator transitions split by episode into 900 training samples and 100 validation samples.

### 3. Closed-loop painting

The planner uses the learned dynamics model inside a receding-horizon loop. The implementation follows the paper's reported structure: PCA–RANSAC stroke initialization, elite softmax sequence updates, covariance adaptation, a horizon of `H=5` for multi-stroke painting, and two executed strokes between replanning steps.

The Flower and Fish tasks use target panels made available with the paper. Their original high-resolution files and per-stroke action annotations were not released, so the stroke actions are inferred by this project and are clearly identified as reproduction data.

## Results

| Component | Result |
|---|---:|
| Stage 1 force RMSE | 0.0166 N at a 0.6 N target |
| Stage 1 contact success | 9/9 validation strokes |
| Stage 1 simulation speed | 9.28× realtime without rendering |
| Stage 2 self-play data | 1,000 transitions |
| Stage 2 validation split | 900 train / 100 validation |
| Stage 2 validation L1 | 0.008487 |
| Flower execution | 17 strokes, weighted gray L1 0.014945 |
| Fish execution | 18 strokes, weighted gray L1 0.014755 |

These values are measurements from the local simulator. The Stage 2 validation score and Stage 3 image errors should not be interpreted as a controlled comparison with the paper's real-robot results.

## Gallery And Video

| Flower target | Flower result |
|---|---|
| ![Flower target](reports/baseline_1107014_corrected_final/flower_target.png) | ![Flower result](reports/baseline_1107014_corrected_final/flower_result.png) |

| Fish target | Fish result |
|---|---|
| ![Fish target](reports/baseline_1107014_corrected_final/fish_target.png) | ![Fish result](reports/baseline_1107014_corrected_final/fish_result.png) |

The showcase videos are continuous MuJoCo replays. They include robot motion, paint loading, washing, canvas contact, brush motion, the live canvas, and force readouts. They are exported at 2× playback speed from the saved execution actions.

- [Flower continuous replay, 2×](reports/baseline_1107014_corrected_final/showcase/flower_painting_2x.mp4)
- [Fish continuous replay, 2×](reports/baseline_1107014_corrected_final/showcase/fish_painting_2x.mp4)
- [Flower → Fish complete replay, 2×](reports/baseline_1107014_corrected_final/showcase/painting_process_2x.mp4)
- [Single-stroke replay, 2×](reports/baseline_1107014_corrected_final/showcase/single_stroke_2x.mp4)
- [Video provenance and integrity checks](reports/baseline_1107014_corrected_final/showcase/video_provenance.json)

For round-by-round montages, numerical tables, force traces, model evaluation, and reproduction notes, see the [full report](reports/final_reproduction/index.html).

## Installation

Create the project-local Conda environment from the repository root:

```bash
conda env create --prefix ./.conda --file environment.yml
conda activate "$PWD/.conda"
```

The project wrapper configures the local source tree, MuJoCo rendering backend, and cache directories:

```bash
bash scripts/run.sh -m pytest -q
```

The test suite currently passes 25 tests. Headless Linux runs use EGL by default; a machine with a working MuJoCo-compatible OpenGL/EGL setup is required for rendered scene checks.

## Reproduce The Stages

Stage 1 builds the scene, runs force/contact and workstation validation, and writes the audit report:

```bash
bash scripts/run.sh scripts/run_stage1.py
```

Stage 2 collects simulator self-play data and trains the dynamics model:

```bash
IMPASTO_STAGE2_CONFIG=configs/stage2_refined.json \
  bash scripts/run.sh scripts/collect_selfplay.py

IMPASTO_STAGE2_CONFIG=configs/stage2_refined.json \
  bash scripts/run.sh scripts/train_dynamics.py
```

Stage 3 runs single-stroke and multi-stroke closed-loop painting:

```bash
IMPASTO_STAGE3_CONFIG=configs/stage3_baseline_1107014_final.json \
  bash scripts/run.sh scripts/run_stage3.py
```

The completed reports and videos are already included, so reading the report does not require retraining or rerunning the experiments.

## Released Model And Dataset

The verified Stage 2 dataset and checkpoint are distributed separately from Git as a [GitHub Release asset](https://github.com/Magic-Pony/IMPASTO/releases/tag/v0.1.0-reproduction-assets). Download and extract them at the repository root if you want to run Stage 3 from the released model:

```bash
gh release download v0.1.0-reproduction-assets \
  --repo Magic-Pony/IMPASTO \
  --pattern impasto_stage2_refined_assets.tar.gz \
  --dir /tmp/impasto-assets

tar -xzf /tmp/impasto-assets/impasto_stage2_refined_assets.tar.gz
```

The archive restores:

```text
data/stage2_refined/stage2_selfplay.npz
outputs/stage2_refined/impasto_unet_best.pt
```

Their SHA-256 values are recorded in [the Stage 2 report](reports/stage2_refined/report.md) and [the final metrics](reports/final_reproduction/metrics.json).

## Repository Layout

- [`src/impasto/`](src/impasto/) — simulator, paint deposition, dynamics model, and planner.
- [`configs/`](configs/) — reproducible stage configurations.
- [`tests/`](tests/) — simulator, dynamics, planner, and reproducibility tests.
- [`reports/scene_audit/`](reports/scene_audit/) — environment, force-control, and workstation validation.
- [`reports/stage2_refined/`](reports/stage2_refined/) — dynamics training and validation artifacts.
- [`reports/baseline_1107014_corrected_final/`](reports/baseline_1107014_corrected_final/) — final painting outputs, round-by-round images, metrics, and videos.
- [`reports/final_reproduction/`](reports/final_reproduction/) — complete three-stage report in HTML, Markdown, and JSON.
- [`papers/`](papers/) — locally archived paper materials, sources, extracted text, and checksums.
- [`docs/fidelity.md`](docs/fidelity.md) — implementation alignment and modeling assumptions.

## Scope And Limitations

This repository focuses on the environment, learned dynamics workflow, and closed-loop painting pipeline described in IMPASTO. The paper remains the reference for the broader experimental context and comparative results.

Several parts of the original system were not fully specified or publicly released. The local implementation therefore uses documented engineering assumptions for brush and fixture dimensions, material parameters, contact behavior, paint deposition, washing, camera calibration, and planner hyperparameters. The simulator is not a substitute for the paper's physical robot, real paint, or real-world self-play data. The transparency module described in the paper is not fully reproduced; the released experiments use the documented opaque simulation branch.

The full report records these assumptions, the exact metrics, and the provenance of the target images and generated actions.

## Citation

```bibtex
@article{wang2026impasto,
  author  = {Wang, Yingke and Li, Hao and Zhu, Yifeng and Yu, Hong-Xing and Goldberg, Ken and Fei-Fei, Li and Wu, Jiajun and Li, Yunzhu and Zhang, Ruohan},
  title   = {IMPASTO: Integrating Model-Based Planning with Learned Dynamics Models for Robotic Oil Painting Reproduction},
  journal = {arXiv preprint arXiv:2603.29315},
  year    = {2026}
}
```

## License And Attribution

This repository contains an independent implementation. The included Panda assets retain their upstream license; see the notices under [`assets/`](assets/). Please cite the paper when referring to the IMPASTO method.
