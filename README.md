# Brain Tumor Segmentation with CNN–ViT H-CSAF

[![CI](https://github.com/Teletetra/brain-tum/actions/workflows/ci.yml/badge.svg)](https://github.com/Teletetra/brain-tum/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

A research-oriented **multi-modal MRI brain tumor segmentation** framework built around a hybrid **CNN + Vision Transformer** encoder, **Hierarchical Cross-Scale Attention Fusion (H-CSAF)**, an edge-aware branch, and deep supervision.

> **Status:** Research implementation / reproducible training scaffold. Benchmark metrics are intentionally not published until a verified BraTS training run is completed.

## Why this project

Brain tumor segmentation requires both **fine-grained anatomical detail** and **global contextual reasoning**. Convolutional networks are strong at local structure, while transformers provide larger receptive fields and long-range spatial interactions.

This project combines both representations in a single encoder-decoder pipeline and treats boundary quality as a first-class objective.

### Core design

| Component | Purpose |
|---|---|
| Lightweight CNN encoder | Local texture, edges, and anatomical detail |
| Vision Transformer encoder | Global spatial context and long-range dependencies |
| H-CSAF | Cross-scale fusion between CNN and transformer features |
| Edge branch | Explicit boundary representation for irregular tumor borders |
| Attention U-Net decoder | High-resolution reconstruction with gated skip fusion |
| Deep supervision | Auxiliary losses at intermediate decoder stages |
| 2.5D input strategy | Adds neighboring-slice context without full 3D memory cost |

## Architecture

```text
                     Multi-modal MRI
                  T1 / T1ce / T2 / FLAIR
                            │
                    2.5D slice formation
                            │
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
      ┌───────────────┐             ┌─────────────────┐
      │ CNN Encoder   │             │ ViT Encoder     │
      │ local detail  │             │ global context  │
      └───────┬───────┘             └────────┬────────┘
              │                              │
              └──────────────┬───────────────┘
                             ▼
                  ┌──────────────────────┐
                  │ H-CSAF Fusion        │
                  │ cross-scale attention│
                  └──────────┬───────────┘
                             │
                  ┌──────────┴───────────┐
                  │                      │
                  ▼                      ▼
          ┌──────────────┐       ┌───────────────┐
          │ Edge Branch  │       │ Feature Path  │
          │ boundaries   │       │ fused scales  │
          └──────┬───────┘       └───────┬───────┘
                 └──────────┬────────────┘
                            ▼
                   ┌──────────────────┐
                   │ Attention Decoder│
                   │ + skip connections│
                   └────────┬─────────┘
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
          Aux head       Aux head       Main head
             │              │              │
             └──────────────┴──────────────┘
                            ▼
                    Tumor segmentation
```

## H-CSAF: cross-scale fusion

**Hierarchical Cross-Scale Attention Fusion** is the central fusion block. At each selected feature scale, CNN and transformer representations are projected into a compatible embedding space and exchanged through cross-attention before being combined.

```text
CNN feature ──► Query ────────────────┐
                                      ├─► Cross Attention ─► fused scale
ViT feature ──► Key / Value ─────────┘
```

The design is intended to preserve CNN locality while injecting transformer context without simply concatenating feature tensors.

## Edge-aware learning

Tumor boundaries can be irregular and low-contrast. A parallel edge branch predicts a boundary representation that is injected into the decoder, encouraging sharper segmentation transitions.

The edge branch is complementary to semantic segmentation rather than replacing it: the main segmentation head remains responsible for the final multi-class prediction.

## Deep supervision

Intermediate decoder outputs are supervised alongside the final prediction. The training objective is represented as:

```text
L_total = w_main · L_main + w_aux1 · L_aux1 + w_aux2 · L_aux2
```

This encourages useful representations at multiple decoder resolutions and can improve gradient flow during training.

## Dataset: BraTS 2021

The intended benchmark is **BraTS 2021** with multi-modal MRI volumes including:

- T1
- T1ce
- T2
- FLAIR

The dataset is **not committed to Git**. Configure its local path in `configs/default.yaml`.

For training, BraTS labels can be remapped from `{0, 1, 2, 4}` to contiguous class IDs `{0, 1, 2, 3}`.

## Preprocessing pipeline

The preprocessing design keeps operations subject-aware and avoids leakage between train and validation/test subjects:

```text
Raw NIfTI volumes
      │
      ├─ modality / metadata validation
      ├─ spatial alignment checks
      ├─ background / skull handling
      ├─ intensity normalization
      ├─ label remapping
      ├─ subject-level split
      └─ 2.5D sample generation
                │
                ▼
            Augmentation
                │
                ▼
             Training
```

Typical augmentation hooks include flips, rotations, elastic deformation, intensity/gamma variation, and configurable CutMix.

## Training pipeline

```text
┌───────────────┐
│ BraTS volumes │
└───────┬───────┘
        ▼
┌─────────────────────┐
│ Preprocess + split  │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 2.5D data loader     │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ CNN + ViT encoders  │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ H-CSAF + edge path  │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Attention decoder   │
│ + deep supervision  │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Loss / backprop      │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Checkpoint + metrics│
└─────────┬───────────┘
          ▼
     Validation
```

## Reference configuration

The default configuration is designed for experimentation on a **12 GB consumer GPU class** system and exposes memory/performance controls through YAML.

```yaml
seed: 42

trainer:
  epochs: 70
  batch_size: 2
  precision: "16-mixed"
  gradient_checkpointing: true
  gradient_clip_norm: 1.0

optimizer:
  name: muon
  lr: 3.0e-4
  weight_decay: 1.0e-2
  warmup_epochs: 5
  scheduler: cosine

model:
  num_classes: 4
  input_channels: 8
  latent_dim: 256
  flash_attention: true
  deep_supervision: true
  edge_branch: true
  h_csaf: true

augmentation:
  flip: true
  rotation: true
  elastic: true
  gamma: true
  cutmix_probability: 0.2
```

> Configuration values are starting points for experimentation, not claimed benchmark-optimal hyperparameters.

## Evaluation

The evaluation layer is designed to report clinically relevant segmentation metrics at both class and aggregate levels:

- **Dice Similarity Coefficient (DSC)**
- **Intersection over Union (IoU)**
- **HD95** (95th percentile Hausdorff distance)

Prediction masks can also be exported for qualitative visualization and error analysis.

## Repository structure

```text
brain-tum/
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/
│   ├── default.yaml
│   └── experiments/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
│   ├── architecture.md
│   └── training.md
├── notebooks/
│   └── 01_brats_exploration.ipynb
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── infer.py
├── src/
│   └── brain_tumor_seg/
│       ├── config.py
│       ├── data/
│       ├── models/
│       │   ├── cnn_encoder.py
│       │   ├── vit_encoder.py
│       │   ├── edge_branch.py
│       │   ├── h_csaf.py
│       │   ├── attention_decoder.py
│       │   └── hybrid_segmentation.py
│       ├── losses/
│       ├── training/
│       ├── evaluation/
│       └── cli.py
├── tests/
├── pyproject.toml
├── Makefile
├── LICENSE
└── README.md
```

## Quick start

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

### 2. Prepare BraTS data

```bash
python scripts/prepare_data.py --config configs/default.yaml
```

### 3. Train

```bash
python scripts/train.py --config configs/default.yaml
```

### 4. Evaluate a checkpoint

```bash
python scripts/evaluate.py \
  --checkpoint artifacts/checkpoints/best.pt \
  --config configs/default.yaml
```

### 5. Run inference

```bash
python scripts/infer.py \
  --checkpoint artifacts/checkpoints/best.pt \
  --input data/processed/example_case \
  --output artifacts/predictions
```

## Reproducibility

Each experiment should retain:

- Git commit SHA
- Dataset/version identifier
- YAML configuration
- Random seed
- Preprocessing configuration
- Model architecture configuration
- Training history
- Best checkpoint
- Validation/test metrics

Generated datasets, checkpoints, predictions, and large experiment artifacts should stay outside source control.

## Engineering practices

- Subject-level train/validation/test separation
- Configuration-driven experiments
- Mixed-precision training
- Gradient checkpointing for memory-constrained hardware
- Gradient clipping and checkpoint recovery
- Modular components for ablation studies
- Unit tests around critical tensor paths
- GitHub Actions CI
- Clear separation between source code and experiment artifacts

## Research roadmap

- Verify full BraTS training/evaluation benchmark
- Add systematic CNN-only / ViT-only / fusion ablations
- Compare Muon and AdamW optimization recipes
- Add MONAI benchmark implementations
- Add test-time augmentation and uncertainty estimation
- Profile inference latency and GPU memory
- Study cross-dataset generalization
- Package inference as a reproducible service

## Responsible use

This repository is a research and engineering project. It is **not a clinical diagnostic system** and should not be used to make patient-care decisions without appropriate clinical validation, regulatory review, and expert oversight.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

**Teletetra** — research implementation and engineering work.