# Brain Tumor Segmentation with CNN-ViT H-CSAF

A research-oriented MRI brain tumor segmentation framework for the BraTS 2021 dataset, combining local convolutional features with global transformer context, edge-aware learning, hierarchical cross-scale attention fusion, and deep supervision.

> **Research project:** hybrid CNN + Vision Transformer architecture for multi-class brain tumor segmentation from multi-modal MRI.

## Overview

Brain tumor segmentation requires both fine anatomical detail and global contextual understanding. This project addresses that challenge with a hybrid encoder-decoder design:

- **Lightweight CNN encoder** for local texture and anatomical detail
- **Vision Transformer encoder** for long-range spatial dependencies
- **FlashAttention / scaled dot-product attention** for efficient transformer attention
- **Parallel edge branch** to preserve tumor boundaries
- **H-CSAF** (Hierarchical Cross-Scale Attention Fusion) to fuse CNN and ViT representations at multiple scales
- **U-Net-style decoder** with attention gates and skip connections
- **Deep supervision** with auxiliary prediction heads at multiple decoder resolutions
- **2.5D MRI processing** to increase contextual information while remaining feasible on a single 12 GB GPU

## Architecture

```text
Multi-modal MRI
   |
   | 2.5D slice construction
   v
+---------------------+       +----------------------+
| Lightweight CNN     |       | Vision Transformer   |
| Encoder             |       | Encoder              |
| local detail        |       | global context       |
+----------+----------+       +----------+-----------+
           |                             |
           | multi-scale features        |
           +-------------+---------------+
                         v
              +----------------------+
              | H-CSAF Fusion        |
              | Cross-scale attention|
              | learnable fusion     |
              +----------+-----------+
                         |
              +----------v-----------+
              | Edge Detection Branch|
              | boundary-aware signal |
              +----------+-----------+
                         |
                         v
              +----------------------+
              | Attention U-Decoder  |
              | skip connections      |
              +----+------+-----+----+
                   |      |     |
                 Aux 1  Aux 2 Main
                   |      |     |
                   +------v-----+
                          |
                    Final logits
                          |
                     Segmentation
```

## H-CSAF

The **Hierarchical Cross-Scale Attention Fusion** module projects CNN and ViT features into a shared latent space and performs cross-attention at corresponding scales.

```text
CNN feature F_i  ----> Query Q_i ----+
                                     |
ViT feature T_i ----> Keys / Values -+--> Cross-Attention --> fused feature
```

The module uses learnable fusion weights rather than relying only on direct concatenation, allowing the network to adapt the contribution of local and global representations at each scale.

## Edge-aware segmentation

A parallel edge branch predicts a boundary probability map from the finest encoder feature. The edge representation is fused with the decoder input to improve tumor boundary localization, particularly around irregular or low-contrast regions.

## Deep supervision

Auxiliary segmentation heads are attached to intermediate decoder stages. The total objective combines the main segmentation loss with auxiliary losses:

`L_total = w_main L_main + w_1 L_aux1 + w_2 L_aux2`

This encourages useful gradient flow throughout the decoder rather than depending exclusively on the final prediction head.

## Dataset

The primary target is **BraTS 2021**, using multi-modal brain MRI volumes including T1, T1ce, T2, and FLAIR. The dataset is intentionally not committed to the repository. Configure its location in `configs/default.yaml`.

## Preprocessing

The pipeline is designed around common MRI preprocessing steps:

1. Verify modality availability and metadata
2. Align modalities using a common spatial reference
3. Skull-stripping / background handling where required
4. Bias-field correction where included by the preprocessing recipe
5. Intensity normalization
6. 2.5D slice construction
7. Label remapping for training classes
8. Subject-level train / validation / test split to prevent leakage

The research setup supports remapping BraTS labels `{0,1,2,4}` to contiguous training classes `{0,1,2,3}`.

## Training pipeline

```text
Raw NIfTI volumes
      |
      v
Subject-level preprocessing
      |
      v
2.5D sample generation
      |
      v
Augmentation
  - flips
  - rotations
  - elastic deformation
  - gamma shifts
  - CutMix (configurable)
      |
      v
Hybrid CNN + ViT encoder
      |
      v
H-CSAF + edge-aware fusion
      |
      v
Attention U-decoder + deep supervision
      |
      v
Segmentation losses
      |
      v
Checkpoint + metrics
      |
      v
Validation / test inference
```

## Training configuration

The reference recipe targets an **NVIDIA RTX 3060 12 GB** class GPU and can be adapted through YAML configuration.

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

## Loss and evaluation

Primary metrics:

- **Dice Similarity Coefficient (DSC)**
- **Intersection over Union (IoU)**
- **HD95** (95th percentile Hausdorff distance)

The evaluation pipeline reports per-class and aggregate metrics and can export prediction masks for qualitative analysis.

## Project structure

```text
.
├── configs/
│   ├── default.yaml
│   └── experiments/
├── data/
│   ├── raw/.gitkeep
│   ├── interim/.gitkeep
│   └── processed/.gitkeep
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
│       ├── __init__.py
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
├── .github/workflows/ci.yml
├── pyproject.toml
├── Makefile
├── LICENSE
└── README.md
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python scripts/prepare_data.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --checkpoint artifacts/checkpoints/best.pt --config configs/default.yaml
```

## Engineering practices

- Subject-level data separation
- Reproducible YAML configuration
- Mixed precision training
- Gradient checkpointing for memory efficiency
- Checkpoint management and best-model tracking
- Modular components for ablation studies
- Unit tests for critical tensor paths
- GitHub Actions CI
- Experiment artifacts separated from source code

## Research roadmap

- Convolutional Variational Autoencoder / uncertainty baselines
- Stronger pretrained initialization
- Multi-scale deformable attention
- Test-time augmentation
- MONAI-based benchmark comparison
- Cross-dataset generalization
- Inference profiling and optimization

## License

MIT License. See `LICENSE` for details.
