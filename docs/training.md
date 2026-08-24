# Training Workflow

## 1. Prepare data

Keep the BraTS dataset outside version control. Convert raw NIfTI volumes into the representation expected by the dataset class and preserve subject identifiers.

## 2. Split by subject

Do not split individual slices across train and validation sets. Subject-level splitting prevents adjacent slices from the same patient appearing in both sets.

## 3. Train

The reference configuration uses mixed precision, gradient checkpointing, batch size 2 and cosine learning-rate decay with a warmup period. Adapt batch size and image resolution to available GPU memory.

## 4. Monitor

Track:

- training and validation loss
- mean Dice
- per-class Dice
- IoU
- HD95
- learning rate
- GPU memory utilization

## 5. Save artifacts

A run should preserve the configuration, model checkpoint, optimizer state, metric history and final evaluation report. Use `artifacts/` locally and keep large generated files out of Git.

## 6. Reproducibility checklist

- fixed random seed
- recorded Git commit
- immutable dataset split
- configuration file
- environment/dependency lock
- checkpoint metadata
- evaluation script and metrics
