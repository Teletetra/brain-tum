from __future__ import annotations

import torch
from torch import Tensor


def dice_loss(logits: Tensor, target: Tensor, smooth: float = 1.0) -> Tensor:
    """Mean soft Dice loss over classes."""
    probs = torch.softmax(logits, dim=1)
    num_classes = logits.shape[1]
    one_hot = torch.nn.functional.one_hot(target.long(), num_classes=num_classes)
    one_hot = one_hot.permute(0, 3, 1, 2).to(dtype=probs.dtype)
    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dims)
    denominator = probs.sum(dims) + one_hot.sum(dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()
