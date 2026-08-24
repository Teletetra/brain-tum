from __future__ import annotations

import numpy as np


def dice_score(pred: np.ndarray, target: np.ndarray, class_id: int) -> float:
    p = pred == class_id
    t = target == class_id
    denom = p.sum() + t.sum()
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(p, t).sum() / denom)


def iou_score(pred: np.ndarray, target: np.ndarray, class_id: int) -> float:
    p = pred == class_id
    t = target == class_id
    union = np.logical_or(p, t).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, t).sum() / union)
