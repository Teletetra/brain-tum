from __future__ import annotations

import numpy as np


def make_25d(volume: np.ndarray, index: int) -> np.ndarray:
    """Construct a 3-slice 2.5D sample around a center slice."""
    if volume.ndim != 3:
        raise ValueError("volume must have shape [H, W, D]")
    depth = volume.shape[-1]
    if not 0 <= index < depth:
        raise IndexError("slice index out of bounds")
    indices = [max(0, index - 1), index, min(depth - 1, index + 1)]
    return np.stack([volume[..., i] for i in indices], axis=0).astype(np.float32)
