from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class HCSAFFusion(nn.Module):
    """Hierarchical cross-scale attention fusion for CNN/ViT features."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, cnn: torch.Tensor, vit: torch.Tensor) -> torch.Tensor:
        if cnn.ndim != 4 or vit.ndim != 4:
            raise ValueError("cnn and vit features must be [B,C,H,W]")
        vit = F.interpolate(vit, size=cnn.shape[-2:], mode="bilinear", align_corners=False)
        b, c, h, w = cnn.shape
        q = self.q_proj(cnn).flatten(2).transpose(1, 2)
        k = self.k_proj(vit).flatten(2).transpose(1, 2)
        v = self.v_proj(vit).flatten(2).transpose(1, 2)
        q = q.view(b, h * w, self.num_heads, c // self.num_heads).transpose(1, 2)
        k = k.view(b, h * w, self.num_heads, c // self.num_heads).transpose(1, 2)
        v = v.view(b, h * w, self.num_heads, c // self.num_heads).transpose(1, 2)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        fused = (attn @ v).transpose(1, 2).reshape(b, h * w, c)
        fused = fused.transpose(1, 2).reshape(b, c, h, w)
        fused = self.out_proj(fused)
        alpha = torch.sigmoid(self.gate)
        return alpha * fused + (1.0 - alpha) * cnn
