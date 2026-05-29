from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class ReversibleInstanceNorm1D(nn.Module):
    """
    Reversible per-sample z-score normalization for 1D sequences.

    Input shape: [B, C, T]
    Stats are computed per sample, per channel over T.
    """

    def __init__(self, num_channels: int = 1, eps: float = 1e-5):
        super().__init__()
        self.num_channels = int(num_channels)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, C, T], got shape={tuple(x.shape)}")
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)
        y = (x - mean) / std
        return y, mean, std

    def inverse(self, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return y * std + mean
