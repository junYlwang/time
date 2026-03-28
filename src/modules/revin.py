from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReversibleInstanceNorm1D(nn.Module):
    """
    Reversible per-sample normalization for 1D sequences.

    Input shape: [B, C, T]
    Stats are computed per sample, per channel over T.
    """

    def __init__(
        self,
        num_channels: int = 1,
        eps: float = 1e-5,
        affine: bool = True,
        init_gamma: float = 1.0,
        init_beta: float = 0.0,
        positive_gamma: bool = False,
    ):
        super().__init__()
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.affine = bool(affine)
        self.positive_gamma = bool(positive_gamma)

        if self.affine:
            if self.positive_gamma:
                # Inverse softplus so that softplus(gamma_raw) ~= init_gamma.
                if init_gamma <= 0:
                    raise ValueError("init_gamma must be > 0 when positive_gamma=True")
                inv = float(torch.log(torch.expm1(torch.tensor(init_gamma))).item())
                self.gamma_raw = nn.Parameter(torch.full((1, self.num_channels, 1), inv))
            else:
                self.gamma = nn.Parameter(torch.full((1, self.num_channels, 1), float(init_gamma)))
            self.beta = nn.Parameter(torch.full((1, self.num_channels, 1), float(init_beta)))

    def _get_gamma(self) -> torch.Tensor:
        if not self.affine:
            raise RuntimeError("Gamma requested when affine=False")
        if self.positive_gamma:
            return F.softplus(self.gamma_raw)
        return self.gamma

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, C, T], got shape={tuple(x.shape)}")
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)
        y = (x - mean) / std
        if self.affine:
            y = y * self._get_gamma() + self.beta
        return y, mean, std

    def inverse(self, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if self.affine:
            y = (y - self.beta) / (self._get_gamma() + self.eps)
        x = y * std + mean
        return x


class ReversibleMeanAbsNorm1D(nn.Module):
    """
    Reversible per-sample mean-absolute scaling for 1D sequences.

    Input shape: [B, C, T]
    Scale is computed per sample, per channel over T:
      s = mean(abs(x), dim=-1, keepdim=True) + eps
    """

    def __init__(
        self,
        num_channels: int = 1,
        eps: float = 1e-5,
        affine: bool = True,
        init_gamma: float = 1.0,
        init_beta: float = 0.0,
        positive_gamma: bool = False,
    ):
        super().__init__()
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.affine = bool(affine)
        self.positive_gamma = bool(positive_gamma)

        if self.affine:
            if self.positive_gamma:
                if init_gamma <= 0:
                    raise ValueError("init_gamma must be > 0 when positive_gamma=True")
                inv = float(torch.log(torch.expm1(torch.tensor(init_gamma))).item())
                self.gamma_raw = nn.Parameter(torch.full((1, self.num_channels, 1), inv))
            else:
                self.gamma = nn.Parameter(torch.full((1, self.num_channels, 1), float(init_gamma)))
            self.beta = nn.Parameter(torch.full((1, self.num_channels, 1), float(init_beta)))

    def _get_gamma(self) -> torch.Tensor:
        if not self.affine:
            raise RuntimeError("Gamma requested when affine=False")
        if self.positive_gamma:
            return F.softplus(self.gamma_raw)
        return self.gamma

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, C, T], got shape={tuple(x.shape)}")
        scale = x.abs().mean(dim=-1, keepdim=True) + self.eps
        y = x / scale
        if self.affine:
            y = y * self._get_gamma() + self.beta
        return y, scale

    def inverse(self, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if self.affine:
            y = (y - self.beta) / (self._get_gamma() + self.eps)
        return y * scale
