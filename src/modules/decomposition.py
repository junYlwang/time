from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn


class TrendResidualDecomposition(nn.Module):
    """
    Decompose a 1D signal into trend and residual components.

    Input shape: [B, C, T]
    Output shapes: trend [B, C, T], residual [B, C, T]

    Trend extraction follows a FEDformer-style mixture of average-pooling
    experts with multiple kernel sizes.
    """

    def __init__(
        self,
        num_channels: int = 1,
        kernel_sizes: Sequence[int] = (15, 31, 63, 127, 255),
        weight_mode: str = "dynamic",
        summary_length: int = 32,
        gating_hidden_dim: int = 64,
    ):
        super().__init__()

        self.num_channels = int(num_channels)
        self.kernel_sizes = self._validate_kernel_sizes(kernel_sizes)
        self.num_experts = len(self.kernel_sizes)
        self.weight_mode = str(weight_mode).strip().lower()
        self.summary_length = int(summary_length)
        self.gating_hidden_dim = int(gating_hidden_dim)

        if self.num_channels <= 0:
            raise ValueError(f"num_channels must be positive, got {self.num_channels}")
        if self.summary_length <= 0:
            raise ValueError(f"summary_length must be positive, got {self.summary_length}")
        if self.gating_hidden_dim <= 0:
            raise ValueError(f"gating_hidden_dim must be positive, got {self.gating_hidden_dim}")
        if self.weight_mode not in {"uniform", "dynamic"}:
            raise ValueError(
                f"Unsupported weight_mode: {self.weight_mode}. "
                "Expected one of: uniform, dynamic."
            )

        self.avg_pools = nn.ModuleList(
            [
                nn.AvgPool1d(
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    count_include_pad=False,
                )
                for kernel_size in self.kernel_sizes
            ]
        )

        if self.weight_mode == "dynamic":
            self.summary_pool = nn.AdaptiveAvgPool1d(self.summary_length)
            summary_dim = self.num_channels * self.summary_length
            self.gating = nn.Sequential(
                nn.Linear(summary_dim, self.gating_hidden_dim),
                nn.GELU(),
                nn.Linear(self.gating_hidden_dim, self.num_experts),
            )
            self._init_dynamic_gating()
        else:
            self.summary_pool = None
            self.gating = None

    @staticmethod
    def _validate_kernel_sizes(kernel_sizes: Iterable[int]) -> tuple[int, ...]:
        values = tuple(int(k) for k in kernel_sizes)
        if not values:
            raise ValueError("kernel_sizes must not be empty")
        for k in values:
            if k <= 0:
                raise ValueError(f"kernel_sizes must be positive, got {k}")
            if k % 2 == 0:
                raise ValueError(
                    f"kernel_sizes must be odd for symmetric same-length pooling, got {k}"
                )
        return values

    def _compute_weights(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_mode == "uniform":
            return x.new_full((x.size(0), self.num_experts), 1.0 / self.num_experts)

        if self.summary_pool is None or self.gating is None:
            raise RuntimeError("Dynamic weighting is not initialized")

        summary = self.summary_pool(x).reshape(x.size(0), -1)
        logits = self.gating(summary)
        return torch.softmax(logits, dim=-1)

    def _init_dynamic_gating(self) -> None:
        if self.gating is None:
            return
        last_linear = self.gating[-1]
        if isinstance(last_linear, nn.Linear):
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, C, T], got shape={tuple(x.shape)}")
        if x.size(1) != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} input channels, got {x.size(1)}"
            )

        trend_candidates = torch.stack([pool(x) for pool in self.avg_pools], dim=1)
        weights = self._compute_weights(x).view(x.size(0), self.num_experts, 1, 1)
        trend = (trend_candidates * weights).sum(dim=1)
        residual = x - trend
        return trend, residual
