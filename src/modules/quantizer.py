from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .stochastic_fsq import IFSQ
from .vq import RVQ


@dataclass
class QuantizerOutput:
    z_q: torch.Tensor
    codes: Optional[torch.Tensor]
    q_loss: torch.Tensor


class InvertibleLayerNorm(nn.Module):
    """Invertible LayerNorm for (B, D, T)."""

    def __init__(self, num_dims: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_dims))
        self.bias = nn.Parameter(torch.zeros(num_dims))

        self.register_buffer("current_mean", None, persistent=False)
        self.register_buffer("current_std", None, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, d, _t = x.shape
        self.current_mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        self.current_std = torch.sqrt(var + self.eps)

        y = (x - self.current_mean) / self.current_std
        w = self.weight.view(1, d, 1)
        b0 = self.bias.view(1, d, 1)
        return y * w + b0

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if self.current_mean is None or self.current_std is None:
            raise RuntimeError("Call forward() before inverse().")
        _b, d, _t = y.shape
        w = self.weight.view(1, d, 1)
        b0 = self.bias.view(1, d, 1)
        x = (y - b0) / w
        return x * self.current_std + self.current_mean


class BaseQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.is_stochastic_quantizer = False

    @property
    def codebook_sizes(self) -> Tuple[int, ...]:
        return ()

    def set_stochastic_mode(self, stochastic: bool, temperature: float) -> None:
        _ = stochastic
        _ = temperature

    def forward(self, latent: torch.Tensor) -> QuantizerOutput:
        raise NotImplementedError


class FSQResidualQuantizer(BaseQuantizer):
    def __init__(self, h):
        super().__init__()
        latent_dim = int(getattr(h, "latent_dim", 16))
        levels_1 = getattr(h, "levels_1", [8, 5, 5, 5])
        levels_2 = getattr(h, "levels_2", [8, 5, 5, 5])
        use_stochastic = bool(getattr(h, "stochastic", False))

        self.quantizer_1 = IFSQ(
            levels=levels_1,
            dim=latent_dim,
            channel_first=True,
            stochastic=use_stochastic,
        )
        self.quantizer_2 = IFSQ(
            levels=levels_2,
            dim=latent_dim,
            channel_first=True,
            stochastic=use_stochastic,
        )
        self.layernorm_1 = InvertibleLayerNorm(latent_dim)
        self.layernorm_2 = InvertibleLayerNorm(latent_dim)
        self.is_stochastic_quantizer = True

    @property
    def codebook_sizes(self) -> Tuple[int, ...]:
        return (int(self.quantizer_1.codebook_size), int(self.quantizer_2.codebook_size))

    def set_stochastic_mode(self, stochastic: bool, temperature: float) -> None:
        self.quantizer_1.stochastic = stochastic
        self.quantizer_1.temperature = temperature
        self.quantizer_2.stochastic = stochastic
        self.quantizer_2.temperature = temperature

    def forward(self, latent: torch.Tensor) -> QuantizerOutput:
        z1, c1 = self.quantizer_1(self.layernorm_1(latent))
        z1 = self.layernorm_1.inverse(z1)

        z2, c2 = self.quantizer_2(self.layernorm_2(latent - z1.detach()))
        z2 = self.layernorm_2.inverse(z2)

        z_q = z1 + z2
        codes = torch.stack([c1, c2], dim=1)
        q_loss = latent.new_zeros(())
        return QuantizerOutput(z_q=z_q, codes=codes, q_loss=q_loss)


class RVQQuantizer(BaseQuantizer):
    def __init__(self, h):
        super().__init__()
        latent_dim = int(getattr(h, "latent_dim", 8))
        num_quantizers = int(getattr(h, "num_quantizers", 2))
        codebook_size = int(getattr(h, "rvq_codebook_size", 1024))
        codebook_dim = int(getattr(h, "rvq_codebook_dim", latent_dim))
        quantize_dropout = bool(getattr(h, "rvq_quantize_dropout", False))
        quantize_dropout_cutoff_index = int(getattr(h, "rvq_quantize_dropout_cutoff_index", 0))

        self.rvq = RVQ(
            dim=latent_dim,
            codebook_dim=codebook_dim,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            quantize_dropout=quantize_dropout,
            quantize_dropout_cutoff_index=quantize_dropout_cutoff_index,
        )
        self.decay = float(getattr(h, "rvq_decay", 0.99))
        self._num_quantizers = num_quantizers
        self._codebook_size = codebook_size

    @property
    def codebook_sizes(self) -> Tuple[int, ...]:
        return tuple([self._codebook_size] * self._num_quantizers)

    def forward(self, latent: torch.Tensor) -> QuantizerOutput:
        z_q, codes, q_loss = self.rvq(latent, decay=self.decay)
        return QuantizerOutput(z_q=z_q, codes=codes, q_loss=q_loss)


class NoQuantizer(BaseQuantizer):
    def forward(self, latent: torch.Tensor) -> QuantizerOutput:
        q_loss = latent.new_zeros(())
        return QuantizerOutput(z_q=latent, codes=None, q_loss=q_loss)


def build_quantizer(h) -> BaseQuantizer:
    quantizer_type = str(getattr(h, "quantizer_type", "fsq")).lower()
    if quantizer_type == "fsq":
        return FSQResidualQuantizer(h)
    if quantizer_type == "rvq":
        return RVQQuantizer(h)
    if quantizer_type == "none":
        return NoQuantizer()
    raise ValueError(f"Unsupported quantizer_type: {quantizer_type}. Expected one of: fsq, rvq, none.")
