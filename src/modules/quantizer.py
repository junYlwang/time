from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .stochastic_fsq import RFSQ
from .vq import RVQ


@dataclass
class QuantizerOutput:
    z_q: torch.Tensor
    codes: Optional[torch.Tensor]
    q_loss: torch.Tensor



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


class RFSQQuantizer(BaseQuantizer):
    def __init__(self, h):
        super().__init__()
        latent_dim = int(getattr(h, "latent_dim", 16))
        num_quantizers = int(getattr(h, "num_quantizers", 2))
        levels = getattr(h, "levels", getattr(h, "levels_1", [8, 5, 5, 5]))
        use_stochastic = bool(getattr(h, "stochastic", False))

        self.rfsq = RFSQ(
            levels=levels,
            dim=latent_dim,
            num_quantizers=num_quantizers,
            channel_first=True,
            stochastic=use_stochastic,
        )
        self._num_quantizers = num_quantizers
        self._codebook_size = int(self.rfsq.codebook_size)
        self.is_stochastic_quantizer = True

    @property
    def codebook_sizes(self) -> Tuple[int, ...]:
        return tuple([self._codebook_size] * self._num_quantizers)

    def set_stochastic_mode(self, stochastic: bool, temperature: float) -> None:
        self.rfsq.set_stochastic(stochastic)
        self.rfsq.set_temperature(temperature)

    def forward(self, latent: torch.Tensor) -> QuantizerOutput:
        z_q, codes = self.rfsq(latent)
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
    quantizer_type = str(getattr(h, "quantizer_type", "rfsq")).lower()
    if quantizer_type == "rfsq":
        return RFSQQuantizer(h)
    if quantizer_type == "rvq":
        return RVQQuantizer(h)
    if quantizer_type == "none":
        return NoQuantizer()
    raise ValueError(f"Unsupported quantizer_type: {quantizer_type}. Expected one of: rfsq, rvq, none.")
