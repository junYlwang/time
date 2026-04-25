from __future__ import annotations

import torch
import torch.nn as nn


class _BaseProbe(nn.Module):
    def __init__(self):
        super().__init__()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _flatten_latent(x: torch.Tensor) -> torch.Tensor:
        b, d, l = x.shape
        return x.reshape(b, d * l)


class LinearProbe(_BaseProbe):
    def __init__(self, h):
        super().__init__()
        in_dim = int(h.latent_dim) * int(h.latent_seq_len)
        self.classifier = nn.Linear(in_dim, int(h.num_classes))

        self.apply(self._init_weights)

    def forward(self, x):
        x = self._flatten_latent(x)
        out = self.classifier(x)

        return out


class MLPProbe(_BaseProbe):
    def __init__(self, h):
        super().__init__()
        in_dim = int(h.latent_dim) * int(h.latent_seq_len)
        hidden_dim = min(in_dim // 2, 512)
        dropout = 0.0

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, int(h.num_classes)),
        )

        self.apply(self._init_weights)

    def forward(self, x):
        x = self._flatten_latent(x)
        out = self.classifier(x)

        return out
