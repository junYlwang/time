from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn.utils import weight_norm

from .backbones import ConvNeXtBlock, DownSamplingBlock
from .stochastic_fsq import IFSQ
from .utils import get_padding, init_weights


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


class Encoder(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h

        self.input_channels = int(getattr(h, "input_channels", 1))
        self.in_dim = int(getattr(h, "encoder_in_dim", 32))
        self.num_layers = list(getattr(h, "encoder_num_layers", [4, 8, 12]))
        self.down_ratio = list(getattr(h, "down_ratio", [2, 2, 2]))
        self.out_dim = self.in_dim * (2 ** len(self.down_ratio))

        embed_ks = int(getattr(h, "encoder_embed_kernel_size", 7))
        self.embed = nn.Conv1d(self.input_channels, self.in_dim, kernel_size=embed_ks, padding=embed_ks // 2)
        self.norm = nn.LayerNorm(self.in_dim, eps=1e-6)

        ds_kernels = list(getattr(h, "seq_encoder_downsample_kernel_size", [6, 6, 6]))

        self.blocks = nn.ModuleList()
        layer_scale_init = [1 / n for n in self.num_layers]
        for i, n in enumerate(self.num_layers):
            cur_dim = self.in_dim * (2 ** i)
            cur_mid = cur_dim * 4
            for _ in range(n):
                self.blocks.append(ConvNeXtBlock(cur_dim, cur_mid, layer_scale_init[i], None))
            k = int(ds_kernels[i])
            s = int(self.down_ratio[i])
            self.blocks.append(
                DownSamplingBlock(cur_dim, cur_dim * 2, k, s, padding=(k - s) // 2)
            )

        self.final_norm = nn.LayerNorm(self.out_dim, eps=1e-6)

        proj_dim = int(getattr(h, "seq_encoder_channel", 128))
        self.out_linear = nn.Linear(self.out_dim, proj_dim)

        out_ks = int(getattr(h, "seq_encoder_output_kernel_size", 11))

        self.output_conv = weight_norm(
            Conv1d(proj_dim, proj_dim // 4, out_ks, 1, padding=get_padding(out_ks, 1))
        )

        latent_ks = int(getattr(h, "latent_output_conv_kernel_size", 11))
        latent_dim = int(getattr(h, "latent_dim", 16))
        self.latent_conv = weight_norm(
            Conv1d(proj_dim // 4, latent_dim, latent_ks, 1, padding=get_padding(latent_ks, 1))
        )

        self.output_conv.apply(init_weights)
        self.latent_conv.apply(init_weights)

        self.num_quantizers = int(getattr(h, "num_quantizers", 2))
        levels_1 = getattr(h, "levels_1", [8, 5, 5, 5])
        self.quantizer_1 = IFSQ(levels=levels_1, dim=latent_dim, channel_first=True, stochastic=bool(getattr(h, "stochastic", False)))
        self.layernorm_1 = InvertibleLayerNorm(latent_dim)

        levels_2 = getattr(h, "levels_2", [8, 5, 5, 5])
        self.quantizer_2 = IFSQ(levels=levels_2, dim=latent_dim, channel_first=True, stochastic=bool(getattr(h, "stochastic", False)))
        self.layernorm_2 = InvertibleLayerNorm(latent_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        # x: [B, C, T], C=1 for raw series
        h = self.embed(x)
        h = self.norm(h.transpose(1, 2)).transpose(1, 2)

        for blk in self.blocks:
            h = blk(h)

        h = self.final_norm(h.transpose(1, 2))
        h = self.out_linear(h).transpose(1, 2)
        h = F.gelu(h)
        h = self.output_conv(h)
        h = F.gelu(h)
        latent = self.latent_conv(h)

        z1, c1 = self.quantizer_1(self.layernorm_1(latent))
        z1 = self.layernorm_1.inverse(z1)

        z2, c2 = self.quantizer_2(self.layernorm_2(latent - z1.detach()))
        z2 = self.layernorm_2.inverse(z2)
        
        zq = z1 + z2
        codes = torch.stack([c1, c2], dim=1)

        return zq, codes
