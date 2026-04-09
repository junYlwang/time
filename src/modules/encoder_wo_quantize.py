from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn.utils import weight_norm

from .backbones import ConvNeXtBlock, DownSamplingBlock
from .utils import get_padding, init_weights


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

        zq = latent
        return zq
