from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn.utils import weight_norm

from .backbones import ConvNeXtBlock, UpSamplingBlock
from .utils import get_padding, init_weights


class TimeSeriesDecoder(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h

        base_dim = int(getattr(h, "decoder_base_dim", 128))
        self.up_ratio = list(getattr(h, "up_ratio", [2, 2, 2]))
        self.in_dim = base_dim * (2 ** len(self.up_ratio))
        self.out_dim = base_dim
        self.num_layers = list(getattr(h, "decoder_num_layers", [12, 8, 4]))

        dec_ch = int(getattr(h, "seq_decoder_channel", getattr(h, "mel_Decoder_channel", 512)))

        latent_ks = int(getattr(h, "latent_input_conv_kernel_size", 11))
        self.latent_input_conv = weight_norm(
            Conv1d(int(getattr(h, "latent_dim", 32)), dec_ch // 4, latent_ks, 1, padding=get_padding(latent_ks, 1))
        )

        in_ks = int(
            getattr(
                h,
                "seq_decoder_input_kernel_size",
                getattr(h, "mel_Decoder_input_kernel_size", 11),
            )
        )
        self.decoder_input_conv = weight_norm(
            Conv1d(dec_ch // 4, dec_ch, in_ks, 1, padding=get_padding(in_ks, 1))
        )

        self.in_linear = nn.Linear(dec_ch, self.in_dim)
        self.norm = nn.LayerNorm(self.in_dim, eps=1e-6)

        dec_kernels = list(
            getattr(
                h,
                "seq_decoder_convnext_kernel_size",
                getattr(h, "mel_Decoder_convnext_kernel_size", [6, 6, 6]),
            )
        )

        self.blocks = nn.ModuleList()
        layer_scale_init = [1 / n for n in self.num_layers]
        for i, n in enumerate(self.num_layers):
            cur_dim = self.in_dim // (2 ** i)
            cur_mid = cur_dim * 2
            k = int(dec_kernels[i])
            s = int(self.up_ratio[i])
            self.blocks.append(
                UpSamplingBlock(cur_dim, cur_dim // 2, k, s, padding=(k - s) // 2)
            )
            for _ in range(n):
                self.blocks.append(ConvNeXtBlock(cur_dim // 2, cur_mid, layer_scale_init[i], None))

        self.final_norm = nn.LayerNorm(self.out_dim, eps=1e-6)

        out_ks = int(
            getattr(
                h,
                "seq_decoder_output_conv_kernel_size",
                getattr(h, "mel_Decoder_output_conv_kernel_size", 11),
            )
        )
        out_channels = int(getattr(h, "output_channels", getattr(h, "decoder_output_channels", 1)))
        self.output_conv = weight_norm(
            Conv1d(self.out_dim, out_channels, out_ks, 1, padding=get_padding(out_ks, 1))
        )

        self.output_conv.apply(init_weights)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        h = F.gelu(latent)
        h = self.latent_input_conv(h)
        h = F.gelu(h)
        h = self.decoder_input_conv(h)
        h = F.gelu(h)
        h = self.in_linear(h.transpose(1, 2))
        h = self.norm(h).transpose(1, 2)

        for blk in self.blocks:
            h = blk(h)

        h = self.final_norm(h.transpose(1, 2)).transpose(1, 2)
        x = self.output_conv(h)
        return x
