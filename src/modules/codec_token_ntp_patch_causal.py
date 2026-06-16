from __future__ import annotations

import torch

from .codec_token_ntp import CodecTokenNTPModel


class PatchCausalCodecTokenNTPModel(CodecTokenNTPModel):
    def encode_batch(self, raw_values: torch.Tensor, valid_lengths: torch.Tensor) -> torch.Tensor:
        x = raw_values
        codec_len = x.size(-1)
        pos = torch.arange(codec_len, device=x.device).view(1, 1, codec_len)
        start = codec_len - valid_lengths.to(x.device).view(-1, 1, 1)
        valid_mask = pos >= start
        count = valid_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        mean = (x * valid_mask).sum(dim=-1, keepdim=True) / count
        var = ((x - mean).pow(2) * valid_mask).sum(dim=-1, keepdim=True) / count
        x = torch.where(valid_mask, (x - mean) / torch.sqrt(var + self.norm_eps), torch.zeros_like(x))

        with torch.no_grad():
            codes = self.quantizer(self.encoder(x, valid_lengths)).codes.long()
        if codes.dim() != 3:
            raise RuntimeError(f"Expected codec codes [B, Q, T], got {tuple(codes.shape)}")
        if codes.size(1) != self.num_quantizers:
            raise RuntimeError(f"Expected {self.num_quantizers} quantizer layers, got {codes.size(1)}")

        offsets = (torch.arange(self.num_quantizers, device=codes.device).view(1, 1, -1) * self.codebook_size).long()
        token_nums = (codes.transpose(1, 2) + offsets).reshape(codes.size(0), -1)
        if int(token_nums.max().item()) >= self.token_id_lookup.numel():
            raise RuntimeError("Codec token id exceeds reserved time-series token vocabulary")
        return self.token_id_lookup[token_nums]
