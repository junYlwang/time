from __future__ import annotations

import torch
import torch.nn as nn


def _compute_valid_token_counts(valid_lengths, downsample_factor, num_quantizers, total_tokens):
    num_latent_steps = total_tokens // num_quantizers
    num_valid_latent = torch.ceil(valid_lengths.float() / downsample_factor).long()
    num_valid_latent = num_valid_latent.clamp(max=num_latent_steps)
    num_valid_tokens = num_valid_latent * num_quantizers
    num_pad_tokens = total_tokens - num_valid_tokens
    return num_valid_tokens, num_pad_tokens


class CodecTokenNTPModel(nn.Module):
    def __init__(
        self,
        llm: nn.Module,
        encoder: nn.Module,
        quantizer: nn.Module,
        decoder: nn.Module,
        token_id_lookup: list[int],
        pad_token_id: int,
        downsample_factor: int,
        codebook_size: int,
        num_quantizers: int,
        norm_eps: float,
    ):
        super().__init__()
        self.llm = llm
        self.encoder = encoder.eval()
        self.quantizer = quantizer.eval()
        self.decoder = decoder.eval()
        self.config = llm.config
        self.pad_token_id = int(pad_token_id)
        self.downsample_factor = int(downsample_factor)
        self.codebook_size = int(codebook_size)
        self.num_quantizers = int(num_quantizers)
        self.norm_eps = float(norm_eps)
        self.register_buffer("token_id_lookup", torch.tensor(token_id_lookup, dtype=torch.long), persistent=False)
        for module in (self.encoder, self.quantizer, self.decoder):
            for param in module.parameters():
                param.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        self.quantizer.eval()
        self.decoder.eval()
        return self

    def save_pretrained(self, *args, **kwargs):
        return self.llm.save_pretrained(*args, **kwargs)

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
            codes = self.quantizer(self.encoder(x)).codes.long()
        if codes.dim() != 3:
            raise RuntimeError(f"Expected codec codes [B, Q, T], got {tuple(codes.shape)}")
        if codes.size(1) != self.num_quantizers:
            raise RuntimeError(f"Expected {self.num_quantizers} quantizer layers, got {codes.size(1)}")

        offsets = (torch.arange(self.num_quantizers, device=codes.device).view(1, 1, -1) * self.codebook_size).long()
        token_nums = (codes.transpose(1, 2) + offsets).reshape(codes.size(0), -1)
        if int(token_nums.max().item()) >= self.token_id_lookup.numel():
            raise RuntimeError("Codec token id exceeds reserved time-series token vocabulary")
        return self.token_id_lookup[token_nums]

    def forward(self, raw_values: torch.Tensor, valid_lengths: torch.Tensor, return_loss: bool = True, **kwargs):
        raw_values = raw_values.float()
        valid_lengths = valid_lengths.long()
        input_ids = self.encode_batch(raw_values, valid_lengths)  # [B, T]
        T = input_ids.size(-1)
        n_valid, n_pad = _compute_valid_token_counts(
            valid_lengths, self.downsample_factor, self.num_quantizers, T,
        )

        # Keep all codec tokens. Padding-derived tokens are placeholders only.
        labels = input_ids.clone()
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)  # [1, T]
        labels[pos < (n_pad + 1).unsqueeze(1)] = -100
        attention_mask = torch.ones_like(input_ids)
        attention_mask[pos < n_pad.unsqueeze(1)] = 0
        return self.llm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
