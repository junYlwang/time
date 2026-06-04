from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .probe import RMSNorm, TransformerBlock


class CodePredictor(nn.Module):
    def __init__(self, h, codebook_sizes: tuple[int, ...]):
        super().__init__()
        latent_dim = int(h.latent_dim)
        d_model = int(h.predictor_d_model)
        nhead = int(h.predictor_nhead)
        num_layers = int(h.predictor_num_layers)
        mlp_ratio = h.predictor_mlp_ratio
        dropout = h.predictor_dropout

        self.num_codebooks = len(codebook_sizes)
        self.in_proj = nn.Linear(latent_dim, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.heads = nn.ModuleList([nn.Linear(d_model, int(size)) for size in codebook_sizes])
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(
        self,
        z_q: torch.Tensor,
        valid_lengths: torch.Tensor,
        downsample_factor: int,
    ) -> list[torch.Tensor]:
        b, _, latent_len = z_q.shape
        token_len = latent_len * self.num_codebooks

        x = z_q.transpose(1, 2).repeat_interleave(self.num_codebooks, dim=1)
        valid_latent = torch.div(
            valid_lengths.to(device=z_q.device, dtype=torch.long) + int(downsample_factor) - 1,
            int(downsample_factor),
            rounding_mode="floor",
        ).clamp(max=latent_len)
        valid_tokens = valid_latent * self.num_codebooks
        pad_tokens = token_len - valid_tokens
        token_positions = torch.arange(token_len, device=z_q.device).view(1, token_len)
        attention_mask = token_positions >= pad_tokens.view(b, 1)

        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask, is_causal=True)
        x = self.norm(x)
        return [head(x) for head in self.heads]


def _ntp_loss_and_accuracy(
    logits_by_layer: list[torch.Tensor],
    codes: torch.Tensor,
    valid_lengths: torch.Tensor,
    downsample_factor: int,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    b, num_codebooks, latent_len = codes.shape
    token_len = latent_len * num_codebooks
    device = codes.device

    targets = codes.transpose(1, 2).reshape(b, token_len).long()
    valid_latent = torch.div(
        valid_lengths.to(device=device, dtype=torch.long) + int(downsample_factor) - 1,
        int(downsample_factor),
        rounding_mode="floor",
    ).clamp(max=latent_len)
    valid_tokens = valid_latent * num_codebooks
    pad_tokens = token_len - valid_tokens

    target_positions = torch.arange(1, token_len, device=device)
    target_is_valid = target_positions.view(1, -1) >= (pad_tokens + 1).view(b, 1)

    total_loss = logits_by_layer[0].new_zeros(())
    total_correct = logits_by_layer[0].new_zeros(())
    total_count = 0
    accuracies: list[torch.Tensor] = []

    for layer_idx, logits in enumerate(logits_by_layer):
        layer_target_mask = (target_positions % num_codebooks == layer_idx).view(1, -1)
        loss_mask = target_is_valid & layer_target_mask
        layer_logits = logits[:, :-1, :][loss_mask]
        layer_targets = targets[:, 1:][loss_mask]
        count = int(layer_targets.numel())
        if count == 0:
            accuracies.append(logits.new_zeros(()))
            continue
        total_loss = total_loss + F.cross_entropy(layer_logits, layer_targets, reduction="sum")
        layer_correct = (layer_logits.argmax(dim=-1) == layer_targets).float().sum()
        total_correct = total_correct + layer_correct
        total_count += count
        accuracies.append(layer_correct / count)

    if total_count == 0:
        loss = logits_by_layer[0].new_zeros(())
        accuracy = logits_by_layer[0].new_zeros(())
    else:
        loss = total_loss / total_count
        accuracy = total_correct / total_count
    return loss, accuracy, accuracies
