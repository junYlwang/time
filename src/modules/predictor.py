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

        self.mask_token = nn.Parameter(torch.zeros(1, latent_dim, 1))
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

    def forward(self, z_q: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
        z_masked = torch.where(mask[:, None, :], self.mask_token.to(dtype=z_q.dtype), z_q)
        x = z_masked.transpose(1, 2)
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return [head(x) for head in self.heads]


def _build_prediction_mask(
    batch_size: int,
    seq_len: int,
    mask_ratio: float,
    random_mask_prob: float,
    device: torch.device,
    mode: str | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    mask_count = max(1, int(round(seq_len * float(mask_ratio))))
    mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)

    if mode == "random":
        use_random = torch.ones(batch_size, device=device, dtype=torch.bool)
    elif mode == "suffix":
        use_random = torch.zeros(batch_size, device=device, dtype=torch.bool)
    else:
        use_random = torch.rand(batch_size, device=device, generator=generator) < float(random_mask_prob)

    suffix_rows = ~use_random
    if suffix_rows.any():
        mask[suffix_rows, seq_len - mask_count:] = True

    for row_idx in torch.nonzero(use_random, as_tuple=False).flatten().tolist():
        indices = torch.randperm(seq_len, device=device, generator=generator)[:mask_count]
        mask[row_idx, indices] = True
    return mask


def _prediction_loss_and_accuracy(
    logits_by_layer: list[torch.Tensor],
    codes: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    losses = []
    accuracies = []
    for layer_idx, logits in enumerate(logits_by_layer):
        targets = codes[:, layer_idx, :].long()
        masked_logits = logits[mask]
        masked_targets = targets[mask]
        losses.append(F.cross_entropy(masked_logits, masked_targets))
        accuracies.append((masked_logits.argmax(dim=-1) == masked_targets).float().mean())
    loss = torch.stack(losses).mean()
    accuracy = torch.stack(accuracies).mean()
    return loss, accuracy, accuracies