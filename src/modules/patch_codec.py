from __future__ import annotations

import torch
import torch.nn as nn

from .backbones import ConvNeXtBlock
from .probe import RMSNorm, TransformerBlock


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected [B, C, T], got shape={tuple(x.shape)}")
    b, c, t = x.shape
    if t % int(patch_size) != 0:
        raise ValueError(f"T must be divisible by patch_size, got T={t}, patch_size={patch_size}")
    n = t // int(patch_size)
    return x.reshape(b, c, n, int(patch_size)).permute(0, 2, 1, 3).reshape(b, n, c * int(patch_size))


def unpatchify(patches: torch.Tensor, channels: int, patch_size: int) -> torch.Tensor:
    if patches.ndim != 3:
        raise ValueError(f"Expected [B, N, C*P], got shape={tuple(patches.shape)}")
    b, n, d = patches.shape
    expected = int(channels) * int(patch_size)
    if d != expected:
        raise ValueError(f"Expected patch dim={expected}, got {d}")
    return patches.reshape(b, n, int(channels), int(patch_size)).permute(0, 2, 1, 3).reshape(
        b, int(channels), n * int(patch_size)
    )


def _valid_patch_mask(valid_lengths: torch.Tensor, num_patches: int, patch_size: int, device: torch.device) -> torch.Tensor:
    valid_patches = torch.div(
        valid_lengths.to(device=device, dtype=torch.long) + int(patch_size) - 1,
        int(patch_size),
        rounding_mode="floor",
    ).clamp(max=int(num_patches))
    pad_patches = int(num_patches) - valid_patches
    positions = torch.arange(int(num_patches), device=device).view(1, int(num_patches))
    return positions >= pad_patches.view(-1, 1)


class PatchLocalEncoder(nn.Module):
    """
    Patch-local encoder.

    Input: [B, C, T], T % patch_size == 0.
    Output: [B, latent_dim, N], N = T / patch_size.
    Each patch is encoded independently by flattening patches into the batch dimension.
    """

    def __init__(self, h):
        super().__init__()
        self.patch_size = int(h.patch_size)
        self.input_channels = int(h.input_channels)
        self.hidden_dim = int(h.patch_local_hidden_dim)
        self.latent_dim = int(h.latent_dim)

        self.input_proj = nn.Conv1d(self.input_channels, self.hidden_dim, kernel_size=1)
        num_blocks = int(h.patch_local_encoder_num_blocks)
        layer_scale_init = 1.0 / num_blocks
        self.blocks = nn.ModuleList(
            [
                ConvNeXtBlock(
                    self.hidden_dim,
                    int(self.hidden_dim * float(h.patch_local_mlp_ratio)),
                    layer_scale_init,
                    None,
                )
                for _ in range(num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(self.hidden_dim, eps=float(h.patch_norm_eps))
        self.out_proj = nn.Linear(self.hidden_dim * self.patch_size, self.latent_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        if c != self.input_channels:
            raise ValueError(f"Expected C={self.input_channels}, got C={c}")
        if t % self.patch_size != 0:
            raise ValueError(f"T must be divisible by patch_size, got T={t}, patch_size={self.patch_size}")
        n = t // self.patch_size
        patches = x.reshape(b, c, n, self.patch_size).permute(0, 2, 1, 3).reshape(
            b * n, c, self.patch_size
        )
        h = self.input_proj(patches)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h.transpose(1, 2)).transpose(1, 2)
        h = h.reshape(b, n, self.hidden_dim * self.patch_size)
        latent = self.out_proj(h)
        return latent.transpose(1, 2)


class PatchLocalDecoder(nn.Module):
    """
    Patch-local decoder.

    Input: [B, latent_dim, N].
    Output: [B, C, N * patch_size].
    Each latent token reconstructs only its own patch.
    """

    def __init__(self, h):
        super().__init__()
        self.patch_size = int(h.patch_size)
        self.output_channels = int(h.output_channels)
        self.hidden_dim = int(h.patch_local_hidden_dim)
        self.latent_dim = int(h.latent_dim)

        self.in_proj = nn.Linear(self.latent_dim, self.hidden_dim * self.patch_size)
        num_blocks = int(h.patch_local_decoder_num_blocks)
        layer_scale_init = 1.0 / num_blocks
        self.blocks = nn.ModuleList(
            [
                ConvNeXtBlock(
                    self.hidden_dim,
                    int(self.hidden_dim * float(h.patch_local_mlp_ratio)),
                    layer_scale_init,
                    None,
                )
                for _ in range(num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(self.hidden_dim, eps=float(h.patch_norm_eps))
        self.output_proj = nn.Conv1d(self.hidden_dim, self.output_channels, kernel_size=1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        b, d, n = latent.shape
        if d != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got {d}")
        h = self.in_proj(latent.transpose(1, 2)).reshape(b * n, self.hidden_dim, self.patch_size)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h.transpose(1, 2)).transpose(1, 2)
        patches = self.output_proj(h).reshape(b, n, self.output_channels * self.patch_size)
        return unpatchify(patches, self.output_channels, self.patch_size)


class PatchCausalEncoder(nn.Module):
    """
    Patch causal encoder.

    Input: [B, C, T], T % patch_size == 0.
    Output: [B, latent_dim, N], N = T / patch_size.
    Patch i can attend only to non-padding patches <= i.
    """

    def __init__(self, h):
        super().__init__()
        self.patch_size = int(h.patch_size)
        self.input_channels = int(h.input_channels)
        self.d_model = int(h.patch_causal_d_model)
        self.latent_dim = int(h.latent_dim)

        self.patch_proj = nn.Linear(self.input_channels * self.patch_size, self.d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=self.d_model,
                    nhead=int(h.patch_causal_nhead),
                    mlp_ratio=float(h.patch_causal_mlp_ratio),
                    dropout=float(h.patch_causal_dropout),
                    norm_eps=float(h.patch_norm_eps),
                )
                for _ in range(int(h.patch_causal_encoder_num_layers))
            ]
        )
        self.norm = RMSNorm(self.d_model, eps=float(h.patch_norm_eps))
        self.out_proj = nn.Linear(self.d_model, self.latent_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor, valid_lengths: torch.Tensor | None = None) -> torch.Tensor:
        patches = patchify(x, self.patch_size)
        h = self.patch_proj(patches)
        attention_mask = None
        if valid_lengths is not None:
            attention_mask = _valid_patch_mask(valid_lengths, h.size(1), self.patch_size, h.device)
        for block in self.blocks:
            h = block(h, attention_mask=attention_mask, is_causal=True)
        h = self.norm(h)
        return self.out_proj(h).transpose(1, 2)


class PatchCausalMAEEncoder(nn.Module):
    """
    Causal patch encoder for MAE pretraining.

    Input: [B, C, T], T % patch_size == 0.
    Output: [B, latent_dim, N], N = T / patch_size.
    mae_patch_mask is [B, N], True means replacing the patch embedding with a
    learned mask embedding before causal self-attention.
    """

    def __init__(self, h):
        super().__init__()
        self.patch_size = int(h.patch_size)
        self.input_channels = int(h.input_channels)
        self.d_model = int(h.patch_causal_d_model)
        self.latent_dim = int(h.latent_dim)

        self.patch_proj = nn.Linear(self.input_channels * self.patch_size, self.d_model)
        self.mask_embedding = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=self.d_model,
                    nhead=int(h.patch_causal_nhead),
                    mlp_ratio=float(h.patch_causal_mlp_ratio),
                    dropout=float(h.patch_causal_dropout),
                    norm_eps=float(h.patch_norm_eps),
                )
                for _ in range(int(h.patch_causal_encoder_num_layers))
            ]
        )
        self.norm = RMSNorm(self.d_model, eps=float(h.patch_norm_eps))
        self.out_proj = nn.Linear(self.d_model, self.latent_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        valid_lengths: torch.Tensor,
        mae_patch_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        patches = patchify(x, self.patch_size)
        h = self.patch_proj(patches)
        if mae_patch_mask is not None:
            h = torch.where(mae_patch_mask.to(device=h.device, dtype=torch.bool).unsqueeze(-1), self.mask_embedding, h)
        attention_mask = _valid_patch_mask(valid_lengths, h.size(1), self.patch_size, h.device)
        for block in self.blocks:
            h = block(h, attention_mask=attention_mask, is_causal=True)
        h = self.norm(h)
        return self.out_proj(h).transpose(1, 2)


class PatchCausalDecoder(nn.Module):
    """
    Patch causal decoder.

    Input: [B, latent_dim, N].
    Output: [B, C, N * patch_size].
    Patch i can reconstruct from quantized tokens <= i only.
    """

    def __init__(self, h):
        super().__init__()
        self.patch_size = int(h.patch_size)
        self.output_channels = int(h.output_channels)
        self.d_model = int(h.patch_causal_d_model)
        self.latent_dim = int(h.latent_dim)

        self.in_proj = nn.Linear(self.latent_dim, self.d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=self.d_model,
                    nhead=int(h.patch_causal_nhead),
                    mlp_ratio=float(h.patch_causal_mlp_ratio),
                    dropout=float(h.patch_causal_dropout),
                    norm_eps=float(h.patch_norm_eps),
                )
                for _ in range(int(h.patch_causal_decoder_num_layers))
            ]
        )
        self.norm = RMSNorm(self.d_model, eps=float(h.patch_norm_eps))
        self.out_proj = nn.Linear(self.d_model, self.output_channels * self.patch_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, latent: torch.Tensor, valid_lengths: torch.Tensor | None = None) -> torch.Tensor:
        h = self.in_proj(latent.transpose(1, 2))
        attention_mask = None
        if valid_lengths is not None:
            attention_mask = _valid_patch_mask(valid_lengths, h.size(1), self.patch_size, h.device)
        for block in self.blocks:
            h = block(h, attention_mask=attention_mask, is_causal=True)
        h = self.norm(h)
        patches = self.out_proj(h)
        return unpatchify(patches, self.output_channels, self.patch_size)
