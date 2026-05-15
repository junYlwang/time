from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class LinearClassifierProbe(_BaseProbe):
    def __init__(self, h):
        super().__init__()
        in_dim = int(h.latent_dim) * int(h.latent_seq_len)
        self.classifier = nn.Linear(in_dim, int(h.num_classes))
        self.apply(self._init_weights)

    def forward(self, x):
        x = self._flatten_latent(x)
        return self.classifier(x)


class MLPClassifierProbe(_BaseProbe):
    def __init__(self, h):
        super().__init__()
        in_dim = int(h.latent_dim) * int(h.latent_seq_len)
        hidden_dim = int(getattr(h, "probe_hidden_dim", min(max(in_dim // 2, 32), 512)))
        dropout = float(getattr(h, "probe_dropout", 0.0))

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, int(h.num_classes)),
        )
        self.apply(self._init_weights)

    def forward(self, x):
        x = self._flatten_latent(x)
        return self.classifier(x)


class MLPForecastProbe(_BaseProbe):
    def __init__(self, h):
        super().__init__()
        in_dim = int(h.latent_dim) * int(h.latent_seq_len)
        hidden_dim = int(getattr(h, "probe_hidden_dim", min(max(in_dim // 2, 32), 512)))
        dropout = float(getattr(h, "probe_dropout", 0.0))
        pred_len = int(h.pred_len)

        self.pred_len = pred_len
        self.forecaster = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len),
        )
        self.apply(self._init_weights)

    def forward(self, x):
        x = self._flatten_latent(x)
        out = self.forecaster(x)
        return out.unsqueeze(1)  # [B, 1, pred_len]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(self.dropout(F.silu(self.w1(x)) * self.w2(x)))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype)[None, None, :, :]
        sin = emb.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.head_dim = self.d_model // self.nhead
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {self.head_dim}")

        self.norm1 = RMSNorm(d_model, eps=norm_eps)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = float(dropout)

        self.norm2 = RMSNorm(d_model, eps=norm_eps)
        ffn_dim = int(d_model * mlp_ratio)
        self.ffn = SwiGLU(d_model, ffn_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        h = self.norm1(x) #prenorm
        qkv = self.qkv(h).view(b, l, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, L, Hd]
        cos, sin = self.rope(l, x.device, x.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).contiguous().view(b, l, d)
        x = x + self.dropout(self.proj(attn))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TransformerForecastProbe(_BaseProbe):
    def __init__(self, h):
        super().__init__()
        latent_dim = int(h.latent_dim)
        latent_seq_len = int(h.latent_seq_len)
        pred_len = int(h.pred_len)
        d_model = int(getattr(h, "transformer_d_model", 256))
        nhead = int(getattr(h, "transformer_nhead", 8))
        num_layers = int(getattr(h, "transformer_num_layers", 4))
        mlp_ratio = float(getattr(h, "transformer_mlp_ratio", 4.0))
        dropout = float(getattr(h, "transformer_dropout", 0.0))

        self.pred_len = pred_len
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
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(latent_seq_len * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )
        self.apply(self._init_weights)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, L, D]
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        out = self.head(x)
        return out.unsqueeze(1)  # [B, 1, pred_len]


# Backward-compatible aliases used by existing UCR code.
LinearProbe = LinearClassifierProbe
MLPProbe = MLPClassifierProbe
