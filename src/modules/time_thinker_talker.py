from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .probe import RMSNorm, SwiGLU, TransformerBlock


class TimeSeriesProjector(nn.Module):
    def __init__(self, h, qwen_hidden_size: int):
        super().__init__()
        latent_dim = int(h.latent_dim)
        hidden_dim = int(h.projector_hidden_dim)
        num_layers = int(h.projector_num_layers)
        dropout = float(h.projector_dropout)
        out_dim = int(qwen_hidden_size)

        layers = []
        in_dim = latent_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3:
            raise ValueError(f"Expected latent [B, D, N], got {tuple(latent.shape)}")
        return self.net(latent.transpose(1, 2))


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, mlp_ratio: float, dropout: float, norm_eps: float):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.head_dim = self.d_model // self.nhead
        self.norm_q = RMSNorm(d_model, eps=norm_eps)
        self.norm_kv = RMSNorm(d_model, eps=norm_eps)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = float(dropout)
        self.norm_ffn = RMSNorm(d_model, eps=norm_eps)
        self.ffn = SwiGLU(d_model, int(d_model * mlp_ratio), dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor | None) -> torch.Tensor:
        b, l, d = x.shape
        _, s, _ = context.shape
        q = self.q_proj(self.norm_q(x)).view(b, l, self.nhead, self.head_dim).transpose(1, 2)
        kv = self.norm_kv(context)
        k = self.k_proj(kv).view(b, s, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(b, s, self.nhead, self.head_dim).transpose(1, 2)
        attn_mask = None
        if context_mask is not None:
            valid = context_mask.to(device=x.device, dtype=torch.bool).view(b, 1, 1, s)
            attn_mask = torch.zeros(b, 1, 1, s, device=x.device, dtype=x.dtype)
            attn_mask = attn_mask.masked_fill(~valid, torch.finfo(x.dtype).min)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(b, l, d)
        x = x + self.dropout(self.out_proj(out))
        x = x + self.dropout(self.ffn(self.norm_ffn(x)))
        return x


class TimeSeriesTalkerBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.self_attn = TransformerBlock(
            d_model=int(h.talker_d_model),
            nhead=int(h.talker_nhead),
            mlp_ratio=float(h.talker_mlp_ratio),
            dropout=float(h.talker_dropout),
            norm_eps=float(h.talker_norm_eps),
        )
        self.cross_attn = CrossAttentionBlock(
            d_model=int(h.talker_d_model),
            nhead=int(h.talker_nhead),
            mlp_ratio=float(h.talker_mlp_ratio),
            dropout=float(h.talker_dropout),
            norm_eps=float(h.talker_norm_eps),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor | None) -> torch.Tensor:
        x = self.self_attn(x, is_causal=True)
        return self.cross_attn(x, context, context_mask)


class TimeSeriesTalker(nn.Module):
    def __init__(self, h, qwen_hidden_size: int):
        super().__init__()
        self.codebook_size = int(h.rvq_codebook_size)
        self.d_model = int(h.talker_d_model)
        self.code_embedding = nn.Embedding(self.codebook_size, self.d_model)
        self.bos = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.context_proj = nn.Linear(int(qwen_hidden_size), self.d_model)
        self.blocks = nn.ModuleList([TimeSeriesTalkerBlock(h) for _ in range(int(h.talker_num_layers))])
        self.norm = RMSNorm(self.d_model, eps=float(h.talker_norm_eps))
        self.head0 = nn.Linear(self.d_model, self.codebook_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias, 0)

    def embed_layer0_prefix(self, layer0_prefix: torch.Tensor) -> torch.Tensor:
        return self.code_embedding(layer0_prefix.long())

    def forward(self, layer0_inputs: torch.Tensor, thinker_hidden: torch.Tensor, thinker_mask: torch.Tensor | None):
        b = layer0_inputs.size(0)
        context = self.context_proj(thinker_hidden)
        bos = self.bos.expand(b, 1, self.d_model)
        if layer0_inputs.size(1) == 0:
            x = bos
        else:
            x = torch.cat([bos, self.embed_layer0_prefix(layer0_inputs)], dim=1)
        for block in self.blocks:
            x = block(x, context, thinker_mask)
        hidden = self.norm(x)
        logits = self.head0(hidden)
        return logits, hidden

    @torch.no_grad()
    def generate_layer0_step(self, prefix: torch.Tensor, thinker_hidden: torch.Tensor, thinker_mask: torch.Tensor | None, temperature: float):
        logits, hidden = self(prefix, thinker_hidden, thinker_mask)
        step_logits = logits[:, -1, :]
        if temperature <= 0.0:
            token = step_logits.argmax(dim=-1)
        else:
            probs = torch.softmax(step_logits / temperature, dim=-1)
            token = torch.multinomial(probs, num_samples=1).squeeze(1)
        return token, hidden[:, -1, :]


class TimeSeriesMTPTransformer(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.num_quantizers = int(h.num_quantizers)
        self.codebook_size = int(h.rvq_codebook_size)
        self.talker_d_model = int(h.talker_d_model)
        self.d_model = int(h.mtp_d_model)
        if self.num_quantizers < 2:
            raise ValueError("TimeSeriesMTPTransformer requires num_quantizers >= 2")
        self.layer0_embedding = nn.Embedding(self.codebook_size, self.d_model)
        self.residual_embeddings = nn.ModuleList(
            [nn.Embedding(self.codebook_size, self.d_model) for _ in range(self.num_quantizers - 2)]
        )
        self.talker_proj = nn.Linear(self.talker_d_model, self.d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.num_quantizers - 1, self.d_model))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=self.d_model,
                    nhead=int(h.mtp_nhead),
                    mlp_ratio=float(h.mtp_mlp_ratio),
                    dropout=float(h.mtp_dropout),
                    norm_eps=float(h.mtp_norm_eps),
                )
                for _ in range(int(h.mtp_num_layers))
            ]
        )
        self.norm = RMSNorm(self.d_model, eps=float(h.mtp_norm_eps))
        self.heads = nn.ModuleList([nn.Linear(self.d_model, self.codebook_size) for _ in range(self.num_quantizers - 1)])
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, talker_hidden: torch.Tensor, layer0_codes: torch.Tensor, residual_teacher: torch.Tensor | None = None):
        b, n, _ = talker_hidden.shape
        q_minus_1 = self.num_quantizers - 1
        h = self.talker_proj(talker_hidden).reshape(b * n, 1, self.d_model)
        layer0 = self.layer0_embedding(layer0_codes.long()).reshape(b * n, 1, self.d_model)
        tokens = [h + layer0]
        if q_minus_1 > 1:
            for idx, emb in enumerate(self.residual_embeddings):
                if residual_teacher is not None and idx < residual_teacher.size(1):
                    prev = residual_teacher[:, idx, :].reshape(b * n).long()
                else:
                    prev = torch.zeros(b * n, dtype=torch.long, device=talker_hidden.device)
                tokens.append(h + emb(prev).unsqueeze(1))
        x = torch.cat(tokens, dim=1) + self.pos
        for block in self.blocks:
            x = block(x, is_causal=True)
        x = self.norm(x)
        logits = [head(x[:, idx, :]).view(b, n, self.codebook_size) for idx, head in enumerate(self.heads)]
        return logits

    @torch.no_grad()
    def generate(self, talker_hidden: torch.Tensor, layer0_codes: torch.Tensor, temperature: float):
        b, n, _ = talker_hidden.shape
        generated = []
        for idx in range(self.num_quantizers - 1):
            teacher = None
            if generated:
                teacher = torch.stack(generated, dim=1)
            logits = self.forward(talker_hidden, layer0_codes, teacher)[idx]
            step_logits = logits[:, :, :]
            if temperature <= 0.0:
                token = step_logits.argmax(dim=-1)
            else:
                probs = torch.softmax(step_logits / temperature, dim=-1)
                token = torch.multinomial(probs.reshape(-1, self.codebook_size), num_samples=1).view(b, n)
            generated.append(token)
        return torch.stack(generated, dim=1)
