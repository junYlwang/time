#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from modules.time_thinker_talker import TimeSeriesMTPTransformer, TimeSeriesProjector, TimeSeriesTalker
from modules.utils import AttrDict


def main() -> int:
    h = AttrDict(
        {
            "latent_dim": 32,
            "projector_hidden_dim": 64,
            "projector_num_layers": 2,
            "projector_dropout": 0.0,
            "rvq_codebook_size": 128,
            "num_quantizers": 4,
            "talker_d_model": 64,
            "talker_num_layers": 2,
            "talker_nhead": 8,
            "talker_mlp_ratio": 2.0,
            "talker_dropout": 0.0,
            "talker_norm_eps": 1.0e-6,
            "mtp_d_model": 32,
            "mtp_num_layers": 2,
            "mtp_nhead": 4,
            "mtp_mlp_ratio": 2.0,
            "mtp_dropout": 0.0,
            "mtp_norm_eps": 1.0e-6,
            "mtp_loss_weight": 1.0,
        }
    )
    torch.manual_seed(1234)
    b = 2
    n_hist = 16
    n_future = 5
    qwen_hidden = 96
    projector = TimeSeriesProjector(h, qwen_hidden)
    talker = TimeSeriesTalker(h, qwen_hidden)
    mtp = TimeSeriesMTPTransformer(h)

    latent = torch.randn(b, h.latent_dim, n_hist)
    thinker_inputs = projector(latent)
    assert thinker_inputs.shape == (b, n_hist, qwen_hidden)

    thinker_hidden = torch.randn(b, n_hist, qwen_hidden)
    thinker_mask = torch.ones(b, n_hist, dtype=torch.bool)
    codes = torch.randint(0, h.rvq_codebook_size, (b, h.num_quantizers, n_future))
    layer0_logits_all, talker_hidden_all = talker(codes[:, 0, :-1], thinker_hidden, thinker_mask)
    layer0_logits = layer0_logits_all[:, :n_future, :]
    talker_hidden = talker_hidden_all[:, :n_future, :]
    assert layer0_logits.shape == (b, n_future, h.rvq_codebook_size)
    assert talker_hidden.shape == (b, n_future, h.talker_d_model)

    mtp_logits = mtp(talker_hidden, codes[:, 0, :], codes[:, 1:, :])
    assert len(mtp_logits) == h.num_quantizers - 1
    for logits in mtp_logits:
        assert logits.shape == (b, n_future, h.rvq_codebook_size)

    loss = F.cross_entropy(layer0_logits.reshape(-1, h.rvq_codebook_size), codes[:, 0, :].reshape(-1))
    for idx, logits in enumerate(mtp_logits, start=1):
        loss = loss + F.cross_entropy(logits.reshape(-1, h.rvq_codebook_size), codes[:, idx, :].reshape(-1))
    loss.backward()

    prefix = torch.empty(b, 0, dtype=torch.long)
    generated = []
    with torch.no_grad():
        for _ in range(n_future):
            token0, hidden = talker.generate_layer0_step(prefix, thinker_hidden, thinker_mask, temperature=0.0)
            generated.append(token0)
            prefix = torch.cat([prefix, token0[:, None]], dim=1)
        layer0 = torch.stack(generated, dim=1)
        logits_all, hidden_all = talker(layer0[:, :-1], thinker_hidden, thinker_mask)
        high = mtp.generate(hidden_all[:, :n_future, :], layer0, temperature=0.0)
        all_codes = torch.cat([layer0[:, None, :], high], dim=1)
        assert all_codes.shape == (b, h.num_quantizers, n_future)

    print("smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
