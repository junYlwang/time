from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_win_sizes(win_sizes: Iterable[int]) -> List[int]:
    out: List[int] = []
    for w in win_sizes:
        w_int = int(w)
        if w_int <= 0:
            raise ValueError(f"win_size must be > 0, got {w_int}")
        if w_int & (w_int - 1):
            raise ValueError(f"win_size must be a power of 2, got {w_int}")
        out.append(w_int)
    if not out:
        raise ValueError("win_sizes must not be empty")
    return out


class MultiScaleLogMagSTFTLoss(nn.Module):
    def __init__(self, win_sizes: Iterable[int]):
        super().__init__()
        self.win_sizes = _validate_win_sizes(win_sizes)

    def forward(self, x_rec: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
        if x_rec.shape != x_ref.shape:
            raise ValueError(f"x_rec and x_ref must have the same shape, got {x_rec.shape} vs {x_ref.shape}")
        if x_rec.ndim != 3:
            raise ValueError(f"Expected [B, C, T], got {x_rec.shape}")

        bsz, channels, time_steps = x_rec.shape
        x_rec_flat = x_rec.reshape(bsz * channels, time_steps)
        x_ref_flat = x_ref.reshape(bsz * channels, time_steps)

        losses = []
        for win_size in self.win_sizes:
            hop_size = win_size // 4
            window = torch.hann_window(win_size, dtype=x_rec.dtype, device=x_rec.device)

            rec_spec = torch.stft(
                x_rec_flat,
                n_fft=win_size,
                hop_length=hop_size,
                win_length=win_size,
                window=window,
                return_complex=True,
            )
            ref_spec = torch.stft(
                x_ref_flat,
                n_fft=win_size,
                hop_length=hop_size,
                win_length=win_size,
                window=window,
                return_complex=True,
            )

            rec_log_mag = torch.log1p(torch.abs(rec_spec))
            ref_log_mag = torch.log1p(torch.abs(ref_spec))
            losses.append(F.l1_loss(rec_log_mag, ref_log_mag))

        return torch.stack(losses).mean()
