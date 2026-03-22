from __future__ import annotations

import random
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .time_moe_dataset import TimeMoEDataset


class TimeSeriesSegmentIterableDataset(IterableDataset):
    """Randomly sample fixed-length 1D segments from TimeMoE/Time-300B dataset."""

    def __init__(
        self,
        ts_dataset: TimeMoEDataset,
        segment_length: int,
        samples_per_epoch: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 1234,
    ):
        super().__init__()
        self.ts_dataset = ts_dataset
        self.segment_length = int(segment_length)
        self.samples_per_epoch = int(samples_per_epoch)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        n_workers = worker.num_workers if worker is not None else 1

        rng_seed = self.seed + 10007 * self.rank + 313 * worker_id
        rng = random.Random(rng_seed)

        total = self.samples_per_epoch
        local_quota = max(1, total // max(1, self.world_size * n_workers))

        n_seqs = len(self.ts_dataset)
        for _ in range(local_quota):
            for _retry in range(16):
                seq_idx = rng.randrange(n_seqs)
                seq = np.asarray(self.ts_dataset[seq_idx], dtype=np.float32)
                if seq.size >= 2:
                    break
            else:
                seq = np.zeros((self.segment_length,), dtype=np.float32)

            if seq.size >= self.segment_length:
                start = rng.randrange(0, seq.size - self.segment_length + 1)
                seg = seq[start : start + self.segment_length]
            else:
                pad = self.segment_length - seq.size
                seg = np.pad(seq, (0, pad), mode="constant", constant_values=0.0)

            # [C, T], C=1
            yield torch.from_numpy(seg).unsqueeze(0)


class TimeSeriesSegmentEvalDataset(Dataset):
    """Deterministic fixed-length eval segments from the first N sequences."""

    def __init__(
        self,
        ts_dataset: TimeMoEDataset,
        segment_length: int,
        max_eval_sequences: int = 256,
    ):
        self.ts_dataset = ts_dataset
        self.segment_length = int(segment_length)
        self.seq_indices = list(range(min(int(max_eval_sequences), len(ts_dataset))))

    def __len__(self) -> int:
        return len(self.seq_indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = np.asarray(self.ts_dataset[self.seq_indices[idx]], dtype=np.float32)
        if seq.size >= self.segment_length:
            seg = seq[: self.segment_length]
        else:
            seg = np.pad(seq, (0, self.segment_length - seq.size), mode="constant", constant_values=0.0)
        return torch.from_numpy(seg).unsqueeze(0)


__all__ = [
    "TimeSeriesSegmentIterableDataset",
    "TimeSeriesSegmentEvalDataset",
]
