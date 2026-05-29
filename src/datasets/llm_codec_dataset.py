from __future__ import annotations

import json
import os
import random
from bisect import bisect_right
import numpy as np
import torch
from torch.utils.data import Dataset

from .time_moe_dataset import TimeMoEDataset


class SplitRawSeriesDataset(Dataset):
    def __init__(
        self,
        split_manifest_path: str,
        split: str,
        segment_length: int,
        max_valid_sequences: int,
        seed: int,
        min_points: int = 504,
    ):
        self.split = str(split)
        self.segment_length = int(segment_length)
        self.max_valid_sequences = int(max_valid_sequences)
        self.seed = int(seed)
        rank = int(os.environ.get("RANK", "0"))
        self.rng = random.Random(self.seed + rank * 1000003)
        self.min_points = int(min_points)

        with open(split_manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.roots = [str(root) for root in manifest[self.split]]
        self.datasets = [TimeMoEDataset(root, normalization_method=None) for root in self.roots]
        self.dataset_cumsum = [0]
        for ds in self.datasets:
            self.dataset_cumsum.append(self.dataset_cumsum[-1] + len(ds))
        self.num_sequences = self.dataset_cumsum[-1]

        self.valid_indices = []
        if self.split in ("valid", "test"):
            rng = random.Random(self.seed + 271828)
            order = list(range(self.num_sequences))
            rng.shuffle(order)
            for seq_idx in order:
                if len(self.valid_indices) >= self.max_valid_sequences:
                    break
                if self.get_sequence_length(seq_idx) > self.min_points:
                    self.valid_indices.append(seq_idx)
            if not self.valid_indices:
                raise ValueError(f"Split '{self.split}' has no sequences longer than {self.min_points}")

    def __len__(self) -> int:
        if self.split == "train":
            return self.num_sequences
        return len(self.valid_indices)

    def get_sequence_length(self, global_seq_idx: int) -> int:
        ds_idx = bisect_right(self.dataset_cumsum, int(global_seq_idx)) - 1
        local_idx = int(global_seq_idx) - self.dataset_cumsum[ds_idx]
        return int(self.datasets[ds_idx].get_sequence_length_by_idx(local_idx))

    def fetch_sequence(self, global_seq_idx: int) -> np.ndarray:
        ds_idx = bisect_right(self.dataset_cumsum, int(global_seq_idx)) - 1
        local_idx = int(global_seq_idx) - self.dataset_cumsum[ds_idx]
        return np.asarray(self.datasets[ds_idx][local_idx], dtype=np.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.split == "train":
            for _ in range(16):
                global_seq_idx = self.rng.randrange(self.num_sequences)
                if self.get_sequence_length(global_seq_idx) > self.min_points:
                    break
            else:
                raise RuntimeError(f"Could not sample a sequence longer than {self.min_points} points")
            seq = self.fetch_sequence(global_seq_idx)
            if seq.size > self.segment_length:
                start = self.rng.randrange(0, seq.size - self.segment_length + 1)
                seq = seq[start:start + self.segment_length]
        else:
            seq = self.fetch_sequence(self.valid_indices[int(idx)])
            if seq.size > self.segment_length:
                seq = seq[:self.segment_length]

        valid_length = int(seq.size)
        x = torch.zeros(1, self.segment_length, dtype=torch.float32)
        x[:, -valid_length:] = torch.from_numpy(seq).view(1, -1)
        return {
            "raw_values": x,
            "valid_lengths": torch.tensor(valid_length, dtype=torch.long),
        }
