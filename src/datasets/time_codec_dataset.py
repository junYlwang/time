from __future__ import annotations

import json
import os
import random
from bisect import bisect_right
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from .time_moe_dataset import TimeMoEDataset


class SplitTimeSeriesCodecDataset(Dataset):
    def __init__(
        self,
        split_manifest_path: str,
        split: str,
        segment_length: int,
        normalization_method=None,
        samples_per_epoch: int = 500000,
        max_valid_sequences: int = 2000,
        seed: int = 1234,
        return_valid_length: bool = False,
        min_input_length: int = 505,
    ):
        self.split = str(split)
        self.segment_length = int(segment_length)
        self.samples_per_epoch = int(samples_per_epoch)
        self.max_valid_sequences = int(max_valid_sequences)
        self.seed = int(seed)
        self.return_valid_length = bool(return_valid_length)
        self.min_input_length = int(min_input_length)
        self.rank = int(os.environ.get("RANK", "0"))
        self._sample_counter = 0

        with open(split_manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.roots = [str(root) for root in manifest[self.split]]
        self.datasets: List[TimeMoEDataset] = [
            TimeMoEDataset(root, normalization_method=normalization_method)
            for root in self.roots
        ]

        self.dataset_cumsum = [0]
        for ds in self.datasets:
            self.dataset_cumsum.append(self.dataset_cumsum[-1] + len(ds))
        self.num_sequences = self.dataset_cumsum[-1]

        self.valid_seq_indices = []
        if self.split != "train":
            self.valid_seq_indices = self._build_valid_indices()

    def __len__(self) -> int:
        if self.split == "train":
            return self.samples_per_epoch
        return len(self.valid_seq_indices)

    def _map_global_seq_idx(self, seq_idx: int):
        ds_idx = bisect_right(self.dataset_cumsum, int(seq_idx)) - 1
        local_idx = int(seq_idx) - self.dataset_cumsum[ds_idx]
        return ds_idx, local_idx

    def _sequence_length(self, global_seq_idx: int) -> int:
        ds_idx, local_idx = self._map_global_seq_idx(global_seq_idx)
        return int(self.datasets[ds_idx].get_sequence_length_by_idx(local_idx))

    def _fetch_seq(self, global_seq_idx: int) -> np.ndarray:
        ds_idx, local_idx = self._map_global_seq_idx(global_seq_idx)
        return np.asarray(self.datasets[ds_idx][local_idx], dtype=np.float32)

    def _rng_for_item(self, idx: int) -> random.Random:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        counter = self._sample_counter
        self._sample_counter += 1
        seed = (
            self.seed
            + self.rank * 1000003
            + worker_id * 1000033
            + int(idx) * 1000037
            + counter * 1000039
        )
        return random.Random(seed)

    def _build_valid_indices(self) -> List[int]:
        rng = random.Random(self.seed + 271828)
        order = list(range(self.num_sequences))
        rng.shuffle(order)

        valid_indices = []
        for global_idx in order:
            if self._sequence_length(global_idx) >= self.min_input_length:
                valid_indices.append(global_idx)
                if len(valid_indices) >= self.max_valid_sequences:
                    break
        if not valid_indices:
            raise RuntimeError(f"No {self.split} sequence with length >= {self.min_input_length}")
        return valid_indices

    def _segment_from_seq(self, seq: np.ndarray, start: int):
        seg = seq[start:start + self.segment_length]
        valid_length = min(int(seg.size), self.segment_length)
        if seg.size < self.segment_length:
            pad_width = self.segment_length - seg.size
            seg = np.pad(seg, (pad_width, 0), mode="constant", constant_values=0.0)
        return seg, valid_length

    def _format_item(self, seg: np.ndarray, valid_length: int):
        x = torch.from_numpy(seg).unsqueeze(0)
        if not self.return_valid_length:
            return x
        return {
            "x": x,
            "valid_length": torch.tensor(int(valid_length), dtype=torch.long),
        }

    def __getitem__(self, idx: int):
        if self.split == "train":
            rng = self._rng_for_item(int(idx))
            for _ in range(1024):
                global_seq_idx = rng.randrange(self.num_sequences)
                if self._sequence_length(global_seq_idx) >= self.min_input_length:
                    seq = self._fetch_seq(global_seq_idx)
                    break
            else:
                raise RuntimeError(f"Could not sample a train sequence with length >= {self.min_input_length}")

            if seq.size >= self.segment_length:
                start = rng.randrange(0, seq.size - self.segment_length + 1)
            else:
                start = 0
        else:
            seq = self._fetch_seq(self.valid_seq_indices[int(idx)])
            start = 0

        seg, valid_length = self._segment_from_seq(seq, start)
        return self._format_item(seg, valid_length)


__all__ = ["SplitTimeSeriesCodecDataset"]
