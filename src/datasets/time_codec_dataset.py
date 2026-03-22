from __future__ import annotations

import json
import random
from bisect import bisect_right
from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .time_moe_dataset import TimeMoEDataset


def _normalize_split_name(split: str) -> str:
    s = split.lower().strip()
    if s == "val":
        return "valid"
    return s


def _extract_split_roots(manifest_obj, split: str) -> List[str]:
    split = _normalize_split_name(split)

    if isinstance(manifest_obj, dict):
        # Try exact key first.
        value = manifest_obj.get(split)
        if value is None and split == "valid":
            value = manifest_obj.get("val")
        if value is None:
            raise KeyError(f"Split '{split}' not found in split manifest")
    elif isinstance(manifest_obj, list):
        # Backward-compatible: plain list means train split.
        if split != "train":
            raise KeyError("List-style manifest only supports split='train'")
        value = manifest_obj
    else:
        raise ValueError("Split manifest must be a dict or a list")

    if isinstance(value, dict):
        value = value.get("roots", value.get("paths", None))

    if not isinstance(value, list):
        raise ValueError(f"Split '{split}' must map to a list of paths")

    roots = [str(p) for p in value]
    if len(roots) == 0:
        raise ValueError(f"Split '{split}' has no dataset roots")
    return roots


class SplitTimeSeriesCodecDataset(Dataset):
    """
    Unified dataset for both training and validation.

    - split='train': random segment sampling from selected roots.
    - split='valid': deterministic fixed segments from selected roots.
    """

    def __init__(
        self,
        split_manifest_path: str,
        split: str,
        segment_length: int,
        normalization_method=None,
        samples_per_epoch: int = 100000,
        max_valid_sequences: int = 2000,
        seed: int = 1234,
    ):
        self.split = _normalize_split_name(split)
        self.segment_length = int(segment_length)
        self.samples_per_epoch = int(samples_per_epoch)
        self.max_valid_sequences = int(max_valid_sequences)
        self.seed = int(seed)
        self.epoch = 0

        with open(split_manifest_path, "r", encoding="utf-8") as f:
            manifest_obj = json.load(f)
        self.roots = _extract_split_roots(manifest_obj, self.split)

        self.datasets: List[TimeMoEDataset] = [
            TimeMoEDataset(root, normalization_method=normalization_method)
            for root in self.roots
        ]

        self.dataset_cumsum = [0]
        for ds in self.datasets:
            self.dataset_cumsum.append(self.dataset_cumsum[-1] + len(ds))
        self.num_sequences = self.dataset_cumsum[-1]
        self.dataset_lengths = [len(ds) for ds in self.datasets]

        if self.num_sequences <= 0:
            raise ValueError(f"Split '{self.split}' has zero sequences")

        # Deterministic validation index set.
        self.valid_seq_indices: Sequence[int] = []
        if self.split in ("valid", "test"):
            self.valid_seq_indices = self._build_valid_indices()

    def _build_valid_indices(self) -> List[int]:
        target_total = min(self.max_valid_sequences, self.num_sequences)
        if target_total <= 0:
            return []

        nonempty = [i for i, n in enumerate(self.dataset_lengths) if n > 0]
        if not nonempty:
            return []

        quotas = [0 for _ in self.dataset_lengths]

        # When budget allows, guarantee at least one sample from each non-empty subset.
        if target_total >= len(nonempty):
            for i in nonempty:
                quotas[i] = 1
            remaining = target_total - len(nonempty)
        else:
            # Not enough budget to cover all subsets: deterministic random pick of subsets.
            remaining = target_total
            pick_rng = random.Random(self.seed + 314159)
            chosen = pick_rng.sample(nonempty, k=target_total)
            for i in chosen:
                quotas[i] = 1
            remaining = 0

        if remaining > 0:
            capacities = [max(0, self.dataset_lengths[i] - quotas[i]) for i in range(len(self.datasets))]
            cap_total = sum(capacities)
            if cap_total > 0:
                before_add = sum(quotas)
                fractions = [remaining * (cap / cap_total) for cap in capacities]
                adds = [int(x) for x in fractions]
                for i, v in enumerate(adds):
                    quotas[i] += min(v, capacities[i])

                used_actual = sum(quotas) - before_add
                left = remaining - used_actual
                if left > 0:
                    # Largest-remainder method, deterministic tie-break by subset index.
                    order = sorted(
                        range(len(self.datasets)),
                        key=lambda i: (fractions[i] - adds[i], -i),
                        reverse=True,
                    )
                    for i in order:
                        if left == 0:
                            break
                        if quotas[i] < self.dataset_lengths[i]:
                            quotas[i] += 1
                            left -= 1

                    # Fallback if rounding/ties leave unused budget.
                    if left > 0:
                        for i in range(len(self.datasets)):
                            if left == 0:
                                break
                            extra_cap = self.dataset_lengths[i] - quotas[i]
                            if extra_cap <= 0:
                                continue
                            take = min(extra_cap, left)
                            quotas[i] += take
                            left -= take

        # Deterministic sample selection inside each subset.
        valid_indices: List[int] = []
        base_rng = random.Random(self.seed + 271828)
        for ds_idx, k in enumerate(quotas):
            if k <= 0:
                continue
            n = self.dataset_lengths[ds_idx]
            start = self.dataset_cumsum[ds_idx]
            if k >= n:
                local = list(range(n))
            else:
                local = base_rng.sample(range(n), k=k)
                local.sort()
            valid_indices.extend(start + li for li in local)

        # Mix subset blocks to avoid fixed per-subset ordering while staying deterministic.
        shuffle_rng = random.Random(self.seed + 1618033)
        shuffle_rng.shuffle(valid_indices)
        return valid_indices

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.split == "train":
            return self.samples_per_epoch
        return len(self.valid_seq_indices)

    def _map_global_seq_idx(self, seq_idx: int):
        ds_idx = bisect_right(self.dataset_cumsum, seq_idx) - 1
        local_idx = seq_idx - self.dataset_cumsum[ds_idx]
        return ds_idx, local_idx

    def _fetch_seq(self, global_seq_idx: int) -> np.ndarray:
        ds_idx, local_idx = self._map_global_seq_idx(global_seq_idx)
        seq = np.asarray(self.datasets[ds_idx][local_idx], dtype=np.float32)
        return seq

    def _segment_from_seq(self, seq: np.ndarray, start: int) -> np.ndarray:
        end = start + self.segment_length
        seg = seq[start:end]
        if seg.size < self.segment_length:
            seg = np.pad(seg, (0, self.segment_length - seg.size), mode="constant", constant_values=0.0)
        return seg

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.split == "train":
            rng = random.Random(self.seed + self.epoch * 1000003 + idx)

            for _ in range(16):
                global_seq_idx = rng.randrange(self.num_sequences)
                seq = self._fetch_seq(global_seq_idx)
                if seq.size >= 2:
                    break
            else:
                seq = np.zeros((self.segment_length,), dtype=np.float32)

            if seq.size >= self.segment_length:
                start = rng.randrange(0, seq.size - self.segment_length + 1)
            else:
                start = 0

            seg = self._segment_from_seq(seq, start)
            return torch.from_numpy(seg).unsqueeze(0)  # [1, T]

        # Validation/Test: deterministic selection and deterministic slicing.
        global_seq_idx = self.valid_seq_indices[idx]
        seq = self._fetch_seq(global_seq_idx)
        seg = self._segment_from_seq(seq, start=0)
        return torch.from_numpy(seg).unsqueeze(0)


__all__ = [
    "SplitTimeSeriesCodecDataset",
]
