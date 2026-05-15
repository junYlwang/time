from __future__ import annotations

import os
import sys
from bisect import bisect_right
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
if _DATA_DIR not in sys.path:
    sys.path.insert(0, _DATA_DIR)

from read_ett import get_ett_csv_path, get_ett_split_indices


DEFAULT_ETT_DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")


@dataclass(frozen=True)
class _VariableWindows:
    dataset_name: str
    feature_name: str
    values: np.ndarray
    num_windows: int


class ETTCodecDataset(Dataset):
    """ETT reconstruction dataset for single-variable codec fine-tuning.

    The CSV values are treated as the codec raw domain. Each ETT variable is
    exposed as an independent univariate sequence, and stride windows are
    returned as [1, segment_length] tensors.
    """

    def __init__(
        self,
        ett_root: str,
        dataset_names: Sequence[str] | None = None,
        split: str = "train",
        segment_length: int = 512,
        stride: int = 1,
    ) -> None:
        self.ett_root = str(ett_root)
        self.dataset_names = tuple(dataset_names or DEFAULT_ETT_DATASETS)
        self.split = str(split).strip().lower()
        self.segment_length = int(segment_length)
        self.stride = int(stride)

        if self.split not in {"train", "val", "valid", "test"}:
            raise ValueError(f"Unsupported split: {split}. Expected train, val, valid, or test.")
        if self.split == "valid":
            self.split = "val"
        if self.segment_length <= 0:
            raise ValueError(f"segment_length must be > 0, got {segment_length}")
        if self.stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}")
        if not self.dataset_names:
            raise ValueError("dataset_names must not be empty")

        self.variables: list[_VariableWindows] = []
        self.cumsum = [0]
        self._load_variables()

        if self.cumsum[-1] <= 0:
            raise ValueError(
                f"No ETT codec windows for split={self.split}, "
                f"segment_length={self.segment_length}, stride={self.stride}"
            )

    def _load_variables(self) -> None:
        for dataset_name in self.dataset_names:
            path = get_ett_csv_path(self.ett_root, dataset_name)
            frame = pd.read_csv(path)
            if "date" not in frame.columns:
                raise ValueError(f"Expected 'date' column in {path}")

            split_indices = get_ett_split_indices(dataset_name, len(frame), seq_len=0)
            start, end = getattr(split_indices, self.split)

            feature_names = [c for c in frame.columns if c != "date"]
            if not feature_names:
                raise ValueError(f"No feature columns found in {path}")

            values_tc = frame.loc[start:end - 1, feature_names].to_numpy(dtype=np.float32)
            if values_tc.shape[0] < self.segment_length:
                continue

            num_windows = (values_tc.shape[0] - self.segment_length) // self.stride + 1
            for feature_idx, feature_name in enumerate(feature_names):
                values = np.ascontiguousarray(values_tc[:, feature_idx], dtype=np.float32)
                self.variables.append(
                    _VariableWindows(
                        dataset_name=str(dataset_name),
                        feature_name=str(feature_name),
                        values=values,
                        num_windows=int(num_windows),
                    )
                )
                self.cumsum.append(self.cumsum[-1] + int(num_windows))

    def __len__(self) -> int:
        return self.cumsum[-1]

    def _map_index(self, idx: int) -> tuple[_VariableWindows, int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        var_idx = bisect_right(self.cumsum, idx) - 1
        local_idx = idx - self.cumsum[var_idx]
        return self.variables[var_idx], local_idx

    def __getitem__(self, idx: int) -> torch.Tensor:
        var, local_idx = self._map_index(int(idx))
        start = local_idx * self.stride
        end = start + self.segment_length
        segment = var.values[start:end]
        return torch.from_numpy(segment).unsqueeze(0)

    def summary(self) -> dict:
        by_dataset: dict[str, dict[str, int]] = {}
        for var in self.variables:
            item = by_dataset.setdefault(var.dataset_name, {"num_variables": 0, "num_windows": 0})
            item["num_variables"] += 1
            item["num_windows"] += int(var.num_windows)
        return {
            "ett_root": self.ett_root,
            "dataset_names": list(self.dataset_names),
            "split": self.split,
            "segment_length": self.segment_length,
            "stride": self.stride,
            "num_variables": len(self.variables),
            "num_windows": len(self),
            "by_dataset": by_dataset,
        }


__all__ = ["DEFAULT_ETT_DATASETS", "ETTCodecDataset"]
