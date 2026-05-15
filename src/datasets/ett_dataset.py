import os
import sys

import torch
from torch.utils.data import Dataset

_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
if _DATA_DIR not in sys.path:
    sys.path.insert(0, _DATA_DIR)

from read_ett import load_ett_forecasting_split


class ETTDataset(Dataset):
    def __init__(
        self,
        ett_root,
        dataset_name,
        split="train",
        seq_len=96,
        pred_len=96,
        stride=1,
        column="OT",
    ):
        self.ett_root = str(ett_root)
        self.dataset_name = str(dataset_name)
        self.split = str(split).lower()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.stride = int(stride)
        self.column = None if column in (None, "__all__") else str(column)

        self.inputs, self.targets, self.metadata = load_ett_forecasting_split(
            ett_root=self.ett_root,
            dataset_name=self.dataset_name,
            split=self.split,
            column=self.column,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            stride=self.stride,
        )

        self.num_windows = int(self.inputs.shape[0])
        self.num_variables = int(self.inputs.shape[1])
        self.feature_names = list(getattr(self.metadata, "feature_names", []))
        if self.column is not None:
            self.selected_feature_names = [self.column]
        else:
            self.selected_feature_names = self.feature_names

        self.inputs = self.inputs.reshape(self.num_windows * self.num_variables, 1, self.seq_len)
        self.targets = self.targets.reshape(self.num_windows * self.num_variables, 1, self.pred_len)
        self.num_channels = 1
        self.num_samples = int(self.inputs.shape[0])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.as_tensor(self.inputs[idx], dtype=torch.float32)
        y = torch.as_tensor(self.targets[idx], dtype=torch.float32)
        return {
            "seq": x,       # [1, seq_len]
            "target": y,    # [1, pred_len]
        }

    def summary(self):
        return {
            "ett_root": self.ett_root,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "selected_column": "__all__" if self.column is None else self.column,
            "feature_names": self.selected_feature_names,
            "num_variables": self.num_variables,
            "num_windows": self.num_windows,
            "num_samples": self.num_samples,
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "stride": self.stride,
        }
