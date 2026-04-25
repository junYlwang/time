import torch
import os
import sys
from torch.utils.data import Dataset

_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
if _DATA_DIR not in sys.path:
    sys.path.insert(0, _DATA_DIR)

from read_ucr import load_ucr_split

class UCRDataset(Dataset):
    def __init__(self, ucr_root, dataset_name, split="TRAIN"):
        self.dataset_name = str(dataset_name)
        self.split = str(split).upper()
        self.sequences, self.labels, self.metadata = load_ucr_split(ucr_root, dataset_name, split)
        if self.metadata.min_length != self.metadata.max_length:
            raise ValueError(
                f"Sequences are not equal length in dataset={self.dataset_name}"
            )
        self.seq_len = self.metadata.max_length
        self.class_names = self.metadata.class_labels
        self.label_to_index = {label: i for i, label in enumerate(self.class_names)}
        self.index_to_label = {i: label for i, label in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        self.label_ids = torch.tensor(
            [self.label_to_index[label] for label in self.labels],
            dtype=torch.long,
        )

    def __len__(self):
        return self.metadata.num_samples

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = torch.as_tensor(seq, dtype=torch.float32).unsqueeze(0) #[B, 1, T]
        y = self.label_ids[idx]
        item = {
            "seq": x,
            "label": y,
        }

        return item
