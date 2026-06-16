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
        sampling_config=None,
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
        self.sampling_config = sampling_config or {}
        self.sequence_entries = None

        with open(split_manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        split_entry = manifest[self.split]
        self.sequence_indices: List[np.ndarray | None] = []
        if isinstance(split_entry, dict) and "sequence_manifest" in split_entry:
            sequence_manifest_path = str(split_entry["sequence_manifest"])
            if not os.path.isabs(sequence_manifest_path):
                sequence_manifest_path = os.path.join(
                    os.path.dirname(os.path.abspath(split_manifest_path)),
                    sequence_manifest_path,
                )
            with open(sequence_manifest_path, "r", encoding="utf-8") as sf:
                sequence_manifest = json.load(sf)
            if sequence_manifest.get("format") != "time300b_sequence_manifest_v1":
                raise ValueError(f"Unsupported sequence manifest format: {sequence_manifest_path}")
            entries = sequence_manifest["datasets"]
            self.sequence_entries = entries
            self.roots = [str(entry["root"]) for entry in entries]
            for entry in entries:
                selection = entry.get("selection", {})
                mode = selection.get("mode", "all")
                if mode == "all":
                    self.sequence_indices.append(None)
                elif mode == "indices_npy":
                    indices_path = str(selection["path"])
                    if not os.path.isabs(indices_path):
                        indices_path = os.path.join(
                            os.path.dirname(os.path.abspath(sequence_manifest_path)),
                            indices_path,
                        )
                    self.sequence_indices.append(np.load(indices_path).astype(np.int64, copy=False))
                else:
                    raise ValueError(f"Unsupported sequence selection mode: {mode}")
        else:
            self.roots = [str(root) for root in split_entry]
            self.sequence_indices = [None for _ in self.roots]
        self.datasets: List[TimeMoEDataset] = [
            TimeMoEDataset(root, normalization_method=normalization_method)
            for root in self.roots
        ]

        self.dataset_cumsum = [0]
        self.dataset_sequence_counts = []
        for ds, indices in zip(self.datasets, self.sequence_indices):
            ds_len = len(ds) if indices is None else int(indices.size)
            self.dataset_sequence_counts.append(int(ds_len))
            self.dataset_cumsum.append(self.dataset_cumsum[-1] + ds_len)
        self.num_sequences = self.dataset_cumsum[-1]
        self._init_sampling_metadata()

        self.valid_seq_indices = []
        if self.split != "train":
            self.valid_seq_indices = self._build_valid_indices()

    def _init_sampling_metadata(self) -> None:
        self.dataset_domains = []
        self.dataset_families = []
        self.dataset_sampling_tokens = []
        for idx, root in enumerate(self.roots):
            if self.sequence_entries is not None:
                entry = self.sequence_entries[idx]
                domain = str(entry.get("domain", os.path.basename(os.path.dirname(root))))
                family = str(entry.get("family", os.path.basename(root)))
                tokens = int(entry.get("selected_tokens", 0))
            else:
                domain = os.path.basename(os.path.dirname(root))
                family = os.path.basename(root)
                tokens = int(self.datasets[idx].get_num_tokens())
            if tokens <= 0:
                tokens = max(1, int(self.dataset_sequence_counts[idx]))
            self.dataset_domains.append(domain)
            self.dataset_families.append(family)
            self.dataset_sampling_tokens.append(tokens)

        self.hierarchical_sampling = False
        self.domain_sampler = None
        self.family_samplers = {}
        self.dataset_samplers = {}
        if self.split != "train":
            return
        strategy = str(self.sampling_config.get("strategy", "uniform_sequence"))
        if strategy != "hierarchical_smooth":
            return
        self.hierarchical_sampling = True
        domain_alpha = float(self.sampling_config.get("domain_alpha", 0.7))
        family_alpha = float(self.sampling_config.get("family_alpha", 0.6))
        dataset_alpha = float(self.sampling_config.get("dataset_alpha", 0.7))

        domain_to_indices = {}
        family_to_indices = {}
        for idx, (domain, family) in enumerate(zip(self.dataset_domains, self.dataset_families)):
            if self.dataset_sequence_counts[idx] <= 0:
                continue
            domain_to_indices.setdefault(domain, []).append(idx)
            family_to_indices.setdefault((domain, family), []).append(idx)

        domain_items = sorted(domain_to_indices)
        domain_weights = [
            sum(self.dataset_sampling_tokens[i] for i in domain_to_indices[d]) ** domain_alpha
            for d in domain_items
        ]
        self.domain_sampler = self._build_sampler(domain_items, domain_weights)

        for domain, indices in domain_to_indices.items():
            families = sorted({self.dataset_families[i] for i in indices})
            family_weights = []
            for family in families:
                family_indices = family_to_indices[(domain, family)]
                family_weights.append(sum(self.dataset_sampling_tokens[i] for i in family_indices) ** family_alpha)
                dataset_weights = [self.dataset_sampling_tokens[i] ** dataset_alpha for i in family_indices]
                self.dataset_samplers[(domain, family)] = self._build_sampler(family_indices, dataset_weights)
            self.family_samplers[domain] = self._build_sampler(families, family_weights)

    @staticmethod
    def _build_sampler(items, weights):
        kept_items = []
        cumsum = []
        total = 0.0
        for item, weight in zip(items, weights):
            weight = float(weight)
            if weight <= 0.0:
                continue
            total += weight
            kept_items.append(item)
            cumsum.append(total)
        if not kept_items:
            raise RuntimeError("Cannot build sampler with no positive weights")
        return kept_items, cumsum

    @staticmethod
    def _sample_from_sampler(rng: random.Random, sampler):
        items, cumsum = sampler
        value = rng.random() * cumsum[-1]
        return items[bisect_right(cumsum, value)]

    def _sample_train_global_seq_idx(self, rng: random.Random) -> int:
        if not self.hierarchical_sampling:
            return rng.randrange(self.num_sequences)
        domain = self._sample_from_sampler(rng, self.domain_sampler)
        family = self._sample_from_sampler(rng, self.family_samplers[domain])
        ds_idx = self._sample_from_sampler(rng, self.dataset_samplers[(domain, family)])
        local_pos = rng.randrange(self.dataset_sequence_counts[ds_idx])
        return self.dataset_cumsum[ds_idx] + local_pos

    def __len__(self) -> int:
        if self.split == "train":
            return self.samples_per_epoch
        return len(self.valid_seq_indices)

    def _map_global_seq_idx(self, seq_idx: int):
        ds_idx = bisect_right(self.dataset_cumsum, int(seq_idx)) - 1
        local_idx = int(seq_idx) - self.dataset_cumsum[ds_idx]
        indices = self.sequence_indices[ds_idx]
        if indices is not None:
            local_idx = int(indices[local_idx])
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
                global_seq_idx = self._sample_train_global_seq_idx(rng)
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
