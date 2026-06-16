from __future__ import annotations

import torch
import numpy as np

from .time_codec_dataset import SplitTimeSeriesCodecDataset


class SplitTimeSeriesForecastDataset(SplitTimeSeriesCodecDataset):
    def __init__(
        self,
        split_manifest_path: str,
        split: str,
        history_length: int,
        max_prediction_length: int,
        prediction_length_choices,
        patch_size: int,
        normalization_method=None,
        samples_per_epoch: int = 1000000,
        max_valid_sequences: int = 60000,
        seed: int = 1234,
        sampling_config=None,
    ):
        self.history_length = int(history_length)
        self.max_prediction_length = int(max_prediction_length)
        self.patch_size = int(patch_size)
        self.prediction_length_choices = sorted(int(x) for x in prediction_length_choices)
        if not self.prediction_length_choices:
            raise ValueError("prediction_length_choices must not be empty")
        if self.prediction_length_choices[-1] != self.max_prediction_length:
            raise ValueError("max_prediction_length must equal max(prediction_length_choices)")
        for pred_len in self.prediction_length_choices:
            if pred_len % self.patch_size != 0:
                raise ValueError("Every prediction length must be divisible by patch_size")
        min_points = 2 * self.prediction_length_choices[0]
        super().__init__(
            split_manifest_path=split_manifest_path,
            split=split,
            segment_length=self.history_length + self.max_prediction_length,
            normalization_method=normalization_method,
            samples_per_epoch=samples_per_epoch,
            max_valid_sequences=max_valid_sequences,
            seed=seed,
            return_valid_length=True,
            min_input_length=min_points,
            sampling_config=sampling_config,
        )

    def _choose_prediction_length(self, seq_len: int) -> int:
        chosen = None
        for pred_len in self.prediction_length_choices:
            if int(seq_len) >= 2 * pred_len:
                chosen = pred_len
            else:
                break
        if chosen is None:
            raise RuntimeError(f"Sequence length {seq_len} is shorter than minimum forecasting length")
        return int(chosen)

    def _format_forecast_item(self, seq: np.ndarray, rng, train: bool):
        seq_len = int(seq.size)
        pred_len = self._choose_prediction_length(seq_len)
        if train:
            start = rng.randrange(pred_len, seq_len - pred_len + 1)
        else:
            start = seq_len - pred_len
        history_start = max(0, start - self.history_length)
        history = seq[history_start:start]
        future = seq[start:start + pred_len]
        history_len = int(history.size)
        future_len = int(future.size)
        if history_len < pred_len or future_len != pred_len:
            raise RuntimeError("Invalid forecasting window construction")

        history_pad = np.zeros(self.history_length, dtype=np.float32)
        history_pad[-history_len:] = history.astype(np.float32, copy=False)
        future_pad = np.zeros(self.max_prediction_length, dtype=np.float32)
        future_pad[:future_len] = future.astype(np.float32, copy=False)
        return {
            "history": torch.from_numpy(history_pad).view(1, -1),
            "future": torch.from_numpy(future_pad).view(1, -1),
            "history_length": torch.tensor(history_len, dtype=torch.long),
            "future_length": torch.tensor(future_len, dtype=torch.long),
            "future_token_length": torch.tensor(future_len // self.patch_size, dtype=torch.long),
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
            return self._format_forecast_item(seq, rng, train=True)

        seq = self._fetch_seq(self.valid_seq_indices[int(idx)])
        rng = self._rng_for_item(int(idx))
        return self._format_forecast_item(seq, rng, train=False)


__all__ = ["SplitTimeSeriesForecastDataset"]
