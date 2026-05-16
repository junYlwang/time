from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from typing import Any

import numpy as np
import torch

_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from datasets.time_moe_dataset import TimeMoEDataset


STD_THRESHOLDS = (10.0, 50.0, 100.0, 1000.0)
ABSMAX_THRESHOLDS = (1e3, 1e4, 1e5, 1e6)
QUANTILES = (0.5, 0.9, 0.95, 0.99, 0.999)


def _load_paths(manifest_path: str, split: str) -> list[tuple[str, str]]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [("train", str(path)) for path in obj]
    if not isinstance(obj, dict):
        raise ValueError("manifest must be a dict of splits or a list of paths")
    if split == "all":
        out = []
        for split_name, paths in obj.items():
            for path in paths:
                out.append((str(split_name), str(path)))
        return out
    paths = obj[split]
    return [(split, str(path)) for path in paths]


def _path_parts(path: str) -> tuple[str, str]:
    name = os.path.basename(path.rstrip(os.sep))
    domain = os.path.basename(os.path.dirname(path.rstrip(os.sep)))
    return domain, name


def _fetch_segment(dataset: TimeMoEDataset, seq_idx: int, segment_length: int, rng: random.Random) -> tuple[np.ndarray, bool]:
    seq = np.asarray(dataset[seq_idx], dtype=np.float32)
    if seq.size >= segment_length:
        start = rng.randrange(0, seq.size - segment_length + 1)
    else:
        start = 0
    seg = seq[start:start + segment_length]
    padded = False
    if seg.size < segment_length:
        seg = np.pad(seg, (0, segment_length - seg.size), mode="constant", constant_values=0.0)
        padded = True
    return seg, padded


def _batch_stats(batch_np: list[np.ndarray], device: torch.device) -> dict[str, np.ndarray]:
    batch = torch.from_numpy(np.stack(batch_np, axis=0)).to(device=device, non_blocking=True)
    finite = torch.isfinite(batch)
    nan_count = torch.isnan(batch).sum(dim=1)
    inf_count = torch.isinf(batch).sum(dim=1)
    finite_count = finite.sum(dim=1).clamp(min=1)
    safe = torch.where(finite, batch, torch.zeros_like(batch))
    mean = safe.sum(dim=1) / finite_count
    centered = torch.where(finite, batch - mean[:, None], torch.zeros_like(batch))
    std = torch.sqrt(centered.square().sum(dim=1) / finite_count)
    absmax = torch.where(finite, batch.abs(), torch.zeros_like(batch)).amax(dim=1)
    min_value = torch.where(finite, batch, torch.full_like(batch, float("inf"))).amin(dim=1)
    max_value = torch.where(finite, batch, torch.full_like(batch, float("-inf"))).amax(dim=1)

    return {
        "std": std.cpu().numpy(),
        "absmax": absmax.cpu().numpy(),
        "mean": mean.cpu().numpy(),
        "min": min_value.cpu().numpy(),
        "max": max_value.cpu().numpy(),
        "nan_count": nan_count.cpu().numpy(),
        "inf_count": inf_count.cpu().numpy(),
        "finite_ratio": (finite.float().mean(dim=1)).cpu().numpy(),
    }


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {f"p{int(q * 1000) if q == 0.999 else int(q * 100)}": float("nan") for q in QUANTILES}
    out = {}
    for q in QUANTILES:
        key = f"p{int(q * 1000)}" if q == 0.999 else f"p{int(q * 100)}"
        out[key] = float(np.quantile(values, q))
    return out


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _profile_one_path(
    split: str,
    path: str,
    segment_length: int,
    samples_per_path: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    start_time = time.time()
    domain, name = _path_parts(path)
    row: dict[str, Any] = {
        "split": split,
        "domain": domain,
        "name": name,
        "path": path,
        "status": "ok",
    }
    try:
        dataset = TimeMoEDataset(path, normalization_method=None)
        num_sequences = len(dataset)
        if num_sequences <= 0:
            raise ValueError("dataset has zero sequences")

        rng = random.Random(seed)
        n_samples = int(samples_per_path) if samples_per_path > 0 else int(num_sequences)
        seq_indices = [rng.randrange(num_sequences) for _ in range(n_samples)]

        std_values = []
        absmax_values = []
        mean_values = []
        min_values = []
        max_values = []
        nan_counts = []
        inf_counts = []
        finite_ratios = []
        padded_count = 0

        batch_np = []
        for seq_idx in seq_indices:
            seg, padded = _fetch_segment(dataset, int(seq_idx), segment_length, rng)
            padded_count += int(padded)
            batch_np.append(seg)
            if len(batch_np) == batch_size:
                stats = _batch_stats(batch_np, device)
                std_values.append(stats["std"])
                absmax_values.append(stats["absmax"])
                mean_values.append(stats["mean"])
                min_values.append(stats["min"])
                max_values.append(stats["max"])
                nan_counts.append(stats["nan_count"])
                inf_counts.append(stats["inf_count"])
                finite_ratios.append(stats["finite_ratio"])
                batch_np = []
        if batch_np:
            stats = _batch_stats(batch_np, device)
            std_values.append(stats["std"])
            absmax_values.append(stats["absmax"])
            mean_values.append(stats["mean"])
            min_values.append(stats["min"])
            max_values.append(stats["max"])
            nan_counts.append(stats["nan_count"])
            inf_counts.append(stats["inf_count"])
            finite_ratios.append(stats["finite_ratio"])

        std_arr = np.concatenate(std_values) if std_values else np.asarray([], dtype=np.float32)
        absmax_arr = np.concatenate(absmax_values) if absmax_values else np.asarray([], dtype=np.float32)
        mean_arr = np.concatenate(mean_values) if mean_values else np.asarray([], dtype=np.float32)
        min_arr = np.concatenate(min_values) if min_values else np.asarray([], dtype=np.float32)
        max_arr = np.concatenate(max_values) if max_values else np.asarray([], dtype=np.float32)
        nan_arr = np.concatenate(nan_counts) if nan_counts else np.asarray([], dtype=np.int64)
        inf_arr = np.concatenate(inf_counts) if inf_counts else np.asarray([], dtype=np.int64)
        finite_ratio_arr = np.concatenate(finite_ratios) if finite_ratios else np.asarray([], dtype=np.float32)
        sampled = int(std_arr.size)

        row.update(
            {
                "num_sequences": int(num_sequences),
                "num_sampled_segments": sampled,
                "segment_length": int(segment_length),
                "padded_ratio": float(padded_count / max(1, sampled)),
                "nonfinite_segment_ratio": float(((nan_arr > 0) | (inf_arr > 0)).mean()) if sampled else float("nan"),
                "nan_segment_ratio": float((nan_arr > 0).mean()) if sampled else float("nan"),
                "inf_segment_ratio": float((inf_arr > 0).mean()) if sampled else float("nan"),
                "finite_ratio_mean": _safe_float(finite_ratio_arr.mean()) if sampled else float("nan"),
                "mean_abs_p99": _safe_float(np.quantile(np.abs(mean_arr), 0.99)) if sampled else float("nan"),
                "min_min": _safe_float(np.min(min_arr)) if sampled else float("nan"),
                "max_max": _safe_float(np.max(max_arr)) if sampled else float("nan"),
                "elapsed_sec": float(time.time() - start_time),
            }
        )
        for key, value in _percentiles(std_arr).items():
            row[f"std_{key}"] = value
        row["std_max"] = _safe_float(std_arr.max()) if sampled else float("nan")
        for key, value in _percentiles(absmax_arr).items():
            row[f"absmax_{key}"] = value
        row["absmax_max"] = _safe_float(absmax_arr.max()) if sampled else float("nan")
        for threshold in STD_THRESHOLDS:
            row[f"std_gt_{threshold:g}_ratio"] = float((std_arr > threshold).mean()) if sampled else float("nan")
        for threshold in ABSMAX_THRESHOLDS:
            row[f"absmax_gt_{threshold:g}_ratio"] = float((absmax_arr > threshold).mean()) if sampled else float("nan")
    except Exception as exc:
        row.update(
            {
                "status": "error",
                "error": repr(exc),
                "num_sequences": 0,
                "num_sampled_segments": 0,
                "segment_length": int(segment_length),
                "elapsed_sec": float(time.time() - start_time),
            }
        )
    return row


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "split",
        "domain",
        "name",
        "path",
        "status",
        "error",
        "num_sequences",
        "num_sampled_segments",
        "segment_length",
        "padded_ratio",
        "nonfinite_segment_ratio",
        "nan_segment_ratio",
        "inf_segment_ratio",
        "finite_ratio_mean",
        "std_p50",
        "std_p90",
        "std_p95",
        "std_p99",
        "std_p999",
        "std_max",
        "absmax_p50",
        "absmax_p90",
        "absmax_p95",
        "absmax_p99",
        "absmax_p999",
        "absmax_max",
        "mean_abs_p99",
        "min_min",
        "max_max",
        "elapsed_sec",
    ]
    keys = set()
    for row in rows:
        keys.update(row)
    return [key for key in preferred if key in keys] + sorted(keys - set(preferred))


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample segment-level scale statistics for Time-300B manifest paths.")
    parser.add_argument("--manifest", default="/mnt/shared-storage-user/wangjunyi/time/configs/data_v1.json")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test", "all"])
    parser.add_argument("--output_csv", default="/mnt/shared-storage-user/wangjunyi/time/data/data_v1_segment_stats.csv")
    parser.add_argument("--segment_length", type=int, default=4096)
    parser.add_argument("--samples_per_path", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--flush_every", type=int, default=1)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    paths = _load_paths(args.manifest, args.split)
    rows: list[dict[str, Any]] = []
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    print(f"Profiling {len(paths)} paths on device={device} -> {args.output_csv}")

    for idx, (split, path) in enumerate(paths, start=1):
        path_seed = int(args.seed) + idx * 1000003
        print(f"[{idx}/{len(paths)}] {split} {path}")
        row = _profile_one_path(
            split=split,
            path=path,
            segment_length=int(args.segment_length),
            samples_per_path=int(args.samples_per_path),
            batch_size=int(args.batch_size),
            seed=path_seed,
            device=device,
        )
        rows.append(row)
        print(
            f"  status={row.get('status')} sampled={row.get('num_sampled_segments')} "
            f"std_p99={row.get('std_p99')} absmax_p99={row.get('absmax_p99')} "
            f"elapsed={row.get('elapsed_sec'):.2f}s"
        )
        if args.flush_every > 0 and idx % int(args.flush_every) == 0:
            with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_fieldnames(rows))
                writer.writeheader()
                writer.writerows(rows)

    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
