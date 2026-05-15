#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from datasets.ett_dataset import ETTDataset
from modules.decoder import Decoder
from modules.encoder_wo_quantize import Encoder
from modules.quantizer import RFSQQuantizer, build_quantizer
from modules.utils import build_input_norm, inverse_revin, load_checkpoint, load_hparams, set_seed


ETT_DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")
SPLITS = ("train", "val", "test")


def _set_quantizer_eval_mode(quantizer) -> None:
    q = quantizer.module if hasattr(quantizer, "module") else quantizer
    if hasattr(q, "set_stochastic_mode"):
        q.set_stochastic_mode(stochastic=False, temperature=0.3)


def _build_models(h, device: torch.device):
    encoder = Encoder(h).to(device)
    quantizer = build_quantizer(h).to(device)
    decoder = Decoder(h).to(device)
    input_norm = build_input_norm(h, device)
    return encoder, quantizer, decoder, input_norm


def _load_codec(h, device: torch.device):
    checkpoint_path = str(getattr(h, "checkpoint_codec", "")).strip()
    inference_cfg = getattr(h, "inference", {}) or {}
    if isinstance(inference_cfg, dict):
        checkpoint_path = str(inference_cfg.get("checkpoint_path", "") or checkpoint_path).strip()
    if not checkpoint_path:
        raise ValueError("Missing codec checkpoint path: set checkpoint_codec or inference.checkpoint_path")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Codec checkpoint not found: {checkpoint_path}")

    encoder, quantizer, decoder, input_norm = _build_models(h, device)
    state = load_checkpoint(checkpoint_path, device)
    encoder.load_state_dict(state["encoder"], strict=True)
    quantizer.load_state_dict(state["quantizer"], strict=True)
    decoder.load_state_dict(state["decoder"], strict=True)
    if "input_norm" in state:
        input_norm.load_state_dict(state["input_norm"], strict=True)

    encoder.eval()
    quantizer.eval()
    decoder.eval()
    input_norm.eval()
    _set_quantizer_eval_mode(quantizer)
    return encoder, quantizer, decoder, input_norm, checkpoint_path


def _unwrap(module):
    return module.module if hasattr(module, "module") else module


def _get_gamma(norm_module):
    mod = _unwrap(norm_module)
    if not getattr(mod, "affine", False):
        return None
    if hasattr(mod, "_get_gamma"):
        return mod._get_gamma()
    return getattr(mod, "gamma")


def _normalize_with_history_stats(h, input_norm, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if not bool(getattr(h, "use_reversible_norm", True)):
        return x

    norm_type = str(getattr(h, "normalization_type", "zscore")).lower()
    if norm_type == "zscore":
        y = (x - mean) / std
    elif norm_type == "mean_abs":
        y = x / std
    else:
        raise ValueError(f"Unsupported normalization_type for token building: {norm_type}")

    mod = _unwrap(input_norm)
    if getattr(mod, "affine", False):
        gamma = _get_gamma(mod)
        y = y * gamma + mod.beta
    return y


def _validate_rfsq_quantizer(quantizer) -> tuple[int, int]:
    q = _unwrap(quantizer)
    if not isinstance(q, RFSQQuantizer):
        raise ValueError(
            f"Only RFSQQuantizer is supported for this token builder, got {type(q).__name__}"
        )
    codebook_sizes = tuple(int(s) for s in q.codebook_sizes)
    if not codebook_sizes:
        raise ValueError("RFSQ quantizer did not expose codebook sizes.")
    if len(set(codebook_sizes)) != 1:
        raise ValueError(f"Expected equal codebook sizes for all quantizers, got {codebook_sizes}")
    return len(codebook_sizes), codebook_sizes[0]


def _codes_to_flat_tokens(codes: torch.Tensor, codebook_size: int) -> list[list[int]]:
    if codes.ndim != 3:
        raise ValueError(f"Expected codes shape [B, Q, L], got {tuple(codes.shape)}")
    bsz, num_quantizers, _latent_len = codes.shape
    offsets = (
        torch.arange(num_quantizers, device=codes.device, dtype=codes.dtype)
        .view(1, num_quantizers, 1)
        * int(codebook_size)
    )
    tokens = (codes.long() + offsets.long()).permute(0, 2, 1).reshape(bsz, -1)
    return tokens.detach().cpu().tolist()


def _rfsq_codes_to_zq(quantizer, codes: torch.Tensor) -> torch.Tensor:
    q = _unwrap(quantizer)
    if not isinstance(q, RFSQQuantizer):
        raise ValueError(f"Only RFSQQuantizer is supported, got {type(q).__name__}")
    if codes.ndim != 3:
        raise ValueError(f"Expected codes shape [B, Q, L], got {tuple(codes.shape)}")
    if codes.size(1) != len(q.rfsq.layers):
        raise ValueError(
            f"Code layer mismatch: codes={codes.size(1)}, quantizer={len(q.rfsq.layers)}"
        )

    z_q = None
    for layer_idx, layer in enumerate(q.rfsq.layers):
        layer_codes = codes[:, layer_idx, :].long()
        code_vectors = layer._indices_to_codes(layer_codes).to(device=codes.device, dtype=torch.float32)
        code_vectors = layer.project_out(code_vectors)
        layer_z = code_vectors.transpose(1, 2).contiguous()
        z_q = layer_z if z_q is None else z_q + layer_z
    return z_q


def _feature_and_window_indices(dataset: ETTDataset, sample_indices: torch.Tensor) -> tuple[list[str], list[int]]:
    num_variables = int(dataset.num_variables)
    names = list(dataset.selected_feature_names)
    feature_indices = (sample_indices % num_variables).detach().cpu().tolist()
    window_indices = (sample_indices // num_variables).detach().cpu().tolist()
    feature_names = [names[int(i)] for i in feature_indices]
    return feature_names, [int(i) for i in window_indices]


def _new_metric_bucket() -> dict[str, float]:
    return {"mae_sum": 0.0, "mse_sum": 0.0, "value_count": 0.0, "sample_count": 0.0}


def _update_metrics(bucket: dict[str, float], pred: torch.Tensor, target: torch.Tensor) -> None:
    diff = pred - target
    bucket["mae_sum"] += float(diff.abs().sum().item())
    bucket["mse_sum"] += float(diff.square().sum().item())
    bucket["value_count"] += float(diff.numel())
    bucket["sample_count"] += float(diff.size(0))


def _finalize_bucket(bucket: dict[str, float]) -> dict[str, Any]:
    value_count = max(1.0, bucket["value_count"])
    return {
        "oracle_mae": float(bucket["mae_sum"] / value_count),
        "oracle_mse": float(bucket["mse_sum"] / value_count),
        "value_count": int(bucket["value_count"]),
        "sample_count": int(bucket["sample_count"]),
    }


def _write_jsonl_record(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _process_dataset_split(
    h,
    split: str,
    dataset_name: str,
    dataset: ETTDataset,
    writer,
    split_sample_offset: int,
    device: torch.device,
    encoder,
    quantizer,
    decoder,
    input_norm,
    codebook_size: int,
    num_quantizers: int,
    downsample_ratio: int,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
    oracle_overall: dict[str, float],
    oracle_by_dataset: dict[str, dict[str, float]],
) -> int:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    use_reversible_norm = bool(getattr(h, "use_reversible_norm", True))
    processed = 0
    dataset_seen = 0

    for batch in loader:
        x = batch["seq"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        original_bsz = x.size(0)

        if max_samples is not None:
            remaining = int(max_samples) - processed
            if remaining <= 0:
                break
            keep = min(original_bsz, remaining)
            x = x[:keep]
            y = y[:keep]
        bsz = x.size(0)
        sample_indices = torch.arange(dataset_seen, dataset_seen + original_bsz, device=device)[:bsz]
        dataset_seen += original_bsz

        with torch.no_grad():
            if use_reversible_norm:
                history_norm, mu, std = input_norm(x)
                future_norm = _normalize_with_history_stats(h, input_norm, y, mu, std)
            else:
                history_norm, mu, std = x, None, None
                future_norm = y

            history_codes = quantizer(encoder(history_norm)).codes
            future_quant_out = quantizer(encoder(future_norm))
            future_codes = future_quant_out.codes
            if history_codes is None or future_codes is None:
                raise RuntimeError("Quantizer did not return discrete codes.")

            input_tokens = _codes_to_flat_tokens(history_codes, codebook_size)
            target_tokens = _codes_to_flat_tokens(future_codes, codebook_size)

            if split == "test":
                future_zq = _rfsq_codes_to_zq(quantizer, future_codes)
                y_hat_norm = decoder(future_zq)
                y_hat = (
                    inverse_revin(input_norm, y_hat_norm, mu, std)
                    if use_reversible_norm
                    else y_hat_norm
                )
                tmin = min(y_hat.shape[-1], y.shape[-1])
                y_hat = y_hat[..., :tmin]
                y_ref = y[..., :tmin]
                _update_metrics(oracle_overall, y_hat, y_ref)
                _update_metrics(oracle_by_dataset[dataset_name], y_hat, y_ref)

        feature_names, window_indices = _feature_and_window_indices(dataset, sample_indices)
        means = (
            mu.detach().reshape(bsz, -1)[:, 0].cpu().tolist()
            if mu is not None
            else [0.0] * bsz
        )
        stds = (
            std.detach().reshape(bsz, -1)[:, 0].cpu().tolist()
            if std is not None
            else [1.0] * bsz
        )

        for local_idx in range(bsz):
            record = {
                "split": split,
                "dataset_name": dataset_name,
                "feature_name": feature_names[local_idx],
                "window_index": window_indices[local_idx],
                "sample_index": split_sample_offset + processed + local_idx,
                "dataset_sample_index": int(sample_indices[local_idx].detach().cpu().item()),
                "input_tokens": input_tokens[local_idx],
                "target_tokens": target_tokens[local_idx],
                "history_mean": float(means[local_idx]),
                "history_std": float(stds[local_idx]),
                "seq_len": int(h.seq_len),
                "pred_len": int(h.pred_len),
                "downsample_ratio": int(downsample_ratio),
                "num_quantizers": int(num_quantizers),
                "codebook_size": int(codebook_size),
            }
            _write_jsonl_record(writer, record)

        processed += bsz
        if max_samples is not None and processed >= int(max_samples):
            break

    return processed


def _parse_csv_arg(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None or not str(value).strip():
        return default
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ETT Codec-LLM token JSONL files.")
    parser.add_argument("--config", type=str, default=os.path.join(_PROJECT_ROOT, "configs", "ett-3-base.yaml"))
    parser.add_argument(
        "--output-root",
        type=str,
        default=os.path.join(
            _PROJECT_ROOT,
            "data",
            "ETT-small-standardized-llm-tokens",
            "rfsq3_hist512_pred96_stride1",
        ),
    )
    parser.add_argument("--datasets", type=str, default=",".join(ETT_DATASETS))
    parser.add_argument("--splits", type=str, default=",".join(SPLITS))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    h = load_hparams(config_path)
    set_seed(int(getattr(h, "seed", 1234)))

    h.seq_len = int(getattr(h, "seq_len", 512))
    h.pred_len = int(getattr(h, "pred_len", 96))
    h.stride = int(getattr(h, "stride", 1))
    h.ett_column = getattr(h, "ett_column", "__all__")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    encoder, quantizer, decoder, input_norm, checkpoint_path = _load_codec(h, device)
    num_quantizers, codebook_size = _validate_rfsq_quantizer(quantizer)
    downsample_ratio = int(math.prod(getattr(h, "down_ratio", [2, 2, 2])))

    if codebook_size != 1000 or num_quantizers != 3:
        raise ValueError(
            "This builder is configured for rfsq3 token ids. "
            f"Got num_quantizers={num_quantizers}, codebook_size={codebook_size}."
        )
    if h.seq_len % downsample_ratio != 0 or h.pred_len % downsample_ratio != 0:
        raise ValueError(
            f"seq_len and pred_len must be divisible by downsample_ratio={downsample_ratio}, "
            f"got seq_len={h.seq_len}, pred_len={h.pred_len}"
        )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    split_names = _parse_csv_arg(args.splits, SPLITS)
    dataset_names = _parse_csv_arg(args.datasets, ETT_DATASETS)
    split_strides = {
        "train": int(getattr(h, "train_stride", h.stride)),
        "val": int(getattr(h, "val_stride", h.stride)),
        "test": int(getattr(h, "test_stride", h.stride)),
    }
    output_paths = {split: output_root / f"{split}.jsonl" for split in split_names}

    existing = [str(path) for path in output_paths.values() if path.exists()]
    metadata_path = output_root / "metadata.json"
    oracle_path = output_root / "test_oracle_reconstruction_metrics.json"
    if metadata_path.exists():
        existing.append(str(metadata_path))
    if oracle_path.exists():
        existing.append(str(oracle_path))
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output files already exist. Re-run with --overwrite to replace them: "
            + ", ".join(existing)
        )

    num_workers = int(args.num_workers) if args.num_workers is not None else 0
    batch_size = int(args.batch_size)
    max_samples = args.max_samples_per_split
    if max_samples is not None and max_samples <= 0:
        max_samples = None

    split_counts = {split: 0 for split in split_names}
    dataset_summaries: list[dict[str, Any]] = []
    oracle_overall = _new_metric_bucket()
    oracle_by_dataset = defaultdict(_new_metric_bucket)

    writers = {}
    try:
        for split, path in output_paths.items():
            writers[split] = open(path, "w", encoding="utf-8")

        for split in split_names:
            for dataset_name in dataset_names:
                split_stride = split_strides[split]
                dataset = ETTDataset(
                    h.ett_root,
                    dataset_name,
                    split=split,
                    seq_len=h.seq_len,
                    pred_len=h.pred_len,
                    stride=split_stride,
                    column=h.ett_column,
                )
                print(
                    f"[Build] split={split} dataset={dataset_name} "
                    f"stride={split_stride} samples={len(dataset)} batch_size={batch_size}"
                )
                written = _process_dataset_split(
                    h=h,
                    split=split,
                    dataset_name=dataset_name,
                    dataset=dataset,
                    writer=writers[split],
                    split_sample_offset=split_counts[split],
                    device=device,
                    encoder=encoder,
                    quantizer=quantizer,
                    decoder=decoder,
                    input_norm=input_norm,
                    codebook_size=codebook_size,
                    num_quantizers=num_quantizers,
                    downsample_ratio=downsample_ratio,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    max_samples=max_samples,
                    oracle_overall=oracle_overall,
                    oracle_by_dataset=oracle_by_dataset,
                )
                split_counts[split] += written
                summary = dataset.summary()
                summary["written_samples"] = int(written)
                dataset_summaries.append(summary)
                print(f"[Done] split={split} dataset={dataset_name} written={written}")
    finally:
        for handle in writers.values():
            handle.close()

    metadata = {
        "config_path": config_path,
        "codec_checkpoint_path": checkpoint_path,
        "ett_root": str(h.ett_root),
        "output_root": str(output_root),
        "datasets": list(dataset_names),
        "splits": list(split_names),
        "ett_column": "__all__" if h.ett_column in (None, "__all__") else str(h.ett_column),
        "seq_len": int(h.seq_len),
        "pred_len": int(h.pred_len),
        "stride": int(h.stride),
        "split_strides": {k: int(v) for k, v in split_strides.items()},
        "downsample_ratio": int(downsample_ratio),
        "num_quantizers": int(num_quantizers),
        "codebook_size": int(codebook_size),
        "llm_vocab_size": int(num_quantizers * codebook_size),
        "history_token_length": int((h.seq_len // downsample_ratio) * num_quantizers),
        "target_token_length": int((h.pred_len // downsample_ratio) * num_quantizers),
        "token_order": "time_major_quantizer_offsets",
        "token_offsets": [int(i * codebook_size) for i in range(num_quantizers)],
        "future_normalization": "history_revin_stats",
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "dataset_summaries": dataset_summaries,
        "max_samples_per_dataset_split": max_samples,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    if "test" in split_names:
        oracle_metrics = {
            "overall": _finalize_bucket(oracle_overall),
            "by_dataset": {
                dataset_name: _finalize_bucket(bucket)
                for dataset_name, bucket in sorted(oracle_by_dataset.items())
            },
            "definition": "Decoder reconstruction from ground-truth future tokens, inverse RevIN with history stats.",
        }
        with open(oracle_path, "w", encoding="utf-8") as f:
            json.dump(oracle_metrics, f, indent=2, ensure_ascii=False)

    print(f"[Saved] output_root={output_root}")
    print(json.dumps({"split_counts": split_counts}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
