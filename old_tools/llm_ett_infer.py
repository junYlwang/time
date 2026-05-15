#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
for _path in (_SRC_DIR, _DATA_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from read_ett import load_ett_forecasting_split
from modules.decoder import Decoder
from modules.quantizer import build_quantizer
from modules.utils import build_input_norm, inverse_revin, load_checkpoint
from llm_ett_tokens_build import _rfsq_codes_to_zq, _validate_rfsq_quantizer


TS_VOCAB_SIZE = 3000
DEFAULT_DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")


class LayeredTimeSeriesTokenLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        token_id_lookup: list[int],
        prompt_length: int,
        num_quantizers: int,
        codebook_size: int,
    ):
        self.prompt_length = int(prompt_length)
        self.num_quantizers = int(num_quantizers)
        self.codebook_size = int(codebook_size)
        self.layer_token_ids = [
            torch.tensor(
                token_id_lookup[layer_idx * self.codebook_size : (layer_idx + 1) * self.codebook_size],
                dtype=torch.long,
            )
            for layer_idx in range(self.num_quantizers)
        ]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        generated_len = max(0, int(input_ids.shape[1]) - self.prompt_length)
        layer_idx = generated_len % self.num_quantizers
        allowed = self.layer_token_ids[layer_idx].to(device=scores.device)
        masked = torch.full_like(scores, -torch.inf)
        masked.index_copy_(1, allowed, scores.index_select(1, allowed))
        return masked


def _resolve_path(path: str | None, *, base_dir: str = _PROJECT_ROOT) -> str | None:
    if path is None:
        return None
    path = str(path)
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _load_token_metadata(data_root: str) -> dict[str, Any]:
    metadata_path = os.path.join(data_root, "metadata.json")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Missing token metadata: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_tokenizer_and_model(cfg: dict[str, Any], adapter_path: str, device: torch.device):
    dtype = torch.bfloat16 if bool(cfg.get("bf16", True)) and device.type == "cuda" else None
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
        local_files_only=bool(cfg.get("local_files_only", True)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model_path = _resolve_path(str(cfg["model_name_or_path"]))
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
        local_files_only=bool(cfg.get("local_files_only", True)),
    )
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.to(device)
    model.eval()
    return tokenizer, model


def _ts_token_maps(tokenizer) -> tuple[list[int], dict[int, int]]:
    tokens = [f"<ts_{idx}>" for idx in range(TS_VOCAB_SIZE)]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    bad = [tok for tok, tok_id in zip(tokens, token_ids) if tok_id is None or int(tok_id) < 0]
    if bad:
        raise RuntimeError(f"Tokenizer is missing time-series token: {bad[0]}")
    id_to_num = {int(token_id): idx for idx, token_id in enumerate(token_ids)}
    return [int(x) for x in token_ids], id_to_num


def _tokens_to_input_ids(tokens: list[int], token_id_lookup: list[int]) -> list[int]:
    return [int(token_id_lookup[int(token_num)]) for token_num in tokens]


def _generated_ids_to_token_nums(
    generated_ids: torch.Tensor,
    id_to_token_num: dict[int, int],
) -> tuple[list[list[int]], int]:
    token_nums: list[list[int]] = []
    invalid = 0
    for row in generated_ids.detach().cpu().tolist():
        out = []
        for token_id in row:
            token_num = id_to_token_num.get(int(token_id))
            if token_num is None:
                invalid += 1
                token_num = 0
            out.append(int(token_num))
        token_nums.append(out)
    return token_nums, invalid


def _token_nums_to_codes(token_nums: list[list[int]], num_quantizers: int, codebook_size: int, device: torch.device) -> torch.Tensor:
    arr = np.asarray(token_nums, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] % num_quantizers != 0:
        raise ValueError(f"Bad generated token shape: {arr.shape}, num_quantizers={num_quantizers}")
    levels = arr % int(codebook_size)
    quantizer_ids = arr // int(codebook_size)
    expected = np.tile(np.arange(num_quantizers, dtype=np.int64), arr.shape[1] // num_quantizers)
    if not np.all(quantizer_ids == expected[None, :]):
        # Greedy generation is constrained to the TS vocab, but not to the layer order.
        # Keep the code value and count order mistakes via caller diagnostics.
        pass
    codes = levels.reshape(arr.shape[0], -1, num_quantizers).transpose(0, 2, 1)
    return torch.as_tensor(codes, dtype=torch.long, device=device)


def _count_layer_order_errors(token_nums: list[list[int]], num_quantizers: int, codebook_size: int) -> int:
    errors = 0
    expected = np.arange(num_quantizers, dtype=np.int64)
    for row in token_nums:
        arr = np.asarray(row, dtype=np.int64)
        if arr.size % num_quantizers != 0:
            errors += int(arr.size)
            continue
        qids = (arr // int(codebook_size)).reshape(-1, num_quantizers)
        errors += int(np.not_equal(qids, expected[None, :]).sum())
    return errors


def _load_codec(codec_config_path: str, device: torch.device):
    codec_cfg = _load_yaml(codec_config_path)
    checkpoint_path = str(codec_cfg.get("checkpoint_codec", "")).strip()
    inference_cfg = codec_cfg.get("inference", {}) or {}
    if isinstance(inference_cfg, dict):
        checkpoint_path = str(inference_cfg.get("checkpoint_path", "") or checkpoint_path).strip()
    checkpoint_path = _resolve_path(checkpoint_path)
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Codec checkpoint not found: {checkpoint_path}")

    class AttrDict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__

    h = AttrDict(codec_cfg)
    quantizer = build_quantizer(h).to(device)
    decoder = Decoder(h).to(device)
    input_norm = build_input_norm(h, device)

    state = load_checkpoint(checkpoint_path, device)
    quantizer.load_state_dict(state["quantizer"], strict=True)
    decoder.load_state_dict(state["decoder"], strict=True)
    if "input_norm" in state:
        input_norm.load_state_dict(state["input_norm"], strict=True)

    quantizer.eval()
    decoder.eval()
    input_norm.eval()
    if hasattr(quantizer, "set_stochastic_mode"):
        quantizer.set_stochastic_mode(stochastic=False, temperature=0.3)
    num_quantizers, codebook_size = _validate_rfsq_quantizer(quantizer)
    return h, quantizer, decoder, input_norm, checkpoint_path, num_quantizers, codebook_size


class ETTGroundTruthCache:
    def __init__(self, ett_root: str, seq_len: int, pred_len: int, stride: int):
        self.ett_root = ett_root
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.stride = int(stride)
        self.cache: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}

    def _load(self, dataset_name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        if dataset_name not in self.cache:
            x, y, metadata = load_ett_forecasting_split(
                ett_root=self.ett_root,
                dataset_name=dataset_name,
                split="test",
                column=None,
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                stride=self.stride,
            )
            self.cache[dataset_name] = (x, y, list(metadata.feature_names))
        return self.cache[dataset_name]

    def get(self, dataset_name: str, feature_name: str, window_index: int) -> tuple[np.ndarray, np.ndarray]:
        x, y, feature_names = self._load(dataset_name)
        feature_idx = feature_names.index(feature_name)
        window_index = int(window_index)
        return x[window_index, feature_idx].copy(), y[window_index, feature_idx].copy()


def _new_bucket() -> dict[str, float]:
    return {"mae_sum": 0.0, "mse_sum": 0.0, "value_count": 0.0, "num_samples": 0.0}


def _update_bucket(bucket: dict[str, float], pred: np.ndarray, target: np.ndarray) -> None:
    diff = pred.astype(np.float64) - target.astype(np.float64)
    bucket["mae_sum"] += float(np.abs(diff).sum())
    bucket["mse_sum"] += float(np.square(diff).sum())
    bucket["value_count"] += float(diff.size)
    bucket["num_samples"] += 1.0


def _finalize_bucket(bucket: dict[str, float]) -> dict[str, Any]:
    value_count = max(1.0, bucket["value_count"])
    return {
        "test_mae": float(bucket["mae_sum"] / value_count),
        "test_mse": float(bucket["mse_sum"] / value_count),
        "value_count": int(bucket["value_count"]),
        "num_samples": int(bucket["num_samples"]),
    }


def _iter_jsonl_batches(path: str, batch_size: int, max_samples: int | None):
    batch = []
    seen = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples is not None and seen >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            batch.append(json.loads(line))
            seen += 1
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def _plot_sample(history: np.ndarray, target: np.ndarray, pred: np.ndarray, save_path: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    hist_len = len(history)
    pred_len = min(len(target), len(pred))
    h_steps = np.arange(hist_len)
    f_steps = np.arange(hist_len, hist_len + pred_len)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(h_steps, history, label="history", color="tab:blue", linewidth=1.3)
    ax.plot(f_steps, target[:pred_len], label="groundtruth", color="tab:green", linewidth=1.3)
    ax.plot(f_steps, pred[:pred_len], label="prediction", color="tab:red", linewidth=1.3)
    ax.axvline(hist_len - 1, color="black", linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("time_step")
    ax.set_ylabel("value")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Infer ETT forecasts with Qwen LoRA codec-token model.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    infer_cfg = cfg.get("inference", {}) or {}
    if not isinstance(infer_cfg, dict):
        raise ValueError("inference must be a mapping in config")

    adapter_path = _resolve_path(infer_cfg.get("adapter_path"))
    if not adapter_path or not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    output_dir = _resolve_path(infer_cfg.get("output_dir"))
    if not output_dir:
        raise ValueError("Missing inference.output_dir in config")
    os.makedirs(output_dir, exist_ok=True)

    data_root = _resolve_path(cfg["data_root"])
    metadata = _load_token_metadata(data_root)
    test_path = os.path.join(data_root, "test.jsonl")
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"Missing test token file: {test_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = _load_tokenizer_and_model(cfg, adapter_path, device)
    token_id_lookup, id_to_token_num = _ts_token_maps(tokenizer)

    codec_config = _resolve_path(infer_cfg.get("codec_config", "configs/ett-3-base.yaml"))
    codec_h, quantizer, decoder, input_norm, codec_checkpoint, num_quantizers, codebook_size = _load_codec(codec_config, device)

    history_len = int(cfg.get("history_token_length", metadata.get("history_token_length", 192)))
    target_len = int(cfg.get("target_token_length", metadata.get("target_token_length", 36)))
    max_new_tokens = int(infer_cfg.get("max_new_tokens", target_len))
    if max_new_tokens != target_len:
        raise ValueError(f"max_new_tokens must equal target_token_length={target_len}, got {max_new_tokens}")

    gt_cache = ETTGroundTruthCache(
        ett_root=str(codec_h.ett_root),
        seq_len=int(metadata.get("seq_len", codec_h.seq_len)),
        pred_len=int(metadata.get("pred_len", codec_h.pred_len)),
        stride=int(metadata.get("stride", 1)),
    )

    logits_processor = LogitsProcessorList([
        LayeredTimeSeriesTokenLogitsProcessor(
            token_id_lookup=token_id_lookup,
            prompt_length=history_len,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
        )
    ])
    batch_size = int(infer_cfg.get("batch_size", 64))
    max_test_samples = infer_cfg.get("max_test_samples")
    max_test_samples = None if max_test_samples in (None, "") else int(max_test_samples)
    do_sample = bool(infer_cfg.get("do_sample", False))
    total_samples = int(metadata.get("split_counts", {}).get("test", 0))
    if max_test_samples is not None:
        total_samples = min(total_samples, int(max_test_samples)) if total_samples > 0 else int(max_test_samples)

    overall = _new_bucket()
    by_dataset = defaultdict(_new_bucket)
    sample_records: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    invalid_token_count = 0
    layer_order_error_count = 0

    with torch.no_grad():
        progress = tqdm(total=total_samples or None, desc="ETT LLM inference", unit="sample")
        for records in _iter_jsonl_batches(test_path, batch_size=batch_size, max_samples=max_test_samples):
            input_ids = [
                _tokens_to_input_ids(record["input_tokens"], token_id_lookup)
                for record in records
            ]
            if any(len(row) != history_len for row in input_ids):
                raise ValueError("Encountered input token length different from history_token_length")
            input_tensor = torch.as_tensor(input_ids, dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_tensor)

            generated = model.generate(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                logits_processor=logits_processor,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=None,
            )
            new_ids = generated[:, input_tensor.size(1) : input_tensor.size(1) + max_new_tokens]
            token_nums, invalid = _generated_ids_to_token_nums(new_ids, id_to_token_num)
            invalid_token_count += invalid
            layer_order_error_count += _count_layer_order_errors(token_nums, num_quantizers, codebook_size)

            codes = _token_nums_to_codes(token_nums, num_quantizers, codebook_size, device)
            z_q = _rfsq_codes_to_zq(quantizer, codes)
            y_hat_norm = decoder(z_q)

            means = torch.as_tensor(
                [float(record["history_mean"]) for record in records],
                dtype=y_hat_norm.dtype,
                device=device,
            ).view(-1, 1, 1)
            stds = torch.as_tensor(
                [float(record["history_std"]) for record in records],
                dtype=y_hat_norm.dtype,
                device=device,
            ).view(-1, 1, 1)
            y_hat = inverse_revin(input_norm, y_hat_norm, means, stds)
            pred_np = y_hat[:, 0, : int(metadata.get("pred_len", 96))].detach().cpu().float().numpy()

            for idx, record in enumerate(records):
                dataset_name = str(record["dataset_name"])
                feature_name = str(record["feature_name"])
                history, target = gt_cache.get(dataset_name, feature_name, int(record["window_index"]))
                pred = pred_np[idx, : len(target)]
                _update_bucket(overall, pred, target)
                _update_bucket(by_dataset[dataset_name], pred, target)

                key = (dataset_name, feature_name)
                if key not in sample_records:
                    sample_records[key] = (history, target, pred.copy())
            progress.update(len(records))
        progress.close()

    samples_dir = os.path.join(output_dir, "samples")
    for (dataset_name, feature_name), (history, target, pred) in sorted(sample_records.items()):
        save_path = os.path.join(samples_dir, dataset_name, f"{feature_name}.png")
        _plot_sample(
            history=history,
            target=target,
            pred=pred,
            save_path=save_path,
            title=f"{dataset_name} | {feature_name}",
        )

    metrics = {
        "adapter_path": adapter_path,
        "model_name_or_path": _resolve_path(cfg["model_name_or_path"]),
        "data_root": data_root,
        "codec_config": codec_config,
        "codec_checkpoint": codec_checkpoint,
        "overall": _finalize_bucket(overall),
        "by_dataset": {
            dataset_name: _finalize_bucket(by_dataset[dataset_name])
            for dataset_name in sorted(set(DEFAULT_DATASETS) | set(by_dataset.keys()))
        },
        "diagnostics": {
            "invalid_token_count": int(invalid_token_count),
            "layer_order_error_count": int(layer_order_error_count),
            "history_token_length": int(history_len),
            "target_token_length": int(target_len),
            "max_new_tokens": int(max_new_tokens),
            "num_quantizers": int(num_quantizers),
            "codebook_size": int(codebook_size),
            "num_sample_plots": int(len(sample_records)),
            "max_test_samples": max_test_samples,
        },
    }
    _write_json(os.path.join(output_dir, "metrics.json"), metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
