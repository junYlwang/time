from __future__ import annotations

import argparse
import json
import os
import sys
import time
from bisect import bisect_right
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from datasets.time_moe_dataset import TimeMoEDataset
from modules.decoder import Decoder
from modules.encoder_wo_quantize import Encoder
from modules.quantizer import build_quantizer
from modules.revin import ReversibleInstanceNorm1D
from modules.utils import load_checkpoint, load_hparams, set_seed


@dataclass
class SampleMeta:
    sample_index: int
    offset: int
    length: int


def _normalize_split_name(split: str) -> str:
    s = str(split).strip().lower()
    if s == "val":
        return "valid"
    return s


def _extract_split_roots(manifest_obj, split: str) -> List[str]:
    split = _normalize_split_name(split)

    if isinstance(manifest_obj, dict):
        value = manifest_obj.get(split)
        if value is None and split == "valid":
            value = manifest_obj.get("val")
        if value is None:
            raise KeyError(f"Split '{split}' not found in split manifest")
    elif isinstance(manifest_obj, list):
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
    if not roots:
        raise ValueError(f"Split '{split}' has no dataset roots")
    return roots


def _load_test_roots(split_manifest_path: str, test_split: str) -> List[str]:
    with open(split_manifest_path, "r", encoding="utf-8") as f:
        manifest_obj = json.load(f)
    return _extract_split_roots(manifest_obj, test_split)


def _subset_name_from_root(root: str) -> str:
    name = os.path.basename(os.path.normpath(root))
    return name if name else "subset"


def _ensure_unique_subset_name(name: str, used: Dict[str, int]) -> str:
    if name not in used:
        used[name] = 1
        return name
    used[name] += 1
    return f"{name}_{used[name]}"


def _safe_int(v, default: int = -1) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _downsample_factor(h) -> int:
    factor = 1
    for ratio in getattr(h, "down_ratio"):
        ratio = int(ratio)
        if ratio <= 0:
            raise ValueError(f"down_ratio values must be positive, got {getattr(h, 'down_ratio')}")
        factor *= ratio
    return factor



def _left_pad_to_multiple(x: np.ndarray, multiple: int) -> Tuple[np.ndarray, int]:
    if multiple <= 0:
        raise ValueError(f"padding multiple must be positive, got {multiple}")
    x = x.astype(np.float32, copy=False)
    pad_len = (-int(x.shape[0])) % multiple
    if pad_len == 0:
        return x, 0
    padded = np.pad(x, (pad_len, 0), mode="constant", constant_values=0.0)
    return padded.astype(np.float32, copy=False), pad_len


def _masked_zscore_left_padded(
    x: torch.Tensor,
    valid_length: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.ndim != 3:
        raise ValueError(f"Expected [B, C, T], got shape={tuple(x.shape)}")
    time_steps = x.size(-1)
    valid_length = int(max(1, min(int(valid_length), time_steps)))
    pos = torch.arange(time_steps, device=x.device).view(1, 1, time_steps)
    valid_mask = pos >= (time_steps - valid_length)
    weight = valid_mask.to(dtype=x.dtype)
    count = weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (x * weight).sum(dim=-1, keepdim=True) / count
    var = ((x - mean).square() * weight).sum(dim=-1, keepdim=True) / count
    std = torch.sqrt(var + float(eps))
    x_in = torch.where(valid_mask, (x - mean) / std, torch.zeros_like(x))
    return x_in, mean, std


class SingleSubsetInferenceDataset(Dataset):
    def __init__(self, root: str, normalization_method=None):
        self.root = root
        self.ds = TimeMoEDataset(root, normalization_method=normalization_method)
        self._cumsum = list(self.ds.cumsum_lengths)

    def __len__(self) -> int:
        return len(self.ds)

    def _read_meta(self, seq_idx: int) -> SampleMeta:
        dataset_idx = bisect_right(self._cumsum, seq_idx) - 1
        local_idx = seq_idx - self._cumsum[dataset_idx]
        child_ds = self.ds.datasets[dataset_idx]

        offset = -1
        length = -1
        if hasattr(child_ds, "seq_infos"):
            seq_info = child_ds.seq_infos[local_idx]
            offset = _safe_int(seq_info.get("offset", -1), default=-1)
            length = _safe_int(seq_info.get("length", -1), default=-1)
        else:
            try:
                length = _safe_int(child_ds.get_sequence_length_by_idx(local_idx), default=-1)
            except Exception:
                length = -1

        if length < 0:
            seq = np.asarray(self.ds[seq_idx], dtype=np.float32)
            length = int(seq.size)

        return SampleMeta(sample_index=int(seq_idx), offset=offset, length=length)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = np.asarray(self.ds[idx], dtype=np.float32)
        meta = self._read_meta(idx)

        return {
            "x": torch.from_numpy(seq).unsqueeze(0),
            "sample_index": torch.tensor(meta.sample_index, dtype=torch.long),
            "offset": torch.tensor(meta.offset, dtype=torch.long),
            "length": torch.tensor(meta.length, dtype=torch.long),
        }


def _plot_sample(
    gt: np.ndarray,
    rec: np.ndarray,
    valid_len: int,
    title: str,
    save_path: str,
    offset: int,
) -> None:
    valid_len = int(max(1, min(valid_len, gt.shape[0], rec.shape[0], 300)))
    x_start = int(offset) if int(offset) >= 0 else 0
    t = np.arange(valid_len) + x_start

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, gt[:valid_len], color="#1f77b4", linewidth=1.5, label="groundtruth")
    ax.plot(t, rec[:valid_len], color="#d62728", linewidth=1.5, label="reconstruction")
    ax.set_title(title)
    ax.set_xlabel("time_index")
    ax.set_ylabel("value")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _build_models(h, device: torch.device):
    encoder = Encoder(h).to(device)
    quantizer = build_quantizer(h).to(device)
    decoder = Decoder(h).to(device)
    input_norm = ReversibleInstanceNorm1D(
        num_channels=int(h.input_channels),
        eps=float(h.revin_eps),
    ).to(device)
    return encoder, quantizer, decoder, input_norm


def _load_codec_checkpoint(h, device: torch.device):
    infer_cfg = getattr(h, "inference", {}) or {}
    if not isinstance(infer_cfg, dict):
        raise ValueError("inference must be a mapping in config")
    checkpoint_path = str(infer_cfg.get("checkpoint_path", "")).strip()
    if not checkpoint_path:
        raise ValueError("Missing inference.checkpoint_path in config")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    encoder, quantizer, decoder, input_norm = _build_models(h, device)

    state = load_checkpoint(checkpoint_path, device)
    encoder.load_state_dict(state["encoder"], strict=True)
    has_quantizer_ckpt = "quantizer" in state
    if has_quantizer_ckpt:
        quantizer.load_state_dict(state["quantizer"], strict=True)
    else:
        quantizer = None
        print(
            "[Warn] No 'quantizer' found in checkpoint. "
            "Using no-quantizer inference path for compatibility."
        )
    decoder.load_state_dict(state["decoder"], strict=True)
    if "input_norm" in state:
        input_norm.load_state_dict(state["input_norm"], strict=True)

    encoder.eval()
    if quantizer is not None:
        quantizer.eval()
    decoder.eval()
    input_norm.eval()

    return encoder, quantizer, decoder, input_norm, checkpoint_path


DEFAULT_WINDOW_LENGTHS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def _ensure_dirs(output_dir: str) -> Tuple[str, str, str, str]:
    metrics_dir = os.path.join(output_dir, "metrics")
    samples_dir = os.path.join(output_dir, "samples")
    recon_dir = os.path.join(output_dir, "reconstructions")
    codes_dir = os.path.join(output_dir, "codes")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(codes_dir, exist_ok=True)
    return metrics_dir, samples_dir, recon_dir, codes_dir


def _parse_window_lengths(infer_cfg: Dict) -> List[int]:
    value = infer_cfg.get("window_lengths", DEFAULT_WINDOW_LENGTHS)
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        lengths = [int(part) for part in parts]
    elif isinstance(value, (list, tuple)):
        lengths = [int(part) for part in value]
    else:
        lengths = [int(value)]
    if not lengths:
        raise ValueError("inference.window_lengths must not be empty")
    if any(length <= 0 for length in lengths):
        raise ValueError(f"window_lengths must be positive, got {lengths}")
    return lengths


def _iter_window_bounds(total_length: int, window_length: int):
    for start in range(0, total_length, window_length):
        end = min(start + window_length, total_length)
        if end > start:
            yield start, end


def _flatten_codes_for_txt(codes: torch.Tensor | None, codebook_size: int) -> np.ndarray:
    if codes is None:
        return np.zeros((0,), dtype=np.int64)
    code_np = codes.detach().cpu().numpy()
    if code_np.ndim != 3 or code_np.shape[0] != 1:
        raise ValueError(f"Expected codes with shape [1, Q, T], got {code_np.shape}")
    code_np = code_np[0].astype(np.int64, copy=False)
    offsets = (np.arange(code_np.shape[0], dtype=np.int64) * int(codebook_size))[:, None]
    return (code_np + offsets).T.reshape(-1)


def _write_codes_txt(path: str, codes: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(" ".join(str(int(code)) for code in codes.tolist()))
        f.write("\n")


def _infer_window(
    window: np.ndarray,
    h,
    device: torch.device,
    encoder,
    quantizer,
    decoder,
    input_norm,
    downsample_factor: int,
) -> Tuple[np.ndarray, np.ndarray]:
    raw_len = int(window.shape[0])
    padded_np, pad_len = _left_pad_to_multiple(window, downsample_factor)
    x = torch.from_numpy(padded_np).view(1, 1, -1).to(device)

    x_in, mu, std = _masked_zscore_left_padded(
        x,
        valid_length=raw_len,
        eps=float(input_norm.eps),
    )

    latent = encoder(x_in)
    if quantizer is not None:
        q_out = quantizer(latent)
        zq = q_out.z_q
        code_flat = _flatten_codes_for_txt(q_out.codes, int(getattr(h, "codebook_size", 1000)))
    else:
        zq = latent
        code_flat = np.zeros((0,), dtype=np.int64)
    x_hat_norm = decoder(zq)
    x_hat = input_norm.inverse(x_hat_norm, mu, std)
    rec_full = x_hat[0, 0].detach().cpu().numpy().astype(np.float32, copy=True)
    return rec_full[pad_len:pad_len + raw_len].copy(), code_flat


def _infer_one_subset(
    subset_root: str,
    subset_name: str,
    h,
    device: torch.device,
    encoder,
    quantizer,
    decoder,
    input_norm,
    recon_dir: str,
    samples_dir: str,
    codes_dir: str,
    window_length: int,
) -> Dict:
    dataset = SingleSubsetInferenceDataset(
        root=subset_root,
        normalization_method=getattr(h, "normalization_method", None),
    )
    n = len(dataset)
    if n == 0:
        raise ValueError(f"Subset '{subset_name}' has no samples: {subset_root}")

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=int(getattr(h, "num_workers", 0)),
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    mae_all = np.zeros((n,), dtype=np.float64)
    mse_all = np.zeros((n,), dtype=np.float64)
    lengths_all = np.zeros((n,), dtype=np.int64)
    offsets_all = np.full((n,), -1, dtype=np.int64)
    valid_len_all = np.zeros((n,), dtype=np.int64)
    window_count_all = np.zeros((n,), dtype=np.int64)
    gt_list: List[np.ndarray] = [np.zeros((1,), dtype=np.float32) for _ in range(n)]
    rec_list: List[np.ndarray] = [np.zeros((1,), dtype=np.float32) for _ in range(n)]
    codes_list: List[np.ndarray] = [np.zeros((0,), dtype=np.int64) for _ in range(n)]

    infer_cfg = getattr(h, "inference", {}) or {}
    downsample_factor = _downsample_factor(h)
    total_abs_error = 0.0
    total_sq_error = 0.0
    total_points = 0

    with torch.no_grad():
        for batch in loader:
            sample_indices = batch["sample_index"].numpy()
            offsets = batch["offset"].numpy()
            lengths = batch["length"].numpy()

            for i, sample_idx in enumerate(sample_indices.tolist()):
                x_np = batch["x"][i, 0].detach().cpu().numpy().astype(np.float32, copy=True)
                raw_len = int(x_np.shape[0])
                rec_np = np.empty_like(x_np)
                code_parts: List[np.ndarray] = []
                window_count = 0

                for start, end in _iter_window_bounds(raw_len, window_length):
                    rec_window, code_flat = _infer_window(
                        x_np[start:end],
                        h=h,
                        device=device,
                        encoder=encoder,
                        quantizer=quantizer,
                        decoder=decoder,
                        input_norm=input_norm,
                        downsample_factor=downsample_factor,
                    )
                    rec_np[start:end] = rec_window
                    code_parts.append(code_flat)
                    window_count += 1

                sample_len = int(lengths[i])
                valid_len = max(1, min(sample_len if sample_len > 0 else raw_len, raw_len, rec_np.shape[0]))
                diff = rec_np[:valid_len] - x_np[:valid_len]
                abs_sum = float(np.sum(np.abs(diff)))
                sq_sum = float(np.sum(np.square(diff)))
                mae = abs_sum / valid_len
                mse = sq_sum / valid_len

                gt_list[sample_idx] = x_np.copy()
                rec_list[sample_idx] = rec_np
                codes_list[sample_idx] = np.concatenate(code_parts) if code_parts else np.zeros((0,), dtype=np.int64)
                mae_all[sample_idx] = mae
                mse_all[sample_idx] = mse
                lengths_all[sample_idx] = sample_len
                offsets_all[sample_idx] = int(offsets[i])
                valid_len_all[sample_idx] = valid_len
                window_count_all[sample_idx] = window_count
                total_abs_error += abs_sum
                total_sq_error += sq_sum
                total_points += valid_len

    subset_npz = os.path.join(recon_dir, f"{subset_name}.npz")
    np.savez_compressed(
        subset_npz,
        groundtruth=np.array(gt_list, dtype=object),
        reconstruction=np.array(rec_list, dtype=object),
        mae_per_sample=mae_all,
        mse_per_sample=mse_all,
        offset=offsets_all,
        length=lengths_all,
        valid_length=valid_len_all,
        window_count=window_count_all,
        subset_root=np.array([subset_root]),
    )

    sample_subset_dir = os.path.join(samples_dir, subset_name)
    code_subset_dir = os.path.join(codes_dir, subset_name)
    os.makedirs(sample_subset_dir, exist_ok=True)
    os.makedirs(code_subset_dir, exist_ok=True)

    vis_num = int(infer_cfg.get("num_visual_samples", 3))
    rng = np.random.default_rng(int(getattr(h, "seed", 1234)))
    choose_k = min(vis_num, n)
    vis_indices = rng.choice(n, size=choose_k, replace=False)
    code_paths = []
    for idx in sorted(vis_indices.tolist()):
        offset = int(offsets_all[idx])
        length = int(lengths_all[idx])
        valid_len = int(valid_len_all[idx])
        if offset >= 0:
            stem = f"offset_{offset}_length_{length}"
            title = f"{subset_name} | window={window_length}, offset={offset}, length={length}"
        else:
            stem = f"index_{idx}_length_{length}"
            title = f"{subset_name} | window={window_length}, index={idx}, length={length}"
        _plot_sample(
            gt=gt_list[idx],
            rec=rec_list[idx],
            valid_len=valid_len,
            title=title,
            save_path=os.path.join(sample_subset_dir, f"{stem}.png"),
            offset=offset,
        )
        code_path = os.path.join(code_subset_dir, f"{stem}.txt")
        _write_codes_txt(code_path, codes_list[idx])
        code_paths.append(code_path)

    mae = total_abs_error / total_points if total_points > 0 else float("nan")
    mse = total_sq_error / total_points if total_points > 0 else float("nan")
    return {
        "subset_name": subset_name,
        "subset_root": subset_root,
        "window_length": int(window_length),
        "num_samples": int(n),
        "num_points": int(total_points),
        "padding_side": "left",
        "padding_value": 0.0,
        "normalization": "valid_points_zscore",
        "downsample_factor": downsample_factor,
        "mae": float(mae),
        "mse": float(mse),
        "npz_path": subset_npz,
        "sample_dir": sample_subset_dir,
        "codes_dir": code_subset_dir,
        "code_paths": code_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    h = load_hparams(args.config)
    set_seed(int(getattr(h, "seed", 1234)))

    infer_cfg = getattr(h, "inference", {}) or {}
    if not isinstance(infer_cfg, dict):
        raise ValueError("inference must be a mapping in config")
    output_dir = str(infer_cfg.get("output_dir", "")).strip()
    if not output_dir:
        raise ValueError("Missing inference.output_dir in config")
    os.makedirs(output_dir, exist_ok=True)

    window_lengths = _parse_window_lengths(infer_cfg)
    test_split = str(infer_cfg.get("test_split", "test"))
    test_roots = _load_test_roots(str(h.split_manifest_path), test_split=test_split)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, quantizer, decoder, input_norm, checkpoint_path = _load_codec_checkpoint(h, device)

    all_window_results = []
    for window_length in window_lengths:
        window_output_dir = os.path.join(output_dir, str(window_length))
        metrics_dir, samples_dir, recon_dir, codes_dir = _ensure_dirs(window_output_dir)
        print(f"[Window] length={window_length} output={window_output_dir}")

        results = []
        used_subset_names: Dict[str, int] = {}
        for subset_root in test_roots:
            raw_name = _subset_name_from_root(subset_root)
            subset_name = _ensure_unique_subset_name(raw_name, used_subset_names)
            print(f"[Infer] window={window_length} subset={subset_name} root={subset_root}")
            res = _infer_one_subset(
                subset_root=subset_root,
                subset_name=subset_name,
                h=h,
                device=device,
                encoder=encoder,
                quantizer=quantizer,
                decoder=decoder,
                input_norm=input_norm,
                recon_dir=recon_dir,
                samples_dir=samples_dir,
                codes_dir=codes_dir,
                window_length=window_length,
            )
            results.append(res)

            metric_path = os.path.join(metrics_dir, f"{subset_name}.json")
            with open(metric_path, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)

            print(
                f"[Done] window={window_length} subset={subset_name} | "
                f"samples={res['num_samples']} | points={res['num_points']} | "
                f"MAE={res['mae']:.6f} | MSE={res['mse']:.6f}"
            )

        summary = {
            "config": args.config,
            "checkpoint_path": checkpoint_path,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "window_length": int(window_length),
            "output_dir": window_output_dir,
            "results": results,
        }
        summary_path = os.path.join(metrics_dir, "all_subsets_metrics.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        all_window_results.append(summary)
        print(f"[Summary] window={window_length} saved: {summary_path}")

    all_summary = {
        "config": args.config,
        "checkpoint_path": checkpoint_path,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "window_lengths": [int(length) for length in window_lengths],
        "results": all_window_results,
    }
    all_summary_path = os.path.join(output_dir, "all_window_metrics.json")
    with open(all_summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)
    print(f"[Summary] Saved: {all_summary_path}")


if __name__ == "__main__":
    main()
