from __future__ import annotations

import argparse
import json
import os
import random
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
from modules.utils import load_checkpoint, load_hparams


@dataclass
class SampleMeta:
    sample_index: int
    offset: int
    length: int


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _inverse_revin(norm_module, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return norm_module.inverse(y, mean, std)


def _set_quantizer_eval_mode(quantizer) -> None:
    if quantizer is None:
        return
    q = quantizer.module if hasattr(quantizer, "module") else quantizer
    if hasattr(q, "set_stochastic_mode"):
        q.set_stochastic_mode(stochastic=False, temperature=0.3)


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
    valid_len = int(max(1, min(valid_len, gt.shape[0], rec.shape[0], 100)))
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
        num_channels=int(getattr(h, "input_channels", 1)),
        eps=float(getattr(h, "revin_eps", 1e-5)),
        affine=bool(getattr(h, "revin_affine", True)),
        init_gamma=float(getattr(h, "revin_init_gamma", 1.0)),
        init_beta=float(getattr(h, "revin_init_beta", 0.0)),
        positive_gamma=bool(getattr(h, "revin_positive_gamma", False)),
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
    _set_quantizer_eval_mode(quantizer)

    return encoder, quantizer, decoder, input_norm, checkpoint_path


def _ensure_dirs(output_dir: str) -> Tuple[str, str, str]:
    metrics_dir = os.path.join(output_dir, "metrics")
    samples_dir = os.path.join(output_dir, "samples")
    recon_dir = os.path.join(output_dir, "reconstructions")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    return metrics_dir, samples_dir, recon_dir


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
) -> Dict:
    dataset = SingleSubsetInferenceDataset(
        root=subset_root,
        normalization_method=getattr(h, "normalization_method", None),
    )
    n = len(dataset)
    if n == 0:
        raise ValueError(f"Subset '{subset_name}' has no samples: {subset_root}")

    # Full-length inference uses batch_size=1 to avoid padding/packing variable-length series.
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
    gt_list: List[np.ndarray] = [np.zeros((1,), dtype=np.float32) for _ in range(n)]
    rec_list: List[np.ndarray] = [np.zeros((1,), dtype=np.float32) for _ in range(n)]

    use_reversible_norm = bool(getattr(h, "use_reversible_norm", True))

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            sample_indices = batch["sample_index"].numpy()
            offsets = batch["offset"].numpy()
            lengths = batch["length"].numpy()

            if use_reversible_norm:
                x_in, mu, std = input_norm(x)
            else:
                x_in, mu, std = x, None, None

            latent = encoder(x_in)
            zq = quantizer(latent).z_q if quantizer is not None else latent
            x_hat_norm = decoder(zq)
            x_hat = _inverse_revin(input_norm, x_hat_norm, mu, std) if use_reversible_norm else x_hat_norm

            tmin = min(x.shape[-1], x_hat.shape[-1])
            x_ref = x[:, 0, :tmin].detach().cpu().numpy()
            x_rec = x_hat[:, 0, :tmin].detach().cpu().numpy()

            for i, sample_idx in enumerate(sample_indices.tolist()):
                sample_len = int(lengths[i])
                valid_len = max(1, min(sample_len if sample_len > 0 else tmin, tmin))

                diff = x_rec[i, :valid_len] - x_ref[i, :valid_len]
                mae = float(np.mean(np.abs(diff)))
                mse = float(np.mean(np.square(diff)))

                gt_list[sample_idx] = x_ref[i].copy()
                rec_list[sample_idx] = x_rec[i].copy()
                mae_all[sample_idx] = mae
                mse_all[sample_idx] = mse
                lengths_all[sample_idx] = sample_len
                offsets_all[sample_idx] = int(offsets[i])
                valid_len_all[sample_idx] = valid_len

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
        subset_root=np.array([subset_root]),
    )

    sample_subset_dir = os.path.join(samples_dir, subset_name)
    os.makedirs(sample_subset_dir, exist_ok=True)

    infer_cfg = getattr(h, "inference", {}) or {}
    vis_num = int(infer_cfg.get("num_visual_samples", 3))
    rng = np.random.default_rng(int(getattr(h, "seed", 1234)))
    choose_k = min(vis_num, n)
    vis_indices = rng.choice(n, size=choose_k, replace=False)
    for idx in sorted(vis_indices.tolist()):
        offset = int(offsets_all[idx])
        length = int(lengths_all[idx])
        valid_len = int(valid_len_all[idx])
        if offset >= 0:
            fn = f"offset_{offset}_length_{length}.png"
            title = f"{subset_name} | offset={offset}, length={length}"
        else:
            fn = f"index_{idx}_length_{length}.png"
            title = f"{subset_name} | index={idx}, length={length}"
        _plot_sample(
            gt=gt_list[idx],
            rec=rec_list[idx],
            valid_len=valid_len,
            title=title,
            save_path=os.path.join(sample_subset_dir, fn),
            offset=offset,
        )

    return {
        "subset_name": subset_name,
        "subset_root": subset_root,
        "num_samples": int(n),
        "mae": float(np.mean(mae_all)),
        "mse": float(np.mean(mse_all)),
        "npz_path": subset_npz,
        "sample_dir": sample_subset_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    h = load_hparams(args.config)
    _set_seed(int(getattr(h, "seed", 1234)))

    infer_cfg = getattr(h, "inference", {}) or {}
    if not isinstance(infer_cfg, dict):
        raise ValueError("inference must be a mapping in config")
    output_dir = str(infer_cfg.get("output_dir", "")).strip()
    if not output_dir:
        raise ValueError("Missing inference.output_dir in config")

    test_split = str(infer_cfg.get("test_split", "test"))
    test_roots = _load_test_roots(str(h.split_manifest_path), test_split=test_split)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, quantizer, decoder, input_norm, checkpoint_path = _load_codec_checkpoint(h, device)

    metrics_dir, samples_dir, recon_dir = _ensure_dirs(output_dir)

    results = []
    used_subset_names: Dict[str, int] = {}
    for subset_root in test_roots:
        raw_name = _subset_name_from_root(subset_root)
        subset_name = _ensure_unique_subset_name(raw_name, used_subset_names)
        print(f"[Infer] subset={subset_name} root={subset_root}")
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
        )
        results.append(res)

        metric_path = os.path.join(metrics_dir, f"{subset_name}.json")
        with open(metric_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

        print(
            f"[Done] subset={subset_name} | samples={res['num_samples']} | "
            f"MAE={res['mae']:.6f} | MSE={res['mse']:.6f}"
        )

    summary = {
        "config": args.config,
        "checkpoint_path": checkpoint_path,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "results": results,
    }
    summary_path = os.path.join(metrics_dir, "all_subsets_metrics.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[Summary] Saved: {summary_path}")


if __name__ == "__main__":
    main()
