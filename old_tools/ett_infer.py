#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import torch

_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
for _path in (_SRC_DIR, _DATA_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from read_ett import list_ett_datasets, load_ett_series
from modules.decoder import Decoder
from modules.encoder_wo_quantize import Encoder
from modules.quantizer import build_quantizer
from modules.utils import build_input_norm, inverse_revin, load_checkpoint, load_hparams, set_seed


def _set_quantizer_eval_mode(quantizer) -> None:
    if quantizer is None:
        return
    q = quantizer.module if hasattr(quantizer, "module") else quantizer
    if hasattr(q, "set_stochastic_mode"):
        q.set_stochastic_mode(stochastic=False, temperature=0.3)


def _build_models(h, device: torch.device):
    encoder = Encoder(h).to(device)
    quantizer = build_quantizer(h).to(device)
    decoder = Decoder(h).to(device)
    input_norm = build_input_norm(h, device)
    return encoder, quantizer, decoder, input_norm


def _load_codec_checkpoint(h, device: torch.device):
    infer_cfg = getattr(h, "inference", {}) or {}
    if not isinstance(infer_cfg, dict):
        raise ValueError("inference must be a mapping in config")

    checkpoint_path = str(
        infer_cfg.get("checkpoint_path", "") or getattr(h, "checkpoint_codec", "")
    ).strip()
    if not checkpoint_path:
        raise ValueError("Missing checkpoint path in config (inference.checkpoint_path or checkpoint_codec)")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    encoder, quantizer, decoder, input_norm = _build_models(h, device)
    state = load_checkpoint(checkpoint_path, device)
    encoder.load_state_dict(state["encoder"], strict=True)
    if "quantizer" in state:
        quantizer.load_state_dict(state["quantizer"], strict=True)
    else:
        quantizer = None
        print("[Warn] No 'quantizer' found in checkpoint; running encoder-decoder only.")
    decoder.load_state_dict(state["decoder"], strict=True)
    if "input_norm" in state:
        input_norm.load_state_dict(state["input_norm"], strict=True)

    encoder.eval()
    decoder.eval()
    input_norm.eval()
    if quantizer is not None:
        quantizer.eval()
        _set_quantizer_eval_mode(quantizer)

    return encoder, quantizer, decoder, input_norm, checkpoint_path


def _plot_reconstruction(
    gt: np.ndarray,
    rec: np.ndarray,
    save_path: str,
    title: str,
    max_plot_length: int | None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it in the `time` environment."
        ) from exc

    if max_plot_length is None or max_plot_length <= 0:
        plot_len = gt.shape[0]
    else:
        plot_len = min(int(max_plot_length), gt.shape[0], rec.shape[0])
    if plot_len <= 0:
        raise ValueError(f"Invalid plot length for {save_path}: {plot_len}")

    t = np.arange(plot_len)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, gt[:plot_len], linewidth=1.2, label="groundtruth", color="#1f77b4")
    ax.plot(t, rec[:plot_len], linewidth=1.2, label="reconstruction", color="#d62728")
    ax.set_title(title)
    ax.set_xlabel("time_index")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _run_reconstruction(
    series_ct: np.ndarray,
    h,
    device: torch.device,
    encoder,
    quantizer,
    decoder,
    input_norm,
) -> tuple[np.ndarray, np.ndarray]:
    if series_ct.ndim != 2:
        raise ValueError(f"Expected series shape [C, T], got {tuple(series_ct.shape)}")

    x = torch.from_numpy(series_ct.astype(np.float32, copy=False)).unsqueeze(1).to(device)  # [C, 1, T]
    use_reversible_norm = bool(getattr(h, "use_reversible_norm", True))

    with torch.no_grad():
        if use_reversible_norm:
            x_in, mu, std = input_norm(x)
        else:
            x_in, mu, std = x, None, None

        latent = encoder(x_in)
        z_q = quantizer(latent).z_q if quantizer is not None else latent
        x_hat_norm = decoder(z_q)
        x_hat = inverse_revin(input_norm, x_hat_norm, mu, std) if use_reversible_norm else x_hat_norm

    tmin = min(x.shape[-1], x_hat.shape[-1])
    gt = x[:, 0, :tmin].detach().cpu().numpy()  # [C, T]
    rec = x_hat[:, 0, :tmin].detach().cpu().numpy()  # [C, T]
    return gt, rec


def _compute_metrics(gt_ct: np.ndarray, rec_ct: np.ndarray, feature_names: list[str]) -> dict[str, Any]:
    if gt_ct.shape != rec_ct.shape:
        raise ValueError(f"Shape mismatch: gt={gt_ct.shape}, rec={rec_ct.shape}")
    if gt_ct.shape[0] != len(feature_names):
        raise ValueError(
            f"Feature name count mismatch: num_features={gt_ct.shape[0]}, names={len(feature_names)}"
        )

    per_variable: list[dict[str, Any]] = []
    for idx, feature_name in enumerate(feature_names):
        diff = rec_ct[idx] - gt_ct[idx]
        per_variable.append(
            {
                "index": int(idx),
                "name": str(feature_name),
                "length": int(gt_ct.shape[1]),
                "mae": float(np.mean(np.abs(diff))),
                "mse": float(np.mean(np.square(diff))),
            }
        )

    all_diff = rec_ct - gt_ct
    return {
        "num_variables": int(gt_ct.shape[0]),
        "sequence_length": int(gt_ct.shape[1]),
        "mae": float(np.mean(np.abs(all_diff))),
        "mse": float(np.mean(np.square(all_diff))),
        "per_variable": per_variable,
    }


def _ensure_dataset_dirs(root_dir: str, dataset_name: str) -> tuple[str, str]:
    dataset_dir = os.path.join(root_dir, dataset_name)
    img_dir = os.path.join(dataset_dir, "img")
    os.makedirs(img_dir, exist_ok=True)
    return dataset_dir, img_dir


def _write_metrics(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _infer_one_dataset(
    ett_root: str,
    dataset_name: str,
    output_root: str,
    max_plot_length: int | None,
    checkpoint_path: str,
    h,
    device: torch.device,
    encoder,
    quantizer,
    decoder,
    input_norm,
) -> None:
    print(f"[Infer] dataset={dataset_name}")
    series_ct, metadata = load_ett_series(
        ett_root=ett_root,
        dataset_name=dataset_name,
        column=None,
        split=None,
        seq_len=0,
    )
    feature_names = list(metadata.feature_names)
    gt_ct, rec_ct = _run_reconstruction(
        series_ct=series_ct,
        h=h,
        device=device,
        encoder=encoder,
        quantizer=quantizer,
        decoder=decoder,
        input_norm=input_norm,
    )

    dataset_dir, img_dir = _ensure_dataset_dirs(output_root, dataset_name)
    for idx, feature_name in enumerate(feature_names):
        save_path = os.path.join(img_dir, f"{idx:02d}_{feature_name}.png")
        _plot_reconstruction(
            gt=gt_ct[idx],
            rec=rec_ct[idx],
            save_path=save_path,
            title=f"{dataset_name} | {feature_name}",
            max_plot_length=max_plot_length,
        )

    metrics = _compute_metrics(gt_ct=gt_ct, rec_ct=rec_ct, feature_names=feature_names)
    metrics_payload = {
        "dataset_name": dataset_name,
        "source_file": metadata.source_file,
        "checkpoint_path": checkpoint_path,
        "overall": {
            "mae": metrics["mae"],
            "mse": metrics["mse"],
            "num_variables": metrics["num_variables"],
            "sequence_length": metrics["sequence_length"],
        },
        "variables": metrics["per_variable"],
    }
    _write_metrics(os.path.join(dataset_dir, "metrics.json"), metrics_payload)
    print(
        f"  variables={metrics['num_variables']} length={metrics['sequence_length']} "
        f"overall_mae={metrics['mae']:.6f} overall_mse={metrics['mse']:.6f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run codec reconstruction inference on full ETT-small datasets.")
    parser.add_argument("--config", required=True, help="Path to ett-base.yaml")
    args = parser.parse_args()

    h = load_hparams(args.config)
    set_seed(int(getattr(h, "seed", 1234)))

    infer_cfg = getattr(h, "inference", {}) or {}
    if not isinstance(infer_cfg, dict):
        raise ValueError("inference must be a mapping in config")

    ett_root = str(getattr(h, "ett_root", "")).strip()
    if not ett_root:
        raise ValueError("Missing ett_root in config")
    output_root = str(infer_cfg.get("output_dir", "")).strip()
    if not output_root:
        raise ValueError("Missing inference.output_dir in config")
    max_plot_length = infer_cfg.get("max_plot_length", 0)
    max_plot_length = None if int(max_plot_length) <= 0 else int(max_plot_length)

    os.makedirs(output_root, exist_ok=True)
    datasets = [name for name in list_ett_datasets(ett_root) if name in {"ETTh1", "ETTh2", "ETTm1", "ETTm2"}]
    datasets.sort()
    if len(datasets) != 4:
        raise ValueError(f"Expected 4 ETT-small datasets, found: {datasets}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    encoder, quantizer, decoder, input_norm, checkpoint_path = _load_codec_checkpoint(h, device)
    print(f"Loaded checkpoint: {checkpoint_path}")

    for dataset_name in datasets:
        _infer_one_dataset(
            ett_root=ett_root,
            dataset_name=dataset_name,
            output_root=output_root,
            max_plot_length=max_plot_length,
            checkpoint_path=checkpoint_path,
            h=h,
            device=device,
            encoder=encoder,
            quantizer=quantizer,
            decoder=decoder,
            input_norm=input_norm,
        )

    print(f"Inference complete. Results written to: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
