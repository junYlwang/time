from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

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

from read_ucr import list_ucr_datasets, load_ucr_split
from modules.decoder import Decoder
from modules.encoder_wo_quantize import Encoder
from modules.quantizer import build_quantizer
from modules.utils import build_input_norm, inverse_revin, load_checkpoint, load_hparams, set_seed


@dataclass
class ReconstructionRecord:
    index: int
    label: str
    length: int
    valid_length: int
    mae: float
    mse: float

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


def _load_codec_checkpoint(h, device: torch.device, checkpoint_override: str | None = None):
    infer_cfg = getattr(h, "inference", {}) or {}
    if not isinstance(infer_cfg, dict):
        raise ValueError("inference must be a mapping in config")
    checkpoint_path = str(checkpoint_override or infer_cfg.get("checkpoint_path", "")).strip()
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
        print("[Warn] No 'quantizer' found in checkpoint; running encoder-decoder only.")
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


def _ensure_dirs(output_dir: str) -> tuple[str, str, str]:
    metrics_dir = os.path.join(output_dir, "metrics")
    samples_dir = os.path.join(output_dir, "samples")
    recon_dir = os.path.join(output_dir, "reconstructions")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    return metrics_dir, samples_dir, recon_dir


def _write_summary_csv(metrics_dir: str, results: List[Dict]) -> str:
    csv_path = os.path.join(metrics_dir, "summary.csv")
    fieldnames = [
        "dataset",
        "split",
        "num_total_samples",
        "num_evaluated_samples",
        "length_min",
        "length_max",
        "length_mean",
        "num_classes",
        "mean_mae",
        "mean_mse",
        "global_mae",
        "global_mse",
        "npz_path",
        "sample_dir",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "dataset": result["dataset"],
                    "split": result["split"],
                    "num_total_samples": result["num_total_samples"],
                    "num_evaluated_samples": result["num_evaluated_samples"],
                    "length_min": result["length"]["min"],
                    "length_max": result["length"]["max"],
                    "length_mean": result["length"]["mean"],
                    "num_classes": result["num_classes"],
                    "mean_mae": result["mean_mae"],
                    "mean_mse": result["mean_mse"],
                    "global_mae": result["global_mae"],
                    "global_mse": result["global_mse"],
                    "npz_path": result["npz_path"],
                    "sample_dir": result["sample_dir"],
                }
            )
    return csv_path


def _plot_sample(
    gt: np.ndarray,
    rec: np.ndarray,
    valid_len: int,
    title: str,
    save_path: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it in the `time` environment "
            "or run with --plot-samples 0."
        ) from exc

    valid_len = int(max(1, min(valid_len, gt.shape[0], rec.shape[0], 200)))
    t = np.arange(valid_len)

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


def _infer_one_series(
    x_np: np.ndarray,
    h,
    device: torch.device,
    encoder,
    quantizer,
    decoder,
    input_norm,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    x = torch.from_numpy(x_np.astype(np.float32, copy=False)).view(1, 1, -1).to(device)
    use_reversible_norm = bool(getattr(h, "use_reversible_norm", True))

    with torch.no_grad():
        if use_reversible_norm:
            x_in, mu, std = input_norm(x)
        else:
            x_in, mu, std = x, None, None

        latent = encoder(x_in)
        zq = quantizer(latent).z_q if quantizer is not None else latent
        x_hat_norm = decoder(zq)
        x_hat = inverse_revin(input_norm, x_hat_norm, mu, std) if use_reversible_norm else x_hat_norm

    tmin = min(x.shape[-1], x_hat.shape[-1])
    gt = x[0, 0, :tmin].detach().cpu().numpy()
    rec = x_hat[0, 0, :tmin].detach().cpu().numpy()
    diff = rec - gt
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(np.square(diff)))
    return gt, rec, mae, mse, int(tmin)


def _select_indices(num_samples: int, max_samples: int | None) -> list[int]:
    if max_samples is None or max_samples <= 0 or max_samples >= num_samples:
        return list(range(num_samples))
    return list(range(int(max_samples)))


def _infer_one_dataset(
    ucr_root: str,
    dataset_name: str,
    split: str,
    h,
    device: torch.device,
    encoder,
    quantizer,
    decoder,
    input_norm,
    output_dirs: tuple[str, str, str],
    max_samples: int | None,
    plot_samples: int,
) -> dict:
    metrics_dir, samples_dir, recon_dir = output_dirs
    sequences, labels, metadata = load_ucr_split(ucr_root, dataset_name, split)
    selected_indices = _select_indices(len(sequences), max_samples=max_samples)

    print(
        f"[Infer] dataset={dataset_name} split={metadata.split} "
        f"total={len(sequences)} selected={len(selected_indices)}"
    )

    per_sample: list[ReconstructionRecord] = []
    gt_list: list[np.ndarray] = []
    rec_list: list[np.ndarray] = []
    array_positions: dict[int, int] = {}
    total_abs_error = 0.0
    total_sq_error = 0.0
    total_points = 0

    for order, idx in enumerate(selected_indices, start=1):
        gt, rec, mae, mse, valid_len = _infer_one_series(
            x_np=sequences[idx],
            h=h,
            device=device,
            encoder=encoder,
            quantizer=quantizer,
            decoder=decoder,
            input_norm=input_norm,
        )
        per_sample.append(
            ReconstructionRecord(
                index=int(idx),
                label=str(labels[idx]),
                length=int(sequences[idx].shape[0]),
                valid_length=int(valid_len),
                mae=mae,
                mse=mse,
            )
        )
        array_positions[int(idx)] = len(gt_list)
        gt_list.append(gt)
        rec_list.append(rec)
        diff = rec[:valid_len] - gt[:valid_len]
        total_abs_error += float(np.abs(diff).sum())
        total_sq_error += float(np.square(diff).sum())
        total_points += int(valid_len)

        if order == len(selected_indices) or order % 10 == 0:
            print(f"  processed {order}/{len(selected_indices)}")

    dataset_samples_dir = os.path.join(samples_dir, dataset_name)
    os.makedirs(dataset_samples_dir, exist_ok=True)
    plot_count = min(max(0, int(plot_samples)), len(per_sample))
    for rec in per_sample[:plot_count]:
        pos = array_positions[rec.index]
        title = f"{dataset_name} | split={metadata.split} | index={rec.index} | label={rec.label}"
        _plot_sample(
            gt=gt_list[pos],
            rec=rec_list[pos],
            valid_len=rec.valid_length,
            title=title,
            save_path=os.path.join(dataset_samples_dir, f"sample_{rec.index:04d}.png"),
        )

    npz_path = os.path.join(recon_dir, f"{dataset_name}_{metadata.split}.npz")
    np.savez_compressed(
        npz_path,
        groundtruth=np.array(gt_list, dtype=object),
        reconstruction=np.array(rec_list, dtype=object),
        labels=np.asarray([record.label for record in per_sample], dtype=object),
        indices=np.asarray([record.index for record in per_sample], dtype=np.int64),
        mae_per_sample=np.asarray([record.mae for record in per_sample], dtype=np.float64),
        mse_per_sample=np.asarray([record.mse for record in per_sample], dtype=np.float64),
        valid_length=np.asarray([record.valid_length for record in per_sample], dtype=np.int64),
    )

    mean_mae = float(np.mean([record.mae for record in per_sample])) if per_sample else 0.0
    mean_mse = float(np.mean([record.mse for record in per_sample])) if per_sample else 0.0
    global_mae = float(total_abs_error / max(1, total_points))
    global_mse = float(total_sq_error / max(1, total_points))

    result = {
        "dataset": dataset_name,
        "split": metadata.split,
        "num_total_samples": metadata.num_samples,
        "num_evaluated_samples": len(per_sample),
        "length": {
            "min": metadata.min_length,
            "max": metadata.max_length,
            "mean": metadata.mean_length,
        },
        "num_classes": metadata.num_classes,
        "mean_mae": mean_mae,
        "mean_mse": mean_mse,
        "global_mae": global_mae,
        "global_mse": global_mse,
        "npz_path": npz_path,
        "sample_dir": dataset_samples_dir,
        "per_sample": [record.__dict__ for record in per_sample],
    }

    metric_path = os.path.join(metrics_dir, f"{dataset_name}_{metadata.split}.json")
    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(
        f"[Done] dataset={dataset_name} split={metadata.split} "
        f"| evaluated={len(per_sample)} | mean_MAE={mean_mae:.6f} "
        f"| mean_MSE={mean_mse:.6f} | global_MAE={global_mae:.6f} | global_MSE={global_mse:.6f}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run codec reconstruction on selected UCR datasets.")
    parser.add_argument("--config", type=str, required=True, help="Path to codec config YAML.")
    parser.add_argument(
        "--ucr-root",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "data", "UCR112"),
        help="Path to the UCR root directory.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="One or more UCR dataset names. If omitted, run all datasets under --ucr-root.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="TRAIN",
        help="Split to run inference on: TRAIN or TEST. Default: TRAIN.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=32,
        help="Evaluate only the first N samples of each selected dataset. Use <=0 for all samples.",
    )
    parser.add_argument(
        "--plot-samples",
        type=int,
        default=3,
        help="Number of reconstruction plots to save per dataset. Use 0 to disable plots.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "runs", "ucr_infer"),
        help="Directory to save metrics and plots.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Optional override for inference.checkpoint_path in the config.",
    )
    args = parser.parse_args()

    h = load_hparams(args.config)
    set_seed(int(getattr(h, "seed", 1234)))

    available = list_ucr_datasets(args.ucr_root)
    if not available:
        raise FileNotFoundError(f"No UCR datasets found under: {args.ucr_root}")

    if args.datasets:
        selected = list(args.datasets)
        missing = sorted(set(selected) - set(available))
        if missing:
            raise FileNotFoundError(f"Requested datasets not found under {args.ucr_root}: {missing}")
    else:
        selected = available
        print(f"[Info] No --datasets specified; running all {len(selected)} datasets.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, quantizer, decoder, input_norm, checkpoint_path = _load_codec_checkpoint(
        h, device, checkpoint_override=args.checkpoint_path
    )

    run_name = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_dir = os.path.join(args.output_dir, run_name)
    output_dirs = _ensure_dirs(output_dir)

    results: List[Dict] = []
    for dataset_name in selected:
        result = _infer_one_dataset(
            ucr_root=args.ucr_root,
            dataset_name=dataset_name,
            split=args.split,
            h=h,
            device=device,
            encoder=encoder,
            quantizer=quantizer,
            decoder=decoder,
            input_norm=input_norm,
            output_dirs=output_dirs,
            max_samples=args.max_samples,
            plot_samples=args.plot_samples,
        )
        results.append(result)

    summary = {
        "config": args.config,
        "checkpoint_path": checkpoint_path,
        "ucr_root": args.ucr_root,
        "datasets": selected,
        "split": str(args.split).upper(),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "results": results,
    }
    summary_path = os.path.join(output_dirs[0], "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    csv_path = _write_summary_csv(output_dirs[0], results)

    print(f"[Summary] Saved: {summary_path}")
    print(f"[Summary] Saved: {csv_path}")


if __name__ == "__main__":
    main()
