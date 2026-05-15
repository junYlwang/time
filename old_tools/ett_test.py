from __future__ import annotations

import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from datasets.ett_dataset import ETTDataset
from modules.encoder_wo_quantize import Encoder
from modules.probe import MLPForecastProbe, TransformerForecastProbe
from modules.quantizer import build_quantizer
from modules.utils import (
    build_input_norm,
    inverse_revin,
    load_checkpoint,
    load_hparams,
    set_seed,
)


def _infer_run_dir_from_config(config_path: str) -> str:
    config_dir = os.path.dirname(os.path.abspath(config_path))
    if os.path.basename(config_dir) != "meta":
        raise ValueError(
            f"Expected config to live under a meta directory, got: {config_path}"
        )
    return os.path.dirname(config_dir)


def _load_best_probe_path(run_dir: str) -> str:
    best_path = os.path.join(run_dir, "checkpoints", "best_checkpoints.json")
    if not os.path.isfile(best_path):
        raise FileNotFoundError(f"Missing best checkpoint record: {best_path}")

    with open(best_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list) or len(records) == 0:
        raise ValueError(f"Empty or invalid best checkpoint record: {best_path}")

    probe_path = records[0].get("probe_path")
    if not probe_path or not os.path.isfile(probe_path):
        raise FileNotFoundError(f"Best probe checkpoint not found: {probe_path}")
    return probe_path


def _build_probe(h, device: torch.device):
    probe_type = str(getattr(h, "probe_type", "mlp")).lower()
    if probe_type == "mlp":
        probe = MLPForecastProbe(h).to(device)
    elif probe_type == "transformer":
        probe = TransformerForecastProbe(h).to(device)
    else:
        raise ValueError(
            f"Unsupported probe_type: {probe_type}. Expected one of: mlp, transformer"
        )
    return probe


def _plot_first_sample(
    x: torch.Tensor,
    y: torch.Tensor,
    y_hat: torch.Tensor,
    save_path: str,
    title: str = "First Test Sample",
) -> None:
    history = x[0, 0].detach().cpu().numpy()
    target = y[0, 0].detach().cpu().numpy()
    pred = y_hat[0, 0].detach().cpu().numpy()

    hist_len = history.shape[0]
    pred_len = target.shape[0]
    hist_steps = list(range(hist_len))
    future_steps = list(range(hist_len, hist_len + pred_len))

    plt.figure(figsize=(12, 4))
    plt.plot(hist_steps, history, label="history", color="tab:blue", linewidth=1.5)
    plt.plot(future_steps, target, label="groundtruth", color="tab:green", linewidth=1.5)
    plt.plot(future_steps, pred, label="prediction", color="tab:red", linewidth=1.5)
    plt.axvline(hist_len - 1, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _first_indices_by_variable(testset: ETTDataset) -> dict[str, int]:
    names = list(getattr(testset, "selected_feature_names", []))
    num_variables = int(getattr(testset, "num_variables", 1))
    if not names or len(names) != num_variables:
        names = [f"var_{idx:02d}" for idx in range(num_variables)]
    return {name: idx for idx, name in enumerate(names)}


def test(h, run_dir: str, probe_checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(h.seed))
    ett_column = getattr(h, "ett_column", "__all__")
    selected_column = "__all__" if ett_column in (None, "__all__") else str(ett_column)

    testset = ETTDataset(
        h.ett_root,
        h.dataset_name,
        split="test",
        seq_len=h.seq_len,
        pred_len=h.pred_len,
        stride=getattr(h, "stride", 1),
        column=ett_column,
    )
    test_loader = DataLoader(
        testset,
        batch_size=int(getattr(h, "eval_batch_size", 32)),
        num_workers=int(h.num_workers),
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    h.latent_seq_len = math.floor(int(h.seq_len) / math.prod(h.down_ratio))

    encoder = Encoder(h).to(device)
    quantizer = build_quantizer(h).to(device)
    input_norm = build_input_norm(h, device)
    probe = _build_probe(h, device)

    state_dict_codec = load_checkpoint(h.checkpoint_codec, device)
    encoder.load_state_dict(state_dict_codec["encoder"], strict=True)
    quantizer.load_state_dict(state_dict_codec["quantizer"], strict=True)
    if "input_norm" in state_dict_codec:
        input_norm.load_state_dict(state_dict_codec["input_norm"], strict=True)

    state_dict_probe = load_checkpoint(probe_checkpoint_path, device)
    probe.load_state_dict(state_dict_probe["probe"], strict=True)

    encoder.eval()
    quantizer.eval()
    input_norm.eval()
    probe.eval()

    mae_sum = 0.0
    mse_sum = 0.0
    value_count = 0
    sample_indices = _first_indices_by_variable(testset)
    pending_samples = dict(sample_indices)
    sample_records: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    seen_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            x = batch["seq"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)

            if bool(getattr(h, "use_reversible_norm", True)):
                x_in, mu, std = input_norm(x)
            else:
                x_in, mu, std = x, None, None

            latent = encoder(x_in)
            quantized_out = quantizer(latent)
            if h.latent_mode == "discrete":
                features = quantized_out.z_q
            elif h.latent_mode == "continuous":
                features = latent
            else:
                raise ValueError("latent_mode must be 'discrete' or 'continuous'")

            y_hat_norm = probe(features)
            y_hat = (
                inverse_revin(input_norm, y_hat_norm, mu, std)
                if bool(getattr(h, "use_reversible_norm", True))
                else y_hat_norm
            )

            diff = y_hat - y
            mae_sum += diff.abs().sum().item()
            mse_sum += diff.square().sum().item()
            value_count += diff.numel()

            batch_size = x.size(0)
            if pending_samples:
                for feature_name, global_idx in list(pending_samples.items()):
                    local_idx = global_idx - seen_samples
                    if 0 <= local_idx < batch_size:
                        sample_records[feature_name] = (
                            x[local_idx : local_idx + 1].detach().cpu(),
                            y[local_idx : local_idx + 1].detach().cpu(),
                            y_hat[local_idx : local_idx + 1].detach().cpu(),
                        )
                        del pending_samples[feature_name]
            seen_samples += batch_size

    if value_count == 0:
        raise RuntimeError("Empty test set: no samples were evaluated.")

    test_mae = mae_sum / value_count
    test_mse = mse_sum / value_count

    metrics = {
        "dataset_name": str(h.dataset_name),
        "latent_mode": str(h.latent_mode),
        "probe_type": str(h.probe_type),
        "ett_column": selected_column,
        "num_variables": int(getattr(testset, "num_variables", 1)),
        "feature_names": list(getattr(testset, "selected_feature_names", [])),
        "num_samples": int(len(testset)),
        "seq_len": int(h.seq_len),
        "pred_len": int(h.pred_len),
        "probe_checkpoint_path": probe_checkpoint_path,
        "test_mae": float(test_mae),
        "test_mse": float(test_mse),
    }

    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    samples_dir = os.path.join(run_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    for feature_name, sample in sample_records.items():
        sample_path = os.path.join(samples_dir, f"{feature_name}.png")
        _plot_first_sample(*sample, save_path=sample_path, title=f"{h.dataset_name} | {feature_name}")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config yaml")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    h = load_hparams(config_path)
    run_dir = _infer_run_dir_from_config(config_path)
    probe_checkpoint_path = _load_best_probe_path(run_dir)

    test(h, run_dir, probe_checkpoint_path)


if __name__ == "__main__":
    main()
