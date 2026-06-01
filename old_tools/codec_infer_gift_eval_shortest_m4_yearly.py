from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import GIFT-Eval before codec_infer adds src/ to sys.path, otherwise the local
# src/datasets package can shadow HuggingFace datasets.
from data.gift_eval.data import Dataset as GiftEvalDataset

_SRC_DIR = os.path.join(_PROJECT_ROOT, 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from modules.decoder import Decoder
from modules.encoder_wo_quantize import Encoder
from modules.quantizer import build_quantizer
from modules.revin import ReversibleInstanceNorm1D
from modules.utils import load_checkpoint, load_hparams, set_seed


DEFAULT_GIFT_EVAL_ROOT = '/mnt/shared-storage-gpfs2/speechllm-share/lishenyi/datasets/gift-eval'



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
    infer_cfg = getattr(h, 'inference', {}) or {}
    if not isinstance(infer_cfg, dict):
        raise ValueError('inference must be a mapping in config')
    checkpoint_path = str(infer_cfg.get('checkpoint_path', '')).strip()
    if not checkpoint_path:
        raise ValueError('Missing inference.checkpoint_path in config')
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'checkpoint not found: {checkpoint_path}')

    encoder, quantizer, decoder, input_norm = _build_models(h, device)

    state = load_checkpoint(checkpoint_path, device)
    encoder.load_state_dict(state['encoder'], strict=True)
    if 'quantizer' in state:
        quantizer.load_state_dict(state['quantizer'], strict=True)
    else:
        quantizer = None
        print("[Warn] No 'quantizer' found in checkpoint. Using no-quantizer inference path for compatibility.")
    decoder.load_state_dict(state['decoder'], strict=True)
    if 'input_norm' in state:
        input_norm.load_state_dict(state['input_norm'], strict=True)

    encoder.eval()
    if quantizer is not None:
        quantizer.eval()
    decoder.eval()
    input_norm.eval()

    return encoder, quantizer, decoder, input_norm, checkpoint_path


def _series_length(target) -> int:
    arr = np.asarray(target)
    if arr.ndim != 1:
        raise ValueError(f'This script expects univariate targets, got shape={arr.shape}')
    return int(arr.shape[-1])


def _load_target_samples(h) -> tuple[List[Dict], Dict]:
    infer_cfg = getattr(h, 'inference', {}) or {}
    if not isinstance(infer_cfg, dict):
        raise ValueError('inference must be a mapping in config')

    gift_eval_root = str(infer_cfg.get('gift_eval_data_root', DEFAULT_GIFT_EVAL_ROOT))
    dataset_name = str(infer_cfg.get('dataset_name', 'm4_yearly'))
    term = str(infer_cfg.get('term', 'short'))
    target_min_history = infer_cfg.get('target_min_test_history', 13)
    target_min_history = None if target_min_history is None else int(target_min_history)

    os.environ['GIFT_EVAL'] = gift_eval_root
    dataset = GiftEvalDataset(name=dataset_name, term=term)
    prediction_length = int(dataset.prediction_length)
    windows = int(dataset.windows)
    heldout_length = prediction_length * windows

    rows = []
    min_seen = None
    for row_index, entry in enumerate(dataset.hf_dataset):
        target = np.asarray(entry['target'], dtype=np.float32)
        series_length = _series_length(target)
        min_test_history = series_length - heldout_length
        if min_test_history <= 0:
            raise ValueError(
                'non-positive min_test_history: '
                f'dataset={dataset_name}, term={term}, row={row_index}, '
                f'series_length={series_length}, prediction_length={prediction_length}, windows={windows}'
            )
        min_seen = min_test_history if min_seen is None else min(min_seen, min_test_history)
        rows.append(
            {
                'row_index': int(row_index),
                'item_id': str(entry.get('item_id', row_index)),
                'target': target,
                'series_length': int(series_length),
                'min_test_history': int(min_test_history),
            }
        )

    selected_history = int(min_seen if target_min_history is None else target_min_history)
    selected = [row for row in rows if row['min_test_history'] == selected_history]
    if not selected:
        raise ValueError(
            f'No samples found for min_test_history={selected_history} in {dataset_name}/{term}; '
            f'minimum observed value is {min_seen}'
        )

    meta = {
        'gift_eval_data_root': gift_eval_root,
        'dataset_name': dataset_name,
        'term': term,
        'target_min_test_history': selected_history,
        'prediction_length': prediction_length,
        'windows': windows,
        'heldout_length': heldout_length,
        'num_samples': len(selected),
    }
    return selected, meta


def _downsample_factor(h) -> int:
    factor = 1
    for ratio in getattr(h, 'down_ratio'):
        ratio = int(ratio)
        if ratio <= 0:
            raise ValueError(f'down_ratio values must be positive, got {getattr(h, "down_ratio")}')
        factor *= ratio
    return factor


def _plot_sample(
    groundtruth: np.ndarray,
    reconstruction: np.ndarray,
    title: str,
    save_path: str,
    num_plot_points: int,
) -> None:
    if num_plot_points <= 0:
        raise ValueError(f"num_plot_points must be positive, got {num_plot_points}")
    gt = groundtruth[: min(num_plot_points, groundtruth.shape[0])]
    rec = reconstruction[: min(num_plot_points, reconstruction.shape[0])]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(gt.shape[0]), gt, color='#1f77b4', linewidth=1.5, label=f'groundtruth ({groundtruth.shape[0]})')
    ax.plot(np.arange(rec.shape[0]), rec, color='#d62728', linewidth=1.5, label=f'reconstruction ({reconstruction.shape[0]})')
    ax.set_title(title)
    ax.set_xlabel('time_index')
    ax.set_ylabel('value')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _ensure_dirs(output_dir: str) -> tuple[str, str, str]:
    metrics_dir = os.path.join(output_dir, 'metrics')
    samples_dir = os.path.join(output_dir, 'samples')
    recon_dir = os.path.join(output_dir, 'reconstructions')
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    return metrics_dir, samples_dir, recon_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()

    h = load_hparams(args.config)
    set_seed(int(getattr(h, 'seed', 1234)))

    infer_cfg = getattr(h, 'inference', {}) or {}
    if not isinstance(infer_cfg, dict):
        raise ValueError('inference must be a mapping in config')
    output_dir = str(infer_cfg.get('output_dir', '')).strip()
    if not output_dir:
        raise ValueError('Missing inference.output_dir in config')

    num_visual_samples = int(infer_cfg.get('num_visual_samples', 16))
    num_plot_points = int(infer_cfg.get('num_plot_points', 20))
    if num_plot_points <= 0:
        raise ValueError(f'num_plot_points must be positive, got {num_plot_points}')
    downsample_factor = _downsample_factor(h)

    samples, dataset_meta = _load_target_samples(h)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, quantizer, decoder, input_norm, checkpoint_path = _load_codec_checkpoint(h, device)
    metrics_dir, samples_dir, recon_dir = _ensure_dirs(output_dir)

    item_ids: List[str] = []
    row_indices = np.zeros((len(samples),), dtype=np.int64)
    input_lengths = np.zeros((len(samples),), dtype=np.int64)
    original_mean = np.zeros((len(samples),), dtype=np.float64)
    original_var = np.zeros((len(samples),), dtype=np.float64)
    pad_lengths = np.zeros((len(samples),), dtype=np.int64)
    normalized_mean = np.zeros((len(samples),), dtype=np.float64)
    normalized_var = np.zeros((len(samples),), dtype=np.float64)
    reconstruction_lengths = np.zeros((len(samples),), dtype=np.int64)
    overlap_lengths = np.zeros((len(samples),), dtype=np.int64)
    mae_all = np.zeros((len(samples),), dtype=np.float64)
    mse_all = np.zeros((len(samples),), dtype=np.float64)
    inputs: List[np.ndarray] = []
    reconstructions: List[np.ndarray] = []

    with torch.no_grad():
        for sample_idx, sample in enumerate(samples):
            x_np = np.asarray(sample['target'], dtype=np.float32)
            raw_len = int(x_np.shape[0])
            x = torch.from_numpy(x_np).view(1, 1, -1).to(device)
            x_norm, mu, std = input_norm(x)
            pad_len = (-raw_len) % downsample_factor
            if pad_len > 0:
                left_pad = torch.zeros((1, 1, pad_len), dtype=x_norm.dtype, device=x_norm.device)
                x_in = torch.cat([left_pad, x_norm], dim=-1)
            else:
                x_in = x_norm

            latent = encoder(x_in)
            zq = quantizer(latent).z_q if quantizer is not None else latent
            x_hat_norm = decoder(zq)
            rec_norm = x_hat_norm[:, :, pad_len:pad_len + raw_len]
            x_hat = input_norm.inverse(rec_norm, mu, std)

            rec_np = x_hat[0, 0].detach().cpu().numpy().astype(np.float32, copy=True)
            overlap_len = int(min(raw_len, rec_np.shape[0]))
            diff = rec_np[:overlap_len] - x_np[:overlap_len]

            item_ids.append(str(sample['item_id']))
            row_indices[sample_idx] = int(sample['row_index'])
            input_lengths[sample_idx] = raw_len
            original_mean[sample_idx] = float(np.mean(x_np))
            original_var[sample_idx] = float(np.var(x_np))
            normalized_input = x_in[0, 0].detach().cpu().numpy()
            pad_lengths[sample_idx] = int(pad_len)
            normalized_mean[sample_idx] = float(np.mean(normalized_input))
            normalized_var[sample_idx] = float(np.var(normalized_input))
            reconstruction_lengths[sample_idx] = int(rec_np.shape[0])
            overlap_lengths[sample_idx] = overlap_len
            mae_all[sample_idx] = float(np.mean(np.abs(diff)))
            mse_all[sample_idx] = float(np.mean(np.square(diff)))
            inputs.append(x_np.copy())
            reconstructions.append(rec_np)

            if sample_idx < num_visual_samples:
                title = (
                    f"{dataset_meta['dataset_name']}/{dataset_meta['term']} | "
                    f"row={sample['row_index']} item={sample['item_id']} | "
                    f"raw={raw_len} rec={rec_np.shape[0]}"
                )
                save_path = os.path.join(samples_dir, f"row_{int(sample['row_index']):05d}_item_{sample['item_id']}.png")
                _plot_sample(x_np, rec_np, title, save_path, num_plot_points=num_plot_points)

    recon_path = os.path.join(recon_dir, 'm4_yearly_short_min_history_samples.npz')
    np.savez_compressed(
        recon_path,
        item_id=np.array(item_ids, dtype=object),
        row_index=row_indices,
        input_length=input_lengths,
        original_mean=original_mean,
        original_var=original_var,
        pad_length=pad_lengths,
        normalized_mean=normalized_mean,
        normalized_var=normalized_var,
        reconstruction_length=reconstruction_lengths,
        overlap_length=overlap_lengths,
        groundtruth=np.array(inputs, dtype=object),
        reconstruction=np.array(reconstructions, dtype=object),
        mae_per_sample=mae_all,
        mse_per_sample=mse_all,
        **{key: np.array([value]) for key, value in dataset_meta.items()},
    )

    summary = {
        'config': args.config,
        'checkpoint_path': checkpoint_path,
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        'dataset': dataset_meta,
        'num_visual_samples': num_visual_samples,
        'num_plot_points': num_plot_points,
        'padding_side': 'left',
        'padding_value': 0.0,
        'normalization': 'valid_points_zscore',
        'downsample_factor': downsample_factor,
        'mae': float(np.mean(mae_all)),
        'mse': float(np.mean(mse_all)),
        'input_lengths': sorted(set(int(x) for x in input_lengths.tolist())),
        'reconstruction_lengths': sorted(set(int(x) for x in reconstruction_lengths.tolist())),
        'overlap_lengths': sorted(set(int(x) for x in overlap_lengths.tolist())),
        'npz_path': recon_path,
        'sample_dir': samples_dir,
    }
    summary_path = os.path.join(metrics_dir, 'm4_yearly_short_min_history_metrics.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"[Done] samples={len(samples)} | "
        f"input_lengths={summary['input_lengths']} | "
        f"reconstruction_lengths={summary['reconstruction_lengths']} | "
        f"MAE={summary['mae']:.6f} | MSE={summary['mse']:.6f}"
    )
    print(f'[Summary] Saved: {summary_path}')


if __name__ == '__main__':
    main()
