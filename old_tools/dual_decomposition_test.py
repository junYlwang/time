from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from datasets.time_moe_dataset import TimeMoEDataset
from modules.decomposition import TrendResidualDecomposition
from modules.utils import load_hparams


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


def _plot_sample(x: np.ndarray, trend: np.ndarray, residual: np.ndarray, title: str, save_path: str) -> None:
    t = np.arange(x.shape[0], dtype=np.int64)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t, x, color="#1f77b4", linewidth=1.2)
    axes[0].set_title("input")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(t, trend, color="#2ca02c", linewidth=1.2)
    axes[1].set_title("trend")
    axes[1].grid(True, alpha=0.2)

    axes[2].plot(t, residual, color="#d62728", linewidth=1.2)
    axes[2].set_title("residual")
    axes[2].grid(True, alpha=0.2)
    axes[2].set_xlabel("time_index")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _build_decomposition_module(h) -> TrendResidualDecomposition:
    cfg = getattr(h, "decomposition", {}) or {}
    if not isinstance(cfg, dict):
        raise ValueError("decomposition must be a mapping in config")
    return TrendResidualDecomposition(
        num_channels=int(cfg.get("num_channels", 1)),
        kernel_sizes=cfg.get("kernel_sizes", [15, 31, 63, 127, 255]),
        weight_mode=str(cfg.get("weight_mode", "uniform")),
        summary_length=int(cfg.get("summary_length", 32)),
        gating_hidden_dim=int(cfg.get("gating_hidden_dim", 64)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    default_config = os.path.join(_PROJECT_ROOT, "configs", "base.yaml")
    parser.add_argument("--config", type=str, default=default_config)
    args = parser.parse_args()

    h = load_hparams(args.config)

    split_manifest_path = str(getattr(h, "split_manifest_path"))
    test_split = str(getattr(h, "test_split", "test"))
    normalization_method = getattr(h, "normalization_method", None)

    test_cfg = getattr(h, "decomposition_test", {}) or {}
    if not isinstance(test_cfg, dict):
        raise ValueError("decomposition_test must be a mapping in config")
    plot_length = int(test_cfg.get("plot_length", 2048))
    output_dir = str(test_cfg.get("output_dir", _PROJECT_ROOT))
    max_samples_per_subset = int(test_cfg.get("max_samples_per_subset", -1))

    os.makedirs(output_dir, exist_ok=True)

    decomp = _build_decomposition_module(h)
    decomp.eval()

    roots = _load_test_roots(split_manifest_path, test_split)
    used_names: Dict[str, int] = {}

    with torch.no_grad():
        for root in roots:
            subset_name = _ensure_unique_subset_name(_subset_name_from_root(root), used_names)
            ds = TimeMoEDataset(root, normalization_method=normalization_method)

            total = len(ds)
            if total <= 0:
                continue

            limit = total if max_samples_per_subset < 0 else min(total, max_samples_per_subset)
            for i in range(limit):
                seq = np.asarray(ds[i], dtype=np.float32)
                if seq.size <= 0:
                    continue

                if seq.size >= plot_length:
                    seg = seq[:plot_length]
                else:
                    seg = np.pad(seq, (0, plot_length - seq.size), mode="constant", constant_values=0.0)

                x = torch.from_numpy(seg).view(1, 1, -1)
                trend, residual = decomp(x)
                trend_np = trend.squeeze(0).squeeze(0).cpu().numpy()
                residual_np = residual.squeeze(0).squeeze(0).cpu().numpy()

                title = f"{subset_name} | idx={i}"
                save_name = f"decomp_{subset_name}_idx{i:06d}.png"
                save_path = os.path.join(output_dir, save_name)
                _plot_sample(seg, trend_np, residual_np, title, save_path)


if __name__ == "__main__":
    main()
