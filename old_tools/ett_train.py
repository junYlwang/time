from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
    _load_topk,
    _save_topk,
    build_env,
    build_input_norm,
    get_state_dict,
    inverse_revin,
    load_checkpoint,
    load_hparams,
    save_checkpoint,
    set_seed,
)


def update_topk_and_prune_min(ckpt_dir: str, keep_k: int, score: float, steps: int):
    if keep_k <= 0:
        return []

    record_path = os.path.join(ckpt_dir, "best_checkpoints.json")
    records = _load_topk(record_path)
    if records is None:
        records = []

    records.append(
        {
            "score": float(score),
            "steps": int(steps),
            "probe_path": os.path.join(ckpt_dir, f"probe_steps={steps:08d}_score={score:.6f}"),
            "state_path": os.path.join(ckpt_dir, f"state_steps={steps:08d}_score={score:.6f}"),
        }
    )
    records.sort(key=lambda r: (float(r["score"]), int(r["steps"])))

    while len(records) > keep_k:
        dropped = records.pop(-1)
        for p in (dropped["probe_path"], dropped["state_path"]):
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass

    _save_topk(record_path, records)
    return records


def _build_probe(h, device: torch.device):
    probe_type = str(getattr(h, "probe_type", "mlp")).lower()
    if probe_type == "mlp":
        probe = MLPForecastProbe(h).to(device)
    elif probe_type == "transformer":
        probe = TransformerForecastProbe(h).to(device)
    else:
        raise ValueError(f"Unsupported probe_type: {probe_type}. Expected one of: mlp, transformer")
    return probe


def train(h):
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(h.seed))
    ett_column = getattr(h, "ett_column", "__all__")

    trainset = ETTDataset(
        h.ett_root,
        h.dataset_name,
        split="train",
        seq_len=h.seq_len,
        pred_len=h.pred_len,
        stride=getattr(h, "stride", 1),
        column=ett_column,
    )
    valset = ETTDataset(
        h.ett_root,
        h.dataset_name,
        split="val",
        seq_len=h.seq_len,
        pred_len=h.pred_len,
        stride=getattr(h, "stride", 1),
        column=ett_column,
    )

    if hasattr(h, "meta_dir"):
        with open(os.path.join(h.meta_dir, "ett_train_dataset.json"), "w", encoding="utf-8") as f:
            json.dump(trainset.summary(), f, indent=2, ensure_ascii=False)
        with open(os.path.join(h.meta_dir, "ett_val_dataset.json"), "w", encoding="utf-8") as f:
            json.dump(valset.summary(), f, indent=2, ensure_ascii=False)

    h.latent_seq_len = math.floor(int(h.seq_len) / math.prod(h.down_ratio))

    train_loader = DataLoader(
        trainset,
        batch_size=int(getattr(h, "train_batch_size", 32)),
        num_workers=int(h.num_workers),
        shuffle=True,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        valset,
        batch_size=int(getattr(h, "eval_batch_size", 32)),
        num_workers=int(h.num_workers),
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    encoder = Encoder(h).to(device)
    quantizer = build_quantizer(h).to(device)
    input_norm = build_input_norm(h, device)
    probe = _build_probe(h, device)

    state_dict_codec = load_checkpoint(h.checkpoint_codec, device)
    encoder.load_state_dict(state_dict_codec["encoder"], strict=True)
    quantizer.load_state_dict(state_dict_codec["quantizer"], strict=True)
    if "input_norm" in state_dict_codec:
        input_norm.load_state_dict(state_dict_codec["input_norm"], strict=True)

    for module in (encoder, quantizer, input_norm):
        for param in module.parameters():
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(h.learning_rate),
        betas=[float(h.adam_b1), float(h.adam_b2)],
    )

    steps_per_epoch = len(train_loader)
    total_steps = int(h.training_epochs) * max(1, steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(h.learning_rate),
        total_steps=total_steps,
        pct_start=float(h.pct_start),
        div_factor=float(h.div_factor),
        final_div_factor=float(h.final_div_factor),
        anneal_strategy="cos",
        last_epoch=-1,
    )

    sw = SummaryWriter(h.logs_dir)

    encoder.eval()
    quantizer.eval()
    input_norm.eval()
    probe.train()

    steps = 0
    criterion = torch.nn.MSELoss()

    for epoch in range(int(h.training_epochs)):
        print(f"Epoch: {epoch + 1}")

        train_loss_sum = 0.0
        train_samples = 0

        for batch in train_loader:
            x = batch["seq"].to(device, non_blocking=True)       # [B, 1, seq_len]
            y = batch["target"].to(device, non_blocking=True)    # [B, 1, pred_len]

            if bool(getattr(h, "use_reversible_norm", True)):
                x_in, mu, std = input_norm(x)
            else:
                x_in, mu, std = x, None, None

            with torch.no_grad():
                latent = encoder(x_in)
                quantized_out = quantizer(latent)
                if h.latent_mode == "discrete":
                    features = quantized_out.z_q
                elif h.latent_mode == "continuous":
                    features = latent
                else:
                    raise ValueError("latent_mode must be 'discrete' or 'continuous'")

            optimizer.zero_grad()
            y_hat_norm = probe(features)
            y_hat = inverse_revin(input_norm, y_hat_norm, mu, std) if bool(getattr(h, "use_reversible_norm", True)) else y_hat_norm
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            bs = x.size(0)
            train_loss_sum += loss.item() * bs
            train_samples += bs
            steps += 1

            if steps % h.summary_interval == 0:
                sw.add_scalar("train/loss", loss.item(), steps)
                sw.add_scalar("train/lr", scheduler.get_last_lr()[0], steps)

        # val
        probe.eval()
        val_mae_sum = 0.0
        val_mse_sum = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
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
                y_hat = inverse_revin(input_norm, y_hat_norm, mu, std) if bool(getattr(h, "use_reversible_norm", True)) else y_hat_norm

                diff = y_hat - y
                bs = x.size(0)
                val_mae_sum += diff.abs().mean(dim=(1, 2)).sum().item()
                val_mse_sum += diff.square().mean(dim=(1, 2)).sum().item()
                val_samples += bs

        val_mae = val_mae_sum / max(1, val_samples)
        val_mse = val_mse_sum / max(1, val_samples)
        val_score = val_mae + val_mse

        sw.add_scalar("val/mae", val_mae, epoch + 1)
        sw.add_scalar("val/mse", val_mse, epoch + 1)
        sw.add_scalar("val/score", val_score, epoch + 1)

        save_checkpoint(
            f"{h.checkpoint_path}/probe_steps={steps:08d}_score={val_score:.6f}",
            {
                "probe": get_state_dict(probe),
            },
        )
        save_checkpoint(
            f"{h.checkpoint_path}/state_steps={steps:08d}_score={val_score:.6f}",
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "steps": steps,
                "epoch": epoch,
            },
        )
        update_topk_and_prune_min(
            h.checkpoint_path,
            int(getattr(h, "keep_topk", 3)),
            val_score,
            steps,
        )
        save_checkpoint(
            f"{h.checkpoint_path}/probe_last",
            {
                "probe": get_state_dict(probe),
            },
        )
        save_checkpoint(
            f"{h.checkpoint_path}/state_last",
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "steps": steps,
                "epoch": epoch,
            },
        )
        probe.train()

    sw.close()
    print(f"{h.dataset_name} training completed, took {time.time() - start:.3f}s")


def main():
    print("Initializing ETT forecasting training process...")
    parser = argparse.ArgumentParser()
    default_config = os.path.join(_PROJECT_ROOT, "configs", "ett-base.yaml")
    parser.add_argument("--config", type=str, default=default_config, help="Path to config yaml")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    h = load_hparams(config_path)

    variant_name = f"{h.dataset_name}-{h.latent_mode}-{h.probe_type}"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    h.run_dir = os.path.join(h.runs_root, h.exp_name, variant_name, timestamp)

    h.logs_dir = os.path.join(h.run_dir, "logs")
    h.checkpoint_path = os.path.join(h.run_dir, "checkpoints")
    h.meta_dir = os.path.join(h.run_dir, "meta")

    os.makedirs(h.logs_dir, exist_ok=True)
    os.makedirs(h.checkpoint_path, exist_ok=True)
    os.makedirs(h.meta_dir, exist_ok=True)

    with open(os.path.join(h.meta_dir, "config_resolved.yaml"), "w", encoding="utf-8") as f:
        import yaml
        yaml.safe_dump(dict(h), f, sort_keys=False, allow_unicode=False)

    build_env(config_path, os.path.basename(config_path), h.meta_dir)
    train(h)


if __name__ == "__main__":
    main()
