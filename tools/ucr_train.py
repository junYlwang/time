from __future__ import annotations

import sys
import torch
import os
import argparse
import json
import yaml
import itertools
import time
import random
import numpy as np
import math

_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
for _path in (_SRC_DIR, _DATA_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from torch.utils.data import DataLoader
from modules.encoder_wo_quantize import Encoder
from modules.quantizer import build_quantizer
from modules.probe import LinearProbe
from modules.utils import load_hparams, build_env, load_checkpoint, save_checkpoint, get_state_dict, \
    set_seed, _build_input_norm, _load_topk, _save_topk

from datasets.ucr_dataset import UCRDataset
from torch.utils.tensorboard import SummaryWriter

def update_topk_and_prune(ckpt_dir: str, keep_k: int, score: float, steps: int):
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
            "probe_path": os.path.join(ckpt_dir, f"probe_steps={steps:08d}_score={score:.4f}"),
            "state_path": os.path.join(ckpt_dir, f"state_steps={steps:08d}_score={score:.4f}"),
        }
    )
    records.sort(key=lambda r: (float(r["score"]), int(r["steps"])), reverse=True)

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

def train(h):

    start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(h.seed))

    trainset = UCRDataset(h.ucr_root, h.dataset_name, "TRAIN")

    evalset = UCRDataset(h.ucr_root, h.dataset_name, "TEST")

    if trainset.class_names != evalset.class_names:
        raise ValueError(
            f"Train/Test class names mismatch for dataset={h.dataset_name}. "
            f"train={trainset.class_names}, test={evalset.class_names}"
        )

    h.num_classes = trainset.num_classes
    h.latent_seq_len = math.floor(trainset.seq_len / math.prod(h.down_ratio))

    train_loader = DataLoader(
        trainset,
        batch_size=int(len(trainset)),
        num_workers=int(h.num_workers),
        shuffle=True,
        pin_memory=True,
        drop_last=False,
    )

    eval_loader = DataLoader(
        evalset,
        batch_size=int(len(evalset)),
        num_workers=int(h.num_workers),
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    encoder = Encoder(h).to(device)
    quantizer = build_quantizer(h).to(device)
    input_norm = _build_input_norm(h, device)
    linear_probe = LinearProbe(h).to(device)

    use_reversible_norm = bool(getattr(h, "use_reversible_norm", True))
    
    state_dict_codec = load_checkpoint(h.checkpoint_codec, device)
    encoder.load_state_dict(state_dict_codec["encoder"], strict=True)
    quantizer.load_state_dict(state_dict_codec["quantizer"], strict=True)
    if "input_norm" in state_dict_codec:
        input_norm.load_state_dict(state_dict_codec["input_norm"], strict=True)
    
    for param in encoder.parameters():
        param.requires_grad = False
    
    for param in quantizer.parameters():
        param.requires_grad = False

    for param in input_norm.parameters():
        param.requires_grad = False


    optimizer = torch.optim.AdamW(
        itertools.chain(linear_probe.parameters()),
        lr=float(h.learning_rate),
        betas=[float(h.adam_b1), float(h.adam_b2)],
    )

    steps_per_epoch = len(train_loader)
    total_steps = h.training_epochs * steps_per_epoch

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
    linear_probe.train()

    steps = 0
    criterion = torch.nn.CrossEntropyLoss()
    last_acc = None

    for epoch in range(0, h.training_epochs):
        if epoch+1 % 100 == 0:
            print("Epoch: {}".format(epoch+1))

        # train
        total_samples = 0
        total_correct = 0

        for i, batch in enumerate(train_loader):
            start_b = time.time()
            x = batch["seq"].to(device, non_blocking=True)
            gt_label = batch["label"].to(device, non_blocking=True)

            if use_reversible_norm:
                x_in, mu, std = input_norm(x)
            else:
                x_in, mu, std = x, None, None

            with torch.no_grad():
                latent = encoder(x_in)
                quantized_out = quantizer(latent)
                if h.latent_mode == "discrete":
                    zq = quantized_out.z_q
                elif h.latent_mode == "continuous":
                    zq = latent
                else:
                    raise ValueError(
                        "latent mode must be 'discrete' or 'continuous' !"
                    )

            optimizer.zero_grad()

            logits = linear_probe(zq)
            loss = criterion(logits, gt_label)

            pred = logits.argmax(dim=1)

            total_samples += x.size(0)
            total_correct += (pred == gt_label).sum().item()


            loss.backward()
            optimizer.step()
            scheduler.step()

            steps += 1

            if sw is not None and steps % h.summary_interval == 0:
                sw.add_scalar("train/loss", loss.item(), steps)
                sw.add_scalar("train/lr", scheduler.get_last_lr()[0], steps)

        train_acc = total_correct / total_samples
        if sw is not None:
            sw.add_scalar("train/acc", train_acc, epoch+1)

        # validation
        linear_probe.eval()
        total_loss = 0.0
        total_samples = 0
        total_correct = 0

        with torch.no_grad():
            for batch in eval_loader:
                x = batch["seq"].to(device, non_blocking=True)
                gt_label = batch["label"].to(device, non_blocking=True)
                if use_reversible_norm:
                    x_in, mu, std = input_norm(x)
                else:
                    x_in, mu, std = x, None, None
                latent = encoder(x_in)
                quantized_out = quantizer(latent)
                if h.latent_mode == "discrete":
                    zq = quantized_out.z_q
                elif h.latent_mode == "continuous":
                    zq = latent
                else:
                    raise ValueError(
                        "latent mode must be 'discrete' or 'continuous' !"
                    )
                logits = linear_probe(zq)
                loss = criterion(logits, gt_label)

                bs = x.size(0)
                total_samples += bs
                total_loss += loss.item() * bs
                pred = logits.argmax(dim=1)
                total_correct += (pred == gt_label).sum().item()

        val_loss = total_loss / total_samples
        val_acc = total_correct / total_samples
        last_acc = val_acc

        if sw is not None :
            sw.add_scalar("val/loss", val_loss, epoch+1)
            sw.add_scalar("val/acc", val_acc, epoch+1)

        score = val_acc
        save_checkpoint(
            f"{h.checkpoint_path}/probe_steps={steps:08d}_score={score:.4f}",
            {
                "linear_probe": get_state_dict(linear_probe),
            },
        )
        save_checkpoint(
            f"{h.checkpoint_path}/state_steps={steps:08d}_score={score:.4f}",
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "steps": steps,
                "epoch": epoch,
            },
        )
        update_topk_and_prune(
            h.checkpoint_path,
            int(getattr(h, "keep_topk", 3)),
            score,
            steps,
        )
        save_checkpoint(
            f"{h.checkpoint_path}/probe_last",
            {
                "linear_probe": get_state_dict(linear_probe),
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

        linear_probe.train()

    if sw is not None:
        sw.close()

    best_records = _load_topk(os.path.join(h.checkpoint_path, "best_checkpoints.json")) or []
    best_acc = None
    best_step = None
    if best_records:
        best_acc = float(best_records[0]["score"])
        best_step = int(best_records[0]["steps"])

    metrics_dir = os.path.join(h.run_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics = {
        "dataset_name": str(h.dataset_name),
        "latent_mode": str(h.latent_mode),
        "best_acc": best_acc,
        "best_step": best_step,
        "last_acc": None if last_acc is None else float(last_acc),
    }
    with open(os.path.join(metrics_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"{h.dataset_name} training completed, took {time.time() - start:.3f}s")

def main():
    print("Initializing UCR training process...")
    parser = argparse.ArgumentParser()
    default_config = os.path.join(_PROJECT_ROOT, "configs", "base.yaml")
    parser.add_argument("--config", type=str, default=default_config, help="Path to config yaml")
    args = parser.parse_args()
    h = load_hparams(args.config)

    runs_root = h.runs_root
    exp_name = h.exp_name
    timestamp = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
    run_id = getattr(h, "run_id", None) or f"{timestamp}__seed{h.seed}__gpu1"
    run_dir = os.path.join(runs_root, exp_name, run_id)

    meta_dir = os.path.join(run_dir, "meta")
    logs_dir = os.path.join(run_dir, "logs")
    ckpt_dir = os.path.join(run_dir, "checkpoints")

    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    try:
        build_env(args.config, "config.yaml", meta_dir)
    except Exception:
        import shutil
        shutil.copyfile(args.config, os.path.join(meta_dir, "config.yaml"))

    with open(os.path.join(meta_dir, "config_resolved.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(h), f, allow_unicode=True, sort_keys=False)

    h.run_dir = run_dir
    h.meta_dir = meta_dir
    h.logs_dir = logs_dir
    h.checkpoint_path = ckpt_dir

    train(h)

if __name__ == "__main__":
    main()
