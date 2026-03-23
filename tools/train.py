from __future__ import annotations

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
import itertools
import os
import random
import socket
import sys
import time
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from modules.encoder import Encoder as TimeSeriesEncoder
from modules.decoder import Decoder as TimeSeriesDecoder
from modules.revin import ReversibleInstanceNorm1D
from modules.utils import load_hparams, build_env, load_checkpoint, save_checkpoint, get_state_dict
from datasets.time_codec_dataset import SplitTimeSeriesCodecDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size <= 1 or not dist.is_initialized():
        return value
    out = value.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out / world_size


def _infer_codec_state_paths(resume_path: str) -> Tuple[str, str]:
    d = os.path.dirname(resume_path)
    base = os.path.basename(resume_path)

    if base.startswith("state_"):
        state_path = resume_path
        codec_path = os.path.join(d, "codec_" + base[len("state_"):])
    elif base.startswith("codec_"):
        codec_path = resume_path
        state_path = os.path.join(d, "state_" + base[len("codec_"):])
    else:
        raise ValueError(f"--resume_from_checkpoint must point to codec_* or state_*, got: {resume_path}")

    return codec_path, state_path


def _set_quantizer_mode(encoder, stochastic: bool, temperature: float) -> None:
    enc = encoder.module if hasattr(encoder, "module") else encoder
    if hasattr(enc, "quantizer_1") and enc.quantizer_1 is not None:
        enc.quantizer_1.stochastic = stochastic
        enc.quantizer_1.temperature = temperature
    if hasattr(enc, "quantizer_2") and enc.quantizer_2 is not None:
        enc.quantizer_2.stochastic = stochastic
        enc.quantizer_2.temperature = temperature


def train(rank: int, local_rank: int, world_size: int, h, resume_from_checkpoint: str = "") -> None:
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    set_seed(int(h.seed) + rank)

    encoder = TimeSeriesEncoder(h).to(device)
    decoder = TimeSeriesDecoder(h).to(device)
    input_norm = ReversibleInstanceNorm1D(
        num_channels=int(getattr(h, "input_channels", 1)),
        eps=float(getattr(h, "revin_eps", 1e-5)),
        affine=bool(getattr(h, "revin_affine", True)),
        init_gamma=float(getattr(h, "revin_init_gamma", 1.0)),
        init_beta=float(getattr(h, "revin_init_beta", 0.0)),
        positive_gamma=bool(getattr(h, "revin_positive_gamma", False)),
    ).to(device)
    use_reversible_norm = bool(getattr(h, "use_reversible_norm", True))

    use_stochastic = bool(getattr(h, "stochastic", False))
    temp_steps = int(getattr(h, "temp_steps", 30000))

    optim_g = torch.optim.AdamW(
        itertools.chain(encoder.parameters(), decoder.parameters(), input_norm.parameters()),
        float(h.learning_rate),
        betas=[float(h.adam_b1), float(h.adam_b2)],
    )

    steps = 0
    cp_codec, cp_state = None, None
    if resume_from_checkpoint:
        cp_codec, cp_state = _infer_codec_state_paths(resume_from_checkpoint)
        if not os.path.isfile(cp_codec):
            raise FileNotFoundError(f"codec checkpoint not found: {cp_codec}")
        if not os.path.isfile(cp_state):
            raise FileNotFoundError(f"state checkpoint not found: {cp_state}")

        state_dict_codec = load_checkpoint(cp_codec, device)
        state_dict_state = load_checkpoint(cp_state, device)

        encoder.load_state_dict(state_dict_codec["encoder"], strict=True)
        decoder.load_state_dict(state_dict_codec["decoder"], strict=True)
        if "input_norm" in state_dict_codec:
            input_norm.load_state_dict(state_dict_codec["input_norm"], strict=True)
        optim_g.load_state_dict(state_dict_state["optim_g"])
        steps = int(state_dict_state["steps"])

    if world_size > 1 and dist.is_initialized():
        ddp_unused = bool(getattr(h, "ddp_find_unused_parameters", False))
        ddp_ids = [local_rank] if device.type == "cuda" else None
        encoder = DDP(encoder, device_ids=ddp_ids, find_unused_parameters=ddp_unused)
        decoder = DDP(decoder, device_ids=ddp_ids, find_unused_parameters=ddp_unused)
        input_norm = DDP(input_norm, device_ids=ddp_ids, find_unused_parameters=ddp_unused)

    trainset = SplitTimeSeriesCodecDataset(
        split_manifest_path=h.split_manifest_path,
        split=getattr(h, "train_split", "train"),
        segment_length=int(h.train_segment_length),
        normalization_method=getattr(h, "normalization_method", None),
        samples_per_epoch=int(h.samples_per_epoch),
        max_valid_sequences=int(getattr(h, "max_valid_sequences", 512)),
        seed=int(h.seed),
    )
    if world_size > 1 and dist.is_initialized():
        train_sampler = DistributedSampler(
            trainset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
    else:
        train_sampler = None
    train_loader = DataLoader(
        trainset,
        batch_size=int(h.train_batch_size),
        num_workers=int(h.num_workers),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(int(h.num_workers) > 0),
    )

    evalset = SplitTimeSeriesCodecDataset(
        split_manifest_path=h.split_manifest_path,
        split=getattr(h, "valid_split", "valid"),
        segment_length=int(h.eval_segment_length),
        normalization_method=getattr(h, "normalization_method", None),
        samples_per_epoch=1,  # unused for valid split
        max_valid_sequences=int(getattr(h, "max_valid_sequences", 512)),
        seed=int(h.seed),
    )
    if world_size > 1 and dist.is_initialized():
        eval_sampler = DistributedSampler(
            evalset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        eval_sampler = None
    eval_loader = DataLoader(
        evalset,
        batch_size=int(h.eval_batch_size),
        num_workers=int(h.num_workers),
        shuffle=False,
        sampler=eval_sampler,
        pin_memory=True,
        drop_last=False,
    )

    total_steps = int(h.max_training_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim_g,
        max_lr=float(h.learning_rate),
        total_steps=total_steps,
        pct_start=float(h.pct_start),
        div_factor=float(h.div_factor),
        final_div_factor=float(h.final_div_factor),
        anneal_strategy="cos",
        last_epoch=-1,
    )

    if steps > 0:
        for _ in range(steps):
            scheduler.step()

    sw = SummaryWriter(h.logs_dir) if rank == 0 else None

    encoder.train()
    decoder.train()
    input_norm.train()

    current_epoch = 0
    trainset.set_epoch(current_epoch)
    if train_sampler is not None:
        train_sampler.set_epoch(current_epoch)
    train_iter = iter(train_loader)

    while steps < total_steps:
        if use_stochastic:
            if steps <= temp_steps:
                current_temp = max(0.3, 1.0 - (steps / max(1, temp_steps)))
            else:
                current_temp = 0.3
            _set_quantizer_mode(encoder, stochastic=True, temperature=current_temp)

        try:
            x = next(train_iter)
        except StopIteration:
            current_epoch += 1
            trainset.set_epoch(current_epoch)
            if train_sampler is not None:
                train_sampler.set_epoch(current_epoch)
            train_iter = iter(train_loader)
            x = next(train_iter)

        start = time.time()

        x = x.to(device, non_blocking=True)  # [B, 1, T]

        optim_g.zero_grad()

        if use_reversible_norm:
            x_in, mu, std = input_norm(x)
        else:
            x_in, mu, std = x, None, None

        z_q, _codes = encoder(x_in)
        x_hat_norm = decoder(z_q)
        x_hat = input_norm.inverse(x_hat_norm, mu, std) if use_reversible_norm else x_hat_norm

        tmin = min(x.size(-1), x_hat.size(-1))
        x_ref = x[..., :tmin]
        x_rec = x_hat[..., :tmin]

        loss_l1 = F.l1_loss(x_rec, x_ref)
        loss_mse = F.mse_loss(x_rec, x_ref)
        if tmin > 1:
            loss_diff = F.l1_loss(torch.diff(x_rec, dim=-1), torch.diff(x_ref, dim=-1))
        else:
            loss_diff = torch.zeros((), device=device)

        total_loss = (
            float(h.recon_l1_weight) * loss_l1
            + float(h.recon_mse_weight) * loss_mse
            + float(h.diff_l1_weight) * loss_diff
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            itertools.chain(encoder.parameters(), decoder.parameters(), input_norm.parameters()),
            max_norm=float(getattr(h, "grad_clip_norm", 1.0)),
        )
        optim_g.step()
        scheduler.step()

        steps += 1

        log_total = reduce_mean(total_loss, world_size).item()
        log_l1 = reduce_mean(loss_l1, world_size).item()
        log_mse = reduce_mean(loss_mse, world_size).item()
        log_diff = reduce_mean(loss_diff, world_size).item()

        if rank == 0 and steps % int(h.stdout_interval) == 0:
            print(
                f"Steps: {steps} | Total: {log_total:.6f} | L1: {log_l1:.6f} | "
                f"MSE: {log_mse:.6f} | Diff: {log_diff:.6f} | s/b: {time.time() - start:.3f}"
            )

        if rank == 0 and sw is not None and steps % int(h.summary_interval) == 0:
            sw.add_scalar("train/total_loss", log_total, steps)
            sw.add_scalar("train/recon_l1", log_l1, steps)
            sw.add_scalar("train/recon_mse", log_mse, steps)
            sw.add_scalar("train/diff_l1", log_diff, steps)
            sw.add_scalar("train/lr", scheduler.get_last_lr()[0], steps)

        if steps % int(h.validation_interval) == 0:
            encoder.eval()
            decoder.eval()
            input_norm.eval()

            if use_stochastic:
                _set_quantizer_mode(encoder, stochastic=False, temperature=0.3)

            eval_loss_sum = torch.zeros((), device=device)
            eval_n = torch.zeros((), device=device)

            with torch.no_grad():
                for xb in eval_loader:
                    xb = xb.to(device, non_blocking=True)
                    if use_reversible_norm:
                        xb_in, xb_mu, xb_std = input_norm(xb)
                    else:
                        xb_in, xb_mu, xb_std = xb, None, None
                    zq, _ = encoder(xb_in)
                    xr_norm = decoder(zq)
                    xr = input_norm.inverse(xr_norm, xb_mu, xb_std) if use_reversible_norm else xr_norm
                    tmin = min(xb.size(-1), xr.size(-1))
                    l = F.l1_loss(xr[..., :tmin], xb[..., :tmin])
                    eval_loss_sum += l
                    eval_n += 1.0

            if world_size > 1 and dist.is_initialized():
                dist.all_reduce(eval_loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_n, op=dist.ReduceOp.SUM)

            eval_l1 = (eval_loss_sum / eval_n.clamp(min=1.0)).item()

            if rank == 0:
                if sw is not None:
                    sw.add_scalar("valid/recon_l1", eval_l1, steps)

                score = -eval_l1
                save_checkpoint(
                    f"{h.checkpoint_path}/codec_steps={steps:08d}_score={score:.6f}",
                    {
                        "encoder": get_state_dict(encoder),
                        "decoder": get_state_dict(decoder),
                        "input_norm": get_state_dict(input_norm),
                    },
                )
                save_checkpoint(
                    f"{h.checkpoint_path}/state_steps={steps:08d}_score={score:.6f}",
                    {
                        "optim_g": optim_g.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "steps": steps,
                    },
                )
                save_checkpoint(
                    f"{h.checkpoint_path}/codec_last",
                    {
                        "encoder": get_state_dict(encoder),
                        "decoder": get_state_dict(decoder),
                        "input_norm": get_state_dict(input_norm),
                    },
                )
                save_checkpoint(
                    f"{h.checkpoint_path}/state_last",
                    {
                        "optim_g": optim_g.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "steps": steps,
                    },
                )

            encoder.train()
            decoder.train()
            input_norm.train()

    if rank == 0 and sw is not None:
        sw.close()


def main() -> None:
    print("Initializing Time-Series Codec Training Process...")
    parser = argparse.ArgumentParser()

    default_config = os.path.join(_PROJECT_ROOT, "configs", "time-codec.yaml")
    parser.add_argument("--config", type=str, default=default_config, help="Path to config YAML")
    parser.add_argument("--resume_from_checkpoint", type=str, default="", help="Path to codec_* or state_* checkpoint")
    args = parser.parse_args()

    h = load_hparams(args.config)

    is_distributed = (
        "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ
    )

    if is_distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        print("=" * 80)
        print(f"[Distributed Setup] Hostname: {socket.gethostname()}")
        print(f"[Distributed Setup] RANK={rank}, WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}")
        print(f"[Distributed Setup] MASTER_ADDR={os.environ.get('MASTER_ADDR', 'NOT SET')}")
        print(f"[Distributed Setup] MASTER_PORT={os.environ.get('MASTER_PORT', 'NOT SET')}")
        print("=" * 80)

        dist.init_process_group(backend=h.ddp_backend, init_method="env://", world_size=world_size, rank=rank)
    else:
        rank, world_size, local_rank = 0, 1, 0

    exp_name = getattr(h, "exp_name", "time-codec")
    runs_root = getattr(h, "runs_root", "runs")
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    run_id = getattr(h, "run_id", None) or f"{timestamp}__seed{h.seed}__gpu{world_size}"
    run_dir = os.path.join(runs_root, exp_name, run_id)

    if is_distributed:
        obj_list = [run_dir if rank == 0 else None]
        dist.broadcast_object_list(obj_list, src=0)
        run_dir = obj_list[0]

    meta_dir = os.path.join(run_dir, "meta")
    logs_dir = os.path.join(run_dir, "logs")
    ckpt_dir = os.path.join(run_dir, "checkpoints")

    if rank == 0:
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

    if is_distributed:
        dist.barrier()

    h.run_dir = run_dir
    h.meta_dir = meta_dir
    h.logs_dir = logs_dir
    h.checkpoint_path = ckpt_dir

    train(rank, local_rank, world_size, h, args.resume_from_checkpoint)

    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
