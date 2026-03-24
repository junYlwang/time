from __future__ import annotations

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
import itertools
import json
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

from modules.encoder_wo_quantize import Encoder
from modules.decoder import Decoder
from modules.quantizer import build_quantizer
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


def _set_quantizer_mode(quantizer, stochastic: bool, temperature: float) -> None:
    q = quantizer.module if hasattr(quantizer, "module") else quantizer
    if hasattr(q, "set_stochastic_mode"):
        q.set_stochastic_mode(stochastic=stochastic, temperature=temperature)


def _inverse_revin(norm_module, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    mod = norm_module.module if hasattr(norm_module, "module") else norm_module
    return mod.inverse(y, mean, std)


def _update_codebook_coverage_masks(codes: torch.Tensor, mask_l1: torch.Tensor, mask_l2: torch.Tensor) -> None:
    # codes: [B, Q, T], where Q=2 for two residual quantizers.
    if codes.dim() != 3 or codes.size(1) < 2:
        raise ValueError(f"Expected codes with shape [B, Q, T] and Q>=2, got {tuple(codes.shape)}")

    idx_l1 = codes[:, 0, :].detach().reshape(-1).long().clamp_(0, mask_l1.numel() - 1)
    idx_l2 = codes[:, 1, :].detach().reshape(-1).long().clamp_(0, mask_l2.numel() - 1)
    mask_l1[idx_l1] = True
    mask_l2[idx_l2] = True


def _compute_global_codebook_coverage(mask: torch.Tensor, world_size: int) -> float:
    global_mask = mask.float()
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(global_mask, op=dist.ReduceOp.MAX)
    used = (global_mask > 0.5).sum().item()
    return float(used / max(1, global_mask.numel()))


def _load_topk(path: str):
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return None


def _save_topk(path: str, records):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


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
            "codec_path": os.path.join(ckpt_dir, f"codec_steps={steps:08d}_score={score:.4f}"),
            "state_path": os.path.join(ckpt_dir, f"state_steps={steps:08d}_score={score:.4f}"),
        }
    )
    records.sort(key=lambda r: (float(r["score"]), int(r["steps"])), reverse=True)

    while len(records) > keep_k:
        dropped = records.pop(-1)
        for p in (dropped["codec_path"], dropped["state_path"]):
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass

    _save_topk(record_path, records)
    return records


def train(rank: int, local_rank: int, world_size: int, h, resume_from_checkpoint: str = "") -> None:
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    set_seed(int(h.seed) + rank)

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
    use_reversible_norm = bool(getattr(h, "use_reversible_norm", True))

    use_stochastic = bool(getattr(h, "stochastic", False))
    temp_steps = int(getattr(h, "temp_steps", 30000))
    quantizer_loss_weight = float(getattr(h, "quantizer_loss_weight", 1.0))
    coverage_interval = max(1, int(getattr(h, "codebook_coverage_interval", getattr(h, "summary_interval", 1000))))

    optim_g = torch.optim.AdamW(
        itertools.chain(encoder.parameters(), quantizer.parameters(), decoder.parameters(), input_norm.parameters()),
        float(h.learning_rate),
        betas=[float(h.adam_b1), float(h.adam_b2)],
    )

    steps = 0
    state_dict_scheduler = None
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
        quantizer.load_state_dict(state_dict_codec["quantizer"], strict=True)
        decoder.load_state_dict(state_dict_codec["decoder"], strict=True)
        if "input_norm" in state_dict_codec:
            input_norm.load_state_dict(state_dict_codec["input_norm"], strict=True)
        optim_g.load_state_dict(state_dict_state["optim_g"])
        steps = int(state_dict_state["steps"])
        if "scheduler" not in state_dict_state:
            raise KeyError("Missing 'scheduler' in state checkpoint during resume")
        state_dict_scheduler = state_dict_state["scheduler"]

    if world_size > 1 and dist.is_initialized():
        ddp_unused = bool(getattr(h, "ddp_find_unused_parameters", False))
        ddp_ids = [local_rank] if device.type == "cuda" else None
        encoder = DDP(encoder, device_ids=ddp_ids, find_unused_parameters=ddp_unused)
        quantizer = DDP(quantizer, device_ids=ddp_ids, find_unused_parameters=ddp_unused)
        decoder = DDP(decoder, device_ids=ddp_ids, find_unused_parameters=ddp_unused)
        input_norm = DDP(input_norm, device_ids=ddp_ids, find_unused_parameters=ddp_unused)

    trainset = SplitTimeSeriesCodecDataset(
        split_manifest_path=h.split_manifest_path,
        split=getattr(h, "train_split", "train"),
        segment_length=int(h.train_segment_length),
        normalization_method=getattr(h, "normalization_method", None),
        samples_per_epoch=int(h.samples_per_epoch),
        max_valid_sequences=int(getattr(h, "max_valid_sequences", 2000)),
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
        max_valid_sequences=int(getattr(h, "max_valid_sequences", 2000)),
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

    if resume_from_checkpoint:
        scheduler.load_state_dict(state_dict_scheduler)

    sw = SummaryWriter(h.logs_dir) if rank == 0 else None

    encoder.train()
    quantizer.train()
    decoder.train()
    input_norm.train()

    quantizer_for_cov = quantizer.module if hasattr(quantizer, "module") else quantizer
    codebook_sizes = tuple(getattr(quantizer_for_cov, "codebook_sizes", ()))
    if len(codebook_sizes) >= 2:
        used_indices_mask_1 = torch.zeros(int(codebook_sizes[0]), device=device, dtype=torch.bool)
        used_indices_mask_2 = torch.zeros(int(codebook_sizes[1]), device=device, dtype=torch.bool)
    else:
        used_indices_mask_1 = None
        used_indices_mask_2 = None

    steps_per_epoch = max(1, len(train_loader))
    current_epoch = steps // steps_per_epoch
    trainset.set_epoch(current_epoch)
    if train_sampler is not None:
        train_sampler.set_epoch(current_epoch)
    train_iter = iter(train_loader)
    steps_in_epoch = steps % steps_per_epoch
    if steps_in_epoch > 0 and bool(getattr(h, "resume_skip_seen_batches", True)):
        for _ in range(steps_in_epoch):
            try:
                next(train_iter)
            except StopIteration:
                current_epoch += 1
                trainset.set_epoch(current_epoch)
                if train_sampler is not None:
                    train_sampler.set_epoch(current_epoch)
                train_iter = iter(train_loader)
                break

    while steps < total_steps:
        if use_stochastic and bool(getattr(quantizer_for_cov, "is_stochastic_quantizer", False)):
            if steps <= temp_steps:
                current_temp = max(0.3, 1.0 - (steps / max(1, temp_steps)))
            else:
                current_temp = 0.3
            _set_quantizer_mode(quantizer, stochastic=True, temperature=current_temp)

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

        latent = encoder(x_in)
        quant_out = quantizer(latent)
        z_q = quant_out.z_q
        _codes = quant_out.codes
        q_loss = quant_out.q_loss
        if _codes is not None and used_indices_mask_1 is not None and used_indices_mask_2 is not None:
            with torch.no_grad():
                _update_codebook_coverage_masks(_codes, used_indices_mask_1, used_indices_mask_2)
        x_hat_norm = decoder(z_q)
        x_hat = _inverse_revin(input_norm, x_hat_norm, mu, std) if use_reversible_norm else x_hat_norm

        tmin = min(x.size(-1), x_hat.size(-1))
        x_ref = x[..., :tmin]
        x_rec = x_hat[..., :tmin]

        loss_smooth_l1 = F.smooth_l1_loss(
            x_rec,
            x_ref,
            beta=float(getattr(h, "smooth_l1_beta", 1.0)),
        )
        if tmin > 1:
            loss_diff = F.l1_loss(torch.diff(x_rec, dim=-1), torch.diff(x_ref, dim=-1))
        else:
            loss_diff = torch.zeros((), device=device)

        total_loss = (
            float(getattr(h, "recon_smooth_l1_weight", getattr(h, "recon_l1_weight", 1.0))) * loss_smooth_l1
            + float(h.diff_l1_weight) * loss_diff
            + quantizer_loss_weight * q_loss
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            itertools.chain(encoder.parameters(), quantizer.parameters(), decoder.parameters(), input_norm.parameters()),
            max_norm=float(getattr(h, "grad_clip_norm", 1.0)),
        )
        optim_g.step()
        scheduler.step()

        steps += 1

        log_total = reduce_mean(total_loss, world_size).item()
        log_smooth_l1 = reduce_mean(loss_smooth_l1, world_size).item()
        log_diff = reduce_mean(loss_diff, world_size).item()
        log_q = reduce_mean(q_loss, world_size).item()

        if rank == 0 and steps % int(h.stdout_interval) == 0:
            print(
                f"Steps: {steps} | Total: {log_total:.4f} | SmoothL1: {log_smooth_l1:.4f} | "
                f"Diff: {log_diff:.4f} | Q: {log_q:.4f} | s/b: {time.time() - start:.3f}"
            )

        if rank == 0 and sw is not None and steps % int(h.summary_interval) == 0:
            sw.add_scalar("train/total_loss", log_total, steps)
            sw.add_scalar("train/recon_smooth_l1", log_smooth_l1, steps)
            sw.add_scalar("train/diff_l1", log_diff, steps)
            sw.add_scalar("train/quantizer_loss", log_q, steps)
            sw.add_scalar("train/lr", scheduler.get_last_lr()[0], steps)

        if (
            steps % coverage_interval == 0
            and used_indices_mask_1 is not None
            and used_indices_mask_2 is not None
        ):
            coverage_l1 = _compute_global_codebook_coverage(used_indices_mask_1, world_size)
            coverage_l2 = _compute_global_codebook_coverage(used_indices_mask_2, world_size)
            if rank == 0:
                if sw is not None:
                    sw.add_scalar("train/codebook_coverage_l1", coverage_l1, steps)
                    sw.add_scalar("train/codebook_coverage_l2", coverage_l2, steps)
                print(
                    f"Steps: {steps} | Codebook Coverage L1: {coverage_l1:.4f} | "
                    f"Codebook Coverage L2: {coverage_l2:.4f}"
                )
            used_indices_mask_1.zero_()
            used_indices_mask_2.zero_()

        if steps % int(h.validation_interval) == 0:
            encoder.eval()
            quantizer.eval()
            decoder.eval()
            input_norm.eval()

            if use_stochastic and bool(getattr(quantizer_for_cov, "is_stochastic_quantizer", False)):
                _set_quantizer_mode(quantizer, stochastic=False, temperature=0.3)

            eval_mae_sum = torch.zeros((), device=device)
            eval_diff_sum = torch.zeros((), device=device)
            eval_n = torch.zeros((), device=device)

            with torch.no_grad():
                for xb in eval_loader:
                    xb = xb.to(device, non_blocking=True)
                    if use_reversible_norm:
                        xb_in, xb_mu, xb_std = input_norm(xb)
                    else:
                        xb_in, xb_mu, xb_std = xb, None, None
                    latent_b = encoder(xb_in)
                    zq = quantizer(latent_b).z_q
                    xr_norm = decoder(zq)
                    xr = _inverse_revin(input_norm, xr_norm, xb_mu, xb_std) if use_reversible_norm else xr_norm
                    tmin = min(xb.size(-1), xr.size(-1))
                    xb_ref = xb[..., :tmin]
                    xr_rec = xr[..., :tmin]
                    mae = F.l1_loss(xr_rec, xb_ref)
                    if tmin > 1:
                        diff = F.l1_loss(torch.diff(xr_rec, dim=-1), torch.diff(xb_ref, dim=-1))
                    else:
                        diff = torch.zeros((), device=device)
                    eval_mae_sum += mae
                    eval_diff_sum += diff
                    eval_n += 1.0

            if world_size > 1 and dist.is_initialized():
                dist.all_reduce(eval_mae_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_diff_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_n, op=dist.ReduceOp.SUM)

            eval_mae = (eval_mae_sum / eval_n.clamp(min=1.0)).item()
            eval_diff = (eval_diff_sum / eval_n.clamp(min=1.0)).item()

            if rank == 0:
                if sw is not None:
                    sw.add_scalar("valid/mae", eval_mae, steps)
                    sw.add_scalar("valid/diff_l1", eval_diff, steps)

                # Checkpoint selection depends only on MAE.
                score = -eval_mae
                save_checkpoint(
                    f"{h.checkpoint_path}/codec_steps={steps:08d}_score={score:.4f}",
                    {
                        "encoder": get_state_dict(encoder),
                        "quantizer": get_state_dict(quantizer),
                        "decoder": get_state_dict(decoder),
                        "input_norm": get_state_dict(input_norm),
                    },
                )
                save_checkpoint(
                    f"{h.checkpoint_path}/state_steps={steps:08d}_score={score:.4f}",
                    {
                        "optim_g": optim_g.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "steps": steps,
                    },
                )
                update_topk_and_prune(
                    h.checkpoint_path,
                    int(getattr(h, "keep_topk", 3)),
                    score,
                    steps,
                )
                save_checkpoint(
                    f"{h.checkpoint_path}/codec_last",
                    {
                        "encoder": get_state_dict(encoder),
                        "quantizer": get_state_dict(quantizer),
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
            quantizer.train()
            decoder.train()
            input_norm.train()

    if rank == 0 and sw is not None:
        sw.close()


def main() -> None:
    print("Initializing Time-Series Codec Training Process...")
    parser = argparse.ArgumentParser()

    default_config = os.path.join(_PROJECT_ROOT, "configs", "time-codec-1000.yaml")
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
