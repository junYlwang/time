from __future__ import annotations

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
import itertools
import json
import math
import os
import random
import socket
import sys
import time
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
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
from modules.loss import MultiScaleLogMagSTFTLoss
from modules.revin import ReversibleInstanceNorm1D
from modules.predictor import CodePredictor, _ntp_loss_and_accuracy
from modules.quantizer import build_quantizer
from modules.utils import load_hparams, build_env, load_checkpoint, save_checkpoint, get_state_dict, \
    set_seed, update_topk_and_prune, reduce_mean, \
    infer_codec_state_paths, _update_codebook_coverage_masks, \
    _compute_global_codebook_coverage, _init_coverage_masks

from datasets.time_codec_dataset import SplitTimeSeriesCodecDataset


def _valid_mask_from_lengths(valid_lengths: torch.Tensor, time_steps: int, device: torch.device) -> torch.Tensor:
    valid_lengths = valid_lengths.to(device=device, dtype=torch.long).clamp(min=0, max=time_steps)
    positions = torch.arange(time_steps, device=device).view(1, 1, time_steps)
    starts = time_steps - valid_lengths.view(-1, 1, 1)
    return positions >= starts


def _masked_input_norm(input_norm, x: torch.Tensor, valid_lengths: torch.Tensor):
    if x.ndim != 3:
        raise ValueError(f"Expected [B, C, T], got shape={tuple(x.shape)}")
    valid_mask = _valid_mask_from_lengths(valid_lengths, x.size(-1), x.device)
    valid = valid_mask.to(dtype=x.dtype)
    count = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (x * valid).sum(dim=-1, keepdim=True) / count
    var = ((x - mean).square() * valid).sum(dim=-1, keepdim=True) / count
    std = torch.sqrt(var + float(getattr(input_norm, "eps", 1.0e-5)))
    y = (x - mean) / std
    y = torch.where(valid_mask, y, torch.zeros_like(y))
    return y, mean, std

def train(rank: int, local_rank: int, world_size: int, h, resume_from_checkpoint: str = "") -> None:
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    set_seed(int(h.seed) + rank)

    encoder = Encoder(h).to(device)
    quantizer = build_quantizer(h).to(device)
    decoder = Decoder(h).to(device)
    input_norm = ReversibleInstanceNorm1D(num_channels=int(h.input_channels), eps=float(h.revin_eps)).to(device)
    quantizer_for_cfg = quantizer.module if hasattr(quantizer, "module") else quantizer
    codebook_sizes = tuple(getattr(quantizer_for_cfg, "codebook_sizes", ()))
    downsample_factor = math.prod(int(x) for x in h.down_ratio)
    predictor = CodePredictor(h, codebook_sizes).to(device)
    stft_loss_fn = MultiScaleLogMagSTFTLoss(win_sizes=h.stft_win_sizes).to(device)

    optim_g = torch.optim.AdamW(
        itertools.chain(
            encoder.parameters(),
            quantizer.parameters(),
            decoder.parameters(),
            predictor.parameters(),
        ),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2],
    )

    total_steps = int(h.max_training_steps)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim_g,
        max_lr=h.learning_rate,
        total_steps=total_steps,
        pct_start=h.pct_start,
        div_factor=h.div_factor,
        final_div_factor=h.final_div_factor,
        anneal_strategy="cos",
        last_epoch=-1,
    )

    steps = 0
    state_dict_scheduler = None
    cp_codec, cp_state = None, None
    if resume_from_checkpoint:
        cp_codec, cp_state = infer_codec_state_paths(resume_from_checkpoint)
        if not os.path.isfile(cp_codec):
            raise FileNotFoundError(f"codec checkpoint not found: {cp_codec}")
        if not os.path.isfile(cp_state):
            raise FileNotFoundError(f"state checkpoint not found: {cp_state}")

        state_dict_codec = load_checkpoint(cp_codec, device)
        state_dict_state = load_checkpoint(cp_state, device)

        encoder.load_state_dict(state_dict_codec["encoder"], strict=True)
        quantizer.load_state_dict(state_dict_codec["quantizer"], strict=True)
        decoder.load_state_dict(state_dict_codec["decoder"], strict=True)
        predictor.load_state_dict(state_dict_codec["predictor"], strict=True)
        if "input_norm" in state_dict_codec:
            input_norm.load_state_dict(state_dict_codec["input_norm"], strict=True)
        optim_g.load_state_dict(state_dict_state["optim_g"])
        steps = int(state_dict_state["steps"])
        if "scheduler" not in state_dict_state:
            raise KeyError("Missing 'scheduler' in state checkpoint during resume")
        scheduler.load_state_dict(state_dict_state["scheduler"])

    if world_size > 1 and dist.is_initialized():
        ddp_ids = [local_rank] if device.type == "cuda" else None
        encoder = DDP(encoder, device_ids=ddp_ids, find_unused_parameters=h.ddp_find_unused_parameters)
        quantizer = DDP(quantizer, device_ids=ddp_ids, find_unused_parameters=h.ddp_find_unused_parameters)
        decoder = DDP(decoder, device_ids=ddp_ids, find_unused_parameters=h.ddp_find_unused_parameters)
        predictor = DDP(predictor, device_ids=ddp_ids, find_unused_parameters=h.ddp_find_unused_parameters)

    trainset = SplitTimeSeriesCodecDataset(
        split_manifest_path=h.split_manifest_path,
        split="train",
        segment_length=int(h.train_segment_length),
        normalization_method=None,
        samples_per_epoch=int(h.samples_per_epoch),
        max_valid_sequences=int(h.max_valid_sequences),
        seed=int(h.seed),
        return_valid_length=True,
        min_input_length=int(h.min_input_length),
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
        split="valid",
        segment_length=int(h.eval_segment_length),
        normalization_method=None,
        samples_per_epoch=1,  # unused for valid split
        max_valid_sequences=h.max_valid_sequences,
        seed=int(h.seed),
        return_valid_length=True,
        min_input_length=int(h.min_input_length),
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

    sw = SummaryWriter(h.logs_dir) if rank == 0 else None

    encoder.train()
    quantizer.train()
    decoder.train()
    predictor.train()

    quantizer_for_cov = quantizer.module if hasattr(quantizer, "module") else quantizer
    codebook_sizes = tuple(getattr(quantizer_for_cov, "codebook_sizes", ()))
    coverage_masks = _init_coverage_masks(codebook_sizes, device)

    train_iter = iter(train_loader)

    while steps < total_steps:
        try:
            x = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x = next(train_iter)

        start = time.time()

        valid_lengths = x["valid_length"].to(device, non_blocking=True).long()
        x = x["x"].to(device, non_blocking=True)  # [B, 1, T]

        optim_g.zero_grad()

        x_in, mu, std = _masked_input_norm(input_norm, x, valid_lengths)

        latent = encoder(x_in)
        quant_out = quantizer(latent)
        z_q = quant_out.z_q
        _codes = quant_out.codes
        q_loss = quant_out.q_loss
        if _codes is not None and coverage_masks:
            with torch.no_grad():
                _update_codebook_coverage_masks(_codes, coverage_masks)
        pred_loss = torch.zeros((), device=device)
        pred_codec_loss = torch.zeros((), device=device)
        pred_acc = torch.zeros((), device=device)
        pred_acc_layers = [torch.zeros((), device=device) for _ in codebook_sizes]
        if h.prediction_loss_weight > 0.0 or h.prediction_codec_loss_weight > 0.0:
            if h.prediction_loss_weight > 0.0:
                pred_logits = predictor(z_q.detach(), valid_lengths, downsample_factor)
                pred_loss, pred_acc, pred_acc_layers = _ntp_loss_and_accuracy(
                    pred_logits, _codes, valid_lengths, downsample_factor
                )
            if h.prediction_codec_loss_weight > 0.0:
                pred_codec_logits = predictor(z_q, valid_lengths, downsample_factor)
                pred_codec_loss, pred_codec_acc, pred_codec_acc_layers = _ntp_loss_and_accuracy(
                    pred_codec_logits, _codes, valid_lengths, downsample_factor
                )

        x_hat_norm = decoder(z_q)
        tmin = min(x.size(-1), x_hat_norm.size(-1))
        
        usage_loss = torch.zeros((), device=device)
        if h.usage_loss_weight > 0.0:
            usage_probs = quantizer_for_cov.rfsq.saved_usage_probs
            for layer_usage_probs in usage_probs:
                layer_usage_prob = layer_usage_probs.mean(dim=(0, 1, 2))
                usage_loss = usage_loss + (
                    layer_usage_prob * ((layer_usage_prob + 1e-8).log() + math.log(h.codebook_size))
                ).sum()
            usage_loss = usage_loss / h.num_quantizers

        if h.reconstruction_loss_weight > 0.0:
            x_ref = x_in[..., :tmin]
            x_rec = x_hat_norm[..., :tmin]
            if h.smooth_l1_weight > 0.0:
                loss_smooth_l1 = F.smooth_l1_loss(
                    x_rec,
                    x_ref,
                    beta=h.smooth_l1_beta,
                )
            else:
                loss_smooth_l1 = torch.zeros((), device=device)
            if h.diff_l1_weight > 0.0 and tmin > 1:
                loss_diff = F.l1_loss(torch.diff(x_rec, dim=-1), torch.diff(x_ref, dim=-1))
            else:
                loss_diff = torch.zeros((), device=device)
            if h.stft_loss_weight > 0.0:
                loss_stft = stft_loss_fn(x_rec, x_ref)
            else:
                loss_stft = torch.zeros((), device=device)
            loss_reconstruction = (
                h.smooth_l1_weight * loss_smooth_l1
                + h.diff_l1_weight * loss_diff
                + h.stft_loss_weight * loss_stft
            )
        else:
            loss_smooth_l1 = torch.zeros((), device=device)
            loss_diff = torch.zeros((), device=device)
            loss_stft = torch.zeros((), device=device)
            loss_reconstruction = torch.zeros((), device=device)

        total_loss = (
            h.reconstruction_loss_weight * loss_reconstruction
            + h.quantizer_loss_weight * q_loss
            + h.prediction_loss_weight * pred_loss
            + h.prediction_codec_loss_weight * pred_codec_loss
            + h.usage_loss_weight * usage_loss
        )

        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(
        #     itertools.chain(encoder.parameters(), quantizer.parameters(), decoder.parameters()),
        #     max_norm=float(getattr(h, "grad_clip_norm", 1.0)),
        # )
        optim_g.step()
        scheduler.step()

        steps += 1

        log_total = reduce_mean(total_loss, world_size).item()
        log_reconstruction = reduce_mean(loss_reconstruction, world_size).item()
        log_smooth_l1 = reduce_mean(loss_smooth_l1, world_size).item()
        log_diff = reduce_mean(loss_diff, world_size).item()
        log_stft = reduce_mean(loss_stft, world_size).item()
        log_q = reduce_mean(q_loss, world_size).item()
        log_pred = reduce_mean(pred_loss, world_size).item()
        log_pred_codec = reduce_mean(pred_codec_loss, world_size).item()
        log_pred_acc = reduce_mean(pred_acc, world_size).item()
        log_pred_acc_layers = [reduce_mean(acc, world_size).item() for acc in pred_acc_layers]
        log_usage = reduce_mean(usage_loss, world_size).item()

        if rank == 0 and steps % int(h.stdout_interval) == 0:
            print(
                f"Steps: {steps} | Total: {log_total:.4f} | "
                f"Rec(S1:{log_smooth_l1:.4f}, D:{log_diff:.4f}, STFT:{log_stft:.4f}, T:{log_reconstruction:.4f}) | "
                f"Q: {log_q:.4f} | Pred:{log_pred:.4f} | PredCodec:{log_pred_codec:.4f} | PredAcc:{log_pred_acc:.4f} | "
                f"Usage loss: {log_usage:.4f} | "
                f"s/b: {time.time() - start:.3f}"
            )

        if rank == 0 and sw is not None and steps % int(h.summary_interval) == 0:
            sw.add_scalar("train/total_loss", log_total, steps)
            sw.add_scalar("train/reconstruction_loss", log_reconstruction, steps)
            sw.add_scalar("train/smooth_l1_loss", log_smooth_l1, steps)
            sw.add_scalar("train/diff_l1_loss", log_diff, steps)
            sw.add_scalar("train/stft_loss", log_stft, steps)
            sw.add_scalar("train/quantizer_loss", log_q, steps)
            sw.add_scalar("train/prediction_loss", log_pred, steps)
            sw.add_scalar("train/prediction_codec_loss", log_pred_codec, steps)
            sw.add_scalar("train/prediction_acc", log_pred_acc, steps)
            sw.add_scalar("train/usage_loss", log_usage, steps)
            for level_idx, acc in enumerate(log_pred_acc_layers, start=1):
                sw.add_scalar(f"train/prediction_acc_l{level_idx}", acc, steps)
            sw.add_scalar("train/lr", scheduler.get_last_lr()[0], steps)

        if steps % int(h.codebook_coverage_interval) == 0 and coverage_masks:
            coverages = [_compute_global_codebook_coverage(mask, world_size) for mask in coverage_masks]
            if rank == 0:
                if sw is not None:
                    for level_idx, coverage in enumerate(coverages, start=1):
                        sw.add_scalar(f"train/codebook_coverage_l{level_idx}", coverage, steps)
                coverage_msg = " | ".join(
                    f"Codebook Coverage L{level_idx}: {coverage:.4f}"
                    for level_idx, coverage in enumerate(coverages, start=1)
                )
                print(f"Steps: {steps} | {coverage_msg}")
            for mask in coverage_masks:
                mask.zero_()

        if steps % int(h.validation_interval) == 0:
            encoder.eval()
            quantizer.eval()
            decoder.eval()
            predictor.eval()

            eval_mae_sum = torch.zeros((), device=device)
            eval_reconstruction_sum = torch.zeros((), device=device)
            eval_smooth_l1_sum = torch.zeros((), device=device)
            eval_diff_sum = torch.zeros((), device=device)
            eval_stft_sum = torch.zeros((), device=device)
            eval_pred_ntp_loss_sum = torch.zeros((), device=device)
            eval_pred_ntp_acc_sum = torch.zeros((), device=device)
            eval_n = torch.zeros((), device=device)

            with torch.no_grad():
                for xb in eval_loader:
                    xb_valid_lengths = xb["valid_length"].to(device, non_blocking=True).long()
                    xb = xb["x"].to(device, non_blocking=True)
                    xb_in, xb_mu, xb_std = _masked_input_norm(input_norm, xb, xb_valid_lengths)
                    latent_b = encoder(xb_in)
                    quant_out_b = quantizer(latent_b)
                    zq = quant_out_b.z_q
                    codes_b = quant_out_b.codes
                    if h.prediction_loss_weight > 0.0:
                        pred_ntp_logits = predictor(zq, xb_valid_lengths, downsample_factor)
                        pred_ntp_loss, pred_ntp_acc, _ = _ntp_loss_and_accuracy(
                            pred_ntp_logits, codes_b, xb_valid_lengths, downsample_factor
                        )
                    else:
                        pred_ntp_loss = torch.zeros((), device=device)
                        pred_ntp_acc = torch.zeros((), device=device)
                    xr_norm = decoder(zq)
                    tmin = min(xb.size(-1), xr_norm.size(-1))
                    xr_rec = xr_norm[..., :tmin]
                    xb_ref = xb_in[..., :tmin]
                    mae = F.l1_loss(xr_rec, xb_ref)

                    if h.reconstruction_loss_weight > 0.0:
                        if h.smooth_l1_weight > 0.0:
                            smooth_l1 = F.smooth_l1_loss(xr_rec, xb_ref, beta=h.smooth_l1_beta)
                        else:
                            smooth_l1 = torch.zeros((), device=device)
                        if h.diff_l1_weight > 0.0 and tmin > 1:
                            diff = F.l1_loss(torch.diff(xr_rec, dim=-1), torch.diff(xb_ref, dim=-1))
                        else:
                            diff = torch.zeros((), device=device)
                        if h.stft_loss_weight > 0.0:
                            stft = stft_loss_fn(xr_rec, xb_ref)
                        else:
                            stft = torch.zeros((), device=device)
                        reconstruction = (
                            h.smooth_l1_weight * smooth_l1
                            + h.diff_l1_weight * diff
                            + h.stft_loss_weight * stft
                        )
                    else:
                        smooth_l1 = torch.zeros((), device=device)
                        diff = torch.zeros((), device=device)
                        stft = torch.zeros((), device=device)
                        reconstruction = torch.zeros((), device=device)
                    eval_mae_sum += mae
                    eval_reconstruction_sum += reconstruction
                    eval_smooth_l1_sum += smooth_l1
                    eval_diff_sum += diff
                    eval_stft_sum += stft
                    eval_pred_ntp_loss_sum += pred_ntp_loss
                    eval_pred_ntp_acc_sum += pred_ntp_acc
                    eval_n += 1.0

            if world_size > 1 and dist.is_initialized():
                dist.all_reduce(eval_mae_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_reconstruction_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_smooth_l1_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_diff_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_stft_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_pred_ntp_loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_pred_ntp_acc_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_n, op=dist.ReduceOp.SUM)

            eval_mae = (eval_mae_sum / eval_n.clamp(min=1.0)).item()
            eval_reconstruction = (eval_reconstruction_sum / eval_n.clamp(min=1.0)).item()
            eval_smooth_l1 = (eval_smooth_l1_sum / eval_n.clamp(min=1.0)).item()
            eval_diff = (eval_diff_sum / eval_n.clamp(min=1.0)).item()
            eval_stft = (eval_stft_sum / eval_n.clamp(min=1.0)).item()
            eval_pred_ntp_loss = (eval_pred_ntp_loss_sum / eval_n.clamp(min=1.0)).item()
            eval_pred_ntp_acc = (eval_pred_ntp_acc_sum / eval_n.clamp(min=1.0)).item()

            if rank == 0:
                if sw is not None:
                    sw.add_scalar("valid/mae", eval_mae, steps)
                    sw.add_scalar("valid/reconstruction_loss", eval_reconstruction, steps)
                    sw.add_scalar("valid/smooth_l1_loss", eval_smooth_l1, steps)
                    sw.add_scalar("valid/diff_l1_loss", eval_diff, steps)
                    sw.add_scalar("valid/stft_loss", eval_stft, steps)
                    sw.add_scalar("valid/prediction_loss_ntp", eval_pred_ntp_loss, steps)
                    sw.add_scalar("valid/prediction_acc_ntp", eval_pred_ntp_acc, steps)

                # Checkpoint selection depends only on MAE.
                score = -eval_mae
                save_checkpoint(
                    f"{h.checkpoint_path}/codec_steps={steps:08d}_score={score:.4f}",
                    {
                        "encoder": get_state_dict(encoder),
                        "quantizer": get_state_dict(quantizer),
                        "decoder": get_state_dict(decoder),
                        "predictor": get_state_dict(predictor),
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
                        "predictor": get_state_dict(predictor),
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
            predictor.train()

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
