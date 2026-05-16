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
from modules.probe import RMSNorm, TransformerBlock
from modules.quantizer import build_quantizer
from modules.utils import load_hparams, build_env, load_checkpoint, save_checkpoint, get_state_dict, \
    set_seed, _set_quantizer_mode, update_topk_and_prune, _build_input_norm, reduce_mean, \
    _infer_codec_state_paths, _inverse_revin, _update_codebook_coverage_masks, \
    _compute_global_codebook_coverage, _init_coverage_masks

from datasets.time_codec_dataset import SplitTimeSeriesCodecDataset


class CodePredictor(nn.Module):
    def __init__(self, h, codebook_sizes: tuple[int, ...]):
        super().__init__()
        latent_dim = int(h.latent_dim)
        d_model = int(h.predictor_d_model)
        nhead = int(h.predictor_nhead)
        num_layers = int(h.predictor_num_layers)
        mlp_ratio = h.predictor_mlp_ratio
        dropout = h.predictor_dropout

        self.mask_token = nn.Parameter(torch.zeros(1, latent_dim, 1))
        self.in_proj = nn.Linear(latent_dim, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.heads = nn.ModuleList([nn.Linear(d_model, int(size)) for size in codebook_sizes])
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, z_q: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
        z_masked = torch.where(mask[:, None, :], self.mask_token.to(dtype=z_q.dtype), z_q)
        x = z_masked.transpose(1, 2)
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return [head(x) for head in self.heads]


def _build_prediction_mask(
    batch_size: int,
    seq_len: int,
    mask_ratio: float,
    random_mask_prob: float,
    device: torch.device,
    mode: str | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    mask_count = max(1, int(round(seq_len * float(mask_ratio))))
    mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)

    if mode == "random":
        use_random = torch.ones(batch_size, device=device, dtype=torch.bool)
    elif mode == "suffix":
        use_random = torch.zeros(batch_size, device=device, dtype=torch.bool)
    else:
        use_random = torch.rand(batch_size, device=device, generator=generator) < float(random_mask_prob)

    suffix_rows = ~use_random
    if suffix_rows.any():
        mask[suffix_rows, seq_len - mask_count:] = True

    for row_idx in torch.nonzero(use_random, as_tuple=False).flatten().tolist():
        indices = torch.randperm(seq_len, device=device, generator=generator)[:mask_count]
        mask[row_idx, indices] = True
    return mask


def _prediction_loss_and_accuracy(
    logits_by_layer: list[torch.Tensor],
    codes: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    losses = []
    accuracies = []
    for layer_idx, logits in enumerate(logits_by_layer):
        targets = codes[:, layer_idx, :].long()
        masked_logits = logits[mask]
        masked_targets = targets[mask]
        losses.append(F.cross_entropy(masked_logits, masked_targets))
        accuracies.append((masked_logits.argmax(dim=-1) == masked_targets).float().mean())
    loss = torch.stack(losses).mean()
    accuracy = torch.stack(accuracies).mean()
    return loss, accuracy, accuracies


def train(rank: int, local_rank: int, world_size: int, h, resume_from_checkpoint: str = "") -> None:
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    set_seed(int(h.seed) + rank)

    encoder = Encoder(h).to(device)
    quantizer = build_quantizer(h).to(device)
    decoder = Decoder(h).to(device)
    input_norm = _build_input_norm(h, device)
    quantizer_for_cfg = quantizer.module if hasattr(quantizer, "module") else quantizer
    codebook_sizes = tuple(getattr(quantizer_for_cfg, "codebook_sizes", ()))
    predictor = CodePredictor(h, codebook_sizes).to(device)
    stft_loss_fn = MultiScaleLogMagSTFTLoss(win_sizes=h.stft_win_sizes).to(device)

    optim_g = torch.optim.AdamW(
        itertools.chain(
            encoder.parameters(),
            quantizer.parameters(),
            decoder.parameters(),
            input_norm.parameters(),
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
        input_norm = DDP(input_norm, device_ids=ddp_ids, find_unused_parameters=h.ddp_find_unused_parameters)
        predictor = DDP(predictor, device_ids=ddp_ids, find_unused_parameters=h.ddp_find_unused_parameters)

    trainset = SplitTimeSeriesCodecDataset(
        split_manifest_path=h.split_manifest_path,
        split="train",
        segment_length=int(h.train_segment_length),
        normalization_method=None,
        samples_per_epoch=int(h.samples_per_epoch),
        max_valid_sequences=int(h.max_valid_sequences),
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
        split="valid",
        segment_length=int(h.eval_segment_length),
        normalization_method=None,
        samples_per_epoch=1,  # unused for valid split
        max_valid_sequences=h.max_valid_sequences,
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

    sw = SummaryWriter(h.logs_dir) if rank == 0 else None

    encoder.train()
    quantizer.train()
    decoder.train()
    input_norm.train()
    predictor.train()

    quantizer_for_cov = quantizer.module if hasattr(quantizer, "module") else quantizer
    codebook_sizes = tuple(getattr(quantizer_for_cov, "codebook_sizes", ()))
    coverage_masks = _init_coverage_masks(codebook_sizes, device)

    steps_per_epoch = max(1, len(train_loader))
    current_epoch = steps // steps_per_epoch
    trainset.set_epoch(current_epoch)
    if train_sampler is not None:
        train_sampler.set_epoch(current_epoch)
    train_iter = iter(train_loader)
    steps_in_epoch = steps % steps_per_epoch
    if steps_in_epoch > 0 and bool(h.resume_skip_seen_batches):
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
        if h.stochastic and bool(getattr(quantizer_for_cov, "is_stochastic_quantizer", False)):
            if steps <= h.temp_steps:
                current_temp = max(0.3, 1.0 - (steps / max(1, h.temp_steps)))
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

        if h.use_reversible_norm:
            x_in, mu, std = input_norm(x)
        else:
            x_in, mu, std = x, None, None

        latent = encoder(x_in)
        quant_out = quantizer(latent)
        z_q = quant_out.z_q
        _codes = quant_out.codes
        q_loss = quant_out.q_loss
        if _codes is not None and coverage_masks:
            with torch.no_grad():
                _update_codebook_coverage_masks(_codes, coverage_masks)
        if h.prediction_loss_weight > 0.0:
            pred_mask = _build_prediction_mask(
                batch_size=z_q.size(0),
                seq_len=z_q.size(-1),
                mask_ratio=h.prediction_mask_ratio,
                random_mask_prob=h.prediction_random_mask_prob,
                device=device,
            )
            pred_logits = predictor(z_q, pred_mask)
            pred_loss, pred_acc, pred_acc_layers = _prediction_loss_and_accuracy(pred_logits, _codes, pred_mask)
        else:
            pred_loss = torch.zeros((), device=device)
            pred_acc = torch.zeros((), device=device)
            pred_acc_layers = [torch.zeros((), device=device) for _ in codebook_sizes]
        x_hat_norm = decoder(z_q)
        tmin = min(x.size(-1), x_hat_norm.size(-1))

        if h.raw_domain_loss_weight > 0.0:
            if h.use_reversible_norm:
                std_for_raw = std.clamp(max=100000.0)
                x_hat = _inverse_revin(input_norm, x_hat_norm, mu, std_for_raw)
            else:
                x_hat = x_hat_norm
            x_ref = x[..., :tmin]
            x_rec = x_hat[..., :tmin]
            if h.raw_smooth_l1_weight > 0.0:
                loss_raw_smooth_l1 = F.smooth_l1_loss(
                    x_rec,
                    x_ref,
                    beta=h.smooth_l1_beta,
                )
            else:
                loss_raw_smooth_l1 = torch.zeros((), device=device)
            if h.raw_diff_l1_weight > 0.0 and tmin > 1:
                loss_raw_diff = F.l1_loss(torch.diff(x_rec, dim=-1), torch.diff(x_ref, dim=-1))
            else:
                loss_raw_diff = torch.zeros((), device=device)
            if h.raw_stft_loss_weight > 0.0:
                loss_raw_stft = stft_loss_fn(x_rec, x_ref)
            else:
                loss_raw_stft = torch.zeros((), device=device)
            loss_raw_total = (
                h.raw_smooth_l1_weight * loss_raw_smooth_l1
                + h.raw_diff_l1_weight * loss_raw_diff
                + h.raw_stft_loss_weight * loss_raw_stft
            )
        else:
            loss_raw_smooth_l1 = torch.zeros((), device=device)
            loss_raw_diff = torch.zeros((), device=device)
            loss_raw_stft = torch.zeros((), device=device)
            loss_raw_total = torch.zeros((), device=device)

        if h.norm_domain_loss_weight > 0.0:
            x_norm_ref = x_in[..., :tmin]
            x_norm_rec = x_hat_norm[..., :tmin]
            if h.norm_smooth_l1_weight > 0.0:
                loss_norm_smooth_l1 = F.smooth_l1_loss(
                    x_norm_rec,
                    x_norm_ref,
                    beta=h.smooth_l1_beta,
                )
            else:
                loss_norm_smooth_l1 = torch.zeros((), device=device)
            if h.norm_diff_l1_weight > 0.0 and tmin > 1:
                loss_norm_diff = F.l1_loss(torch.diff(x_norm_rec, dim=-1), torch.diff(x_norm_ref, dim=-1))
            else:
                loss_norm_diff = torch.zeros((), device=device)
            if h.norm_stft_loss_weight > 0.0:
                loss_norm_stft = stft_loss_fn(x_norm_rec, x_norm_ref)
            else:
                loss_norm_stft = torch.zeros((), device=device)
            loss_norm_total = (
                h.norm_smooth_l1_weight * loss_norm_smooth_l1
                + h.norm_diff_l1_weight * loss_norm_diff
                + h.norm_stft_loss_weight * loss_norm_stft
            )
        else:
            loss_norm_smooth_l1 = torch.zeros((), device=device)
            loss_norm_diff = torch.zeros((), device=device)
            loss_norm_stft = torch.zeros((), device=device)
            loss_norm_total = torch.zeros((), device=device)

        total_loss = (
            h.raw_domain_loss_weight * loss_raw_total
            + h.norm_domain_loss_weight * loss_norm_total
            + h.quantizer_loss_weight * q_loss
            + h.prediction_loss_weight * pred_loss
        )

        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(
        #     itertools.chain(encoder.parameters(), quantizer.parameters(), decoder.parameters(), input_norm.parameters()),
        #     max_norm=float(getattr(h, "grad_clip_norm", 1.0)),
        # )
        optim_g.step()
        scheduler.step()

        steps += 1

        log_total = reduce_mean(total_loss, world_size).item()
        log_raw_total = reduce_mean(loss_raw_total, world_size).item()
        log_raw_smooth_l1 = reduce_mean(loss_raw_smooth_l1, world_size).item()
        log_raw_diff = reduce_mean(loss_raw_diff, world_size).item()
        log_raw_stft = reduce_mean(loss_raw_stft, world_size).item()
        log_norm_total = reduce_mean(loss_norm_total, world_size).item()
        log_norm_smooth_l1 = reduce_mean(loss_norm_smooth_l1, world_size).item()
        log_norm_diff = reduce_mean(loss_norm_diff, world_size).item()
        log_norm_stft = reduce_mean(loss_norm_stft, world_size).item()
        log_q = reduce_mean(q_loss, world_size).item()
        log_pred = reduce_mean(pred_loss, world_size).item()
        log_pred_acc = reduce_mean(pred_acc, world_size).item()
        log_pred_acc_layers = [reduce_mean(acc, world_size).item() for acc in pred_acc_layers]

        if rank == 0 and steps % int(h.stdout_interval) == 0:
            print(
                f"Steps: {steps} | Total: {log_total:.4f} | "
                f"Raw(S1:{log_raw_smooth_l1:.4f}, D:{log_raw_diff:.4f}, STFT:{log_raw_stft:.4f}, T:{log_raw_total:.4f}) | "
                f"Norm(S1:{log_norm_smooth_l1:.4f}, D:{log_norm_diff:.4f}, STFT:{log_norm_stft:.4f}, T:{log_norm_total:.4f}) | "
                f"Q: {log_q:.4f} | Pred:{log_pred:.4f} | PredAcc:{log_pred_acc:.4f} | "
                f"s/b: {time.time() - start:.3f}"
            )

        if rank == 0 and sw is not None and steps % int(h.summary_interval) == 0:
            sw.add_scalar("train/total_loss", log_total, steps)
            sw.add_scalar("train/raw/total_loss", log_raw_total, steps)
            sw.add_scalar("train/raw/smooth_l1_loss", log_raw_smooth_l1, steps)
            sw.add_scalar("train/raw/diff_l1_loss", log_raw_diff, steps)
            sw.add_scalar("train/raw/stft_loss", log_raw_stft, steps)
            sw.add_scalar("train/norm/total_loss", log_norm_total, steps)
            sw.add_scalar("train/norm/smooth_l1_loss", log_norm_smooth_l1, steps)
            sw.add_scalar("train/norm/diff_l1_loss", log_norm_diff, steps)
            sw.add_scalar("train/norm/stft_loss", log_norm_stft, steps)
            sw.add_scalar("train/quantizer_loss", log_q, steps)
            sw.add_scalar("train/prediction_loss", log_pred, steps)
            sw.add_scalar("train/prediction_acc", log_pred_acc, steps)
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
            input_norm.eval()
            predictor.eval()

            if h.stochastic and bool(getattr(quantizer_for_cov, "is_stochastic_quantizer", False)):
                _set_quantizer_mode(quantizer, stochastic=False, temperature=0.3)

            eval_mae_sum = torch.zeros((), device=device)
            eval_raw_total_sum = torch.zeros((), device=device)
            eval_raw_smooth_l1_sum = torch.zeros((), device=device)
            eval_raw_diff_sum = torch.zeros((), device=device)
            eval_raw_stft_sum = torch.zeros((), device=device)
            eval_norm_total_sum = torch.zeros((), device=device)
            eval_norm_smooth_l1_sum = torch.zeros((), device=device)
            eval_norm_diff_sum = torch.zeros((), device=device)
            eval_norm_stft_sum = torch.zeros((), device=device)
            eval_pred_random_loss_sum = torch.zeros((), device=device)
            eval_pred_random_acc_sum = torch.zeros((), device=device)
            eval_pred_suffix_loss_sum = torch.zeros((), device=device)
            eval_pred_suffix_acc_sum = torch.zeros((), device=device)
            eval_n = torch.zeros((), device=device)
            eval_random_generator = torch.Generator(device=device)
            eval_random_generator.manual_seed(int(steps))

            with torch.no_grad():
                for xb in eval_loader:
                    xb = xb.to(device, non_blocking=True)
                    if h.use_reversible_norm:
                        xb_in, xb_mu, xb_std = input_norm(xb)
                    else:
                        xb_in, xb_mu, xb_std = xb, None, None
                    latent_b = encoder(xb_in)
                    quant_out_b = quantizer(latent_b)
                    zq = quant_out_b.z_q
                    codes_b = quant_out_b.codes
                    if h.prediction_loss_weight > 0.0:
                        random_mask = _build_prediction_mask(
                            batch_size=zq.size(0),
                            seq_len=zq.size(-1),
                            mask_ratio=h.prediction_mask_ratio,
                            random_mask_prob=h.prediction_random_mask_prob,
                            device=device,
                            mode="random",
                            generator=eval_random_generator,
                        )
                        suffix_mask = _build_prediction_mask(
                            batch_size=zq.size(0),
                            seq_len=zq.size(-1),
                            mask_ratio=h.prediction_mask_ratio,
                            random_mask_prob=h.prediction_random_mask_prob,
                            device=device,
                            mode="suffix",
                        )
                        random_logits = predictor(zq, random_mask)
                        suffix_logits = predictor(zq, suffix_mask)
                        pred_random_loss, pred_random_acc, _ = _prediction_loss_and_accuracy(
                            random_logits, codes_b, random_mask
                        )
                        pred_suffix_loss, pred_suffix_acc, _ = _prediction_loss_and_accuracy(
                            suffix_logits, codes_b, suffix_mask
                        )
                    else:
                        pred_random_loss = torch.zeros((), device=device)
                        pred_random_acc = torch.zeros((), device=device)
                        pred_suffix_loss = torch.zeros((), device=device)
                        pred_suffix_acc = torch.zeros((), device=device)
                    xr_norm = decoder(zq)
                    tmin = min(xb.size(-1), xr_norm.size(-1))
                    xb_ref = xb[..., :tmin]
                    xr_norm_rec = xr_norm[..., :tmin]
                    xb_norm_ref = xb_in[..., :tmin]
                    mae = F.l1_loss(xr_norm_rec, xb_norm_ref)

                    if h.raw_domain_loss_weight > 0.0:
                        if h.use_reversible_norm:
                            xb_std_for_raw = xb_std.clamp(max=100000.0)
                            xr = _inverse_revin(input_norm, xr_norm, xb_mu, xb_std_for_raw)
                        else:
                            xr = xr_norm
                        xr_rec = xr[..., :tmin]
                        if h.raw_smooth_l1_weight > 0.0:
                            raw_smooth_l1 = F.smooth_l1_loss(xr_rec, xb_ref, beta=h.smooth_l1_beta)
                        else:
                            raw_smooth_l1 = torch.zeros((), device=device)
                        if h.raw_diff_l1_weight > 0.0 and tmin > 1:
                            raw_diff = F.l1_loss(torch.diff(xr_rec, dim=-1), torch.diff(xb_ref, dim=-1))
                        else:
                            raw_diff = torch.zeros((), device=device)
                        if h.raw_stft_loss_weight > 0.0:
                            raw_stft = stft_loss_fn(xr_rec, xb_ref)
                        else:
                            raw_stft = torch.zeros((), device=device)
                        raw_total = (
                            h.raw_smooth_l1_weight * raw_smooth_l1
                            + h.raw_diff_l1_weight * raw_diff
                            + h.raw_stft_loss_weight * raw_stft
                        )
                    else:
                        raw_smooth_l1 = torch.zeros((), device=device)
                        raw_diff = torch.zeros((), device=device)
                        raw_stft = torch.zeros((), device=device)
                        raw_total = torch.zeros((), device=device)

                    if h.norm_domain_loss_weight > 0.0:
                        if h.norm_smooth_l1_weight > 0.0:
                            norm_smooth_l1 = F.smooth_l1_loss(xr_norm_rec, xb_norm_ref, beta=h.smooth_l1_beta)
                        else:
                            norm_smooth_l1 = torch.zeros((), device=device)
                        if h.norm_diff_l1_weight > 0.0 and tmin > 1:
                            norm_diff = F.l1_loss(torch.diff(xr_norm_rec, dim=-1), torch.diff(xb_norm_ref, dim=-1))
                        else:
                            norm_diff = torch.zeros((), device=device)
                        if h.norm_stft_loss_weight > 0.0:
                            norm_stft = stft_loss_fn(xr_norm_rec, xb_norm_ref)
                        else:
                            norm_stft = torch.zeros((), device=device)
                        norm_total = (
                            h.norm_smooth_l1_weight * norm_smooth_l1
                            + h.norm_diff_l1_weight * norm_diff
                            + h.norm_stft_loss_weight * norm_stft
                        )
                    else:
                        norm_smooth_l1 = torch.zeros((), device=device)
                        norm_diff = torch.zeros((), device=device)
                        norm_stft = torch.zeros((), device=device)
                        norm_total = torch.zeros((), device=device)
                    eval_mae_sum += mae
                    eval_raw_total_sum += raw_total
                    eval_raw_smooth_l1_sum += raw_smooth_l1
                    eval_raw_diff_sum += raw_diff
                    eval_raw_stft_sum += raw_stft
                    eval_norm_total_sum += norm_total
                    eval_norm_smooth_l1_sum += norm_smooth_l1
                    eval_norm_diff_sum += norm_diff
                    eval_norm_stft_sum += norm_stft
                    eval_pred_random_loss_sum += pred_random_loss
                    eval_pred_random_acc_sum += pred_random_acc
                    eval_pred_suffix_loss_sum += pred_suffix_loss
                    eval_pred_suffix_acc_sum += pred_suffix_acc
                    eval_n += 1.0

            if world_size > 1 and dist.is_initialized():
                dist.all_reduce(eval_mae_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_raw_total_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_raw_smooth_l1_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_raw_diff_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_raw_stft_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_norm_total_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_norm_smooth_l1_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_norm_diff_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_norm_stft_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_pred_random_loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_pred_random_acc_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_pred_suffix_loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_pred_suffix_acc_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_n, op=dist.ReduceOp.SUM)

            eval_mae = (eval_mae_sum / eval_n.clamp(min=1.0)).item()
            eval_raw_total = (eval_raw_total_sum / eval_n.clamp(min=1.0)).item()
            eval_raw_smooth_l1 = (eval_raw_smooth_l1_sum / eval_n.clamp(min=1.0)).item()
            eval_raw_diff = (eval_raw_diff_sum / eval_n.clamp(min=1.0)).item()
            eval_raw_stft = (eval_raw_stft_sum / eval_n.clamp(min=1.0)).item()
            eval_norm_total = (eval_norm_total_sum / eval_n.clamp(min=1.0)).item()
            eval_norm_smooth_l1 = (eval_norm_smooth_l1_sum / eval_n.clamp(min=1.0)).item()
            eval_norm_diff = (eval_norm_diff_sum / eval_n.clamp(min=1.0)).item()
            eval_norm_stft = (eval_norm_stft_sum / eval_n.clamp(min=1.0)).item()
            eval_pred_random_loss = (eval_pred_random_loss_sum / eval_n.clamp(min=1.0)).item()
            eval_pred_random_acc = (eval_pred_random_acc_sum / eval_n.clamp(min=1.0)).item()
            eval_pred_suffix_loss = (eval_pred_suffix_loss_sum / eval_n.clamp(min=1.0)).item()
            eval_pred_suffix_acc = (eval_pred_suffix_acc_sum / eval_n.clamp(min=1.0)).item()

            if rank == 0:
                if sw is not None:
                    sw.add_scalar("valid/mae", eval_mae, steps)
                    sw.add_scalar("valid/raw/total_loss", eval_raw_total, steps)
                    sw.add_scalar("valid/raw/smooth_l1_loss", eval_raw_smooth_l1, steps)
                    sw.add_scalar("valid/raw/diff_l1_loss", eval_raw_diff, steps)
                    sw.add_scalar("valid/raw/stft_loss", eval_raw_stft, steps)
                    sw.add_scalar("valid/norm/total_loss", eval_norm_total, steps)
                    sw.add_scalar("valid/norm/smooth_l1_loss", eval_norm_smooth_l1, steps)
                    sw.add_scalar("valid/norm/diff_l1_loss", eval_norm_diff, steps)
                    sw.add_scalar("valid/norm/stft_loss", eval_norm_stft, steps)
                    sw.add_scalar("valid/prediction_loss_random", eval_pred_random_loss, steps)
                    sw.add_scalar("valid/prediction_loss_suffix", eval_pred_suffix_loss, steps)
                    sw.add_scalar("valid/prediction_acc_random", eval_pred_random_acc, steps)
                    sw.add_scalar("valid/prediction_acc_suffix", eval_pred_suffix_acc, steps)

                # Checkpoint selection depends only on MAE.
                score = -eval_mae
                save_checkpoint(
                    f"{h.checkpoint_path}/codec_steps={steps:08d}_score={score:.4f}",
                    {
                        "encoder": get_state_dict(encoder),
                        "quantizer": get_state_dict(quantizer),
                        "decoder": get_state_dict(decoder),
                        "input_norm": get_state_dict(input_norm),
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
                        "input_norm": get_state_dict(input_norm),
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
            input_norm.train()
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
