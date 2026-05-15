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
from typing import Iterable, Tuple

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

from datasets.time_codec_dataset import SplitTimeSeriesCodecDataset
from modules.decoder import Decoder
from modules.decomposition import TrendResidualDecomposition
from modules.encoder_wo_quantize import Encoder
from modules.loss import MultiScaleLogMagSTFTLoss
from modules.quantizer import build_quantizer
from modules.revin import ReversibleInstanceNorm1D, ReversibleMeanAbsNorm1D
from modules.utils import AttrDict, build_env, get_state_dict, load_checkpoint, load_hparams, save_checkpoint


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


def _build_input_norm(h, device: torch.device):
    norm_type = str(getattr(h, "normalization_type", "zscore")).lower()
    if norm_type == "zscore":
        mod = ReversibleInstanceNorm1D(
            num_channels=int(getattr(h, "input_channels", 1)),
            eps=float(getattr(h, "revin_eps", 1e-5)),
            affine=bool(getattr(h, "revin_affine", True)),
            init_gamma=float(getattr(h, "revin_init_gamma", 1.0)),
            init_beta=float(getattr(h, "revin_init_beta", 0.0)),
            positive_gamma=bool(getattr(h, "revin_positive_gamma", False)),
        )
    elif norm_type == "mean_abs":
        mod = ReversibleMeanAbsNorm1D(
            num_channels=int(getattr(h, "input_channels", 1)),
            eps=float(getattr(h, "revin_eps", 1e-5)),
            affine=bool(getattr(h, "revin_affine", True)),
            init_gamma=float(getattr(h, "revin_init_gamma", 1.0)),
            init_beta=float(getattr(h, "revin_init_beta", 0.0)),
            positive_gamma=bool(getattr(h, "revin_positive_gamma", False)),
        )
    else:
        raise ValueError(f"Unsupported normalization_type: {norm_type}. Expected one of: zscore, mean_abs")
    return mod.to(device)


def _build_decomposition(h, device: torch.device) -> TrendResidualDecomposition:
    cfg = getattr(h, "decomposition", {}) or {}
    if not isinstance(cfg, dict):
        raise ValueError("decomposition must be a mapping in config")
    return TrendResidualDecomposition(
        num_channels=int(cfg.get("num_channels", getattr(h, "input_channels", 1))),
        kernel_sizes=cfg.get("kernel_sizes", [15, 31, 63, 127, 255]),
        weight_mode=str(cfg.get("weight_mode", "dynamic")),
        summary_length=int(cfg.get("summary_length", 32)),
        gating_hidden_dim=int(cfg.get("gating_hidden_dim", 64)),
    ).to(device)


def _branch_value(h, prefix: str, key: str, default=None):
    branch_key = f"{prefix}_{key}"
    if hasattr(h, branch_key):
        return getattr(h, branch_key)
    if hasattr(h, key):
        return getattr(h, key)
    return default


def _build_branch_quantizer(h, prefix: str, device: torch.device):
    cfg = AttrDict(
        {
            "latent_dim": int(getattr(h, "latent_dim", 16)),
            "quantizer_type": str(_branch_value(h, prefix, "quantizer_type", "rfsq")),
            "num_quantizers": int(_branch_value(h, prefix, "num_quantizers", 2)),
            "stochastic": bool(_branch_value(h, prefix, "stochastic", False)),
            "levels_1": _branch_value(h, prefix, "levels_1", [8, 5, 5, 5]),
            "levels_2": _branch_value(h, prefix, "levels_2", [8, 5, 5, 5]),
            "rvq_codebook_size": int(_branch_value(h, prefix, "rvq_codebook_size", 1024)),
            "rvq_codebook_dim": int(_branch_value(h, prefix, "rvq_codebook_dim", getattr(h, "latent_dim", 16))),
            "rvq_decay": float(_branch_value(h, prefix, "rvq_decay", 0.99)),
            "rvq_quantize_dropout": bool(_branch_value(h, prefix, "rvq_quantize_dropout", False)),
            "rvq_quantize_dropout_cutoff_index": int(
                _branch_value(h, prefix, "rvq_quantize_dropout_cutoff_index", 0)
            ),
        }
    )
    return build_quantizer(cfg).to(device)


def _iter_trainable_parameters(modules: Iterable[torch.nn.Module]):
    for module in modules:
        for param in module.parameters():
            if param.requires_grad:
                yield param


def _maybe_ddp(module: torch.nn.Module, device: torch.device, local_rank: int, ddp_unused: bool):
    has_trainable_params = any(param.requires_grad for param in module.parameters())
    if not has_trainable_params:
        return module
    ddp_ids = [local_rank] if device.type == "cuda" else None
    return DDP(module, device_ids=ddp_ids, find_unused_parameters=ddp_unused)


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module


def _update_codebook_coverage_masks(codes: torch.Tensor, masks: list[torch.Tensor]) -> None:
    if codes.dim() != 3:
        raise ValueError(f"Expected codes with shape [B, Q, T], got {tuple(codes.shape)}")
    if codes.size(1) != len(masks):
        raise ValueError(
            f"Mismatch between code layers ({codes.size(1)}) and coverage masks ({len(masks)})"
        )
    for level_idx, mask in enumerate(masks):
        indices = codes[:, level_idx, :].detach().reshape(-1).long().clamp_(0, mask.numel() - 1)
        mask[indices] = True


def _compute_global_codebook_coverage(mask: torch.Tensor, world_size: int) -> float:
    global_mask = mask.float()
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(global_mask, op=dist.ReduceOp.MAX)
    used = (global_mask > 0.5).sum().item()
    return float(used / max(1, global_mask.numel()))


def _init_coverage_masks(quantizer: torch.nn.Module, device: torch.device):
    quantizer_impl = _unwrap(quantizer)
    codebook_sizes = tuple(getattr(quantizer_impl, "codebook_sizes", ()))
    return [
        torch.zeros(int(size), device=device, dtype=torch.bool)
        for size in codebook_sizes
    ]


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


def _compute_domain_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth_l1_beta: float,
    smooth_weight: float,
    diff_weight: float,
    stft_weight: float,
    stft_loss_fn: torch.nn.Module,
):
    tmin = min(pred.size(-1), target.size(-1))
    pred_t = pred[..., :tmin]
    target_t = target[..., :tmin]
    smooth = F.smooth_l1_loss(pred_t, target_t, beta=smooth_l1_beta)
    if tmin > 1:
        diff = F.l1_loss(torch.diff(pred_t, dim=-1), torch.diff(target_t, dim=-1))
    else:
        diff = torch.zeros((), device=pred.device)
    if stft_weight > 0.0:
        stft = stft_loss_fn(pred_t, target_t)
    else:
        stft = torch.zeros((), device=pred.device)
    total = smooth_weight * smooth + diff_weight * diff + stft_weight * stft
    return {
        "total": total,
        "smooth_l1": smooth,
        "diff": diff,
        "stft": stft,
        "pred": pred_t,
        "target": target_t,
    }


def train(rank: int, local_rank: int, world_size: int, h, resume_from_checkpoint: str = "") -> None:
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    set_seed(int(h.seed) + rank)

    encoder_trend = Encoder(h).to(device)
    quantizer_trend = _build_branch_quantizer(h, "trend", device)
    decoder_trend = Decoder(h).to(device)
    encoder_residual = Encoder(h).to(device)
    quantizer_residual = _build_branch_quantizer(h, "residual", device)
    decoder_residual = Decoder(h).to(device)
    input_norm = _build_input_norm(h, device)
    decomposition = _build_decomposition(h, device)
    use_reversible_norm = bool(getattr(h, "use_reversible_norm", True))

    trend_use_stochastic = bool(_branch_value(h, "trend", "stochastic", False))
    residual_use_stochastic = bool(_branch_value(h, "residual", "stochastic", False))
    temp_steps = int(getattr(h, "temp_steps", 30000))
    quantizer_loss_weight = float(getattr(h, "quantizer_loss_weight", 1.0))
    raw_domain_loss_weight = float(getattr(h, "raw_domain_loss_weight", 1.0))
    norm_domain_loss_weight = float(getattr(h, "norm_domain_loss_weight", 1.0))
    raw_smooth_l1_weight = float(getattr(h, "raw_smooth_l1_weight", 1.0))
    raw_diff_l1_weight = float(getattr(h, "raw_diff_l1_weight", 1.0))
    raw_stft_loss_weight = float(getattr(h, "raw_stft_loss_weight", 1.0))
    norm_smooth_l1_weight = float(getattr(h, "norm_smooth_l1_weight", 1.0))
    norm_diff_l1_weight = float(getattr(h, "norm_diff_l1_weight", 1.0))
    norm_stft_loss_weight = float(getattr(h, "norm_stft_loss_weight", 1.0))
    trend_loss_weight = float(getattr(h, "trend_loss_weight", 1.0))
    residual_loss_weight = float(getattr(h, "residual_loss_weight", 1.0))
    smooth_l1_beta = float(getattr(h, "smooth_l1_beta", 1.0))
    stft_win_sizes = list(getattr(h, "stft_win_sizes", [128, 256, 512]))
    stft_loss_fn = MultiScaleLogMagSTFTLoss(win_sizes=stft_win_sizes).to(device)
    coverage_interval = max(1, int(getattr(h, "codebook_coverage_interval", getattr(h, "summary_interval", 1000))))

    trainable_modules = [
        encoder_trend,
        quantizer_trend,
        decoder_trend,
        encoder_residual,
        quantizer_residual,
        decoder_residual,
        input_norm,
        decomposition,
    ]
    optim_g = torch.optim.AdamW(
        _iter_trainable_parameters(trainable_modules),
        float(h.learning_rate),
        betas=[float(h.adam_b1), float(h.adam_b2)],
    )

    steps = 0
    state_dict_scheduler = None
    if resume_from_checkpoint:
        cp_codec, cp_state = _infer_codec_state_paths(resume_from_checkpoint)
        if not os.path.isfile(cp_codec):
            raise FileNotFoundError(f"codec checkpoint not found: {cp_codec}")
        if not os.path.isfile(cp_state):
            raise FileNotFoundError(f"state checkpoint not found: {cp_state}")

        state_dict_codec = load_checkpoint(cp_codec, device)
        state_dict_state = load_checkpoint(cp_state, device)

        encoder_trend.load_state_dict(state_dict_codec["encoder_trend"], strict=True)
        quantizer_trend.load_state_dict(state_dict_codec["quantizer_trend"], strict=True)
        decoder_trend.load_state_dict(state_dict_codec["decoder_trend"], strict=True)
        encoder_residual.load_state_dict(state_dict_codec["encoder_residual"], strict=True)
        quantizer_residual.load_state_dict(state_dict_codec["quantizer_residual"], strict=True)
        decoder_residual.load_state_dict(state_dict_codec["decoder_residual"], strict=True)
        if "input_norm" in state_dict_codec:
            input_norm.load_state_dict(state_dict_codec["input_norm"], strict=True)
        if "decomposition" in state_dict_codec:
            decomposition.load_state_dict(state_dict_codec["decomposition"], strict=True)

        optim_g.load_state_dict(state_dict_state["optim_g"])
        steps = int(state_dict_state["steps"])
        if "scheduler" not in state_dict_state:
            raise KeyError("Missing 'scheduler' in state checkpoint during resume")
        state_dict_scheduler = state_dict_state["scheduler"]

    if world_size > 1 and dist.is_initialized():
        ddp_unused = bool(getattr(h, "ddp_find_unused_parameters", False))
        encoder_trend = _maybe_ddp(encoder_trend, device, local_rank, ddp_unused)
        quantizer_trend = _maybe_ddp(quantizer_trend, device, local_rank, ddp_unused)
        decoder_trend = _maybe_ddp(decoder_trend, device, local_rank, ddp_unused)
        encoder_residual = _maybe_ddp(encoder_residual, device, local_rank, ddp_unused)
        quantizer_residual = _maybe_ddp(quantizer_residual, device, local_rank, ddp_unused)
        decoder_residual = _maybe_ddp(decoder_residual, device, local_rank, ddp_unused)
        input_norm = _maybe_ddp(input_norm, device, local_rank, ddp_unused)
        decomposition = _maybe_ddp(decomposition, device, local_rank, ddp_unused)

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
        samples_per_epoch=1,
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

    for module in (
        encoder_trend,
        quantizer_trend,
        decoder_trend,
        encoder_residual,
        quantizer_residual,
        decoder_residual,
        input_norm,
        decomposition,
    ):
        module.train()

    trend_masks = _init_coverage_masks(quantizer_trend, device)
    residual_masks = _init_coverage_masks(quantizer_residual, device)

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
        if trend_use_stochastic:
            current_temp = max(0.3, 1.0 - (steps / max(1, temp_steps))) if steps <= temp_steps else 0.3
            _set_quantizer_mode(quantizer_trend, stochastic=True, temperature=current_temp)
        if residual_use_stochastic:
            current_temp = max(0.3, 1.0 - (steps / max(1, temp_steps))) if steps <= temp_steps else 0.3
            _set_quantizer_mode(quantizer_residual, stochastic=True, temperature=current_temp)

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
        x = x.to(device, non_blocking=True)
        optim_g.zero_grad()

        if use_reversible_norm:
            x_in, mu, std = input_norm(x)
        else:
            x_in, mu, std = x, None, None

        trend_target_norm, residual_target_norm = decomposition(x_in)

        trend_quant_out = quantizer_trend(encoder_trend(trend_target_norm))
        residual_quant_out = quantizer_residual(encoder_residual(residual_target_norm))

        trend_hat_norm = decoder_trend(trend_quant_out.z_q)
        residual_hat_norm = decoder_residual(residual_quant_out.z_q)
        x_hat_norm = trend_hat_norm + residual_hat_norm
        x_hat = _inverse_revin(input_norm, x_hat_norm, mu, std) if use_reversible_norm else x_hat_norm

        if trend_quant_out.codes is not None and trend_masks:
            with torch.no_grad():
                _update_codebook_coverage_masks(trend_quant_out.codes, trend_masks)
        if residual_quant_out.codes is not None and residual_masks:
            with torch.no_grad():
                _update_codebook_coverage_masks(residual_quant_out.codes, residual_masks)

        overall_raw = _compute_domain_losses(
            x_hat,
            x,
            smooth_l1_beta,
            raw_smooth_l1_weight,
            raw_diff_l1_weight,
            raw_stft_loss_weight,
            stft_loss_fn,
        )
        overall_norm = _compute_domain_losses(
            x_hat_norm,
            x_in,
            smooth_l1_beta,
            norm_smooth_l1_weight,
            norm_diff_l1_weight,
            norm_stft_loss_weight,
            stft_loss_fn,
        )
        trend_norm = _compute_domain_losses(
            trend_hat_norm,
            trend_target_norm,
            smooth_l1_beta,
            norm_smooth_l1_weight,
            norm_diff_l1_weight,
            norm_stft_loss_weight,
            stft_loss_fn,
        )
        residual_norm = _compute_domain_losses(
            residual_hat_norm,
            residual_target_norm,
            smooth_l1_beta,
            norm_smooth_l1_weight,
            norm_diff_l1_weight,
            norm_stft_loss_weight,
            stft_loss_fn,
        )

        overall_total = raw_domain_loss_weight * overall_raw["total"] + norm_domain_loss_weight * overall_norm["total"]
        q_loss_total = trend_quant_out.q_loss + residual_quant_out.q_loss
        total_loss = (
            overall_total
            + trend_loss_weight * trend_norm["total"]
            + residual_loss_weight * residual_norm["total"]
            + quantizer_loss_weight * q_loss_total
        )

        total_loss.backward()
        optim_g.step()
        scheduler.step()
        steps += 1

        log_total = reduce_mean(total_loss, world_size).item()
        log_overall_raw_total = reduce_mean(overall_raw["total"], world_size).item()
        log_overall_raw_s1 = reduce_mean(overall_raw["smooth_l1"], world_size).item()
        log_overall_raw_diff = reduce_mean(overall_raw["diff"], world_size).item()
        log_overall_raw_stft = reduce_mean(overall_raw["stft"], world_size).item()
        log_overall_norm_total = reduce_mean(overall_norm["total"], world_size).item()
        log_overall_norm_s1 = reduce_mean(overall_norm["smooth_l1"], world_size).item()
        log_overall_norm_diff = reduce_mean(overall_norm["diff"], world_size).item()
        log_overall_norm_stft = reduce_mean(overall_norm["stft"], world_size).item()
        log_trend_norm_total = reduce_mean(trend_norm["total"], world_size).item()
        log_trend_norm_s1 = reduce_mean(trend_norm["smooth_l1"], world_size).item()
        log_trend_norm_diff = reduce_mean(trend_norm["diff"], world_size).item()
        log_trend_norm_stft = reduce_mean(trend_norm["stft"], world_size).item()
        log_residual_norm_total = reduce_mean(residual_norm["total"], world_size).item()
        log_residual_norm_s1 = reduce_mean(residual_norm["smooth_l1"], world_size).item()
        log_residual_norm_diff = reduce_mean(residual_norm["diff"], world_size).item()
        log_residual_norm_stft = reduce_mean(residual_norm["stft"], world_size).item()
        log_q_trend = reduce_mean(trend_quant_out.q_loss, world_size).item()
        log_q_residual = reduce_mean(residual_quant_out.q_loss, world_size).item()
        log_q_total = reduce_mean(q_loss_total, world_size).item()

        if rank == 0 and steps % int(h.stdout_interval) == 0:
            print(
                f"Steps: {steps} | Total: {log_total:.4f} | "
                f"OverallRaw(S1:{log_overall_raw_s1:.4f}, D:{log_overall_raw_diff:.4f}, STFT:{log_overall_raw_stft:.4f}, T:{log_overall_raw_total:.4f}) | "
                f"OverallNorm(S1:{log_overall_norm_s1:.4f}, D:{log_overall_norm_diff:.4f}, STFT:{log_overall_norm_stft:.4f}, T:{log_overall_norm_total:.4f}) | "
                f"TrendNorm(T:{log_trend_norm_total:.4f}) | ResidualNorm(T:{log_residual_norm_total:.4f}) | "
                f"Q(T:{log_q_trend:.4f}, R:{log_q_residual:.4f}, Sum:{log_q_total:.4f}) | s/b: {time.time() - start:.3f}"
            )

        if rank == 0 and sw is not None and steps % int(h.summary_interval) == 0:
            sw.add_scalar("train/total_loss", log_total, steps)
            sw.add_scalar("train/overall/raw/total_loss", log_overall_raw_total, steps)
            sw.add_scalar("train/overall/raw/smooth_l1_loss", log_overall_raw_s1, steps)
            sw.add_scalar("train/overall/raw/diff_l1_loss", log_overall_raw_diff, steps)
            sw.add_scalar("train/overall/raw/stft_loss", log_overall_raw_stft, steps)
            sw.add_scalar("train/overall/norm/total_loss", log_overall_norm_total, steps)
            sw.add_scalar("train/overall/norm/smooth_l1_loss", log_overall_norm_s1, steps)
            sw.add_scalar("train/overall/norm/diff_l1_loss", log_overall_norm_diff, steps)
            sw.add_scalar("train/overall/norm/stft_loss", log_overall_norm_stft, steps)
            sw.add_scalar("train/trend/norm/total_loss", log_trend_norm_total, steps)
            sw.add_scalar("train/trend/norm/smooth_l1_loss", log_trend_norm_s1, steps)
            sw.add_scalar("train/trend/norm/diff_l1_loss", log_trend_norm_diff, steps)
            sw.add_scalar("train/trend/norm/stft_loss", log_trend_norm_stft, steps)
            sw.add_scalar("train/residual/norm/total_loss", log_residual_norm_total, steps)
            sw.add_scalar("train/residual/norm/smooth_l1_loss", log_residual_norm_s1, steps)
            sw.add_scalar("train/residual/norm/diff_l1_loss", log_residual_norm_diff, steps)
            sw.add_scalar("train/residual/norm/stft_loss", log_residual_norm_stft, steps)
            sw.add_scalar("train/quantizer/trend_loss", log_q_trend, steps)
            sw.add_scalar("train/quantizer/residual_loss", log_q_residual, steps)
            sw.add_scalar("train/quantizer/total_loss", log_q_total, steps)
            sw.add_scalar("train/lr", scheduler.get_last_lr()[0], steps)

        if steps % coverage_interval == 0:
            if trend_masks:
                trend_coverages = [
                    _compute_global_codebook_coverage(mask, world_size)
                    for mask in trend_masks
                ]
                if rank == 0:
                    if sw is not None:
                        for level_idx, coverage in enumerate(trend_coverages, start=1):
                            sw.add_scalar(f"train/codebook_coverage_trend_l{level_idx}", coverage, steps)
                    coverage_msg = " | ".join(
                        f"L{level_idx}: {coverage:.4f}"
                        for level_idx, coverage in enumerate(trend_coverages, start=1)
                    )
                    print(f"Steps: {steps} | Trend Codebook Coverage {coverage_msg}")
                for mask in trend_masks:
                    mask.zero_()
            if residual_masks:
                residual_coverages = [
                    _compute_global_codebook_coverage(mask, world_size)
                    for mask in residual_masks
                ]
                if rank == 0:
                    if sw is not None:
                        for level_idx, coverage in enumerate(residual_coverages, start=1):
                            sw.add_scalar(f"train/codebook_coverage_residual_l{level_idx}", coverage, steps)
                    coverage_msg = " | ".join(
                        f"L{level_idx}: {coverage:.4f}"
                        for level_idx, coverage in enumerate(residual_coverages, start=1)
                    )
                    print(f"Steps: {steps} | Residual Codebook Coverage {coverage_msg}")
                for mask in residual_masks:
                    mask.zero_()

        if steps % int(h.validation_interval) == 0:
            for module in (
                encoder_trend,
                quantizer_trend,
                decoder_trend,
                encoder_residual,
                quantizer_residual,
                decoder_residual,
                input_norm,
                decomposition,
            ):
                module.eval()

            if trend_use_stochastic:
                _set_quantizer_mode(quantizer_trend, stochastic=False, temperature=0.3)
            if residual_use_stochastic:
                _set_quantizer_mode(quantizer_residual, stochastic=False, temperature=0.3)

            eval_mae_sum = torch.zeros((), device=device)
            eval_overall_raw_total_sum = torch.zeros((), device=device)
            eval_overall_raw_s1_sum = torch.zeros((), device=device)
            eval_overall_raw_diff_sum = torch.zeros((), device=device)
            eval_overall_raw_stft_sum = torch.zeros((), device=device)
            eval_overall_norm_total_sum = torch.zeros((), device=device)
            eval_overall_norm_s1_sum = torch.zeros((), device=device)
            eval_overall_norm_diff_sum = torch.zeros((), device=device)
            eval_overall_norm_stft_sum = torch.zeros((), device=device)
            eval_trend_norm_total_sum = torch.zeros((), device=device)
            eval_trend_norm_s1_sum = torch.zeros((), device=device)
            eval_trend_norm_diff_sum = torch.zeros((), device=device)
            eval_trend_norm_stft_sum = torch.zeros((), device=device)
            eval_residual_norm_total_sum = torch.zeros((), device=device)
            eval_residual_norm_s1_sum = torch.zeros((), device=device)
            eval_residual_norm_diff_sum = torch.zeros((), device=device)
            eval_residual_norm_stft_sum = torch.zeros((), device=device)
            eval_n = torch.zeros((), device=device)

            with torch.no_grad():
                for xb in eval_loader:
                    xb = xb.to(device, non_blocking=True)
                    if use_reversible_norm:
                        xb_in, xb_mu, xb_std = input_norm(xb)
                    else:
                        xb_in, xb_mu, xb_std = xb, None, None

                    trend_target_norm_b, residual_target_norm_b = decomposition(xb_in)
                    trend_hat_norm_b = decoder_trend(quantizer_trend(encoder_trend(trend_target_norm_b)).z_q)
                    residual_hat_norm_b = decoder_residual(quantizer_residual(encoder_residual(residual_target_norm_b)).z_q)
                    x_hat_norm_b = trend_hat_norm_b + residual_hat_norm_b
                    x_hat_b = _inverse_revin(input_norm, x_hat_norm_b, xb_mu, xb_std) if use_reversible_norm else x_hat_norm_b

                    overall_raw_b = _compute_domain_losses(
                        x_hat_b,
                        xb,
                        smooth_l1_beta,
                        raw_smooth_l1_weight,
                        raw_diff_l1_weight,
                        raw_stft_loss_weight,
                        stft_loss_fn,
                    )
                    overall_norm_b = _compute_domain_losses(
                        x_hat_norm_b,
                        xb_in,
                        smooth_l1_beta,
                        norm_smooth_l1_weight,
                        norm_diff_l1_weight,
                        norm_stft_loss_weight,
                        stft_loss_fn,
                    )
                    trend_norm_b = _compute_domain_losses(
                        trend_hat_norm_b,
                        trend_target_norm_b,
                        smooth_l1_beta,
                        norm_smooth_l1_weight,
                        norm_diff_l1_weight,
                        norm_stft_loss_weight,
                        stft_loss_fn,
                    )
                    residual_norm_b = _compute_domain_losses(
                        residual_hat_norm_b,
                        residual_target_norm_b,
                        smooth_l1_beta,
                        norm_smooth_l1_weight,
                        norm_diff_l1_weight,
                        norm_stft_loss_weight,
                        stft_loss_fn,
                    )

                    mae = F.l1_loss(overall_raw_b["pred"], overall_raw_b["target"])
                    eval_mae_sum += mae
                    eval_overall_raw_total_sum += overall_raw_b["total"]
                    eval_overall_raw_s1_sum += overall_raw_b["smooth_l1"]
                    eval_overall_raw_diff_sum += overall_raw_b["diff"]
                    eval_overall_raw_stft_sum += overall_raw_b["stft"]
                    eval_overall_norm_total_sum += overall_norm_b["total"]
                    eval_overall_norm_s1_sum += overall_norm_b["smooth_l1"]
                    eval_overall_norm_diff_sum += overall_norm_b["diff"]
                    eval_overall_norm_stft_sum += overall_norm_b["stft"]
                    eval_trend_norm_total_sum += trend_norm_b["total"]
                    eval_trend_norm_s1_sum += trend_norm_b["smooth_l1"]
                    eval_trend_norm_diff_sum += trend_norm_b["diff"]
                    eval_trend_norm_stft_sum += trend_norm_b["stft"]
                    eval_residual_norm_total_sum += residual_norm_b["total"]
                    eval_residual_norm_s1_sum += residual_norm_b["smooth_l1"]
                    eval_residual_norm_diff_sum += residual_norm_b["diff"]
                    eval_residual_norm_stft_sum += residual_norm_b["stft"]
                    eval_n += 1.0

            if world_size > 1 and dist.is_initialized():
                for tensor in (
                    eval_mae_sum,
                    eval_overall_raw_total_sum,
                    eval_overall_raw_s1_sum,
                    eval_overall_raw_diff_sum,
                    eval_overall_raw_stft_sum,
                    eval_overall_norm_total_sum,
                    eval_overall_norm_s1_sum,
                    eval_overall_norm_diff_sum,
                    eval_overall_norm_stft_sum,
                    eval_trend_norm_total_sum,
                    eval_trend_norm_s1_sum,
                    eval_trend_norm_diff_sum,
                    eval_trend_norm_stft_sum,
                    eval_residual_norm_total_sum,
                    eval_residual_norm_s1_sum,
                    eval_residual_norm_diff_sum,
                    eval_residual_norm_stft_sum,
                    eval_n,
                ):
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

            denom = eval_n.clamp(min=1.0)
            eval_mae = (eval_mae_sum / denom).item()
            eval_overall_raw_total = (eval_overall_raw_total_sum / denom).item()
            eval_overall_raw_s1 = (eval_overall_raw_s1_sum / denom).item()
            eval_overall_raw_diff = (eval_overall_raw_diff_sum / denom).item()
            eval_overall_raw_stft = (eval_overall_raw_stft_sum / denom).item()
            eval_overall_norm_total = (eval_overall_norm_total_sum / denom).item()
            eval_overall_norm_s1 = (eval_overall_norm_s1_sum / denom).item()
            eval_overall_norm_diff = (eval_overall_norm_diff_sum / denom).item()
            eval_overall_norm_stft = (eval_overall_norm_stft_sum / denom).item()
            eval_trend_norm_total = (eval_trend_norm_total_sum / denom).item()
            eval_trend_norm_s1 = (eval_trend_norm_s1_sum / denom).item()
            eval_trend_norm_diff = (eval_trend_norm_diff_sum / denom).item()
            eval_trend_norm_stft = (eval_trend_norm_stft_sum / denom).item()
            eval_residual_norm_total = (eval_residual_norm_total_sum / denom).item()
            eval_residual_norm_s1 = (eval_residual_norm_s1_sum / denom).item()
            eval_residual_norm_diff = (eval_residual_norm_diff_sum / denom).item()
            eval_residual_norm_stft = (eval_residual_norm_stft_sum / denom).item()

            if rank == 0:
                if sw is not None:
                    sw.add_scalar("valid/mae", eval_mae, steps)
                    sw.add_scalar("valid/overall/raw/total_loss", eval_overall_raw_total, steps)
                    sw.add_scalar("valid/overall/raw/smooth_l1_loss", eval_overall_raw_s1, steps)
                    sw.add_scalar("valid/overall/raw/diff_l1_loss", eval_overall_raw_diff, steps)
                    sw.add_scalar("valid/overall/raw/stft_loss", eval_overall_raw_stft, steps)
                    sw.add_scalar("valid/overall/norm/total_loss", eval_overall_norm_total, steps)
                    sw.add_scalar("valid/overall/norm/smooth_l1_loss", eval_overall_norm_s1, steps)
                    sw.add_scalar("valid/overall/norm/diff_l1_loss", eval_overall_norm_diff, steps)
                    sw.add_scalar("valid/overall/norm/stft_loss", eval_overall_norm_stft, steps)
                    sw.add_scalar("valid/trend/norm/total_loss", eval_trend_norm_total, steps)
                    sw.add_scalar("valid/trend/norm/smooth_l1_loss", eval_trend_norm_s1, steps)
                    sw.add_scalar("valid/trend/norm/diff_l1_loss", eval_trend_norm_diff, steps)
                    sw.add_scalar("valid/trend/norm/stft_loss", eval_trend_norm_stft, steps)
                    sw.add_scalar("valid/residual/norm/total_loss", eval_residual_norm_total, steps)
                    sw.add_scalar("valid/residual/norm/smooth_l1_loss", eval_residual_norm_s1, steps)
                    sw.add_scalar("valid/residual/norm/diff_l1_loss", eval_residual_norm_diff, steps)
                    sw.add_scalar("valid/residual/norm/stft_loss", eval_residual_norm_stft, steps)

                score = -eval_mae
                codec_state = {
                    "encoder_trend": get_state_dict(encoder_trend),
                    "quantizer_trend": get_state_dict(quantizer_trend),
                    "decoder_trend": get_state_dict(decoder_trend),
                    "encoder_residual": get_state_dict(encoder_residual),
                    "quantizer_residual": get_state_dict(quantizer_residual),
                    "decoder_residual": get_state_dict(decoder_residual),
                    "input_norm": get_state_dict(input_norm),
                    "decomposition": get_state_dict(decomposition),
                }
                save_checkpoint(f"{h.checkpoint_path}/codec_steps={steps:08d}_score={score:.4f}", codec_state)
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
                save_checkpoint(f"{h.checkpoint_path}/codec_last", codec_state)
                save_checkpoint(
                    f"{h.checkpoint_path}/state_last",
                    {
                        "optim_g": optim_g.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "steps": steps,
                    },
                )

            for module in (
                encoder_trend,
                quantizer_trend,
                decoder_trend,
                encoder_residual,
                quantizer_residual,
                decoder_residual,
                input_norm,
                decomposition,
            ):
                module.train()

    if rank == 0 and sw is not None:
        sw.close()


def main() -> None:
    print("Initializing Dual-Path Time-Series Codec Training Process...")
    parser = argparse.ArgumentParser()
    default_config = os.path.join(_PROJECT_ROOT, "configs", "dual-codec-rfsq-zscore.yaml")
    parser.add_argument("--config", type=str, default=default_config, help="Path to config YAML")
    parser.add_argument("--resume_from_checkpoint", type=str, default="", help="Path to codec_* or state_* checkpoint")
    args = parser.parse_args()

    h = load_hparams(args.config)
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ

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

    exp_name = getattr(h, "exp_name", "dual-codec")
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
