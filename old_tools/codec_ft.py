from __future__ import annotations

import argparse
import itertools
import json
import os
import socket
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from datasets.ett_codec_dataset import DEFAULT_ETT_DATASETS, ETTCodecDataset
from modules.decoder import Decoder
from modules.encoder_wo_quantize import Encoder
from modules.loss import MultiScaleLogMagSTFTLoss
from modules.quantizer import build_quantizer
from modules.utils import (
    _build_input_norm,
    _compute_global_codebook_coverage,
    _init_coverage_masks,
    _inverse_revin,
    _set_quantizer_mode,
    _update_codebook_coverage_masks,
    build_env,
    get_state_dict,
    load_checkpoint,
    load_hparams,
    reduce_mean,
    save_checkpoint,
    set_seed,
    update_topk_and_prune,
)


def _get_finetune_checkpoint(h) -> str:
    finetune_cfg = getattr(h, "finetune", {}) or {}
    if not isinstance(finetune_cfg, dict):
        raise ValueError("finetune must be a mapping in config")
    checkpoint_path = str(
        finetune_cfg.get("checkpoint_path", "") or getattr(h, "checkpoint_codec", "")
    ).strip()
    if not checkpoint_path:
        raise ValueError("Missing finetune.checkpoint_path or checkpoint_codec in config")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Codec checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _dataset_names_from_config(h) -> tuple[str, ...]:
    names = getattr(h, "ett_dataset_names", None)
    if names is None:
        names = getattr(h, "dataset_names", None)
    if names is None:
        return DEFAULT_ETT_DATASETS
    if isinstance(names, str):
        return tuple(x.strip() for x in names.split(",") if x.strip())
    return tuple(str(x) for x in names)


def _build_dataset(h, split: str) -> ETTCodecDataset:
    return ETTCodecDataset(
        ett_root=str(h.ett_root),
        dataset_names=_dataset_names_from_config(h),
        split=split,
        segment_length=int(getattr(h, "segment_length", getattr(h, "train_segment_length", 512))),
        stride=int(getattr(h, "stride", 1)),
    )


def _load_codec_weights(h, device: torch.device, encoder, quantizer, decoder, input_norm) -> str:
    checkpoint_path = _get_finetune_checkpoint(h)
    state = load_checkpoint(checkpoint_path, device)
    encoder.load_state_dict(state["encoder"], strict=True)
    quantizer.load_state_dict(state["quantizer"], strict=True)
    decoder.load_state_dict(state["decoder"], strict=True)
    if "input_norm" in state:
        input_norm.load_state_dict(state["input_norm"], strict=True)
    return checkpoint_path


def _run_validation(
    h,
    device: torch.device,
    world_size: int,
    eval_loader: DataLoader,
    encoder,
    quantizer,
    decoder,
    input_norm,
    stft_loss_fn,
) -> dict[str, float]:
    use_reversible_norm = bool(getattr(h, "use_reversible_norm", True))
    smooth_l1_beta = float(getattr(h, "smooth_l1_beta", 1.0))
    raw_smooth_l1_weight = float(getattr(h, "raw_smooth_l1_weight", 1.0))
    raw_diff_l1_weight = float(getattr(h, "raw_diff_l1_weight", 1.0))
    raw_stft_loss_weight = float(getattr(h, "raw_stft_loss_weight", 1.0))
    norm_smooth_l1_weight = float(getattr(h, "norm_smooth_l1_weight", 1.0))
    norm_diff_l1_weight = float(getattr(h, "norm_diff_l1_weight", 1.0))
    norm_stft_loss_weight = float(getattr(h, "norm_stft_loss_weight", 1.0))

    sums = {
        "mae": torch.zeros((), device=device),
        "numel": torch.zeros((), device=device),
        "raw_total": torch.zeros((), device=device),
        "raw_smooth_l1": torch.zeros((), device=device),
        "raw_diff": torch.zeros((), device=device),
        "raw_stft": torch.zeros((), device=device),
        "norm_total": torch.zeros((), device=device),
        "norm_smooth_l1": torch.zeros((), device=device),
        "norm_diff": torch.zeros((), device=device),
        "norm_stft": torch.zeros((), device=device),
        "batches": torch.zeros((), device=device),
    }

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
            xb_norm_ref = xb_in[..., :tmin]
            xr_norm_rec = xr_norm[..., :tmin]

            raw_smooth_l1 = F.smooth_l1_loss(xr_rec, xb_ref, beta=smooth_l1_beta)
            norm_smooth_l1 = F.smooth_l1_loss(xr_norm_rec, xb_norm_ref, beta=smooth_l1_beta)
            if tmin > 1:
                raw_diff = F.l1_loss(torch.diff(xr_rec, dim=-1), torch.diff(xb_ref, dim=-1))
                norm_diff = F.l1_loss(torch.diff(xr_norm_rec, dim=-1), torch.diff(xb_norm_ref, dim=-1))
            else:
                raw_diff = torch.zeros((), device=device)
                norm_diff = torch.zeros((), device=device)
            raw_stft = stft_loss_fn(xr_rec, xb_ref) if raw_stft_loss_weight > 0.0 else torch.zeros((), device=device)
            norm_stft = stft_loss_fn(xr_norm_rec, xb_norm_ref) if norm_stft_loss_weight > 0.0 else torch.zeros((), device=device)

            raw_total = (
                raw_smooth_l1_weight * raw_smooth_l1
                + raw_diff_l1_weight * raw_diff
                + raw_stft_loss_weight * raw_stft
            )
            norm_total = (
                norm_smooth_l1_weight * norm_smooth_l1
                + norm_diff_l1_weight * norm_diff
                + norm_stft_loss_weight * norm_stft
            )

            sums["mae"] += torch.abs(xr_rec - xb_ref).sum()
            sums["numel"] += torch.tensor(float(xr_rec.numel()), device=device)
            sums["raw_total"] += raw_total
            sums["raw_smooth_l1"] += raw_smooth_l1
            sums["raw_diff"] += raw_diff
            sums["raw_stft"] += raw_stft
            sums["norm_total"] += norm_total
            sums["norm_smooth_l1"] += norm_smooth_l1
            sums["norm_diff"] += norm_diff
            sums["norm_stft"] += norm_stft
            sums["batches"] += 1.0

    if world_size > 1 and dist.is_initialized():
        for value in sums.values():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)

    numel = sums["numel"].clamp(min=1.0)
    batches = sums["batches"].clamp(min=1.0)
    return {
        "mae": (sums["mae"] / numel).item(),
        "raw_total": (sums["raw_total"] / batches).item(),
        "raw_smooth_l1": (sums["raw_smooth_l1"] / batches).item(),
        "raw_diff": (sums["raw_diff"] / batches).item(),
        "raw_stft": (sums["raw_stft"] / batches).item(),
        "norm_total": (sums["norm_total"] / batches).item(),
        "norm_smooth_l1": (sums["norm_smooth_l1"] / batches).item(),
        "norm_diff": (sums["norm_diff"] / batches).item(),
        "norm_stft": (sums["norm_stft"] / batches).item(),
    }


def train(rank: int, local_rank: int, world_size: int, h) -> None:
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    set_seed(int(h.seed) + rank)

    encoder = Encoder(h).to(device)
    quantizer = build_quantizer(h).to(device)
    decoder = Decoder(h).to(device)
    input_norm = _build_input_norm(h, device)
    checkpoint_path = _load_codec_weights(h, device, encoder, quantizer, decoder, input_norm)

    if rank == 0:
        print(f"Loaded finetune source checkpoint: {checkpoint_path}")

    trainset = _build_dataset(h, split=getattr(h, "train_split", "train"))
    evalset = _build_dataset(h, split=getattr(h, "valid_split", "val"))

    if rank == 0:
        os.makedirs(h.meta_dir, exist_ok=True)
        with open(os.path.join(h.meta_dir, "ett_codec_train_dataset.json"), "w", encoding="utf-8") as f:
            json.dump(trainset.summary(), f, indent=2, ensure_ascii=False)
        with open(os.path.join(h.meta_dir, "ett_codec_valid_dataset.json"), "w", encoding="utf-8") as f:
            json.dump(evalset.summary(), f, indent=2, ensure_ascii=False)
        print(f"Train windows: {len(trainset)} | Valid windows: {len(evalset)}")

    train_sampler = (
        DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        if world_size > 1 and dist.is_initialized()
        else None
    )
    eval_dataset = evalset
    if world_size > 1 and dist.is_initialized():
        eval_dataset = Subset(evalset, list(range(rank, len(evalset), world_size)))

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
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(h.eval_batch_size),
        num_workers=int(h.num_workers),
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(int(h.num_workers) > 0),
    )

    if len(train_loader) <= 0:
        raise ValueError("Empty train_loader after batching; lower train_batch_size or check dataset.")

    if world_size > 1 and dist.is_initialized():
        ddp_unused = bool(getattr(h, "ddp_find_unused_parameters", False))
        ddp_ids = [local_rank] if device.type == "cuda" else None
        encoder = DDP(encoder, device_ids=ddp_ids, find_unused_parameters=ddp_unused)
        quantizer = DDP(quantizer, device_ids=ddp_ids, find_unused_parameters=ddp_unused)
        decoder = DDP(decoder, device_ids=ddp_ids, find_unused_parameters=ddp_unused)
        input_norm = DDP(input_norm, device_ids=ddp_ids, find_unused_parameters=ddp_unused)

    optim_g = torch.optim.AdamW(
        itertools.chain(encoder.parameters(), quantizer.parameters(), decoder.parameters(), input_norm.parameters()),
        float(h.learning_rate),
        betas=[float(h.adam_b1), float(h.adam_b2)],
    )

    max_epochs = int(getattr(h, "max_training_epochs", getattr(h, "training_epochs", 1)))
    configured_steps = int(getattr(h, "max_training_steps", 0) or 0)
    total_steps = configured_steps if configured_steps > 0 else max_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim_g,
        max_lr=float(h.learning_rate),
        total_steps=max(1, total_steps),
        pct_start=float(h.pct_start),
        div_factor=float(h.div_factor),
        final_div_factor=float(h.final_div_factor),
        anneal_strategy="cos",
        last_epoch=-1,
    )

    use_reversible_norm = bool(getattr(h, "use_reversible_norm", True))
    use_stochastic = bool(getattr(h, "stochastic", False))
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
    smooth_l1_beta = float(getattr(h, "smooth_l1_beta", 1.0))
    stft_loss_fn = MultiScaleLogMagSTFTLoss(
        win_sizes=list(getattr(h, "stft_win_sizes", [128, 256, 512]))
    ).to(device)
    coverage_interval = max(1, int(getattr(h, "codebook_coverage_interval", getattr(h, "summary_interval", 1000))))
    validation_interval = int(getattr(h, "validation_interval", 0) or 0)

    sw = SummaryWriter(h.logs_dir) if rank == 0 else None

    encoder.train()
    quantizer.train()
    decoder.train()
    input_norm.train()

    quantizer_for_cov = quantizer.module if hasattr(quantizer, "module") else quantizer
    coverage_masks = _init_coverage_masks(tuple(getattr(quantizer_for_cov, "codebook_sizes", ())), device)
    steps = 0

    for epoch in range(max_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for x in train_loader:
            if steps >= total_steps:
                break

            if use_stochastic and bool(getattr(quantizer_for_cov, "is_stochastic_quantizer", False)):
                current_temp = max(0.3, 1.0 - (steps / max(1, temp_steps))) if steps <= temp_steps else 0.3
                _set_quantizer_mode(quantizer, stochastic=True, temperature=current_temp)

            start_time = time.time()
            x = x.to(device, non_blocking=True)
            optim_g.zero_grad()

            if use_reversible_norm:
                x_in, mu, std = input_norm(x)
            else:
                x_in, mu, std = x, None, None

            latent = encoder(x_in)
            quant_out = quantizer(latent)
            z_q = quant_out.z_q
            q_loss = quant_out.q_loss
            if quant_out.codes is not None and coverage_masks:
                with torch.no_grad():
                    _update_codebook_coverage_masks(quant_out.codes, coverage_masks)

            x_hat_norm = decoder(z_q)
            x_hat = _inverse_revin(input_norm, x_hat_norm, mu, std) if use_reversible_norm else x_hat_norm

            tmin = min(x.size(-1), x_hat.size(-1))
            x_ref = x[..., :tmin]
            x_rec = x_hat[..., :tmin]
            x_norm_ref = x_in[..., :tmin]
            x_norm_rec = x_hat_norm[..., :tmin]

            loss_raw_smooth_l1 = F.smooth_l1_loss(x_rec, x_ref, beta=smooth_l1_beta)
            loss_norm_smooth_l1 = F.smooth_l1_loss(x_norm_rec, x_norm_ref, beta=smooth_l1_beta)
            if tmin > 1:
                loss_raw_diff = F.l1_loss(torch.diff(x_rec, dim=-1), torch.diff(x_ref, dim=-1))
                loss_norm_diff = F.l1_loss(torch.diff(x_norm_rec, dim=-1), torch.diff(x_norm_ref, dim=-1))
            else:
                loss_raw_diff = torch.zeros((), device=device)
                loss_norm_diff = torch.zeros((), device=device)
            loss_raw_stft = stft_loss_fn(x_rec, x_ref) if raw_stft_loss_weight > 0.0 else torch.zeros((), device=device)
            loss_norm_stft = stft_loss_fn(x_norm_rec, x_norm_ref) if norm_stft_loss_weight > 0.0 else torch.zeros((), device=device)

            loss_raw_total = (
                raw_smooth_l1_weight * loss_raw_smooth_l1
                + raw_diff_l1_weight * loss_raw_diff
                + raw_stft_loss_weight * loss_raw_stft
            )
            loss_norm_total = (
                norm_smooth_l1_weight * loss_norm_smooth_l1
                + norm_diff_l1_weight * loss_norm_diff
                + norm_stft_loss_weight * loss_norm_stft
            )
            total_loss = (
                raw_domain_loss_weight * loss_raw_total
                + norm_domain_loss_weight * loss_norm_total
                + quantizer_loss_weight * q_loss
            )

            total_loss.backward()
            grad_clip_norm = float(getattr(h, "grad_clip_norm", 0.0))
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(encoder.parameters(), quantizer.parameters(), decoder.parameters(), input_norm.parameters()),
                    max_norm=grad_clip_norm,
                )
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

            if rank == 0 and steps % int(h.stdout_interval) == 0:
                print(
                    f"Epoch: {epoch + 1}/{max_epochs} | Steps: {steps}/{total_steps} | "
                    f"Total: {log_total:.4f} | "
                    f"Raw(S1:{log_raw_smooth_l1:.4f}, D:{log_raw_diff:.4f}, STFT:{log_raw_stft:.4f}, T:{log_raw_total:.4f}) | "
                    f"Norm(S1:{log_norm_smooth_l1:.4f}, D:{log_norm_diff:.4f}, STFT:{log_norm_stft:.4f}, T:{log_norm_total:.4f}) | "
                    f"Q: {log_q:.4f} | s/b: {time.time() - start_time:.3f}"
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
                sw.add_scalar("train/lr", scheduler.get_last_lr()[0], steps)

            if steps % coverage_interval == 0 and coverage_masks:
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

            if validation_interval > 0 and steps % validation_interval == 0:
                _validate_and_save(
                    h, rank, world_size, steps, encoder, quantizer, decoder, input_norm,
                    eval_loader, device, stft_loss_fn, sw, optim_g, scheduler,
                )

        _validate_and_save(
            h, rank, world_size, steps, encoder, quantizer, decoder, input_norm,
            eval_loader, device, stft_loss_fn, sw, optim_g, scheduler,
        )

        if steps >= total_steps:
            break

    if rank == 0 and sw is not None:
        sw.close()


def _validate_and_save(
    h,
    rank: int,
    world_size: int,
    steps: int,
    encoder,
    quantizer,
    decoder,
    input_norm,
    eval_loader: DataLoader,
    device: torch.device,
    stft_loss_fn,
    sw: SummaryWriter | None,
    optim_g: torch.optim.Optimizer,
    scheduler,
) -> None:
    encoder.eval()
    quantizer.eval()
    decoder.eval()
    input_norm.eval()

    _set_quantizer_mode(quantizer, stochastic=False, temperature=0.3)
    metrics = _run_validation(
        h=h,
        device=device,
        world_size=world_size,
        eval_loader=eval_loader,
        encoder=encoder,
        quantizer=quantizer,
        decoder=decoder,
        input_norm=input_norm,
        stft_loss_fn=stft_loss_fn,
    )

    if rank == 0:
        print(
            f"Validation | Steps: {steps} | MAE: {metrics['mae']:.6f} | "
            f"RawT: {metrics['raw_total']:.4f} | NormT: {metrics['norm_total']:.4f}"
        )
        if sw is not None:
            sw.add_scalar("valid/mae", metrics["mae"], steps)
            sw.add_scalar("valid/raw/total_loss", metrics["raw_total"], steps)
            sw.add_scalar("valid/raw/smooth_l1_loss", metrics["raw_smooth_l1"], steps)
            sw.add_scalar("valid/raw/diff_l1_loss", metrics["raw_diff"], steps)
            sw.add_scalar("valid/raw/stft_loss", metrics["raw_stft"], steps)
            sw.add_scalar("valid/norm/total_loss", metrics["norm_total"], steps)
            sw.add_scalar("valid/norm/smooth_l1_loss", metrics["norm_smooth_l1"], steps)
            sw.add_scalar("valid/norm/diff_l1_loss", metrics["norm_diff"], steps)
            sw.add_scalar("valid/norm/stft_loss", metrics["norm_stft"], steps)

        score = -float(metrics["mae"])
        codec_state = {
            "encoder": get_state_dict(encoder),
            "quantizer": get_state_dict(quantizer),
            "decoder": get_state_dict(decoder),
            "input_norm": get_state_dict(input_norm),
        }
        state = {
            "optim_g": optim_g.state_dict(),
            "scheduler": scheduler.state_dict(),
            "steps": int(steps),
            "valid_metrics": metrics,
        }
        save_checkpoint(f"{h.checkpoint_path}/codec_steps={steps:08d}_score={score:.4f}", codec_state)
        save_checkpoint(f"{h.checkpoint_path}/state_steps={steps:08d}_score={score:.4f}", state)
        update_topk_and_prune(h.checkpoint_path, int(getattr(h, "keep_topk", 3)), score, steps)
        save_checkpoint(f"{h.checkpoint_path}/codec_last", codec_state)
        save_checkpoint(f"{h.checkpoint_path}/state_last", state)

    encoder.train()
    quantizer.train()
    decoder.train()
    input_norm.train()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a time-series codec on standardized ETT train windows.")
    parser.add_argument("--config", type=str, required=True, help="Path to ETT codec fine-tuning YAML.")
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

    exp_name = getattr(h, "exp_name", "ett-codec-ft")
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

    train(rank, local_rank, world_size, h)

    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
