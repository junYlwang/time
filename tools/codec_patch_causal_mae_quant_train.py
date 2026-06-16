from __future__ import annotations

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
import itertools
import os
import socket
import sys
import time

import torch
import torch.distributed as dist
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
from modules.loss import MultiScaleLogMagSTFTLoss
from modules.patch_codec import PatchCausalMAEEncoder as Encoder, PatchCausalDecoder as Decoder
from modules.quantizer import build_quantizer
from modules.revin import ReversibleInstanceNorm1D
from modules.utils import (
    _compute_global_codebook_coverage,
    _init_coverage_masks,
    _update_codebook_coverage_masks,
    build_env,
    get_state_dict,
    infer_codec_state_paths,
    load_checkpoint,
    load_hparams,
    masked_diff_l1_loss,
    masked_input_norm,
    masked_smooth_l1_loss,
    reduce_mean,
    save_checkpoint,
    set_seed,
    update_topk_and_prune,
)


def train(rank: int, local_rank: int, world_size: int, h, resume_from_checkpoint: str = "") -> None:
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    set_seed(int(h.seed) + rank)

    encoder = Encoder(h).to(device)
    quantizer = build_quantizer(h).to(device)
    decoder = Decoder(h).to(device)
    input_norm = ReversibleInstanceNorm1D(num_channels=int(h.input_channels), eps=float(h.revin_eps)).to(device)
    stft_loss_fn = MultiScaleLogMagSTFTLoss(win_sizes=h.stft_win_sizes).to(device)
    train_decoder = bool(h.train_quant_stage_decoder)
    optim_param_groups = [
        {"params": quantizer.parameters(), "lr": h.quant_stage_learning_rate},
    ]
    max_lrs = [h.quant_stage_learning_rate]
    if train_decoder:
        optim_param_groups.append(
            {"params": decoder.parameters(), "lr": h.quant_stage_decoder_learning_rate}
        )
        max_lrs.append(h.quant_stage_decoder_learning_rate)
    optim_g = torch.optim.AdamW(
        optim_param_groups,
        betas=[h.adam_b1, h.adam_b2],
    )
    total_steps = int(h.quant_stage_max_training_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim_g,
        max_lr=max_lrs,
        total_steps=total_steps,
        pct_start=h.pct_start,
        div_factor=h.div_factor,
        final_div_factor=h.final_div_factor,
        anneal_strategy="cos",
        last_epoch=-1,
    )

    steps = 0
    if resume_from_checkpoint:
        cp_codec, cp_state = infer_codec_state_paths(resume_from_checkpoint)
        state_dict_codec = load_checkpoint(cp_codec, device)
        state_dict_state = load_checkpoint(cp_state, device)
        encoder.load_state_dict(state_dict_codec["encoder"], strict=True)
        quantizer.load_state_dict(state_dict_codec["quantizer"], strict=True)
        decoder.load_state_dict(state_dict_codec["decoder"], strict=True)
        optim_g.load_state_dict(state_dict_state["optim_g"])
        scheduler.load_state_dict(state_dict_state["scheduler"])
        steps = int(state_dict_state["steps"])
    else:
        state_dict_codec = load_checkpoint(h.stage1_checkpoint_path, device)
        encoder.load_state_dict(state_dict_codec["encoder"], strict=True)
        if h.load_stage1_decoder:
            decoder.load_state_dict(state_dict_codec["decoder"], strict=True)

    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    if not train_decoder:
        for p in decoder.parameters():
            p.requires_grad_(False)
        decoder.eval()

    if world_size > 1 and dist.is_initialized():
        ddp_ids = [local_rank] if device.type == "cuda" else None
        quantizer = DDP(quantizer, device_ids=ddp_ids, find_unused_parameters=h.ddp_find_unused_parameters)
        if train_decoder:
            decoder = DDP(decoder, device_ids=ddp_ids, find_unused_parameters=h.ddp_find_unused_parameters)

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
        sampling_config=getattr(h, "sampling", None),
    )
    train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if world_size > 1 and dist.is_initialized() else None
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
        samples_per_epoch=1,
        max_valid_sequences=h.max_valid_sequences,
        seed=int(h.seed),
        return_valid_length=True,
        min_input_length=int(h.min_input_length),
        sampling_config=getattr(h, "sampling", None),
    )
    eval_sampler = DistributedSampler(evalset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if world_size > 1 and dist.is_initialized() else None
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
    quantizer.train()
    if train_decoder:
        decoder.train()
    else:
        decoder.eval()
    quantizer_for_cov = quantizer.module if hasattr(quantizer, "module") else quantizer
    codebook_sizes = tuple(getattr(quantizer_for_cov, "codebook_sizes", ()))
    coverage_masks = _init_coverage_masks(codebook_sizes, device)
    train_iter = iter(train_loader)

    while steps < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        start = time.time()
        valid_lengths = batch["valid_length"].to(device, non_blocking=True).long()
        x = batch["x"].to(device, non_blocking=True)
        optim_g.zero_grad()

        x_in, _, _, valid_mask = masked_input_norm(input_norm, x, valid_lengths)
        with torch.no_grad():
            latent = encoder(x_in, valid_lengths, None)
        quant_out = quantizer(latent)
        z_q = quant_out.z_q
        codes = quant_out.codes
        q_loss = quant_out.q_loss
        if codes is not None and coverage_masks:
            with torch.no_grad():
                _update_codebook_coverage_masks(codes, coverage_masks)
        x_hat_norm = decoder(z_q, valid_lengths)

        tmin = min(x.size(-1), x_hat_norm.size(-1))
        valid_mask_t = valid_mask[..., :tmin]
        x_ref = x_in[..., :tmin]
        x_rec = x_hat_norm[..., :tmin]

        loss_smooth_l1 = masked_smooth_l1_loss(x_rec, x_ref, valid_mask_t, beta=h.smooth_l1_beta) if h.smooth_l1_weight > 0.0 else torch.zeros((), device=device)
        loss_diff = masked_diff_l1_loss(x_rec, x_ref, valid_mask_t) if h.diff_l1_weight > 0.0 and tmin > 1 else torch.zeros((), device=device)
        loss_stft = stft_loss_fn(x_rec, x_ref) if h.stft_loss_weight > 0.0 else torch.zeros((), device=device)
        loss_reconstruction = h.smooth_l1_weight * loss_smooth_l1 + h.diff_l1_weight * loss_diff + h.stft_loss_weight * loss_stft

        total_loss = h.reconstruction_loss_weight * loss_reconstruction + h.quantizer_loss_weight * q_loss
        total_loss.backward()
        optim_g.step()
        scheduler.step()
        steps += 1

        log_total = reduce_mean(total_loss, world_size).item()
        log_reconstruction = reduce_mean(loss_reconstruction, world_size).item()
        log_smooth_l1 = reduce_mean(loss_smooth_l1, world_size).item()
        log_diff = reduce_mean(loss_diff, world_size).item()
        log_stft = reduce_mean(loss_stft, world_size).item()
        log_q = reduce_mean(q_loss, world_size).item()

        if rank == 0 and steps % int(h.stdout_interval) == 0:
            print(
                f"Steps: {steps} | Total: {log_total:.4f} | "
                f"Rec(S1:{log_smooth_l1:.4f}, D:{log_diff:.4f}, STFT:{log_stft:.4f}, T:{log_reconstruction:.4f}) | "
                f"Q:{log_q:.4f} | s/b: {time.time() - start:.3f}"
            )

        if rank == 0 and sw is not None and steps % int(h.summary_interval) == 0:
            sw.add_scalar("train/total_loss", log_total, steps)
            sw.add_scalar("train/reconstruction_loss", log_reconstruction, steps)
            sw.add_scalar("train/smooth_l1_loss", log_smooth_l1, steps)
            sw.add_scalar("train/diff_l1_loss", log_diff, steps)
            sw.add_scalar("train/stft_loss", log_stft, steps)
            sw.add_scalar("train/quantizer_loss", log_q, steps)
            sw.add_scalar("train/lr", scheduler.get_last_lr()[0], steps)

        if steps % int(h.codebook_coverage_interval) == 0 and coverage_masks:
            coverages = [_compute_global_codebook_coverage(mask, world_size) for mask in coverage_masks]
            if rank == 0:
                if sw is not None:
                    for level_idx, coverage in enumerate(coverages, start=1):
                        sw.add_scalar(f"train/codebook_coverage_l{level_idx}", coverage, steps)
                coverage_msg = " | ".join(f"Codebook Coverage L{level_idx}: {coverage:.4f}" for level_idx, coverage in enumerate(coverages, start=1))
                print(f"Steps: {steps} | {coverage_msg}")
            for mask in coverage_masks:
                mask.zero_()

        if steps % int(h.validation_interval) == 0:
            quantizer.eval()
            decoder.eval()
            eval_mae_sum = torch.zeros((), device=device)
            eval_reconstruction_sum = torch.zeros((), device=device)
            eval_n = torch.zeros((), device=device)
            with torch.no_grad():
                for xb in eval_loader:
                    xb_valid_lengths = xb["valid_length"].to(device, non_blocking=True).long()
                    xb = xb["x"].to(device, non_blocking=True)
                    xb_in, _, _, xb_valid_mask = masked_input_norm(input_norm, xb, xb_valid_lengths)
                    latent_b = encoder(xb_in, xb_valid_lengths, None)
                    zq = quantizer(latent_b).z_q
                    xr_norm = decoder(zq, xb_valid_lengths)
                    tmin = min(xb.size(-1), xr_norm.size(-1))
                    xr_rec = xr_norm[..., :tmin]
                    xb_ref = xb_in[..., :tmin]
                    xb_valid_mask_t = xb_valid_mask[..., :tmin]
                    mae_weight = xb_valid_mask_t.to(dtype=xr_rec.dtype)
                    mae = (torch.abs(xr_rec - xb_ref) * mae_weight).sum() / mae_weight.sum().clamp_min(1.0)
                    smooth_l1 = masked_smooth_l1_loss(xr_rec, xb_ref, xb_valid_mask_t, beta=h.smooth_l1_beta) if h.smooth_l1_weight > 0.0 else torch.zeros((), device=device)
                    diff = masked_diff_l1_loss(xr_rec, xb_ref, xb_valid_mask_t) if h.diff_l1_weight > 0.0 and tmin > 1 else torch.zeros((), device=device)
                    stft = stft_loss_fn(xr_rec, xb_ref) if h.stft_loss_weight > 0.0 else torch.zeros((), device=device)
                    reconstruction = h.smooth_l1_weight * smooth_l1 + h.diff_l1_weight * diff + h.stft_loss_weight * stft
                    eval_mae_sum += mae
                    eval_reconstruction_sum += reconstruction
                    eval_n += 1.0

            if world_size > 1 and dist.is_initialized():
                dist.all_reduce(eval_mae_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_reconstruction_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_n, op=dist.ReduceOp.SUM)

            eval_mae = (eval_mae_sum / eval_n.clamp(min=1.0)).item()
            eval_reconstruction = (eval_reconstruction_sum / eval_n.clamp(min=1.0)).item()
            if rank == 0:
                if sw is not None:
                    sw.add_scalar("valid/mae", eval_mae, steps)
                    sw.add_scalar("valid/reconstruction_loss", eval_reconstruction, steps)
                score = -eval_mae
                save_checkpoint(f"{h.checkpoint_path}/codec_steps={steps:08d}_score={score:.4f}", {"encoder": encoder.state_dict(), "quantizer": get_state_dict(quantizer), "decoder": get_state_dict(decoder) if train_decoder else decoder.state_dict()})
                save_checkpoint(f"{h.checkpoint_path}/state_steps={steps:08d}_score={score:.4f}", {"optim_g": optim_g.state_dict(), "scheduler": scheduler.state_dict(), "steps": steps})
                update_topk_and_prune(h.checkpoint_path, int(h.keep_topk), score, steps)
                save_checkpoint(f"{h.checkpoint_path}/codec_last", {"encoder": encoder.state_dict(), "quantizer": get_state_dict(quantizer), "decoder": get_state_dict(decoder) if train_decoder else decoder.state_dict()})
                save_checkpoint(f"{h.checkpoint_path}/state_last", {"optim_g": optim_g.state_dict(), "scheduler": scheduler.state_dict(), "steps": steps})

            encoder.eval()
            quantizer.train()
            if train_decoder:
                decoder.train()
            else:
                decoder.eval()

    if rank == 0 and sw is not None:
        sw.close()


def main() -> None:
    print("Initializing Quantizer Patch Codec Training Process...")
    parser = argparse.ArgumentParser()
    default_config = os.path.join(_PROJECT_ROOT, "configs", "codec-patch-causal-mae-v4.yaml")
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

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    run_id = f"{timestamp}__seed{h.seed}__gpu{world_size}"
    run_dir = os.path.join(h.runs_root, h.exp_name, run_id)
    if is_distributed:
        obj_list = [run_dir if rank == 0 else None]
        dist.broadcast_object_list(obj_list, src=0)
        run_dir = obj_list[0]

    h.run_dir = run_dir
    h.meta_dir = os.path.join(run_dir, "meta")
    h.logs_dir = os.path.join(run_dir, "logs")
    h.checkpoint_path = os.path.join(run_dir, "checkpoints")
    if rank == 0:
        os.makedirs(h.meta_dir, exist_ok=True)
        os.makedirs(h.logs_dir, exist_ok=True)
        os.makedirs(h.checkpoint_path, exist_ok=True)
        build_env(args.config, "config.yaml", h.meta_dir)
        with open(os.path.join(h.meta_dir, "config_resolved.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(dict(h), f, allow_unicode=True, sort_keys=False)
    if is_distributed:
        dist.barrier()

    train(rank, local_rank, world_size, h, args.resume_from_checkpoint)
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
