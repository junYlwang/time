#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM

_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from datasets.time_forecast_dataset import SplitTimeSeriesForecastDataset
from modules.patch_codec import PatchCausalMAEEncoder as Encoder, PatchCausalDecoder as Decoder
from modules.quantizer import build_quantizer
from modules.revin import ReversibleInstanceNorm1D
from modules.time_thinker_talker import TimeSeriesMTPTransformer, TimeSeriesProjector, TimeSeriesTalker
from modules.utils import get_state_dict, load_checkpoint, load_hparams, masked_input_norm, reduce_mean, save_checkpoint, set_seed


def setup_distributed(h):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend=str(h.ddp_backend))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
    return rank, local_rank, world_size


def make_run_dir(h, rank: int, world_size: int):
    run_root = os.path.abspath(os.path.join(str(h.runs_root), str(h.exp_name)))
    os.makedirs(run_root, exist_ok=True)
    start_time = time.time()
    if world_size <= 1:
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    else:
        key = os.environ.get("TORCHELASTIC_RUN_ID") or os.environ.get("MASTER_PORT") or "single"
        key = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(key)).strip("_") or "single"
        coord_path = os.path.join(run_root, f".run_id_{key}.json")
        if rank == 0:
            run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
            with open(coord_path + ".tmp", "w", encoding="utf-8") as f:
                json.dump({"run_id": run_id, "created_at": time.time()}, f, ensure_ascii=False)
            os.replace(coord_path + ".tmp", coord_path)
        else:
            payload = None
            while time.time() - start_time < float(h.run_dir_wait_seconds):
                if os.path.isfile(coord_path):
                    with open(coord_path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    break
                time.sleep(0.2)
            if payload is None:
                raise TimeoutError(f"Timed out waiting for rank0 run id: {coord_path}")
            run_id = str(payload["run_id"])
    h.run_root = run_root
    h.run_id = run_id
    h.output_dir = os.path.join(run_root, run_id)
    h.logs_dir = os.path.join(h.output_dir, "logs")
    h.checkpoint_dir = os.path.join(h.output_dir, "checkpoints")
    if rank == 0:
        os.makedirs(h.checkpoint_dir, exist_ok=True)
        with open(os.path.join(h.output_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(dict(h), f, allow_unicode=True, sort_keys=False)
    if dist.is_initialized():
        dist.barrier()


def normalize_with_history(history: torch.Tensor, future: torch.Tensor, input_norm, history_lengths: torch.Tensor, future_lengths: torch.Tensor):
    history_norm, mean, std, history_mask = masked_input_norm(input_norm, history, history_lengths)
    pos = torch.arange(future.size(-1), device=future.device).view(1, 1, -1)
    future_mask = pos < future_lengths.to(future.device).view(-1, 1, 1)
    future_norm = torch.where(future_mask, (future - mean) / std, torch.zeros_like(future))
    return history_norm, future_norm, history_mask.squeeze(1), future_mask.squeeze(1)


def build_target_codes(encoder, quantizer, future_norm: torch.Tensor):
    full_lengths = torch.full((future_norm.size(0),), future_norm.size(-1), dtype=torch.long, device=future_norm.device)
    latent = encoder(future_norm, full_lengths, None)
    return quantizer(latent).codes.long()


def patch_mask_from_lengths(lengths: torch.Tensor, total_points: int, patch_size: int) -> torch.Tensor:
    num_tokens = int(total_points) // int(patch_size)
    token_lengths = torch.div(lengths.to(dtype=torch.long) + int(patch_size) - 1, int(patch_size), rounding_mode="floor").clamp(max=num_tokens)
    positions = torch.arange(num_tokens, device=lengths.device).view(1, num_tokens)
    starts = num_tokens - token_lengths.view(-1, 1)
    return positions >= starts


def masked_ce(logits: torch.Tensor, target: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    b, n, c = logits.shape
    loss = F.cross_entropy(logits.reshape(b * n, c), target.reshape(b * n), reduction="none").view(b, n)
    weight = token_mask.to(dtype=loss.dtype)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def compute_loss(layer0_logits, mtp_logits, target_codes, token_mask, h):
    loss0 = masked_ce(layer0_logits, target_codes[:, 0, :], token_mask)
    mtp_losses = []
    for idx, logits in enumerate(mtp_logits, start=1):
        mtp_losses.append(masked_ce(logits, target_codes[:, idx, :], token_mask))
    loss_mtp = torch.stack(mtp_losses).mean() if mtp_losses else layer0_logits.new_zeros(())
    return loss0 + float(h.mtp_loss_weight) * loss_mtp, loss0, loss_mtp


def save_stage_checkpoint(path, projector, talker, mtp, optim, scheduler, steps: int, metrics: dict):
    save_checkpoint(path, {
        "projector": get_state_dict(projector),
        "talker": get_state_dict(talker),
        "mtp": get_state_dict(mtp),
        "optim": optim.state_dict(),
        "scheduler": scheduler.state_dict(),
        "steps": int(steps),
        "metrics": metrics,
    })


def make_dataset(h, split: str):
    return SplitTimeSeriesForecastDataset(
        split_manifest_path=str(h.split_manifest_path),
        split=split,
        history_length=int(h.history_length),
        max_prediction_length=int(h.max_prediction_length),
        prediction_length_choices=list(h.prediction_length_choices),
        patch_size=int(h.patch_size),
        normalization_method=None,
        samples_per_epoch=int(h.samples_per_epoch),
        max_valid_sequences=int(h.max_valid_sequences),
        seed=int(h.seed),
        sampling_config=getattr(h, "sampling", None),
    )


def forward_batch(batch, encoder, quantizer, projector, thinker, talker, mtp, input_norm, h, device):
    history = batch["history"].to(device, non_blocking=True)
    future = batch["future"].to(device, non_blocking=True)
    history_lengths = batch["history_length"].to(device, non_blocking=True).long()
    future_lengths = batch["future_length"].to(device, non_blocking=True).long()
    future_token_lengths = batch["future_token_length"].to(device, non_blocking=True).long()
    max_future_patches = int(h.max_prediction_length) // int(h.patch_size)
    token_mask = torch.arange(max_future_patches, device=device).view(1, -1) < future_token_lengths.view(-1, 1)

    history_norm, future_norm, _, _ = normalize_with_history(history, future, input_norm, history_lengths, future_lengths)
    thinker_mask = patch_mask_from_lengths(history_lengths, int(h.history_length), int(h.patch_size))
    with torch.no_grad():
        target_codes = build_target_codes(encoder, quantizer, future_norm)
    history_latent = encoder(history_norm, history_lengths, None)
    thinker_embeds = projector(history_latent)
    thinker_out = thinker(inputs_embeds=thinker_embeds, attention_mask=thinker_mask.long(), output_hidden_states=True, use_cache=False)
    thinker_hidden = thinker_out.hidden_states[-1]
    layer0_logits_all, talker_hidden_all = talker(target_codes[:, 0, :-1], thinker_hidden, thinker_mask)
    layer0_logits = layer0_logits_all[:, :max_future_patches, :]
    talker_hidden = talker_hidden_all[:, :max_future_patches, :]
    mtp_logits = mtp(talker_hidden, target_codes[:, 0, :], target_codes[:, 1:, :])
    return compute_loss(layer0_logits, mtp_logits, target_codes, token_mask, h)


def train(h) -> int:
    rank, local_rank, world_size = setup_distributed(h)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    set_seed(int(h.seed) + rank)
    make_run_dir(h, rank, world_size)
    if int(h.history_length) % int(h.patch_size) != 0:
        raise ValueError("history_length must be divisible by patch_size")
    if int(h.max_prediction_length) % int(h.patch_size) != 0:
        raise ValueError("max_prediction_length must be divisible by patch_size")

    dtype = torch.bfloat16 if bool(h.bf16) else None
    thinker = AutoModelForCausalLM.from_pretrained(str(h.qwen_model_path), torch_dtype=dtype, trust_remote_code=bool(h.trust_remote_code), local_files_only=bool(h.local_files_only)).to(device)
    thinker.config.use_cache = False
    thinker.eval()
    for p in thinker.parameters():
        p.requires_grad_(False)
    qwen_hidden_size = int(thinker.config.hidden_size)

    encoder = Encoder(h).to(device)
    quantizer = build_quantizer(h).to(device)
    decoder = Decoder(h).to(device)
    state = load_checkpoint(str(h.codec_checkpoint_path), device)
    encoder.load_state_dict(state["encoder"], strict=True)
    quantizer.load_state_dict(state["quantizer"], strict=True)
    decoder.load_state_dict(state["decoder"], strict=True)
    encoder.eval(); quantizer.eval(); decoder.eval()
    for module in (encoder, quantizer, decoder):
        for p in module.parameters():
            p.requires_grad_(False)

    projector = TimeSeriesProjector(h, qwen_hidden_size).to(device)
    talker = TimeSeriesTalker(h, qwen_hidden_size).to(device)
    mtp = TimeSeriesMTPTransformer(h).to(device)
    input_norm = ReversibleInstanceNorm1D(num_channels=int(h.input_channels), eps=float(h.revin_eps)).to(device)

    if world_size > 1 and dist.is_initialized():
        ddp_ids = [local_rank] if device.type == "cuda" else None
        projector = DDP(projector, device_ids=ddp_ids, find_unused_parameters=bool(h.ddp_find_unused_parameters))
        talker = DDP(talker, device_ids=ddp_ids, find_unused_parameters=bool(h.ddp_find_unused_parameters))
        mtp = DDP(mtp, device_ids=ddp_ids, find_unused_parameters=bool(h.ddp_find_unused_parameters))

    trainset = make_dataset(h, str(h.train_split))
    train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if world_size > 1 and dist.is_initialized() else None
    train_loader = DataLoader(trainset, batch_size=int(h.train_batch_size), num_workers=int(h.num_workers), shuffle=(train_sampler is None), sampler=train_sampler, pin_memory=True, drop_last=True, persistent_workers=(int(h.num_workers) > 0))
    evalset = make_dataset(h, str(h.valid_split))
    eval_sampler = DistributedSampler(evalset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if world_size > 1 and dist.is_initialized() else None
    eval_loader = DataLoader(evalset, batch_size=int(h.eval_batch_size), num_workers=int(h.num_workers), shuffle=False, sampler=eval_sampler, pin_memory=True)

    params = list(projector.parameters()) + list(talker.parameters()) + list(mtp.parameters())
    optim = torch.optim.AdamW(params, lr=float(h.learning_rate), betas=[float(h.adam_b1), float(h.adam_b2)], weight_decay=float(h.weight_decay))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=float(h.learning_rate), total_steps=int(h.max_steps), pct_start=float(h.pct_start), div_factor=float(h.div_factor), final_div_factor=float(h.final_div_factor), anneal_strategy="cos")
    sw = SummaryWriter(h.logs_dir) if rank == 0 else None
    train_iter = iter(train_loader)
    steps = 0
    best_eval = None

    while steps < int(h.max_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        start = time.time()
        optim.zero_grad(set_to_none=True)
        total_loss, loss0, loss_mtp = forward_batch(batch, encoder, quantizer, projector, thinker, talker, mtp, input_norm, h, device)
        total_loss.backward()
        if float(h.max_grad_norm) > 0:
            torch.nn.utils.clip_grad_norm_(params, float(h.max_grad_norm))
        optim.step(); scheduler.step(); steps += 1

        log_total = reduce_mean(total_loss, world_size).item(); log_l0 = reduce_mean(loss0, world_size).item(); log_mtp = reduce_mean(loss_mtp, world_size).item()
        if rank == 0 and steps % int(h.stdout_interval) == 0:
            print(f"Steps: {steps} | Total: {log_total:.4f} | L0: {log_l0:.4f} | MTP: {log_mtp:.4f} | s/b: {time.time() - start:.3f}")
        if rank == 0 and sw is not None and steps % int(h.summary_interval) == 0:
            sw.add_scalar("train/total_loss", log_total, steps); sw.add_scalar("train/layer0_loss", log_l0, steps); sw.add_scalar("train/mtp_loss", log_mtp, steps); sw.add_scalar("train/lr", scheduler.get_last_lr()[0], steps)

        if steps % int(h.validation_interval) == 0:
            projector.eval(); talker.eval(); mtp.eval()
            eval_loss = torch.zeros((), device=device); eval_l0 = torch.zeros((), device=device); eval_mtp = torch.zeros((), device=device); eval_n = torch.zeros((), device=device)
            with torch.no_grad():
                for eb in eval_loader:
                    etotal, el0, emtp = forward_batch(eb, encoder, quantizer, projector, thinker, talker, mtp, input_norm, h, device)
                    bs = eb["history"].size(0); eval_loss += etotal * bs; eval_l0 += el0 * bs; eval_mtp += emtp * bs; eval_n += bs
            if dist.is_initialized():
                for t in (eval_loss, eval_l0, eval_mtp, eval_n): dist.all_reduce(t, op=dist.ReduceOp.SUM)
            eval_loss = eval_loss / eval_n.clamp_min(1); eval_l0 = eval_l0 / eval_n.clamp_min(1); eval_mtp = eval_mtp / eval_n.clamp_min(1)
            if rank == 0:
                print(f"Validation steps={steps} | Total: {eval_loss.item():.4f} | L0: {eval_l0.item():.4f} | MTP: {eval_mtp.item():.4f}")
                if sw is not None:
                    sw.add_scalar("eval/total_loss", eval_loss.item(), steps); sw.add_scalar("eval/layer0_loss", eval_l0.item(), steps); sw.add_scalar("eval/mtp_loss", eval_mtp.item(), steps)
                metrics = {"eval_loss": float(eval_loss.item()), "eval_layer0_loss": float(eval_l0.item()), "eval_mtp_loss": float(eval_mtp.item())}
                save_stage_checkpoint(os.path.join(h.checkpoint_dir, "stage1_last"), projector, talker, mtp, optim, scheduler, steps, metrics)
                if best_eval is None or float(eval_loss.item()) < best_eval:
                    best_eval = float(eval_loss.item())
                    save_stage_checkpoint(os.path.join(h.checkpoint_dir, "stage1_best"), projector, talker, mtp, optim, scheduler, steps, metrics)
            projector.train(); talker.train(); mtp.train()

    if rank == 0:
        save_stage_checkpoint(os.path.join(h.checkpoint_dir, "stage1_final"), projector, talker, mtp, optim, scheduler, steps, {"global_step": steps})
    if dist.is_initialized(): dist.destroy_process_group()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train stage1 dense time-series Thinker-Talker.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    return train(load_hparams(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
