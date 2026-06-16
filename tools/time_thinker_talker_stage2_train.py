#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import yaml
from peft import LoraConfig, TaskType, get_peft_model
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
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from modules.patch_codec import PatchCausalMAEEncoder as Encoder, PatchCausalDecoder as Decoder
from modules.quantizer import build_quantizer
from modules.revin import ReversibleInstanceNorm1D
from modules.time_thinker_talker import TimeSeriesMTPTransformer, TimeSeriesProjector, TimeSeriesTalker
from modules.utils import get_state_dict, load_checkpoint, load_hparams, reduce_mean, save_checkpoint, set_seed
from time_thinker_talker_stage1_train import forward_batch, make_dataset, make_run_dir, setup_distributed


def save_stage2_checkpoint(path, projector, talker, mtp, optim, scheduler, steps: int, metrics: dict):
    save_checkpoint(path, {
        "projector": get_state_dict(projector),
        "talker": get_state_dict(talker),
        "mtp": get_state_dict(mtp),
        "optim": optim.state_dict(),
        "scheduler": scheduler.state_dict(),
        "steps": int(steps),
        "metrics": metrics,
    })


def train(h) -> int:
    rank, local_rank, world_size = setup_distributed(h)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    set_seed(int(h.seed) + rank)
    make_run_dir(h, rank, world_size)
    if int(h.max_history_length) % int(h.patch_size) != 0:
        raise ValueError("max_history_length must be divisible by patch_size")
    if int(h.max_prediction_length) % int(h.patch_size) != 0:
        raise ValueError("max_prediction_length must be divisible by patch_size")

    dtype = torch.bfloat16 if bool(h.bf16) else None
    thinker = AutoModelForCausalLM.from_pretrained(str(h.qwen_model_path), torch_dtype=dtype, trust_remote_code=bool(h.trust_remote_code), local_files_only=bool(h.local_files_only)).to(device)
    thinker.config.use_cache = False
    if bool(h.gradient_checkpointing):
        thinker.gradient_checkpointing_enable()
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(h.lora_r),
        lora_alpha=int(h.lora_alpha),
        lora_dropout=float(h.lora_dropout),
        target_modules=list(h.lora_target_modules),
        bias=str(h.lora_bias),
    )
    thinker = get_peft_model(thinker, lora_cfg)
    qwen_hidden_size = int(thinker.config.hidden_size)
    if rank == 0:
        thinker.print_trainable_parameters()

    encoder = Encoder(h).to(device)
    quantizer = build_quantizer(h).to(device)
    decoder = Decoder(h).to(device)
    codec_state = load_checkpoint(str(h.codec_checkpoint_path), device)
    encoder.load_state_dict(codec_state["encoder"], strict=True)
    quantizer.load_state_dict(codec_state["quantizer"], strict=True)
    decoder.load_state_dict(codec_state["decoder"], strict=True)
    encoder.eval(); quantizer.eval(); decoder.eval()
    for module in (encoder, quantizer, decoder):
        for p in module.parameters():
            p.requires_grad_(False)

    projector = TimeSeriesProjector(h, qwen_hidden_size).to(device)
    talker = TimeSeriesTalker(h, qwen_hidden_size).to(device)
    mtp = TimeSeriesMTPTransformer(h).to(device)
    stage1_state = load_checkpoint(str(h.stage1_checkpoint_path), device)
    projector.load_state_dict(stage1_state["projector"], strict=True)
    talker.load_state_dict(stage1_state["talker"], strict=True)
    mtp.load_state_dict(stage1_state["mtp"], strict=True)
    input_norm = ReversibleInstanceNorm1D(num_channels=int(h.input_channels), eps=float(h.revin_eps)).to(device)

    if world_size > 1 and dist.is_initialized():
        ddp_ids = [local_rank] if device.type == "cuda" else None
        thinker = DDP(thinker, device_ids=ddp_ids, find_unused_parameters=bool(h.ddp_find_unused_parameters))
        projector = DDP(projector, device_ids=ddp_ids, find_unused_parameters=bool(h.ddp_find_unused_parameters))
        talker = DDP(talker, device_ids=ddp_ids, find_unused_parameters=bool(h.ddp_find_unused_parameters))
        mtp = DDP(mtp, device_ids=ddp_ids, find_unused_parameters=bool(h.ddp_find_unused_parameters))

    trainset = make_dataset(h, str(h.train_split))
    train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if world_size > 1 and dist.is_initialized() else None
    train_loader = DataLoader(trainset, batch_size=int(h.train_batch_size), num_workers=int(h.num_workers), shuffle=(train_sampler is None), sampler=train_sampler, pin_memory=True, drop_last=True, persistent_workers=(int(h.num_workers) > 0))
    evalset = make_dataset(h, str(h.valid_split))
    eval_sampler = DistributedSampler(evalset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if world_size > 1 and dist.is_initialized() else None
    eval_loader = DataLoader(evalset, batch_size=int(h.eval_batch_size), num_workers=int(h.num_workers), shuffle=False, sampler=eval_sampler, pin_memory=True)

    params = [p for p in thinker.parameters() if p.requires_grad] + list(projector.parameters()) + list(talker.parameters()) + list(mtp.parameters())
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
            thinker.eval(); projector.eval(); talker.eval(); mtp.eval()
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
                save_stage2_checkpoint(os.path.join(h.checkpoint_dir, "stage2_last"), projector, talker, mtp, optim, scheduler, steps, metrics)
                model_to_save = thinker.module if hasattr(thinker, "module") else thinker
                model_to_save.save_pretrained(os.path.join(h.checkpoint_dir, "qwen_lora_last"))
                if best_eval is None or float(eval_loss.item()) < best_eval:
                    best_eval = float(eval_loss.item())
                    save_stage2_checkpoint(os.path.join(h.checkpoint_dir, "stage2_best"), projector, talker, mtp, optim, scheduler, steps, metrics)
                    model_to_save.save_pretrained(os.path.join(h.checkpoint_dir, "qwen_lora_best"))
            thinker.train(); projector.train(); talker.train(); mtp.train()

    if rank == 0:
        save_stage2_checkpoint(os.path.join(h.checkpoint_dir, "stage2_final"), projector, talker, mtp, optim, scheduler, steps, {"global_step": steps})
        model_to_save = thinker.module if hasattr(thinker, "module") else thinker
        model_to_save.save_pretrained(os.path.join(h.checkpoint_dir, "qwen_lora_final"))
    if dist.is_initialized(): dist.destroy_process_group()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train stage2 dense time-series Thinker-Talker with Qwen LoRA.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    return train(load_hparams(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
