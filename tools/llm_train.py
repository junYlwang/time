#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any

_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

from src.datasets.llm_codec_dataset import SplitRawSeriesDataset
from src.modules.codec_token_ntp import CodecTokenNTPModel
from src.modules.decoder import Decoder
from src.modules.encoder_wo_quantize import Encoder
from src.modules.quantizer import build_quantizer
from src.modules.utils import AttrDict, load_checkpoint


class BestAdapterSaver(TrainerCallback):
    def __init__(self, output_dir: str, tokenizer):
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.best_loss = None

    def save_adapter(self, model, subdir: str, metrics: dict[str, Any]):
        save_dir = os.path.join(self.output_dir, subdir)
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return control
        metrics = metrics or {}
        if "eval_loss" not in metrics:
            return control
        val_loss = float(metrics["eval_loss"])
        payload = {"val_loss": val_loss, "step": int(state.global_step), "epoch": float(state.epoch or 0.0), "metrics": metrics}
        self.save_adapter(kwargs["model"], "last_adapter", payload)
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.save_adapter(kwargs["model"], "best_adapter", payload)
            with open(os.path.join(self.output_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if getattr(state, "is_world_process_zero", True):
            self.save_adapter(kwargs["model"], "last_adapter", {"global_step": int(state.global_step), "epoch": float(state.epoch or 0.0)})
        return control


def train(h: AttrDict) -> int:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    model_path = str(h.model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=bool(h.trust_remote_code),
        local_files_only=bool(h.local_files_only),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ts_vocab_size = int(h.ts_vocab_size)
    ts_tokens = [f"<ts_{idx}>" for idx in range(ts_vocab_size)]
    tokenizer.add_tokens(ts_tokens, special_tokens=False)
    token_id_lookup = tokenizer.convert_tokens_to_ids(ts_tokens)
    if any(tok_id is None or int(tok_id) < 0 for tok_id in token_id_lookup):
        raise RuntimeError("Failed to add time-series tokens to tokenizer")

    dtype = None
    if bool(h.bf16):
        dtype = torch.bfloat16
    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=bool(h.trust_remote_code),
        local_files_only=bool(h.local_files_only),
    )
    llm.resize_token_embeddings(len(tokenizer))
    llm.config.pad_token_id = tokenizer.pad_token_id
    if bool(h.gradient_checkpointing):
        llm.config.use_cache = False
        llm.gradient_checkpointing_enable()

    trainable_token_indices = None
    if bool(h.trainable_ts_token_embeddings):
        if bool(getattr(llm.config, "tie_word_embeddings", False)):
            trainable_token_indices = token_id_lookup
        else:
            trainable_token_indices = {"embed_tokens": token_id_lookup, "lm_head": token_id_lookup}
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(h.lora_r),
        lora_alpha=int(h.lora_alpha),
        lora_dropout=float(h.lora_dropout),
        target_modules=list(h.lora_target_modules),
        modules_to_save=list(h.modules_to_save) or None,
        trainable_token_indices=trainable_token_indices,
        bias=str(h.lora_bias),
    )
    llm = get_peft_model(llm, lora_cfg)
    if rank == 0:
        llm.print_trainable_parameters()

    encoder = Encoder(h)
    quantizer = build_quantizer(h)
    decoder = Decoder(h)
    checkpoint_path = str(h.codec_checkpoint_path)
    state = load_checkpoint(checkpoint_path, "cpu")
    encoder.load_state_dict(state["encoder"], strict=True)
    quantizer.load_state_dict(state["quantizer"], strict=True)
    decoder.load_state_dict(state["decoder"], strict=True)

    downsample_factor = int(h.downsample_factor)
    num_quantizers = int(h.num_quantizers)
    codebook_size = int(h.codebook_size)
    if num_quantizers * codebook_size > ts_vocab_size:
        raise ValueError(f"Need {num_quantizers * codebook_size} ts tokens, but ts_vocab_size={ts_vocab_size}")

    model = CodecTokenNTPModel(
        llm=llm,
        encoder=encoder,
        quantizer=quantizer,
        decoder=decoder,
        token_id_lookup=token_id_lookup,
        pad_token_id=tokenizer.pad_token_id,
        downsample_factor=downsample_factor,
        codebook_size=codebook_size,
        num_quantizers=num_quantizers,
        norm_eps=float(h.revin_eps),
    )

    split_manifest_path = str(h.split_manifest_path)

    train_dataset = SplitRawSeriesDataset(
        split_manifest_path=split_manifest_path,
        split=str(h.train_split),
        segment_length=int(h.train_segment_length),
        max_valid_sequences=int(h.max_valid_sequences),
        seed=int(h.seed),
        min_points=int(h.min_train_points),
    )
    val_dataset = SplitRawSeriesDataset(
        split_manifest_path=split_manifest_path,
        split=str(h.valid_split),
        segment_length=int(h.valid_segment_length),
        max_valid_sequences=int(h.max_valid_sequences),
        seed=int(h.seed),
        min_points=int(h.min_valid_points),
    )

    training_args = TrainingArguments(
        output_dir=str(h.output_dir),
        do_train=True,
        do_eval=True,
        eval_strategy=str(h.eval_strategy),
        eval_steps=h.eval_steps,
        save_strategy="no",
        logging_strategy=str(h.logging_strategy),
        logging_steps=int(h.logging_steps),
        per_device_train_batch_size=int(h.per_device_train_batch_size),
        per_device_eval_batch_size=int(h.per_device_eval_batch_size),
        gradient_accumulation_steps=int(h.gradient_accumulation_steps),
        learning_rate=float(h.learning_rate),
        max_steps=int(h.max_steps),
        warmup_ratio=float(h.warmup_ratio),
        weight_decay=float(h.weight_decay),
        # max_grad_norm=float(h.max_grad_norm),
        lr_scheduler_type=str(h.lr_scheduler_type),
        bf16=bool(h.bf16),
        fp16=bool(h.fp16),
        report_to=list(h.report_to),
        label_names=[],
        remove_unused_columns=False,
        dataloader_num_workers=int(h.dataloader_num_workers),
        dataloader_pin_memory=bool(h.dataloader_pin_memory),
        ddp_find_unused_parameters=bool(h.ddp_find_unused_parameters),
        seed=int(h.seed),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[BestAdapterSaver(str(h.output_dir), tokenizer)],
    )
    trainer.train()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train LLM NTP on dynamically encoded time-series codec tokens.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        h = AttrDict(yaml.safe_load(f))

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
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
            deadline = start_time + float(h.run_dir_wait_seconds)
            while time.time() < deadline:
                if os.path.isfile(coord_path):
                    with open(coord_path, "r", encoding="utf-8") as f:
                        candidate = json.load(f)
                    if float(candidate.get("created_at", 0.0)) >= start_time - 30.0:
                        payload = candidate
                        break
                time.sleep(0.2)
            if payload is None:
                raise TimeoutError(f"Timed out waiting for rank0 run id: {coord_path}")
            run_id = str(payload["run_id"])

    h.run_root = run_root
    h.run_id = run_id
    h.output_dir = os.path.join(run_root, run_id)
    h.logging_dir = os.path.join(h.output_dir, "logs")
    os.makedirs(h.output_dir, exist_ok=True)
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", h.logging_dir)
    if rank == 0:
        with open(os.path.join(h.output_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(dict(h), f, allow_unicode=True, sort_keys=False)
    return train(h)


if __name__ == "__main__":
    raise SystemExit(main())
