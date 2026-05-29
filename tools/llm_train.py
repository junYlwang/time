#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset

_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from datasets.time_moe_dataset import TimeMoEDataset
from modules.encoder_wo_quantize import Encoder
from modules.quantizer import build_quantizer
from modules.utils import AttrDict, load_checkpoint

TS_VOCAB_SIZE = 2000
IGNORE_INDEX = -100


class SplitRawSeriesDataset(Dataset):
    def __init__(
        self,
        split_manifest_path: str,
        split: str,
        segment_length: int,
        samples_per_epoch: int,
        max_valid_sequences: int,
        seed: int,
        min_points: int = 8,
    ):
        self.split = str(split)
        self.segment_length = int(segment_length)
        self.samples_per_epoch = int(samples_per_epoch)
        self.max_valid_sequences = int(max_valid_sequences)
        self.seed = int(seed)
        self.min_points = int(min_points)
        self.epoch = 0

        with open(split_manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if isinstance(manifest, list):
            if self.split != "train":
                raise KeyError("List-style split manifest only supports split='train'")
            roots = manifest
        else:
            roots = manifest.get(self.split)
            if roots is None and self.split == "val":
                roots = manifest.get("valid")
            if roots is None:
                raise KeyError(f"Split '{self.split}' not found in split manifest")
        if not isinstance(roots, list) or len(roots) == 0:
            raise ValueError(f"Split '{self.split}' must contain at least one dataset root")

        self.roots = [str(root) for root in roots]
        self.datasets = [TimeMoEDataset(root, normalization_method=None) for root in self.roots]
        self.dataset_cumsum = [0]
        for ds in self.datasets:
            self.dataset_cumsum.append(self.dataset_cumsum[-1] + len(ds))
        self.num_sequences = self.dataset_cumsum[-1]
        if self.num_sequences <= 0:
            raise ValueError(f"Split '{self.split}' has zero sequences")

        self.valid_indices = []
        if self.split in ("valid", "val", "test"):
            rng = random.Random(self.seed + 271828)
            order = list(range(self.num_sequences))
            rng.shuffle(order)
            for seq_idx in order:
                if len(self.valid_indices) >= self.max_valid_sequences:
                    break
                if self.get_sequence_length(seq_idx) > self.min_points:
                    self.valid_indices.append(seq_idx)
            if not self.valid_indices:
                raise ValueError(f"Split '{self.split}' has no sequences longer than {self.min_points}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.split == "train":
            return self.samples_per_epoch
        return len(self.valid_indices)

    def get_sequence_length(self, global_seq_idx: int) -> int:
        ds_idx = bisect_right(self.dataset_cumsum, int(global_seq_idx)) - 1
        local_idx = int(global_seq_idx) - self.dataset_cumsum[ds_idx]
        return int(self.datasets[ds_idx].get_sequence_length_by_idx(local_idx))

    def fetch_sequence(self, global_seq_idx: int) -> np.ndarray:
        ds_idx = bisect_right(self.dataset_cumsum, int(global_seq_idx)) - 1
        local_idx = int(global_seq_idx) - self.dataset_cumsum[ds_idx]
        return np.asarray(self.datasets[ds_idx][local_idx], dtype=np.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.split == "train":
            rng = random.Random(self.seed + self.epoch * 1000003 + int(idx))
            for _ in range(128):
                global_seq_idx = rng.randrange(self.num_sequences)
                if self.get_sequence_length(global_seq_idx) > self.min_points:
                    break
            else:
                raise RuntimeError(f"Could not sample a sequence longer than {self.min_points} points")
            seq = self.fetch_sequence(global_seq_idx)
            if seq.size > self.segment_length:
                start = rng.randrange(0, seq.size - self.segment_length + 1)
                seq = seq[start:start + self.segment_length]
        else:
            seq = self.fetch_sequence(self.valid_indices[int(idx)])
            if seq.size > self.segment_length:
                seq = seq[:self.segment_length]

        if seq.size <= self.min_points:
            raise RuntimeError(f"Sequence with length <= {self.min_points} reached LLM NTP dataset")
        return {
            "values": torch.from_numpy(seq).view(1, -1),
            "valid_length": torch.tensor(int(seq.size), dtype=torch.long),
        }


@dataclass
class RawSeriesCollator:
    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        lengths = torch.tensor([int(item["valid_length"]) for item in features], dtype=torch.long)
        max_len = int(lengths.max().item())
        values = torch.zeros(len(features), 1, max_len, dtype=torch.float32)
        for i, item in enumerate(features):
            length = int(lengths[i].item())
            values[i, :, -length:] = item["values"][:, :length]
        return {"raw_values": values, "valid_lengths": lengths}


class CodecTokenNTPModel(nn.Module):
    def __init__(
        self,
        llm: nn.Module,
        encoder: nn.Module,
        quantizer: nn.Module,
        token_id_lookup: list[int],
        pad_token_id: int,
        downsample_factor: int,
        codebook_size: int,
        num_quantizers: int,
        norm_eps: float,
    ):
        super().__init__()
        self.llm = llm
        self.encoder = encoder.eval()
        self.quantizer = quantizer.eval()
        self.config = llm.config
        self.pad_token_id = int(pad_token_id)
        self.downsample_factor = int(downsample_factor)
        self.codebook_size = int(codebook_size)
        self.num_quantizers = int(num_quantizers)
        self.norm_eps = float(norm_eps)
        self.register_buffer("token_id_lookup", torch.tensor(token_id_lookup, dtype=torch.long), persistent=False)
        for module in (self.encoder, self.quantizer):
            for param in module.parameters():
                param.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        self.quantizer.eval()
        return self

    def save_pretrained(self, *args, **kwargs):
        return self.llm.save_pretrained(*args, **kwargs)

    def _encode_group(self, raw_values: torch.Tensor, valid_lengths: torch.Tensor, rows: torch.Tensor) -> list[torch.Tensor]:
        group_lengths = valid_lengths[rows]
        codec_len = int(((int(group_lengths.max().item()) + self.downsample_factor - 1) // self.downsample_factor) * self.downsample_factor)
        x = raw_values.new_zeros(rows.numel(), raw_values.size(1), codec_len)
        for local_i, row in enumerate(rows.tolist()):
            length = int(valid_lengths[row].item())
            real = raw_values[row, :, -length:]
            x[local_i, :, -length:] = real

        pos = torch.arange(codec_len, device=x.device).view(1, 1, codec_len)
        start = codec_len - group_lengths.to(x.device).view(-1, 1, 1)
        valid_mask = pos >= start
        count = valid_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        mean = (x * valid_mask).sum(dim=-1, keepdim=True) / count
        var = ((x - mean).pow(2) * valid_mask).sum(dim=-1, keepdim=True) / count
        x = torch.where(valid_mask, (x - mean) / torch.sqrt(var + self.norm_eps), torch.zeros_like(x))

        with torch.no_grad():
            codes = self.quantizer(self.encoder(x)).codes.long()
        if codes.dim() != 3:
            raise RuntimeError(f"Expected codec codes [B, Q, T], got {tuple(codes.shape)}")
        if codes.size(1) != self.num_quantizers:
            raise RuntimeError(f"Expected {self.num_quantizers} quantizer layers, got {codes.size(1)}")

        offsets = (torch.arange(self.num_quantizers, device=codes.device).view(1, -1) * self.codebook_size).long()
        encoded = []
        for local_i, length in enumerate(group_lengths.tolist()):
            latent_len = (int(length) + self.downsample_factor - 1) // self.downsample_factor
            sample_codes = codes[local_i, :, -latent_len:].transpose(0, 1)
            token_nums = (sample_codes + offsets).reshape(-1)
            if int(token_nums.max().item()) >= self.token_id_lookup.numel():
                raise RuntimeError("Codec token id exceeds reserved time-series token vocabulary")
            encoded.append(self.token_id_lookup[token_nums])
        return encoded

    def forward(self, raw_values: torch.Tensor, valid_lengths: torch.Tensor, **kwargs):
        raw_values = raw_values.float()
        valid_lengths = valid_lengths.long()
        if torch.any(valid_lengths <= self.downsample_factor):
            raise RuntimeError("LLM NTP samples must contain more than one codec token")

        encoded_by_row: list[torch.Tensor | None] = [None for _ in range(raw_values.size(0))]
        codec_lengths = ((valid_lengths + self.downsample_factor - 1) // self.downsample_factor) * self.downsample_factor
        for codec_len in torch.unique(codec_lengths).tolist():
            rows = torch.nonzero(codec_lengths == int(codec_len), as_tuple=False).flatten()
            encoded = self._encode_group(raw_values, valid_lengths, rows)
            for row, ids in zip(rows.tolist(), encoded):
                encoded_by_row[row] = ids

        lengths = torch.tensor([int(ids.numel()) for ids in encoded_by_row if ids is not None], device=raw_values.device)
        if torch.any(lengths < 2):
            raise RuntimeError("NTP requires at least two LLM tokens per sample")
        max_len = int(lengths.max().item())
        input_ids = torch.full((raw_values.size(0), max_len), self.pad_token_id, dtype=torch.long, device=raw_values.device)
        attention_mask = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        for i, ids in enumerate(encoded_by_row):
            if ids is None:
                raise RuntimeError("Internal error: missing encoded sample")
            seq_len = int(ids.numel())
            input_ids[i, :seq_len] = ids
            attention_mask[i, :seq_len] = 1
            labels[i, :seq_len] = ids
        return self.llm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train LLM NTP on dynamically encoded time-series codec tokens.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {args.config}")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    run_root = os.path.abspath(os.path.join(str(cfg["runs_root"]), str(cfg["exp_name"])))
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
            deadline = start_time + float(cfg.get("run_dir_wait_seconds", 600))
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

    output_dir = os.path.join(run_root, run_id)
    logging_dir = os.path.join(output_dir, "logs")
    os.makedirs(output_dir, exist_ok=True)
    cfg["output_dir"] = output_dir
    cfg["logging_dir"] = logging_dir
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", logging_dir)

    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    model_path = os.path.abspath(str(cfg["model_name_or_path"])) if not os.path.isabs(str(cfg["model_name_or_path"])) else str(cfg["model_name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
        local_files_only=bool(cfg.get("local_files_only", True)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ts_tokens = [f"<ts_{idx}>" for idx in range(TS_VOCAB_SIZE)]
    tokenizer.add_tokens(ts_tokens, special_tokens=False)
    token_id_lookup = tokenizer.convert_tokens_to_ids(ts_tokens)
    if any(tok_id is None or int(tok_id) < 0 for tok_id in token_id_lookup):
        raise RuntimeError("Failed to add time-series tokens to tokenizer")

    dtype = torch.bfloat16 if bool(cfg.get("bf16", True)) else None
    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
        local_files_only=bool(cfg.get("local_files_only", True)),
    )
    llm.resize_token_embeddings(len(tokenizer))
    llm.config.pad_token_id = tokenizer.pad_token_id
    if bool(cfg.get("gradient_checkpointing", True)):
        llm.config.use_cache = False
        llm.gradient_checkpointing_enable()

    trainable_token_indices = None
    if bool(cfg.get("trainable_ts_token_embeddings", True)):
        if bool(getattr(llm.config, "tie_word_embeddings", False)):
            trainable_token_indices = token_id_lookup
        else:
            trainable_token_indices = {"embed_tokens": token_id_lookup, "lm_head": token_id_lookup}
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        target_modules=list(cfg.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])),
        modules_to_save=list(cfg.get("modules_to_save", [])) or None,
        trainable_token_indices=trainable_token_indices,
        bias=str(cfg.get("lora_bias", "none")),
    )
    llm = get_peft_model(llm, lora_cfg)
    if rank == 0:
        llm.print_trainable_parameters()

    codec_config_path = str(cfg.get("codec_config_path", cfg.get("codec_hparams_path", "")))
    if not codec_config_path:
        raise ValueError("Config must define codec_config_path")
    codec_config_path = codec_config_path if os.path.isabs(codec_config_path) else os.path.abspath(os.path.join(_PROJECT_ROOT, codec_config_path))
    with open(codec_config_path, "r", encoding="utf-8") as f:
        codec_cfg = yaml.safe_load(f)
    h = AttrDict(codec_cfg)
    encoder = Encoder(h)
    quantizer = build_quantizer(h)
    checkpoint_path = str(cfg.get("codec_checkpoint_path", codec_cfg.get("inference", {}).get("checkpoint_path", "")))
    if not checkpoint_path:
        raise ValueError("Config must define codec_checkpoint_path, or codec config inference.checkpoint_path")
    state = load_checkpoint(checkpoint_path, "cpu")
    encoder.load_state_dict(state["encoder"], strict=True)
    quantizer.load_state_dict(state["quantizer"], strict=True)

    downsample_factor = int(cfg.get("downsample_factor", np.prod(list(h.down_ratio))))
    num_quantizers = int(getattr(quantizer, "_num_quantizers", int(h.num_quantizers)))
    codebook_size = int(getattr(quantizer, "_codebook_size", int(getattr(h, "codebook_size", 1000))))
    if num_quantizers * codebook_size > TS_VOCAB_SIZE:
        raise ValueError(f"Need {num_quantizers * codebook_size} ts tokens, but TS_VOCAB_SIZE={TS_VOCAB_SIZE}")

    model = CodecTokenNTPModel(
        llm=llm,
        encoder=encoder,
        quantizer=quantizer,
        token_id_lookup=token_id_lookup,
        pad_token_id=tokenizer.pad_token_id,
        downsample_factor=downsample_factor,
        codebook_size=codebook_size,
        num_quantizers=num_quantizers,
        norm_eps=float(getattr(h, "revin_eps", 1.0e-5)),
    )

    split_manifest_path = str(cfg.get("split_manifest_path", codec_cfg.get("split_manifest_path", "")))
    if not split_manifest_path:
        raise ValueError("Config must define split_manifest_path, or codec config split_manifest_path")
    split_manifest_path = split_manifest_path if os.path.isabs(split_manifest_path) else os.path.abspath(os.path.join(_PROJECT_ROOT, split_manifest_path))
    segment_length = int(cfg.get("segment_length", cfg.get("train_segment_length", codec_cfg.get("train_segment_length", 512))))
    train_dataset = SplitRawSeriesDataset(
        split_manifest_path=split_manifest_path,
        split=str(cfg.get("train_split", codec_cfg.get("train_split", "train"))),
        segment_length=segment_length,
        samples_per_epoch=int(args.max_train_samples or cfg.get("max_train_samples", cfg.get("samples_per_epoch", codec_cfg.get("samples_per_epoch", 500000)))),
        max_valid_sequences=int(cfg.get("max_valid_sequences", codec_cfg.get("max_valid_sequences", 2000))),
        seed=int(cfg.get("seed", 1234)),
        min_points=downsample_factor,
    )
    val_dataset = SplitRawSeriesDataset(
        split_manifest_path=split_manifest_path,
        split=str(cfg.get("valid_split", codec_cfg.get("valid_split", "valid"))),
        segment_length=int(cfg.get("eval_segment_length", cfg.get("segment_length", codec_cfg.get("eval_segment_length", segment_length)))),
        samples_per_epoch=1,
        max_valid_sequences=int(args.max_val_samples or cfg.get("max_val_samples", cfg.get("max_valid_sequences", codec_cfg.get("max_valid_sequences", 2000)))),
        seed=int(cfg.get("seed", 1234)),
        min_points=downsample_factor,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        eval_strategy=str(cfg.get("eval_strategy", "steps")),
        eval_steps=cfg.get("eval_steps", None),
        save_strategy="no",
        logging_strategy=str(cfg.get("logging_strategy", "steps")),
        logging_steps=int(cfg.get("logging_steps", 10)),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 16)),
        learning_rate=float(cfg.get("learning_rate", 2.0e-4)),
        num_train_epochs=float(cfg.get("num_train_epochs", 3)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        lr_scheduler_type=str(cfg.get("lr_scheduler_type", "cosine")),
        bf16=bool(cfg.get("bf16", True)),
        fp16=bool(cfg.get("fp16", False)),
        report_to=list(cfg.get("report_to", ["tensorboard"])),
        label_names=[],
        remove_unused_columns=False,
        dataloader_num_workers=int(cfg.get("dataloader_num_workers", 0)),
        dataloader_pin_memory=bool(cfg.get("dataloader_pin_memory", True)),
        ddp_find_unused_parameters=bool(cfg.get("ddp_find_unused_parameters", False)),
        seed=int(cfg.get("seed", 1234)),
    )

    class EpochSetter:
        from transformers import TrainerCallback

        class Callback(TrainerCallback):
            def on_epoch_begin(self, args, state, control, train_dataloader=None, **kwargs):
                if train_dataloader is not None and hasattr(train_dataloader.dataset, "set_epoch"):
                    train_dataloader.dataset.set_epoch(int(state.epoch or 0))
                return control

    class BestAdapterSaver:
        from transformers import TrainerCallback

        class Callback(TrainerCallback):
            def __init__(self, output_dir: str, tokenizer, config_payload: dict[str, Any]):
                self.output_dir = output_dir
                self.tokenizer = tokenizer
                self.config_payload = config_payload
                self.best_loss = None

            def save_adapter(self, model, subdir: str, metrics: dict[str, Any]):
                save_dir = os.path.join(self.output_dir, subdir)
                os.makedirs(save_dir, exist_ok=True)
                model.save_pretrained(save_dir)
                self.tokenizer.save_pretrained(save_dir)
                with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)
                with open(os.path.join(save_dir, "adapter_metadata.json"), "w", encoding="utf-8") as f:
                    json.dump(self.config_payload, f, indent=2, ensure_ascii=False)

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

    config_payload = dict(cfg)
    config_payload.update({
        "config_path": os.path.abspath(args.config),
        "run_id": run_id,
        "resolved_output_dir": output_dir,
        "resolved_logging_dir": logging_dir,
        "split_manifest_path": split_manifest_path,
        "codec_config_path": codec_config_path,
        "codec_checkpoint_path": checkpoint_path,
        "segment_length": segment_length,
        "downsample_factor": downsample_factor,
        "num_quantizers": num_quantizers,
        "codebook_size": codebook_size,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "ts_vocab_size": TS_VOCAB_SIZE,
        "ts_token_format": "<ts_{id}>",
        "ts_token_id_range": [0, TS_VOCAB_SIZE - 1],
        "adapter_load_note": "Load tokenizer from this adapter directory, resize base model embeddings to len(tokenizer), then load the PEFT adapter.",
        "global_batch_size": int(cfg.get("per_device_train_batch_size", 1)) * world_size * int(cfg.get("gradient_accumulation_steps", 16)),
    })

    if rank == 0:
        with open(os.path.join(output_dir, "training_args.json"), "w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2, ensure_ascii=False)
        print(json.dumps({
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "segment_length": segment_length,
            "downsample_factor": downsample_factor,
            "num_quantizers": num_quantizers,
            "codebook_size": codebook_size,
            "output_dir": output_dir,
        }, indent=2, ensure_ascii=False))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=RawSeriesCollator(),
        callbacks=[EpochSetter.Callback(), BestAdapterSaver.Callback(output_dir, tokenizer, config_payload)],
    )
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
