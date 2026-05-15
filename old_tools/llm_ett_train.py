#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset


_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))


TS_VOCAB_SIZE = 3000
IGNORE_INDEX = -100


def _is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _resolve_path(path: str, *, base_dir: str = _PROJECT_ROOT) -> str:
    path = str(path)
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _safe_name(text: str) -> str:
    keep = []
    for ch in str(text):
        keep.append(ch if ch.isalnum() or ch in ("-", "_", ".") else "_")
    return "".join(keep).strip("_") or "run"


def _coordination_path(run_root: str) -> str:
    key = os.environ.get("TORCHELASTIC_RUN_ID") or os.environ.get("MASTER_PORT") or "single"
    key = _safe_name(key)
    return os.path.join(run_root, f".run_id_{key}.json")


def _build_run_root(cfg: dict[str, Any]) -> str:
    if "runs_root" in cfg and "exp_name" in cfg:
        return _resolve_path(os.path.join(str(cfg["runs_root"]), str(cfg["exp_name"])))
    raise ValueError("Config must define runs_root + exp_name, or legacy output_dir.")


def _build_run_dir(cfg: dict[str, Any], run_root: str) -> tuple[str, str]:
    os.makedirs(run_root, exist_ok=True)
    start_time = time.time()

    if _world_size() <= 1:
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        return os.path.join(run_root, run_id), run_id

    coord_path = _coordination_path(run_root)
    if _rank() == 0:
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        payload = {"run_id": run_id, "created_at": time.time()}
        tmp_path = coord_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_path, coord_path)
    else:
        deadline = start_time + float(cfg.get("run_dir_wait_seconds", 600))
        payload = None
        while time.time() < deadline:
            if os.path.isfile(coord_path):
                try:
                    with open(coord_path, "r", encoding="utf-8") as f:
                        candidate = json.load(f)
                    if float(candidate.get("created_at", 0.0)) >= start_time - 30.0:
                        payload = candidate
                        break
                except Exception:
                    pass
            time.sleep(0.2)
        if payload is None:
            raise TimeoutError(f"Timed out waiting for rank0 run id: {coord_path}")
        run_id = str(payload["run_id"])

    return os.path.join(run_root, run_id), run_id


class TokenDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        token_id_lookup: list[int],
        history_token_length: int,
        target_token_length: int,
        max_samples: int | None = None,
    ):
        self.path = str(path)
        self.token_id_lookup = token_id_lookup
        self.history_token_length = int(history_token_length)
        self.target_token_length = int(target_token_length)
        self.max_samples = None if max_samples is None else int(max_samples)
        self.offsets = self._scan_offsets()
        self._fp = None

    def _scan_offsets(self) -> list[int]:
        offsets: list[int] = []
        with open(self.path, "rb") as f:
            while True:
                if self.max_samples is not None and len(offsets) >= self.max_samples:
                    break
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)
        if not offsets:
            raise ValueError(f"No samples found in {self.path}")
        return offsets

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fp"] = None
        return state

    def _file(self):
        if self._fp is None:
            self._fp = open(self.path, "rb")
        return self._fp

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        fp = self._file()
        fp.seek(self.offsets[int(idx)])
        record = json.loads(fp.readline())

        input_tokens = record["input_tokens"]
        target_tokens = record["target_tokens"]
        if len(input_tokens) != self.history_token_length:
            raise ValueError(
                f"Bad input token length in {self.path}: "
                f"got {len(input_tokens)}, expected {self.history_token_length}"
            )
        if len(target_tokens) != self.target_token_length:
            raise ValueError(
                f"Bad target token length in {self.path}: "
                f"got {len(target_tokens)}, expected {self.target_token_length}"
            )

        input_ids = [self._map_ts_token(t) for t in input_tokens]
        target_ids = [self._map_ts_token(t) for t in target_tokens]
        labels = [IGNORE_INDEX] * len(input_ids) + target_ids
        input_ids = input_ids + target_ids
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def _map_ts_token(self, token_num: int) -> int:
        token_num = int(token_num)
        if token_num < 0 or token_num >= len(self.token_id_lookup):
            raise ValueError(f"Time-series token out of range [0, {len(self.token_id_lookup) - 1}]: {token_num}")
        return int(self.token_id_lookup[token_num])


@dataclass
class FixedLengthCollator:
    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
            "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
        }


class BestAdapterSaver:
    """Trainer callback that saves best/last PEFT adapter without base model shards."""

    def __init__(self, output_dir: str, tokenizer, config_payload: dict[str, Any]):
        from transformers import TrainerCallback

        class _Callback(TrainerCallback):
            def __init__(self, outer):
                self.outer = outer

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                metrics = metrics or {}
                if not getattr(state, "is_world_process_zero", True):
                    return control
                val_loss = metrics.get("eval_loss")
                if val_loss is None:
                    return control
                val_loss = float(val_loss)
                current_step = int(state.global_step)
                self.outer.last_eval_loss = val_loss
                self.outer.last_eval_step = current_step
                self.outer.save_adapter(
                    kwargs["model"],
                    "last_adapter",
                    {
                        "last_val_loss": val_loss,
                        "last_eval_step": current_step,
                        "epoch": float(state.epoch or 0.0),
                        "metrics": metrics,
                    },
                )
                if self.outer.best_loss is None or val_loss < self.outer.best_loss:
                    self.outer.best_loss = val_loss
                    self.outer.best_step = current_step
                    self.outer.save_adapter(
                        kwargs["model"],
                        "best_adapter",
                        {
                            "best_val_loss": val_loss,
                            "best_step": self.outer.best_step,
                            "epoch": float(state.epoch or 0.0),
                            "metrics": metrics,
                        },
                    )
                return control

            def on_train_end(self, args, state, control, **kwargs):
                if not getattr(state, "is_world_process_zero", True):
                    return control
                self.outer.save_adapter(
                    kwargs["model"],
                    "last_adapter",
                    {
                        "global_step": int(state.global_step),
                        "epoch": float(state.epoch or 0.0),
                        "last_val_loss": self.outer.last_eval_loss,
                        "last_eval_step": self.outer.last_eval_step,
                        "best_val_loss": self.outer.best_loss,
                        "best_step": self.outer.best_step,
                    },
                )
                return control

        self.output_dir = str(output_dir)
        self.tokenizer = tokenizer
        self.config_payload = config_payload
        self.best_loss: float | None = None
        self.best_step: int | None = None
        self.last_eval_loss: float | None = None
        self.last_eval_step: int | None = None
        self.callback = _Callback(self)

    def save_adapter(self, model, subdir: str, metrics: dict[str, Any]) -> None:
        save_dir = os.path.join(self.output_dir, subdir)
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        _dump_json(os.path.join(save_dir, "metrics.json"), metrics)

        if subdir == "best_adapter":
            _dump_json(os.path.join(self.output_dir, "best_metrics.json"), metrics)
        elif subdir == "last_adapter":
            _dump_json(os.path.join(self.output_dir, "last_metrics.json"), metrics)


def _build_tokenizer_and_model(cfg: dict[str, Any]):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = _resolve_path(cfg["model_name_or_path"])
    bf16 = bool(cfg.get("bf16", True))
    dtype = torch.bfloat16 if bf16 else None

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
    bad = [tok for tok, tok_id in zip(ts_tokens, token_id_lookup) if tok_id is None or tok_id < 0]
    if bad:
        raise RuntimeError(f"Failed to add time-series tokens, first bad token: {bad[0]}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
        local_files_only=bool(cfg.get("local_files_only", True)),
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    if bool(cfg.get("gradient_checkpointing", True)):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    trainable_token_indices = None
    if bool(cfg.get("trainable_ts_token_embeddings", True)):
        if bool(getattr(model.config, "tie_word_embeddings", False)):
            trainable_token_indices = token_id_lookup
        else:
            trainable_token_indices = {
                "embed_tokens": token_id_lookup,
                "lm_head": token_id_lookup,
            }

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        target_modules=list(cfg.get(
            "lora_target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )),
        modules_to_save=list(cfg.get("modules_to_save", [])) or None,
        trainable_token_indices=trainable_token_indices,
        bias=str(cfg.get("lora_bias", "none")),
    )
    model = get_peft_model(model, lora_cfg)
    if _is_rank0():
        model.print_trainable_parameters()

    return tokenizer, model, token_id_lookup


def _load_token_metadata(data_root: str) -> dict[str, Any]:
    metadata_path = os.path.join(data_root, "metadata.json")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Missing token metadata: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_training_arguments(cfg: dict[str, Any]):
    from transformers import TrainingArguments

    output_dir = _resolve_path(cfg["output_dir"])
    logging_dir = _resolve_path(cfg.get("logging_dir", os.path.join(output_dir, "logs")))
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", logging_dir)
    return TrainingArguments(
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
        label_names=list(cfg.get("label_names", ["labels"])),
        remove_unused_columns=False,
        dataloader_num_workers=int(cfg.get("dataloader_num_workers", 0)),
        dataloader_pin_memory=bool(cfg.get("dataloader_pin_memory", True)),
        ddp_find_unused_parameters=bool(cfg.get("ddp_find_unused_parameters", False)),
        seed=int(cfg.get("seed", 1234)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Qwen3 LoRA on ETT codec tokens.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    args = parser.parse_args()

    cfg = _load_yaml(args.config)

    run_root = _build_run_root(cfg)
    output_dir, run_id = _build_run_dir(cfg, run_root)
    cfg["run_root"] = run_root
    cfg["output_dir"] = output_dir
    cfg["run_id"] = run_id
    cfg["logging_dir"] = os.path.join(output_dir, "logs")
    os.makedirs(output_dir, exist_ok=True)

    tokenizer, model, token_id_lookup = _build_tokenizer_and_model(cfg)

    data_root = _resolve_path(cfg["data_root"])
    metadata = _load_token_metadata(data_root)
    history_len = int(cfg.get("history_token_length", metadata.get("history_token_length", 192)))
    target_len = int(cfg.get("target_token_length", metadata.get("target_token_length", 36)))

    train_max = args.max_train_samples if args.max_train_samples is not None else cfg.get("max_train_samples")
    val_max = args.max_val_samples if args.max_val_samples is not None else cfg.get("max_val_samples")

    train_dataset = TokenDataset(
        os.path.join(data_root, "train.jsonl"),
        token_id_lookup=token_id_lookup,
        history_token_length=history_len,
        target_token_length=target_len,
        max_samples=train_max,
    )
    val_dataset = TokenDataset(
        os.path.join(data_root, "val.jsonl"),
        token_id_lookup=token_id_lookup,
        history_token_length=history_len,
        target_token_length=target_len,
        max_samples=val_max,
    )

    if _is_rank0():
        sample = train_dataset[0]
        if len(sample["input_ids"]) != history_len + target_len:
            raise RuntimeError("Bad sample length after token mapping.")
        if sample["labels"][:history_len] != [IGNORE_INDEX] * history_len:
            raise RuntimeError("History labels are not masked with -100.")
        if tokenizer.convert_tokens_to_ids("<ts_0>") == tokenizer.unk_token_id:
            raise RuntimeError("<ts_0> was not added as a valid tokenizer token.")
        if tokenizer.convert_tokens_to_ids("<ts_2999>") == tokenizer.unk_token_id:
            raise RuntimeError("<ts_2999> was not added as a valid tokenizer token.")

    from transformers import Trainer

    training_args = _build_training_arguments(cfg)
    config_payload = dict(cfg)
    config_payload.update(
        {
            "config_path": os.path.abspath(args.config),
            "run_id": run_id,
            "resolved_run_root": run_root,
            "resolved_output_dir": output_dir,
            "resolved_logging_dir": cfg["logging_dir"],
            "resolved_data_root": data_root,
            "token_metadata": metadata,
            "history_token_length": history_len,
            "target_token_length": target_len,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "ts_vocab_size": TS_VOCAB_SIZE,
            "global_batch_size": (
                int(cfg.get("per_device_train_batch_size", 1))
                * int(os.environ.get("WORLD_SIZE", "1"))
                * int(cfg.get("gradient_accumulation_steps", 16))
            ),
        }
    )

    saver = BestAdapterSaver(output_dir, tokenizer, config_payload)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=FixedLengthCollator(),
        callbacks=[saver.callback],
    )

    if _is_rank0():
        _dump_json(os.path.join(output_dir, "training_args.json"), config_payload)
        print(
            json.dumps(
                {
                    "train_samples": len(train_dataset),
                    "val_samples": len(val_dataset),
                    "sequence_length": history_len + target_len,
                    "history_token_length": history_len,
                    "target_token_length": target_len,
                    "run_id": run_id,
                    "output_dir": output_dir,
                    "logging_dir": cfg["logging_dir"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
