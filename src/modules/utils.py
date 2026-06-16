import glob
import os
import json
import argparse
import yaml
import matplotlib
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.utils import weight_norm
matplotlib.use("Agg")
import matplotlib.pylab as plt
import shutil
import random
import numpy as np
from typing import Tuple

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load_hparams(config_path: str):
    """Load hyperparameters from a YAML (preferred) or JSON config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.endswith((".yaml", ".yml")):
            data = yaml.safe_load(f)
        elif config_path.endswith(".json"):
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file type: {config_path}")
    if data is None:
        raise ValueError(f"Empty config file: {config_path}")
    return AttrDict(data)

def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    #print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    #print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def get_state_dict(model):
    if hasattr(model, 'module'):
        return model.module.state_dict()
    return model.state_dict()

def infer_codec_state_paths(resume_path: str):
    """
    输入一个路径（codec_* 或 state_*），返回 (codec_path, state_path)
    """
    base = os.path.basename(resume_path)
    d = os.path.dirname(resume_path)

    if base.startswith("codec_"):
        codec_path = resume_path
        state_path = os.path.join(d, "state_" + base[len("codec_"):])
    elif base.startswith("state_"):
        state_path = resume_path
        codec_path = os.path.join(d, "codec_" + base[len("state_"):])
    else:
        raise ValueError(f"resume_from_checkpoint must point to codec_* or state_*, got: {resume_path}")

    return codec_path, state_path

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def inverse_revin(norm_module, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    mod = norm_module.module if hasattr(norm_module, "module") else norm_module
    return mod.inverse(y, mean, std)


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size <= 1 or not dist.is_initialized():
        return value
    out = value.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out / world_size



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


def _init_coverage_masks(codebook_sizes, device: torch.device) -> list[torch.Tensor]:
    return [
        torch.zeros(int(size), device=device, dtype=torch.bool)
        for size in tuple(codebook_sizes)
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


def valid_mask_from_lengths(valid_lengths: torch.Tensor, time_steps: int, device: torch.device) -> torch.Tensor:
    valid_lengths = valid_lengths.to(device=device, dtype=torch.long).clamp(min=0, max=time_steps)
    positions = torch.arange(time_steps, device=device).view(1, 1, time_steps)
    starts = time_steps - valid_lengths.view(-1, 1, 1)
    return positions >= starts


def masked_input_norm(input_norm, x: torch.Tensor, valid_lengths: torch.Tensor):
    if x.ndim != 3:
        raise ValueError(f"Expected [B, C, T], got shape={tuple(x.shape)}")
    valid_mask = valid_mask_from_lengths(valid_lengths, x.size(-1), x.device)
    valid = valid_mask.to(dtype=x.dtype)
    count = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (x * valid).sum(dim=-1, keepdim=True) / count
    var = ((x - mean).square() * valid).sum(dim=-1, keepdim=True) / count
    std = torch.sqrt(var + float(getattr(input_norm, "eps", 1.0e-5)))
    y = (x - mean) / std
    y = torch.where(valid_mask, y, torch.zeros_like(y))
    return y, mean, std, valid_mask


def masked_smooth_l1_loss(x_rec: torch.Tensor, x_ref: torch.Tensor, mask: torch.Tensor, beta: float) -> torch.Tensor:
    loss = F.smooth_l1_loss(x_rec, x_ref, beta=beta, reduction="none")
    weight = mask.to(dtype=loss.dtype)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def masked_diff_l1_loss(x_rec: torch.Tensor, x_ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if x_rec.size(-1) <= 1:
        return torch.zeros((), device=x_rec.device)
    diff_mask = mask[..., 1:] & mask[..., :-1]
    if not bool(diff_mask.any()):
        return torch.zeros((), device=x_rec.device)
    loss = torch.abs(torch.diff(x_rec, dim=-1) - torch.diff(x_ref, dim=-1))
    weight = diff_mask.to(dtype=loss.dtype)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)
