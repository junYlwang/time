import glob
import os
import json
import argparse
import yaml
import matplotlib
import torch
from torch.nn.utils import weight_norm
matplotlib.use("Agg")
import matplotlib.pylab as plt
import shutil

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
        
def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


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
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


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