from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from modules.utils import load_hparams


ETT_DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")
LATENT_MODES = ("continuous", "discrete")
PROBE_TYPES = ("mlp",)


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _build_tasks(base_cfg: dict, config_root: Path) -> list[dict]:
    tasks = []
    for dataset_name in ETT_DATASETS:
        for latent_mode in LATENT_MODES:
            for probe_type in PROBE_TYPES:
                cfg = dict(base_cfg)
                cfg["dataset_name"] = dataset_name
                cfg["latent_mode"] = latent_mode
                cfg["probe_type"] = probe_type
                cfg["ett_column"] = "__all__"

                cfg_name = f"{dataset_name}__{latent_mode}__{probe_type}.yaml"
                cfg_path = config_root / cfg_name
                _write_yaml(cfg_path, cfg)

                tasks.append(
                    {
                        "dataset_name": dataset_name,
                        "latent_mode": latent_mode,
                        "probe_type": probe_type,
                        "config_path": cfg_path,
                    }
                )
    return tasks


def _launch_task(task: dict) -> subprocess.Popen:
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "tools", "ett_train.py"),
        "--config",
        str(task["config_path"]),
    ]
    return subprocess.Popen(cmd, cwd=PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    default_config = os.path.join(PROJECT_ROOT, "configs", "ett-3-base.yaml")
    parser.add_argument("--config", type=str, default=default_config, help="Path to base ETT config yaml")
    args = parser.parse_args()

    base_cfg = load_hparams(args.config)
    base_cfg_dict = dict(base_cfg)

    base_runs_root = str(base_cfg.runs_root)
    if not os.path.isabs(base_runs_root):
        base_runs_root = os.path.join(PROJECT_ROOT, base_runs_root)

    suite_root = Path(base_runs_root) / str(base_cfg.exp_name)
    suite_root.mkdir(parents=True, exist_ok=True)
    config_root = suite_root / "configs"

    tasks = _build_tasks(base_cfg_dict, config_root)
    print(f"Prepared {len(tasks)} ETT runs under {suite_root}")

    pending = list(tasks)
    running: list[dict] = []
    max_concurrent = len(tasks)

    while pending or running:
        while pending and len(running) < max_concurrent:
            task = pending.pop(0)
            proc = _launch_task(task)
            task["proc"] = proc
            running.append(task)
            print(
                f"Started {task['dataset_name']} "
                f"[{task['latent_mode']} | {task['probe_type']}] pid={proc.pid}"
            )

        still_running = []
        for task in running:
            proc = task["proc"]
            ret = proc.poll()
            if ret is None:
                still_running.append(task)
                continue

            print(
                f"Finished {task['dataset_name']} "
                f"[{task['latent_mode']} | {task['probe_type']}] exit_code={ret}"
            )
        running = still_running

        if running:
            time.sleep(5)


if __name__ == "__main__":
    main()
