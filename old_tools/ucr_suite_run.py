from __future__ import annotations

import argparse
import csv
import json
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


def _list_ucr_datasets(ucr_root: str) -> list[str]:
    root = Path(ucr_root)
    if not root.is_dir():
        raise FileNotFoundError(f"UCR root does not exist: {ucr_root}")
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _load_metrics(metrics_path: Path) -> dict | None:
    if not metrics_path.is_file():
        return None
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sanitize_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in text)


def _build_tasks(base_cfg: dict, suite_root: Path, datasets: list[str]) -> list[dict]:
    config_dir = suite_root / "configs"
    tasks = []

    for dataset_name in datasets:
        for latent_mode in ("continuous", "discrete"):
            exp_name = f"ucr-{dataset_name}-{latent_mode}"
            cfg = dict(base_cfg)
            cfg["runs_root"] = str(suite_root)
            cfg["exp_name"] = exp_name
            cfg["dataset_name"] = dataset_name
            cfg["latent_mode"] = latent_mode

            cfg_path = config_dir / f"{dataset_name}__{latent_mode}.yaml"
            _write_yaml(cfg_path, cfg)

            tasks.append(
                {
                    "dataset_name": dataset_name,
                    "latent_mode": latent_mode,
                    "config_path": cfg_path,
                    "exp_name": exp_name,
                }
            )
    return tasks


def _launch_task(task: dict) -> subprocess.Popen:
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "tools", "ucr_train.py"),
        "--config",
        str(task["config_path"]),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
    )
    return proc


def _collect_summary(suite_root: Path, summary_path: Path) -> None:
    rows: dict[str, dict[str, float | None]] = {}
    for metrics_path in sorted(suite_root.glob("ucr-*/*/metrics/metrics.json")):
        metrics = _load_metrics(metrics_path)
        if not metrics:
            continue
        dataset_name = str(metrics["dataset_name"])
        latent_mode = str(metrics["latent_mode"])
        best_acc = metrics.get("best_acc")
        rows.setdefault(
            dataset_name,
            {
                "dataset_name": dataset_name,
                "continuous_acc": None,
                "discrete_acc": None,
            },
        )
        if latent_mode == "continuous":
            rows[dataset_name]["continuous_acc"] = best_acc
        elif latent_mode == "discrete":
            rows[dataset_name]["discrete_acc"] = best_acc

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset_name", "continuous_acc", "discrete_acc"],
        )
        writer.writeheader()
        for dataset_name in sorted(rows):
            writer.writerow(rows[dataset_name])


def main() -> None:
    parser = argparse.ArgumentParser()
    default_config = os.path.join(PROJECT_ROOT, "configs", "base.yaml")
    parser.add_argument("--config", type=str, default=default_config, help="Path to base config yaml")
    parser.add_argument("--max-concurrent", type=int, required=True, help="Maximum number of concurrent runs")
    args = parser.parse_args()

    if args.max_concurrent <= 0:
        raise ValueError("--max-concurrent must be > 0")

    base_cfg = load_hparams(args.config)
    base_cfg_dict = dict(base_cfg)

    ucr_root = str(base_cfg.ucr_root)
    datasets = _list_ucr_datasets(ucr_root)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    suite_name = f"{_sanitize_name(str(base_cfg.exp_name))}-{timestamp}"

    base_runs_root = str(base_cfg.runs_root)
    if not os.path.isabs(base_runs_root):
        base_runs_root = os.path.join(PROJECT_ROOT, base_runs_root)
    suite_root = Path(base_runs_root) / suite_name
    suite_root.mkdir(parents=True, exist_ok=True)

    tasks = _build_tasks(base_cfg_dict, suite_root, datasets)
    print(f"Prepared {len(tasks)} runs under {suite_root}")

    pending = list(tasks)
    running: list[dict] = []

    while pending or running:
        while pending and len(running) < args.max_concurrent:
            task = pending.pop(0)
            proc = _launch_task(task)
            task["proc"] = proc
            running.append(task)
            print(f"Started {task['dataset_name']} [{task['latent_mode']}] pid={proc.pid}")

        still_running = []
        for task in running:
            proc = task["proc"]
            ret = proc.poll()
            if ret is None:
                still_running.append(task)
                continue

            print(
                f"Finished {task['dataset_name']} [{task['latent_mode']}] "
                f"exit_code={ret}"
            )
        running = still_running

        if running:
            time.sleep(5)

    summary_path = suite_root / "summary.csv"
    _collect_summary(suite_root, summary_path)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
