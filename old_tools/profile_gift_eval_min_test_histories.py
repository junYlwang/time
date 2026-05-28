from __future__ import annotations

import os
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

warnings.simplefilter(action="ignore", category=FutureWarning)

DATA_ROOT = "/mnt/shared-storage-gpfs2/speechllm-share/lishenyi/datasets/gift-eval"

SHORT_DATASETS = """
m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly
electricity/15T electricity/H electricity/D electricity/W
solar/10T solar/H solar/D solar/W
hospital covid_deaths us_births/D us_births/M us_births/W
saugeenday/D saugeenday/M saugeenday/W
temperature_rain_with_missing
kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D
car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W
LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D
SZ_TAXI/15T SZ_TAXI/H
M_DENSE/H M_DENSE/D
ett1/15T ett1/H ett1/D ett1/W
ett2/15T ett2/H ett2/D ett2/W
jena_weather/10T jena_weather/H jena_weather/D
bitbrains_fast_storage/5T bitbrains_fast_storage/H
bitbrains_rnd/5T bitbrains_rnd/H
bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H
""".split()

MED_LONG_DATASETS = """
electricity/15T electricity/H
solar/10T solar/H
kdd_cup_2018_with_missing/H
LOOP_SEATTLE/5T LOOP_SEATTLE/H
SZ_TAXI/15T
M_DENSE/H
ett1/15T ett1/H
ett2/15T ett2/H
jena_weather/10T jena_weather/H
bitbrains_fast_storage/5T
bitbrains_rnd/5T
bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H
""".split()


def _import_dataset_class():
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from data.gift_eval.data import Dataset

    return Dataset


def _official_configs() -> list[tuple[str, str]]:
    configs = [(name, "short") for name in SHORT_DATASETS]
    configs.extend((name, term) for name in MED_LONG_DATASETS for term in ("medium", "long"))
    return configs


def _series_length(target) -> int:
    array = np.asarray(target)
    if array.ndim == 0:
        raise ValueError(f"target must be at least 1-D, got shape {array.shape}")
    return int(array.shape[-1])


def _print_quantiles(counter: Counter[int]) -> None:
    values = np.array(
        [history for history, count in sorted(counter.items()) for _ in range(count)],
        dtype=np.int64,
    )
    quantiles = [
        ("min", np.min(values)),
        ("mean", np.mean(values)),
        ("25%", np.quantile(values, 0.25)),
        ("50%", np.quantile(values, 0.50)),
        ("75%", np.quantile(values, 0.75)),
        ("99%", np.quantile(values, 0.99)),
        ("max", np.max(values)),
    ]
    left_width = max(len("statistic"), *(len(name) for name, _ in quantiles), len("count"))
    right_width = max(
        len("min_test_history"),
        *(len(f"{value:g}") for _, value in quantiles),
        len(str(len(values))),
    )

    print(f"{'statistic':>{left_width}}  {'min_test_history':>{right_width}}")
    print(f"{'-' * left_width}  {'-' * right_width}")
    for name, value in quantiles:
        print(f"{name:>{left_width}}  {value:>{right_width}g}")
    print(f"{'count':>{left_width}}  {len(values):>{right_width}}")


def main() -> None:
    os.environ["GIFT_EVAL"] = DATA_ROOT
    Dataset = _import_dataset_class()

    counter: Counter[int] = Counter()
    configs = _official_configs()
    for name, term in configs:
        dataset_path = Path(DATA_ROOT) / name
        if not (dataset_path / "dataset_info.json").is_file():
            raise FileNotFoundError(f"missing GIFT-Eval dataset: {dataset_path}")

        dataset = Dataset(name=name, term=term)
        heldout_length = int(dataset.prediction_length) * int(dataset.windows)
        for row_index, entry in enumerate(dataset.hf_dataset):
            min_test_history = _series_length(entry["target"]) - heldout_length
            if min_test_history <= 0:
                raise ValueError(
                    "non-positive min_test_history: "
                    f"dataset={name}, term={term}, row={row_index}, "
                    f"series_length={_series_length(entry['target'])}, "
                    f"prediction_length={dataset.prediction_length}, windows={dataset.windows}"
                )
            counter[int(min_test_history)] += 1

    _print_quantiles(counter)


if __name__ == "__main__":
    main()
