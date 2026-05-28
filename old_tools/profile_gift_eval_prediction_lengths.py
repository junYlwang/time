from __future__ import annotations

import os
import sys
import warnings
from collections import Counter
from pathlib import Path

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


def _print_table(counter: Counter[int]) -> None:
    rows = sorted(counter.items())
    left_width = max(len("prediction_length"), *(len(str(k)) for k, _ in rows))
    right_width = max(len("num_samples"), *(len(str(v)) for _, v in rows), len(str(sum(counter.values()))))

    print(f"{'prediction_length':>{left_width}}  {'num_samples':>{right_width}}")
    print(f"{'-' * left_width}  {'-' * right_width}")
    for prediction_length, num_samples in rows:
        print(f"{prediction_length:>{left_width}}  {num_samples:>{right_width}}")
    print(f"{'TOTAL':>{left_width}}  {sum(counter.values()):>{right_width}}")


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
        counter[int(dataset.prediction_length)] += len(dataset.hf_dataset)

    _print_table(counter)


if __name__ == "__main__":
    main()
