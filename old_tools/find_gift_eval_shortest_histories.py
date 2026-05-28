from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

from profile_gift_eval_min_test_histories import (
    DATA_ROOT,
    _import_dataset_class,
    _official_configs,
    _series_length,
)


def _print_rows(rows: list[dict[str, int | str]]) -> None:
    columns = [
        "min_test_history",
        "dataset",
        "term",
        "num_samples",
        "example_row",
        "example_item_id",
        "series_length",
        "prediction_length",
        "windows",
        "heldout_length",
    ]
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows))
        for column in columns
    }
    print("  ".join(f"{column:>{widths[column]}}" for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(f"{str(row[column]):>{widths[column]}}" for column in columns))


def main() -> None:
    os.environ["GIFT_EVAL"] = DATA_ROOT
    Dataset = _import_dataset_class()

    best_history: int | None = None
    groups: dict[tuple[str, str, int, int, int, int], dict[str, int | str]] = {}
    counts: defaultdict[tuple[str, str, int, int, int, int], int] = defaultdict(int)

    for name, term in _official_configs():
        dataset_path = Path(DATA_ROOT) / name
        if not (dataset_path / "dataset_info.json").is_file():
            raise FileNotFoundError(f"missing GIFT-Eval dataset: {dataset_path}")

        dataset = Dataset(name=name, term=term)
        prediction_length = int(dataset.prediction_length)
        windows = int(dataset.windows)
        heldout_length = prediction_length * windows

        for row_index, entry in enumerate(dataset.hf_dataset):
            series_length = _series_length(entry["target"])
            min_test_history = series_length - heldout_length
            if min_test_history <= 0:
                raise ValueError(
                    "non-positive min_test_history: "
                    f"dataset={name}, term={term}, row={row_index}, "
                    f"series_length={series_length}, "
                    f"prediction_length={prediction_length}, windows={windows}"
                )

            if best_history is None or min_test_history < best_history:
                best_history = min_test_history
                groups.clear()
                counts.clear()

            if min_test_history == best_history:
                key = (name, term, series_length, prediction_length, windows, heldout_length)
                counts[key] += 1
                groups.setdefault(
                    key,
                    {
                        "min_test_history": min_test_history,
                        "dataset": name,
                        "term": term,
                        "num_samples": 0,
                        "example_row": row_index,
                        "example_item_id": entry.get("item_id", ""),
                        "series_length": series_length,
                        "prediction_length": prediction_length,
                        "windows": windows,
                        "heldout_length": heldout_length,
                    },
                )

    rows = []
    for key, row in groups.items():
        row = dict(row)
        row["num_samples"] = counts[key]
        rows.append(row)
    rows.sort(key=lambda row: (str(row["dataset"]), str(row["term"])))
    _print_rows(rows)


if __name__ == "__main__":
    main()
