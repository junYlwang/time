from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

SPLIT_CONFIG_PATH = Path('/mnt/shared-storage-user/wangjunyi/time/configs/data_v1.json')


def _load_train_roots() -> list[str]:
    with SPLIT_CONFIG_PATH.open('r', encoding='utf-8') as f:
        split_config = json.load(f)
    roots = split_config.get('train')
    if not isinstance(roots, list) or not roots:
        raise ValueError(f"Missing non-empty 'train' list in {SPLIT_CONFIG_PATH}")
    return [str(root) for root in roots]


def _lengths_from_meta(meta_path: Path) -> list[int]:
    with meta_path.open('r', encoding='utf-8') as f:
        meta = json.load(f)

    scales = meta.get('scales')
    if not isinstance(scales, list):
        raise ValueError(f"Missing list field 'scales' in {meta_path}")

    lengths = []
    for idx, item in enumerate(scales):
        if not isinstance(item, dict) or 'length' not in item:
            raise ValueError(f"Missing scales[{idx}].length in {meta_path}")
        length = int(item['length'])
        if length <= 0:
            raise ValueError(f"Non-positive length in {meta_path}: scales[{idx}].length={length}")
        lengths.append(length)

    num_sequences = int(meta.get('num_sequences', len(lengths)))
    if num_sequences != len(lengths):
        raise ValueError(
            f"num_sequences mismatch in {meta_path}: num_sequences={num_sequences}, "
            f"len(scales)={len(lengths)}"
        )
    return lengths


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f'{value:.2f}'


def _print_table(rows: Iterable[tuple[str, str]]) -> None:
    rows = list(rows)
    key_width = max(len(key) for key, _ in rows)
    val_width = max(len(value) for _, value in rows)
    sep = f"+-{'-' * key_width}-+-{'-' * val_width}-+"
    print(sep)
    print(f"| {'metric'.ljust(key_width)} | {'value'.rjust(val_width)} |")
    print(sep)
    for key, value in rows:
        print(f"| {key.ljust(key_width)} | {value.rjust(val_width)} |")
    print(sep)


def _print_count_table(
    rows: Iterable[tuple[str, str]],
    left_header: str,
    right_header: str,
) -> None:
    rows = list(rows)
    key_width = max([len(left_header)] + [len(key) for key, _ in rows])
    val_width = max([len(right_header)] + [len(value) for _, value in rows])
    sep = f"+-{'-' * key_width}-+-{'-' * val_width}-+"
    print(sep)
    print(f"| {left_header.ljust(key_width)} | {right_header.rjust(val_width)} |")
    print(sep)
    for key, value in rows:
        print(f"| {key.ljust(key_width)} | {value.rjust(val_width)} |")
    print(sep)


def main() -> None:
    all_lengths: list[int] = []
    length_counts: Counter[int] = Counter()
    num_datasets = 0
    for root in _load_train_roots():
        meta_path = Path(root) / 'meta.json'
        if not meta_path.is_file():
            raise FileNotFoundError(f'meta.json not found: {meta_path}')
        lengths = _lengths_from_meta(meta_path)
        all_lengths.extend(lengths)
        length_counts.update(lengths)
        num_datasets += 1

    if not all_lengths:
        raise ValueError('No train sequence lengths found')

    arr = np.asarray(all_lengths, dtype=np.float64)
    rows = [
        ('num_datasets', str(num_datasets)),
        ('num_samples', str(len(all_lengths))),
        ('mean', _format_number(float(np.mean(arr)))),
        ('min', _format_number(float(np.min(arr)))),
        ('5%', _format_number(float(np.percentile(arr, 5)))),
        ('25%', _format_number(float(np.percentile(arr, 25)))),
        ('50%', _format_number(float(np.percentile(arr, 50)))),
        ('75%', _format_number(float(np.percentile(arr, 75)))),
        ('99%', _format_number(float(np.percentile(arr, 99)))),
        ('max', _format_number(float(np.max(arr)))),
    ]
    _print_table(rows)

    shortest_rows = [
        (str(length), str(length_counts[length]))
        for length in sorted(length_counts)[:10]
    ]
    print('\nShortest 10 lengths:')
    _print_count_table(shortest_rows, left_header='length', right_header='num_samples')


if __name__ == '__main__':
    main()
