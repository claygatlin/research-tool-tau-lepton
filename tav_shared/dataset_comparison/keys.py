"""
Primary-key construction, sorting, and aligned joins for dataset comparison.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from tav_shared.dataset_comparison.config import load_comparison_config


def _column_array(data: Mapping[str, Any], column: str) -> np.ndarray:
    if column not in data:
        raise KeyError(f"Missing column for primary key: {column}")
    return np.asarray(data[column]).reshape(-1)


def build_rounded_key_columns(
    data: Mapping[str, Any],
    key_columns: Sequence[str] | None = None,
    *,
    round_digits: Mapping[str, int] | None = None,
) -> np.ndarray:
    """
    Build sortable primary-key tuples from kinematic columns.

    Rounding stabilizes floating-point jitter across MC vs data exports.
    """
    cfg = load_comparison_config()
    keys = list(key_columns or cfg.get("cms_dimuon_primary_keys", []))
    digits = dict(cfg.get("key_round_digits", {}))
    if round_digits:
        digits.update(round_digits)

    if not keys:
        raise ValueError("key_columns must be non-empty")

    n = min(_column_array(data, keys[0]).size, *(_column_array(data, k).size for k in keys[1:]))
    if n == 0:
        return np.empty((0, len(keys)), dtype=float)

    cols: list[np.ndarray] = []
    for key in keys:
        arr = _column_array(data, key)[:n].astype(float)
        nd = digits.get(key)
        if nd is not None:
            arr = np.round(arr, int(nd))
        cols.append(arr)
    return np.column_stack(cols)


def sort_by_primary_keys(
    data: Mapping[str, Any],
    key_columns: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Return column dict sorted lexicographically by primary keys."""
    cfg = load_comparison_config()
    keys = list(key_columns or cfg.get("cms_dimuon_primary_keys", []))
    key_matrix = build_rounded_key_columns(data, keys)
    n = key_matrix.shape[0]
    out: dict[str, np.ndarray] = {}
    for col, arr in data.items():
        flat = np.asarray(arr).reshape(-1)
        if flat.size == n:
            out[col] = flat
    if n == 0:
        return out
    order = np.lexsort(tuple(reversed([key_matrix[:, i] for i in range(key_matrix.shape[1])])))
    return {col: vals[order] for col, vals in out.items()}


def align_by_primary_keys(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    key_columns: Sequence[str] | None = None,
    *,
    feature_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Inner-join two event tables on rounded primary keys.

    Avoids naive row-by-row zip when MC and data row orders differ.
    """
    cfg = load_comparison_config()
    keys = list(key_columns or cfg.get("cms_dimuon_primary_keys", []))
    features = list(feature_columns or cfg.get("cms_dimuon_features", []))

    left_sorted = sort_by_primary_keys(left, keys)
    right_sorted = sort_by_primary_keys(right, keys)
    n_left = _column_array(left_sorted, keys[0]).size if keys[0] in left_sorted else 0
    n_right = _column_array(right_sorted, keys[0]).size if keys[0] in right_sorted else 0

    if n_left == 0 or n_right == 0:
        return {
            "left": {k: np.array([]) for k in features},
            "right": {k: np.array([]) for k in features},
            "n_left": n_left,
            "n_right": n_right,
            "n_matched": 0,
            "n_left_unmatched": n_left,
            "n_right_unmatched": n_right,
            "primary_keys": keys,
        }

    left_keys = build_rounded_key_columns(left_sorted, keys)
    right_keys = build_rounded_key_columns(right_sorted, keys)

    # Map right rows by key tuple string for O(n) join
    right_index: dict[str, int] = {}
    for idx, row in enumerate(right_keys):
        token = "|".join(f"{v:.12g}" for v in row)
        right_index.setdefault(token, idx)

    left_idx: list[int] = []
    right_idx: list[int] = []
    for i, row in enumerate(left_keys):
        token = "|".join(f"{v:.12g}" for v in row)
        j = right_index.get(token)
        if j is not None:
            left_idx.append(i)
            right_idx.append(j)

    matched_left = {k: _column_array(left_sorted, k)[left_idx] for k in features if k in left_sorted}
    matched_right = {k: _column_array(right_sorted, k)[right_idx] for k in features if k in right_sorted}
    n_matched = len(left_idx)

    return {
        "left": matched_left,
        "right": matched_right,
        "n_left": n_left,
        "n_right": n_right,
        "n_matched": n_matched,
        "n_left_unmatched": n_left - n_matched,
        "n_right_unmatched": n_right - n_matched,
        "primary_keys": keys,
    }


def feature_matrix(
    data: Mapping[str, Any],
    feature_columns: Sequence[str],
) -> np.ndarray:
    """Stack feature columns into an (n_events, n_features) matrix."""
    cols = [_column_array(data, name).astype(float) for name in feature_columns]
    n = min(c.size for c in cols)
    if n == 0:
        return np.empty((0, len(feature_columns)), dtype=float)
    return np.column_stack([c[:n] for c in cols])