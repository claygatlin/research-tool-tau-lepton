"""
Feature scaling utilities (StandardScaler / MinMaxScaler) for fair comparisons.

Numpy-only implementations with sklearn-compatible ``fit`` / ``transform`` API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

ScalerName = Literal["standard", "minmax", "none"]


def _as_2d(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


@dataclass
class StandardScaler:
    """Zero-mean, unit-variance scaling per column."""

    with_mean: bool = True
    with_std: bool = True
    mean_: np.ndarray = field(default_factory=lambda: np.array([]))
    scale_: np.ndarray = field(default_factory=lambda: np.array([]))

    def fit(self, X: np.ndarray) -> StandardScaler:
        x = _as_2d(X)
        self.mean_ = x.mean(axis=0) if self.with_mean else np.zeros(x.shape[1])
        if self.with_std:
            std = x.std(axis=0, ddof=0)
            self.scale_ = np.where(std > 0.0, std, 1.0)
        else:
            self.scale_ = np.ones(x.shape[1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        x = _as_2d(X)
        if self.mean_.size == 0:
            raise RuntimeError("StandardScaler is not fitted")
        return (x - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


@dataclass
class MinMaxScaler:
    """Scale each column to [0, 1] using training min/max."""

    feature_range: tuple[float, float] = (0.0, 1.0)
    min_: np.ndarray = field(default_factory=lambda: np.array([]))
    scale_: np.ndarray = field(default_factory=lambda: np.array([]))

    def fit(self, X: np.ndarray) -> MinMaxScaler:
        x = _as_2d(X)
        data_min = x.min(axis=0)
        data_max = x.max(axis=0)
        data_range = data_max - data_min
        data_range = np.where(data_range > 0.0, data_range, 1.0)
        fr_min, fr_max = self.feature_range
        self.scale_ = (fr_max - fr_min) / data_range
        self.min_ = fr_min - data_min * self.scale_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        x = _as_2d(X)
        if self.min_.size == 0:
            raise RuntimeError("MinMaxScaler is not fitted")
        return x * self.scale_ + self.min_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def build_scaler(name: ScalerName) -> StandardScaler | MinMaxScaler | None:
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    return None


def fit_transform_pair(
    left: np.ndarray,
    right: np.ndarray,
    *,
    method: ScalerName = "standard",
    fit_reference: str = "combined",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit scaler on reference data and transform both sides."""
    if method == "none":
        return left, right, {"scaler": "none"}

    scaler = build_scaler(method)
    assert scaler is not None
    left_2d = _as_2d(left)
    right_2d = _as_2d(right)
    ref = fit_reference.lower()
    if ref == "left":
        fit_x = left_2d
    elif ref == "right":
        fit_x = right_2d
    else:
        fit_x = np.vstack([left_2d, right_2d]) if left_2d.size and right_2d.size else left_2d

    scaler.fit(fit_x)
    return (
        scaler.transform(left_2d),
        scaler.transform(right_2d),
        {
            "scaler": method,
            "fit_reference": ref,
            "n_fit_rows": int(fit_x.shape[0]),
            "n_features": int(fit_x.shape[1]) if fit_x.ndim == 2 else 1,
        },
    )