"""
End-to-end dataset comparison pipeline: align, scale, tolerant diff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from tav_shared.dataset_comparison.config import load_comparison_config
from tav_shared.dataset_comparison.keys import (
    align_by_primary_keys,
    feature_matrix,
    sort_by_primary_keys,
)
from tav_shared.dataset_comparison.scaling import ScalerName, fit_transform_pair
from tav_shared.dataset_comparison.tolerance import (
    compare_scalar_observables,
    tolerant_diff,
)


@dataclass
class DatasetComparisonReport:
    """Summary of a standardized Data vs MC (or model vs data) comparison."""

    comparison_type: str
    n_left: int = 0
    n_right: int = 0
    n_matched: int = 0
    scaler: str = "none"
    primary_keys: list[str] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)
    within_tolerance: bool = True
    feature_diffs: dict[str, Any] = field(default_factory=dict)
    scalar_diffs: dict[str, Any] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "comparison_type": self.comparison_type,
            "n_left": self.n_left,
            "n_right": self.n_right,
            "n_matched": self.n_matched,
            "scaler": self.scaler,
            "primary_keys": self.primary_keys,
            "feature_columns": self.feature_columns,
            "within_tolerance": self.within_tolerance,
            "feature_diffs": self.feature_diffs,
            "scalar_diffs": self.scalar_diffs,
            "messages": self.messages,
        }


def _align_histograms(
    left: np.ndarray,
    right: np.ndarray,
    *,
    bin_edges: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    l = np.asarray(left, dtype=float).ravel()
    r = np.asarray(right, dtype=float).ravel()
    if bin_edges is not None:
        edges = np.asarray(bin_edges, dtype=float).ravel()
        n_bins = max(len(edges) - 1, 0)
        l = l[:n_bins]
        r = r[:n_bins]
        return l, r, edges
    n = min(l.size, r.size)
    return l[:n], r[:n], None


def compare_histogram_pair(
    left_hist: np.ndarray,
    right_hist: np.ndarray,
    *,
    bin_edges: np.ndarray | None = None,
    scaler: ScalerName | None = None,
    normalize: bool = True,
) -> dict[str, Any]:
    """
    Compare two histograms on aligned bins with optional scaling.

    When ``normalize=True``, compares shape (fraction of total) not raw counts.
    """
    cfg = load_comparison_config()
    method: ScalerName = scaler or cfg.get("histogram_scaler", "minmax")  # type: ignore[assignment]
    fit_ref = str(cfg.get("fit_reference", "combined"))

    left, right, edges = _align_histograms(left_hist, right_hist, bin_edges=bin_edges)
    if left.size == 0:
        return {"verdict": "UNDERPOWERED", "n_bins": 0}

    if normalize:
        left_sum = max(float(left.sum()), 1.0e-30)
        right_sum = max(float(right.sum()), 1.0e-30)
        left = left / left_sum
        right = right / right_sum

    left_s, right_s, scale_meta = fit_transform_pair(
        left.reshape(-1, 1),
        right.reshape(-1, 1),
        method=method,
        fit_reference=fit_ref,
    )
    left_s = left_s.ravel()
    right_s = right_s.ravel()

    tol_label = "histogram_fraction"
    diff = tolerant_diff(left_s, right_s, label=tol_label)
    l1 = float(np.sum(np.abs(left - right)))
    return {
        "n_bins": int(left.size),
        "bin_edges": edges.tolist() if edges is not None else None,
        "normalized": normalize,
        "scaler": scale_meta,
        "l1_distance_raw": l1,
        "scaled_diff": diff,
        "within_tolerance": bool(diff.get("fraction_within", 0.0) >= 0.95)
        if isinstance(diff.get("within_tolerance"), np.ndarray)
        else bool(diff.get("within_tolerance", False)),
        "mean_left": float(np.mean(left)),
        "mean_right": float(np.mean(right)),
    }


def prepare_datasets_for_comparison(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    feature_columns: Sequence[str] | None = None,
    primary_keys: Sequence[str] | None = None,
    scaler: ScalerName | None = None,
    fit_reference: str | None = None,
) -> tuple[dict[str, Any], DatasetComparisonReport]:
    """
    Align event tables by primary keys, scale features, and compute tolerant diffs.

    Intended for MC vs empirical dimuon kinematics before solver input.
    """
    cfg = load_comparison_config()
    features = list(feature_columns or cfg.get("cms_dimuon_features", []))
    keys = list(primary_keys or cfg.get("cms_dimuon_primary_keys", []))
    method: ScalerName = scaler or cfg.get("scaler", "standard")  # type: ignore[assignment]
    fit_ref = fit_reference or str(cfg.get("fit_reference", "combined"))

    aligned = align_by_primary_keys(left, right, keys, feature_columns=features)
    report = DatasetComparisonReport(
        comparison_type="feature_table",
        n_left=int(aligned["n_left"]),
        n_right=int(aligned["n_right"]),
        n_matched=int(aligned["n_matched"]),
        primary_keys=keys,
        feature_columns=features,
        scaler=method,
    )

    if report.n_matched == 0:
        report.within_tolerance = False
        report.messages.append("No primary-key matches between left and right tables")
        return {"aligned": aligned, "scaled_left": {}, "scaled_right": {}}, report

    left_mat = feature_matrix(aligned["left"], features)
    right_mat = feature_matrix(aligned["right"], features)
    left_s, right_s, scale_meta = fit_transform_pair(
        left_mat,
        right_mat,
        method=method,
        fit_reference=fit_ref,
    )
    report.scaler = str(scale_meta.get("scaler", method))

    feature_diffs: dict[str, Any] = {}
    all_ok = True
    for idx, name in enumerate(features):
        rep = tolerant_diff(left_s[:, idx], right_s[:, idx], label="kinematic_gev")
        feature_diffs[name] = rep
        frac = rep.get("fraction_within", 0.0)
        if isinstance(frac, float) and frac < 0.9:
            all_ok = False

    report.feature_diffs = feature_diffs
    report.within_tolerance = all_ok
    n_left_unmatched = int(aligned["n_left_unmatched"])
    n_right_unmatched = int(aligned["n_right_unmatched"])
    if n_left_unmatched:
        report.messages.append(f"{n_left_unmatched} left rows unmatched on primary keys")
    if n_right_unmatched:
        report.messages.append(f"{n_right_unmatched} right rows unmatched on primary keys")

    return {
        "aligned": aligned,
        "scaled_left": {features[i]: left_s[:, i] for i in range(len(features))},
        "scaled_right": {features[i]: right_s[:, i] for i in range(len(features))},
        "scaler_metadata": scale_meta,
    }, report


def compare_aggregate_metrics(
    left_metrics: Mapping[str, float],
    right_metrics: Mapping[str, float],
    *,
    tolerance_map: Mapping[str, str] | None = None,
) -> DatasetComparisonReport:
    """Tolerant comparison of scalar summary metrics (χ², amplitudes, mass gaps)."""
    result = compare_scalar_observables(left_metrics, right_metrics, tolerance_labels=tolerance_map)
    report = DatasetComparisonReport(
        comparison_type="scalar_metrics",
        n_left=len(left_metrics),
        n_right=len(right_metrics),
        n_matched=result["n_compared"],
        within_tolerance=bool(result["all_within_tolerance"]),
        scalar_diffs=result["comparisons"],
    )
    return report


def compare_dimuon_kinematics(
    data_events: Mapping[str, Any],
    mc_events: Mapping[str, Any],
    *,
    scaler: ScalerName | None = None,
) -> dict[str, Any]:
    """
    High-level Data vs MC dimuon comparison with sort, scale, and tolerant diff.
    """
    data_sorted = sort_by_primary_keys(data_events)
    mc_sorted = sort_by_primary_keys(mc_events)
    prepared, report = prepare_datasets_for_comparison(
        data_sorted,
        mc_sorted,
        scaler=scaler,
    )
    return {
        "dataset_comparison": report.as_dict(),
        "prepared": prepared,
    }


def compare_model_predictions_to_observed(
    predicted: Mapping[str, float],
    observed: Mapping[str, float],
) -> dict[str, Any]:
    """Tolerant check of theoretical_model outputs vs observed anchors."""
    tol_map = {
        "mass_gap_mev": "mass_gap_mev",
        "mass_gap": "mass_gap_mev",
        "spectral_dim": "amplitude",
    }
    left = {k: float(v) for k, v in predicted.items() if isinstance(v, (int, float))}
    right = {k: float(v) for k, v in observed.items() if isinstance(v, (int, float))}
    keys = [k for k in ("mass_gap_mev", "mass_gap", "spectral_dim") if k in left and k in right]
    if "mass_gap" in keys and "mass_gap_mev" in keys:
        keys = [k for k in keys if k != "mass_gap"]
    report = compare_aggregate_metrics(
        {k: left[k] for k in keys},
        {k: right.get(k, right.get("mass_gap_mev" if k == "mass_gap" else k, 0.0)) for k in keys},
        tolerance_map=tol_map,
    )
    return report.as_dict()