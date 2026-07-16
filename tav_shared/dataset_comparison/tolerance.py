"""
Tolerance-aware numeric diff utilities for solver and kinematic comparisons.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from tav_shared.dataset_comparison.config import load_comparison_config, tolerance_for


def tolerant_diff(
    a: float | np.ndarray,
    b: float | np.ndarray,
    *,
    atol: float | None = None,
    rtol: float | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """
    Compare scalars or arrays with combined absolute and relative tolerance.

    ``within_tolerance`` is True when ``|a-b| <= atol + rtol * max(|a|, |b|)``.
    """
    cfg = load_comparison_config()
    tol = cfg.get("tolerances", {})
    if label and label in tol:
        atol = float(tol[label]) if atol is None else atol
    if atol is None:
        atol = float(tol.get("absolute", 1.0e-9))
    if rtol is None:
        rtol = float(tol.get("relative", 1.0e-6))

    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    delta = aa - bb
    scale = np.maximum(np.maximum(np.abs(aa), np.abs(bb)), 1.0e-30)
    threshold = atol + rtol * scale
    within = np.abs(delta) <= threshold

    if aa.size == 1 and bb.size == 1:
        return {
            "a": float(aa.reshape(-1)[0]),
            "b": float(bb.reshape(-1)[0]),
            "delta": float(delta.reshape(-1)[0]),
            "atol": float(atol),
            "rtol": float(rtol),
            "within_tolerance": bool(within.reshape(-1)[0]),
            "label": label,
        }

    return {
        "delta": delta,
        "atol": float(atol),
        "rtol": float(rtol),
        "within_tolerance": within,
        "n_within": int(np.sum(within)),
        "n_total": int(within.size),
        "fraction_within": float(np.mean(within)) if within.size else 1.0,
        "max_abs_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "label": label,
    }


def tolerant_allclose(
    a: np.ndarray,
    b: np.ndarray,
    *,
    atol: float | None = None,
    rtol: float | None = None,
    label: str | None = None,
) -> bool:
    report = tolerant_diff(a, b, atol=atol, rtol=rtol, label=label)
    within = report["within_tolerance"]
    if isinstance(within, np.ndarray):
        return bool(np.all(within))
    return bool(within)


def compare_mass_gap_mev(
    predicted_mev: float,
    observed_mev: float,
    *,
    atol_mev: float | None = None,
) -> dict[str, Any]:
    """Dimuon / tau mass-gap check with MeV-scale tolerance."""
    if atol_mev is None:
        atol_mev = tolerance_for("mass_gap_mev")
    return tolerant_diff(predicted_mev, observed_mev, atol=atol_mev, rtol=0.0, label="mass_gap_mev")


def compare_scalar_observables(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    tolerance_labels: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Compare named scalars (subharmonic, chi2, amplitudes) with per-key tolerances."""
    labels = dict(tolerance_labels or {})
    default_label = "amplitude"
    comparisons: dict[str, Any] = {}
    all_ok = True
    for key in sorted(set(left.keys()) & set(right.keys())):
        tol_label = labels.get(key, default_label)
        rep = tolerant_diff(float(left[key]), float(right[key]), label=tol_label)
        comparisons[key] = rep
        if not rep["within_tolerance"]:
            all_ok = False
    return {
        "comparisons": comparisons,
        "all_within_tolerance": all_ok,
        "n_compared": len(comparisons),
    }