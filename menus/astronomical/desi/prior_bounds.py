"""
Shared prior bounds and evidence corrections for Tau-SB / aDE pipelines.

Used by nested sampling (dynesty), fit_tau_sb_model, compare_models, and JAX likelihood.
"""

from __future__ import annotations

import math
from typing import Any, Literal

PriorMode = Literal["nested", "fit", "mock_recovery"]

# Canonical bounds (dict style — single source of truth)
PRIOR_BOUNDS: dict[str, tuple[float, float]] = {
    "ade_amplitude_frac": (0.0, 0.05),
    "ade_z_star": (0.05, 2.5),
    "ade_omega": (0.5, 4.0),
    "tau_sb_amplitude_frac_nested": (-0.05, 0.05),
    "tau_sb_amplitude_frac_fit": (-0.1, 0.1),
    "hier_frac_nested_center": (0.003, 0.007),
    "hier_frac_nested_halfwidth": (0.0, 0.0005),
    "hier_frac_fit": (-0.05, 0.05),
    "hier_frac_mock_recovery": (-0.1, 0.1),
}

# Back-compat aliases (nested_sampling / scanner imports)
ADE_AOSC_MAX_FRAC: float = PRIOR_BOUNDS["ade_amplitude_frac"][1]
ADE_ZSTAR_MIN: float = PRIOR_BOUNDS["ade_z_star"][0]
ADE_ZSTAR_MAX: float = PRIOR_BOUNDS["ade_z_star"][1]
ADE_OMEGA_MIN: float = PRIOR_BOUNDS["ade_omega"][0]
ADE_OMEGA_MAX: float = PRIOR_BOUNDS["ade_omega"][1]
GEOMETRIC_PRIOR_SIGMA_FRAC: float = 0.01

SPARSE_N_THRESHOLD: int = 24


def ade_amplitude_bounds() -> tuple[float, float]:
    return PRIOR_BOUNDS["ade_amplitude_frac"]


def ade_z_star_bounds() -> tuple[float, float]:
    return PRIOR_BOUNDS["ade_z_star"]


def ade_omega_bounds() -> tuple[float, float]:
    return PRIOR_BOUNDS["ade_omega"]


def tau_amplitude_frac_bounds(*, mode: PriorMode = "fit") -> tuple[float, float]:
    if mode == "nested":
        return PRIOR_BOUNDS["tau_sb_amplitude_frac_nested"]
    return PRIOR_BOUNDS["tau_sb_amplitude_frac_fit"]


def hier_frac_bounds(
    *,
    mode: PriorMode = "fit",
    center: float | None = None,
) -> tuple[float, float]:
    if mode == "mock_recovery":
        if center is not None:
            half = max(0.008, abs(float(center)) * 2.0)
            lo, hi = PRIOR_BOUNDS["hier_frac_mock_recovery"]
            return (
                max(lo, float(center) - half),
                min(hi, float(center) + half),
            )
        return PRIOR_BOUNDS["hier_frac_mock_recovery"]
    if mode == "fit":
        if center is not None:
            lo, hi = PRIOR_BOUNDS["hier_frac_fit"]
            half = max(abs(hi), abs(lo))
            return (float(center) - half, float(center) + half)
        return PRIOR_BOUNDS["hier_frac_fit"]
    # nested: tight window around geometric center
    c_lo, c_hi = PRIOR_BOUNDS["hier_frac_nested_center"]
    hw = PRIOR_BOUNDS["hier_frac_nested_halfwidth"][1]
    hier_center = center if center is not None else 0.5 * (c_lo + c_hi)
    return (hier_center - hw, hier_center + hw)


def bic_penalty_per_model(k_params: int, n_data: int) -> float:
    """BIC complexity penalty ½ k ln(n)."""
    return 0.5 * float(k_params) * math.log(max(int(n_data), 2))


def sparse_occam_penalty(n_data: int, ndim_tau: int, ndim_ade: int) -> float:
    """
    Extra Occam penalty when n is small (prior volume washout guard).

    Penalizes the higher-parameter model by ½Δk·ln(n) for n < SPARSE_N_THRESHOLD.
    """
    if n_data >= SPARSE_N_THRESHOLD or ndim_ade <= ndim_tau:
        return 0.0
    delta_k = float(ndim_ade - ndim_tau)
    return 0.5 * delta_k * math.log(max(int(n_data), 2))


def compute_bic_evidence_delta(
    log_evidence_tau: float,
    log_evidence_ade: float,
    *,
    k_tau: int,
    k_ade: int,
    n_data: int,
) -> dict[str, Any]:
    """Always-on BIC-corrected evidence comparison."""
    pen_tau = bic_penalty_per_model(k_tau, n_data)
    pen_ade = bic_penalty_per_model(k_ade, n_data)
    delta_bic = (float(log_evidence_tau) - pen_tau) - (
        float(log_evidence_ade) - pen_ade
    )
    return {
        "delta_log_evidence_bic": float(delta_bic),
        "favored_model_bic": "Tau-SB" if delta_bic > 0 else "aDE",
        "bic_penalty_tau": pen_tau,
        "bic_penalty_ade": pen_ade,
        "k_tau": int(k_tau),
        "k_ade": int(k_ade),
    }


def apply_evidence_corrections(
    *,
    log_evidence_tau: float | None = None,
    log_evidence_ade: float | None = None,
    delta_log_evidence_tau_minus_ade: float | None = None,
    n_data: int,
    ndim_tau: int,
    ndim_ade: int,
) -> dict[str, Any]:
    """
    Raw ΔlnZ plus BIC (always-on) and sparse Occam (n < 24) diagnostics.
    """
    if delta_log_evidence_tau_minus_ade is not None:
        delta_raw = float(delta_log_evidence_tau_minus_ade)
    elif log_evidence_tau is not None and log_evidence_ade is not None:
        delta_raw = float(log_evidence_tau) - float(log_evidence_ade)
    else:
        raise ValueError(
            "Provide delta_log_evidence_tau_minus_ade or both log evidences"
        )

    if log_evidence_tau is None or log_evidence_ade is None:
        # Reconstruct for BIC from delta: Δ_bic = Δ_raw - ½(k_τ - k_aDE)ln(n)
        logn = math.log(max(int(n_data), 2))
        delta_bic = delta_raw - 0.5 * (int(ndim_tau) - int(ndim_ade)) * logn
        pen_tau = bic_penalty_per_model(ndim_tau, n_data)
        pen_ade = bic_penalty_per_model(ndim_ade, n_data)
        bic = {
            "delta_log_evidence_bic": float(delta_bic),
            "favored_model_bic": "Tau-SB" if delta_bic > 0 else "aDE",
            "bic_penalty_tau": pen_tau,
            "bic_penalty_ade": pen_ade,
            "k_tau": int(ndim_tau),
            "k_ade": int(ndim_ade),
        }
    else:
        bic = compute_bic_evidence_delta(
            log_evidence_tau,
            log_evidence_ade,
            k_tau=ndim_tau,
            k_ade=ndim_ade,
            n_data=n_data,
        )

    sparse_pen = sparse_occam_penalty(n_data, ndim_tau, ndim_ade)
    delta_sparse = delta_raw - sparse_pen

    return {
        "delta_log_evidence_tau_minus_ade": delta_raw,
        **bic,
        "delta_log_evidence_occam_corrected": float(delta_sparse),
        "occam_penalty_tau_minus_ade": sparse_pen,
        "favored_model_occam": "Tau-SB" if delta_sparse > 0 else "aDE",
        "favored_model_sparse": "Tau-SB" if delta_sparse > 0 else "aDE",
        "sparse_n_threshold": SPARSE_N_THRESHOLD,
    }


def apply_occam_evidence_correction(
    delta_log_evidence_tau_minus_ade: float,
    *,
    n_data: int,
    ndim_tau: int,
    ndim_ade: int,
) -> dict[str, Any]:
    """Back-compat wrapper — returns BIC + sparse diagnostics."""
    return apply_evidence_corrections(
        delta_log_evidence_tau_minus_ade=delta_log_evidence_tau_minus_ade,
        n_data=n_data,
        ndim_tau=ndim_tau,
        ndim_ade=ndim_ade,
    )