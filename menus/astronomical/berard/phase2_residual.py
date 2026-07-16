"""
Pre-registered Phase 2 residual analyzer (tau-cosmology PREREGISTRATION.md §7).

Fixed log-periodic frequency 1/ln(7) in ln-radius; no frequency search for claim.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

from menus.astronomical.berard.halo_models import LN7

PHASE2_R0_KPC = 1.0
PHASE2_FREQ = 1.0 / LN7
PHASE2_A_PRIOR_SCALE_KMS = 10.0
PHASE2_LN_B_THRESHOLD = 5.0


def log_periodic_residual(
    r_kpc: np.ndarray,
    amplitude: float,
    phase: float,
    *,
    r0_kpc: float = PHASE2_R0_KPC,
) -> np.ndarray:
    """Δv(r) = A · sin(2π · ln(r/r0) / ln(7) + φ)."""
    r = np.maximum(np.asarray(r_kpc, dtype=float), 1e-6)
    arg = 2.0 * math.pi * np.log(r / float(r0_kpc)) / LN7 + float(phase)
    return float(amplitude) * np.sin(arg)


def free_frequency_sinusoid(
    r_kpc: np.ndarray,
    amplitude: float,
    phase: float,
    freq: float,
    *,
    r0_kpc: float = PHASE2_R0_KPC,
) -> np.ndarray:
    """Free-frequency cross-check sinusoid in ln(r)."""
    r = np.maximum(np.asarray(r_kpc, dtype=float), 1e-6)
    arg = 2.0 * math.pi * float(freq) * np.log(r / float(r0_kpc)) + float(phase)
    return float(amplitude) * np.sin(arg)


def _chi2(
    residual: np.ndarray,
    model: np.ndarray,
    sigma: np.ndarray,
) -> float:
    err = np.maximum(np.asarray(sigma, dtype=float), 1.0)
    return float(np.sum(((residual - model) / err) ** 2))


def _ln_b_approx(delta_chi2: float) -> float:
    """Rough ln Bayes factor from nested Δχ² (Laplace, equal priors)."""
    return 0.5 * float(delta_chi2)


def run_phase2_residual_analysis(
    r_kpc: np.ndarray,
    v_obs: np.ndarray,
    v_model_smooth: np.ndarray,
    e_vobs: np.ndarray,
    *,
    r0_kpc: float = PHASE2_R0_KPC,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Phase 2 on pre-registered fixed-frequency log-periodic residual test.

    Runs only after Phase 1 smooth model is finalized.
    """
    r = np.asarray(r_kpc, dtype=float)
    dv = np.asarray(v_obs, dtype=float) - np.asarray(v_model_smooth, dtype=float)
    sig = np.maximum(np.asarray(e_vobs, dtype=float), 1.0)

    chi2_null = _chi2(dv, np.zeros_like(dv), sig)

    # Fixed 1/ln(7) fit
    def _fixed(rad, amp, phi):
        return log_periodic_residual(rad, amp, phi, r0_kpc=r0_kpc)

    try:
        popt, _ = curve_fit(
            _fixed,
            r,
            dv,
            p0=[5.0, 0.0],
            bounds=([0.0, -2 * math.pi], [30.0, 2 * math.pi]),
            maxfev=20000,
        )
        amp_fixed, phi_fixed = float(popt[0]), float(popt[1])
    except (RuntimeError, ValueError):
        amp_fixed, phi_fixed = 0.0, 0.0

    model_fixed = log_periodic_residual(r, amp_fixed, phi_fixed, r0_kpc=r0_kpc)
    chi2_fixed = _chi2(dv, model_fixed, sig)
    ln_b_fixed = _ln_b_approx(chi2_null - chi2_fixed)

    # Free-frequency cross-check
    def _free(rad, amp, phi, freq):
        return free_frequency_sinusoid(rad, amp, phi, freq, r0_kpc=r0_kpc)

    try:
        pfree, _ = curve_fit(
            _free,
            r,
            dv,
            p0=[5.0, 0.0, PHASE2_FREQ],
            bounds=([0.0, -2 * math.pi, 0.05], [30.0, 2 * math.pi, 0.5]),
            maxfev=30000,
        )
        amp_free, phi_free, freq_free = map(float, pfree)
    except (RuntimeError, ValueError):
        amp_free, phi_free, freq_free = 0.0, 0.0, PHASE2_FREQ

    model_free = free_frequency_sinusoid(r, amp_free, phi_free, freq_free, r0_kpc=r0_kpc)
    chi2_free = _chi2(dv, model_free, sig)

    freq_shift = abs(freq_free - PHASE2_FREQ) / PHASE2_FREQ
    coincidental = freq_shift > 0.25 and chi2_free + 2.0 < chi2_fixed

    if ln_b_fixed > PHASE2_LN_B_THRESHOLD and not coincidental:
        verdict = "PHASE2_DETECTION_CANDIDATE"
    elif ln_b_fixed > 3.0 and not coincidental:
        verdict = "PHASE2_MARGINAL_HINT"
    else:
        verdict = "PHASE2_NULL"

    report: dict[str, Any] = {
        "phase": 2,
        "preregistration": {
            "r0_kpc": float(r0_kpc),
            "frequency_fixed": PHASE2_FREQ,
            "frequency_label": "1/ln(7)",
            "ln_b_threshold": PHASE2_LN_B_THRESHOLD,
        },
        "residual_rms_kms": float(np.sqrt(np.mean(dv**2))),
        "chi2_null": chi2_null,
        "fixed_frequency_fit": {
            "amplitude_kms": amp_fixed,
            "phase_rad": phi_fixed,
            "chi2": chi2_fixed,
            "ln_b_vs_null": ln_b_fixed,
        },
        "free_frequency_crosscheck": {
            "amplitude_kms": amp_free,
            "phase_rad": phi_free,
            "frequency": freq_free,
            "chi2": chi2_free,
            "freq_shift_fraction": freq_shift,
            "rejects_fixed_as_coincidence": coincidental,
        },
        "verdict": verdict,
        "interpretation": (
            "Fixed-frequency detection requires ln B > 5 vs null AND free-frequency "
            "must not prefer a materially different frequency (pre-reg §7)."
        ),
        "upper_limit_amplitude_kms": amp_fixed if verdict == "PHASE2_NULL" else None,
    }

    if verbose:
        print("=" * 60)
        print("PHASE 2 — Pre-registered residual (1/ln 7)")
        print(f"  Residual RMS     : {report['residual_rms_kms']:.2f} km/s")
        print(f"  Fixed A, φ       : {amp_fixed:.2f} km/s, {phi_fixed:.2f} rad")
        print(f"  ln B (fixed)     : {ln_b_fixed:.2f}")
        print(f"  Free freq        : {freq_free:.4f} (shift {100*freq_shift:.1f}%)")
        print(f"  Verdict          : {verdict}")
        print("=" * 60)
    return report