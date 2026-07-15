"""DESI Tau-SB statistical diagnostics: s-grid, bootstrap, null model, γ jackknife."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal

def oversample_s_phase_coherent(
    s: np.ndarray,
    residuals: np.ndarray,
    *,
    n_target: int = 96,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Interpolate (s, residuals) onto a uniform dense s-grid.

    Used when n BAO points is too small for 1/7 Lomb resolution (Nyquist washout).
    Returns (s_dense, y_dense, was_oversampled).
    """
    s = np.asarray(s, dtype=float)
    y = np.asarray(residuals, dtype=float)
    if len(s) >= n_target or len(s) < 3:
        return s, y, False
    order = np.argsort(s)
    s_sorted = s[order]
    y_sorted = y[order]
    s_dense = np.linspace(float(s_sorted[0]), float(s_sorted[-1]), int(n_target))
    y_dense = np.interp(s_dense, s_sorted, y_sorted)
    return s_dense, y_dense, True


def diagnose_frequency_grid(
    s_array: np.ndarray,
    *,
    period: float = 7.0,
    n_cycles_min: float = 0.8,
) -> dict[str, Any]:
    """Check whether s-space span supports a 1/period oscillation search."""
    s = np.asarray(s_array, dtype=float)
    s_min, s_max = float(s.min()), float(s.max())
    s_range = s_max - s_min
    cycles_possible = s_range / period if period > 0 else 0.0
    recommended_n_bins = max(50, int(cycles_possible * 25))

    return {
        "s_min": s_min,
        "s_max": s_max,
        "s_range": s_range,
        "period_s_units": period,
        "cycles_possible": cycles_possible,
        "recommended_n_bins": recommended_n_bins,
        "warning": cycles_possible < n_cycles_min,
        "warning_message": (
            "s-range spans < {:.1f} full 1/{} cycles; periodicity search is underpowered".format(
                n_cycles_min, int(period) if period == int(period) else period
            )
            if cycles_possible < n_cycles_min
            else None
        ),
    }


def bootstrap_lomb_scargle_pvalue(
    s: np.ndarray,
    residual: np.ndarray,
    *,
    freq: float,
    n_bootstrap: int = 2000,
    n_freq: int = 200,
    seed: int = 42,
) -> dict[str, float]:
    """Permutation bootstrap p-value for Lomb-Scargle power at a target frequency."""
    s = np.asarray(s, dtype=float)
    y = np.asarray(residual, dtype=float)
    if len(s) < 3:
        return {"observed_power": 0.0, "bootstrap_p_value": 1.0, "n_bootstrap": 0}

    f_lo = max(0.02, freq * 0.4)
    f_hi = min(0.5, freq * 2.5)
    freqs = np.linspace(f_lo, f_hi, max(n_freq, 50))
    power = signal.lombscargle(s, y, freqs, normalize=True)
    idx = int(np.argmin(np.abs(freqs - freq)))
    observed = float(power[idx])

    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        perm = y[rng.permutation(len(y))]
        boot[i] = float(signal.lombscargle(s, perm, freqs, normalize=True)[idx])

    return {
        "observed_power": observed,
        "bootstrap_p_value": float(np.mean(boot >= observed)),
        "n_bootstrap": float(n_bootstrap),
    }


def _lomb_scargle_power_at_frequency(
    s: np.ndarray,
    y: np.ndarray,
    freq: float,
) -> tuple[float, str]:
    """Lomb-Scargle power at a single frequency (astropy preferred, scipy fallback)."""
    try:
        from astropy.timeseries import LombScargle

        power = float(LombScargle(s, y).power(float(freq)))
        return power, "astropy"
    except ImportError:
        f_lo = max(0.02, float(freq) * 0.4)
        f_hi = min(0.5, float(freq) * 2.5)
        freqs = np.linspace(f_lo, f_hi, 200)
        power_grid = signal.lombscargle(s, y, freqs, normalize=True)
        idx = int(np.argmin(np.abs(freqs - float(freq))))
        return float(power_grid[idx]), "scipy"


def search_higher_harmonics(
    residuals: np.ndarray,
    s: np.ndarray,
    *,
    fundamental_freq: float = 1.0 / 7.0,
    max_harmonic: int = 5,
    n_bootstrap: int = 2000,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Search for power at higher harmonics of the fundamental 1/7 frequency.

    Returns permutation-bootstrap p-values for each harmonic h=1..max_harmonic
    at frequency ``h * fundamental_freq`` on the (s, residuals) grid.

    Per-harmonic entries live in ``result["harmonics"][h]`` with keys
    ``frequency``, ``observed_power``, and ``bootstrap_p_value``.
    """
    s = np.asarray(s, dtype=float)
    resid = np.asarray(residuals, dtype=float)
    if len(s) != len(resid):
        raise ValueError(f"s length {len(s)} != residuals length {len(resid)}")
    if len(resid) < 3:
        return {
            "fundamental_freq": float(fundamental_freq),
            "max_harmonic": int(max_harmonic),
            "n_bootstrap": int(n_bootstrap),
            "harmonics": {},
            "any_significant_at_5pct": False,
            "note": f"n={len(resid)} too few for Lomb-Scargle (need n≥3)",
        }

    fundamental = float(fundamental_freq)
    rng = np.random.default_rng(seed)
    backend = "unknown"
    harmonics: dict[int, dict[str, float]] = {}

    for h in range(1, max_harmonic + 1):
        freq = h * fundamental
        observed_power, backend = _lomb_scargle_power_at_frequency(s, resid, freq)

        bootstrap_powers = np.empty(n_bootstrap, dtype=float)
        for i in range(n_bootstrap):
            y_perm = resid[rng.permutation(len(resid))]
            bootstrap_powers[i], _ = _lomb_scargle_power_at_frequency(s, y_perm, freq)

        p_value = float(np.mean(bootstrap_powers >= observed_power))
        harmonics[h] = {
            "frequency": float(freq),
            "observed_power": float(observed_power),
            "bootstrap_p_value": p_value,
            "significant_at_5pct": p_value < 0.05,
        }
        if verbose:
            print(
                f"Harmonic {h} (f={freq:.4f}): "
                f"Power = {observed_power:.2e}, Bootstrap p = {p_value:.4f}"
            )

    any_sig = any(entry["significant_at_5pct"] for entry in harmonics.values())
    return {
        "fundamental_freq": fundamental,
        "max_harmonic": int(max_harmonic),
        "n_bootstrap": int(n_bootstrap),
        "lomb_scargle_backend": backend,
        "harmonics": harmonics,
        "any_significant_at_5pct": bool(any_sig),
    }


def _bootstrap_lomb_pvalue_at_freq(
    s: np.ndarray,
    resid: np.ndarray,
    freq: float,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    """Permutation-bootstrap Lomb-Scargle p-value at a single target frequency."""
    observed_power, _ = _lomb_scargle_power_at_frequency(s, resid, freq)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        y_perm = resid[rng.permutation(len(resid))]
        boot[i], _ = _lomb_scargle_power_at_frequency(s, y_perm, freq)
    return {
        "frequency": float(freq),
        "observed_power": float(observed_power),
        "bootstrap_p_value": float(np.mean(boot >= observed_power)),
    }


def search_whim_web_signatures(
    residuals: np.ndarray,
    s: np.ndarray,
    n_bootstrap: int = 2000,
    *,
    seed: int = 42,
    gamma: float | None = None,
    web_scale_mpc: float | None = None,
    s_reference: float = 50.0,
    tau_period: float = 7.0,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Search for 1D/web-like correlations or scale features in residuals.

    Implements the sketch workflow (Pearson corr along ``s`` + targeted
    Lomb-Scargle at web scales) and extends it with bootstrap p-values,
    a :func:`~tau_sb_desi_scanner.whim_web_correction` template correlation,
    and data-driven BEC-well frequencies from the observed ``s`` span.

    The original placeholder ``web_freq = 1 / (web_scale / s_reference)`` is
    recorded in ``web_freq_placeholder``; prefer ``lomb_scargle["bec_well"]``
    (``f ≈ 1 / s_span``) on finite BAO vectors.
    """
    from scipy.stats import pearsonr, spearmanr

    s_arr = np.asarray(s, dtype=float)
    resid = np.asarray(residuals, dtype=float)
    if len(s_arr) != len(resid):
        raise ValueError(f"s length {len(s_arr)} != residuals length {len(resid)}")

    n = len(resid)
    result: dict[str, Any] = {
        "n_data": int(n),
        "n_bootstrap": int(n_bootstrap),
        "s_span": float(np.max(s_arr) - np.min(s_arr)) if n else 0.0,
    }

    if n < 3:
        result["note"] = f"n={n} too few for WHIM/web signature search (need n≥3)"
        if verbose:
            print(f"[WHIM/Web Signature] {result['note']}")
        return result

    try:
        from menus.astronomical.desi.scanner import (
            BEC_COHERENCE_LENGTH_MPC,
            TSB_RD_GAMMA_FALLBACK,
            whim_web_correction,
        )

        bec_mpc = float(web_scale_mpc if web_scale_mpc is not None else BEC_COHERENCE_LENGTH_MPC)
        gamma_used = float(gamma if gamma is not None else TSB_RD_GAMMA_FALLBACK)
    except ImportError:
        bec_mpc = float(web_scale_mpc if web_scale_mpc is not None else 5.5)
        gamma_used = float(gamma if gamma is not None else 8.8511)
        whim_web_correction = None  # type: ignore[assignment,misc]

    result["web_scale_mpc"] = bec_mpc
    result["gamma_used"] = gamma_used
    s_ref = float(s_reference)
    f_placeholder = (1.0 / (bec_mpc / s_ref)) if bec_mpc > 0 and s_ref > 0 else float("nan")
    result["web_freq_placeholder"] = {
        "web_scale_mpc": bec_mpc,
        "s_reference": s_ref,
        "frequency": float(f_placeholder),
        "note": "Sketch formula 1/(L_BEC/s_ref); use bec_well (1/s_span) on real data",
    }

    corr_s, p_s = pearsonr(s_arr, resid)
    result["pearson_s"] = {
        "correlation": float(corr_s),
        "p_value": float(p_s),
        "significant_at_5pct": float(p_s) < 0.05,
    }
    if verbose:
        print(
            f"[WHIM/Web Signature] Pearson corr(s, residuals) = {corr_s:.4f} (p={p_s:.4f})"
        )

    try:
        rho, p_rho = spearmanr(s_arr, resid)
        result["spearman_s"] = {
            "correlation": float(rho),
            "p_value": float(p_rho),
            "significant_at_5pct": float(p_rho) < 0.05,
        }
        if verbose:
            print(
                f"[WHIM/Web Signature] Spearman corr(s, residuals) = {rho:.4f} (p={p_rho:.4f})"
            )
    except Exception:
        result["spearman_s"] = None

    if whim_web_correction is not None:
        template = whim_web_correction(s_arr, gamma_used)
        corr_t, p_t = pearsonr(template, resid)
        result["pearson_whim_template"] = {
            "correlation": float(corr_t),
            "p_value": float(p_t),
            "significant_at_5pct": float(p_t) < 0.05,
        }
        if verbose:
            print(
                f"[WHIM/Web Signature] Pearson corr(whim_template, residuals) = "
                f"{corr_t:.4f} (p={p_t:.4f})"
            )

    s_span = float(result["s_span"])
    f_filament = 1.0 / (2.0 * float(tau_period))
    f_bec = (1.0 / s_span) if s_span > 0 else float("nan")
    f_bec_h2 = (2.0 / s_span) if s_span > 0 else float("nan")

    targets = {
        "filament_2tau": f_filament,
        "bec_well": f_bec,
        "bec_well_h2": f_bec_h2,
    }
    lomb: dict[str, dict[str, float]] = {}
    backend = "unknown"
    seed_offsets = {"filament_2tau": 0, "bec_well": 1, "bec_well_h2": 2}
    for name, freq in targets.items():
        if not np.isfinite(freq) or freq <= 0:
            continue
        boot = _bootstrap_lomb_pvalue_at_freq(
            s_arr,
            resid,
            freq,
            n_bootstrap=n_bootstrap,
            seed=seed + seed_offsets.get(name, 0),
        )
        boot["significant_at_5pct"] = boot["bootstrap_p_value"] < 0.05
        lomb[name] = boot
        _, backend = _lomb_scargle_power_at_frequency(s_arr, resid, freq)
        if verbose:
            print(
                f"[WHIM/Web Lomb] {name} (f={freq:.4f}): "
                f"Power = {boot['observed_power']:.2e}, "
                f"Bootstrap p = {boot['bootstrap_p_value']:.4f}"
            )

    result["target_frequencies"] = targets
    result["lomb_scargle"] = lomb
    result["lomb_scargle_backend"] = backend
    result["any_lomb_significant_at_5pct"] = any(
        v.get("significant_at_5pct") for v in lomb.values()
    )

    if verbose:
        if not lomb:
            print("[WHIM/Web] No valid target frequencies for Lomb-Scargle search.")
        else:
            print(
                f"[WHIM/Web] Placeholder f (1/(L_BEC/s_ref)) = {f_placeholder:.4f} "
                f"(s_ref={s_ref:.1f}); data-driven bec_well f = {f_bec:.4f}"
            )

    return result


def estimate_tsb_percolation(
    n_data: int,
    *,
    tau_phases: int = 7,
    effective_dim: float = 2.5,
    p_c_2d: float = 0.59,
) -> dict[str, Any]:
    """
    Simplified 2.5D percolation estimate for Tau-cylinder bubble filling.

    Uses ``n_effective_sites ≈ n_data × tau_phases`` and a rough
    ``p_c ≈ p_c,2D / (d_eff - 1)`` scaling for a 2.5D-like lattice.
    """
    n = max(int(n_data), 0)
    n_sites = n * int(tau_phases)
    denom = float(effective_dim) - 1.0
    p_c_approx = float(p_c_2d / denom) if denom > 0 else float("nan")
    return {
        "estimated_p_c": p_c_approx,
        "n_effective_sites": int(n_sites),
        "n_data": int(n),
        "tau_phases": int(tau_phases),
        "effective_dim": float(effective_dim),
        "p_c_2d_reference": float(p_c_2d),
        "note": "Simplified estimate from 2.5D bubble percolation on Tau Cylinder",
    }


def fit_null_model(y: np.ndarray, cov: np.ndarray) -> tuple[float, float]:
    """Weighted-mean null model χ² using full covariance."""
    y = np.asarray(y, dtype=float)
    cov = np.asarray(cov, dtype=float)
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        from menus.astronomical.desi.analysis import apply_tikhonov_regularization

        cov, _ = apply_tikhonov_regularization(cov)
        cov_inv = np.linalg.inv(cov)
    weights = np.diag(cov_inv)
    null_value = float(np.average(y, weights=weights))
    resid = y - null_value
    chi2_null = float(resid @ cov_inv @ resid)
    return null_value, chi2_null


def compute_model_comparison(
    chi2_model: float,
    chi2_null: float,
    *,
    n_data: int,
    n_params_model: int,
) -> dict[str, float]:
    """ΔAIC/ΔBIC vs constant null; includes reduced χ²."""
    k_model = n_params_model
    k_null = 1
    aic_model = chi2_model + 2 * k_model
    aic_null = chi2_null + 2 * k_null
    bic_model = chi2_model + k_model * np.log(max(n_data, 1))
    bic_null = chi2_null + k_null * np.log(max(n_data, 1))
    dof = max(n_data - k_model, 1)
    return {
        "chi2_model": float(chi2_model),
        "chi2_null": float(chi2_null),
        "delta_chi2_vs_null": float(chi2_null - chi2_model),
        "delta_aic_vs_null": float(aic_model - aic_null),
        "delta_bic_vs_null": float(bic_model - bic_null),
        "reduced_chi2_model": float(chi2_model / dof),
        "n_data": float(n_data),
        "n_params_model": float(k_model),
        "dof": float(n_data - k_model),
    }


def gamma_jackknife_std(
    z: np.ndarray,
    *,
    n_hier: float,
    delta_n: float,
    period: float,
    calibrate_fn,
) -> dict[str, float]:
    """Leave-one-out spread on auto-calibrated γ."""
    z = np.asarray(z, dtype=float)
    if len(z) < 3:
        return {"gamma_median": float(calibrate_fn(z)), "gamma_std": 0.0, "n_jackknife": 0}

    gammas = []
    for i in range(len(z)):
        mask = np.ones(len(z), dtype=bool)
        mask[i] = False
        z_sub = z[mask]
        gammas.append(float(calibrate_fn(z_sub)))
    g = np.array(gammas, dtype=float)
    return {
        "gamma_median": float(np.median(g)),
        "gamma_std": float(np.std(g, ddof=1)) if len(g) > 1 else 0.0,
        "n_jackknife": float(len(g)),
    }


def tav_harmonic_check_robust(
    s: np.ndarray,
    residuals: np.ndarray,
    *,
    period: float = 7.0,
    phase_tolerance: float = 0.05,
    min_peaks: int = 1,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Lomb-Scargle + bootstrap 1/period check on (s, residuals).

    ``tav_resonance.analyze_tav_harmonics`` needs n≥8; for smaller n we still
    run Lomb-Scargle and report ``n_freq_bins`` from the search grid.
    """
    s = np.asarray(s, dtype=float)
    resid = np.asarray(residuals, dtype=float)
    n = len(resid)
    diag = diagnose_frequency_grid(s, period=period)
    expected_f = 1.0 / period

    s_lomb, resid_lomb, oversampled = oversample_s_phase_coherent(
        s, resid, n_target=max(96, int(diag["recommended_n_bins"]))
    )
    if oversampled and n < 12:
        diag = diagnose_frequency_grid(s_lomb, period=period)

    boot = bootstrap_lomb_scargle_pvalue(
        s_lomb, resid_lomb, freq=expected_f, n_bootstrap=n_bootstrap, seed=seed
    )

    result: dict[str, Any] = {
        "method": "lomb_scargle_bootstrap",
        "n_data": n,
        "n_lomb_grid": int(len(s_lomb)),
        "s_oversampled": oversampled,
        "n_freq_bins": int(max(50, diag["recommended_n_bins"])),
        "cycles_possible": diag["cycles_possible"],
        "s_grid_warning": diag["warning"],
        "s_grid_warning_message": diag.get("warning_message"),
        "power_at_expected": boot["observed_power"],
        "bootstrap_p_value": boot["bootstrap_p_value"],
        "lomb_detected": boot["bootstrap_p_value"] < 0.05,
        "harmonic_peaks": [],
        "tav_resonance_detected": False,
        "tav_resonance_skipped": n < 8,
        "tav_skip_reason": None,
    }

    if n < 8:
        result["tav_skip_reason"] = (
            f"tav_resonance periodogram requires n≥8 points (have n={n}); "
            "using Lomb-Scargle bootstrap only"
        )
        result["underpowered_warning"] = (
            f"n={n} BAO points: Lomb-Scargle and model comparison are underpowered; "
            "prefer bootstrap p-values over asymptotic approximations"
        )
        return result

    try:
        from tav_shared.tav_resonance_bootstrap import ensure_tav_resonance_importable

        ensure_tav_resonance_importable()
        from tav_resonance import analyze_tav_harmonics

        freqs_t, _power_t, peaks, detected = analyze_tav_harmonics(
            resid,
            phase_tolerance=phase_tolerance,
            min_peaks=min_peaks,
        )
        result["method"] = "lomb_scargle_bootstrap+tav_periodogram"
        result["tav_resonance_detected"] = bool(detected)
        result["harmonic_peaks"] = peaks
        result["tav_n_freq_bins"] = int(freqs_t.size)
        result["tav_resonance_skipped"] = False
        result["tav_skip_reason"] = None
    except ImportError:
        result["tav_skip_reason"] = "tav_resonance package not installed"
    except Exception as exc:
        result["tav_skip_reason"] = f"tav_resonance error: {exc}"

    return result


def assess_scan_power(
    *,
    n_data: int,
    cycles_possible: float,
    auto_calibrate_gamma: bool,
    n_params: int | None = None,
    tav_skipped: bool = False,
    min_n_recommended: int = 12,
) -> dict[str, Any]:
    """
    Summarize statistical power limitations for a DESI Tau-SB scan.

    Returns a list of issues with severity: critical | high | medium.
    """
    issues: list[dict[str, str]] = []

    if n_data < 8:
        issues.append(
            {
                "issue": "n_data",
                "severity": "critical",
                "message": (
                    f"Only n={n_data} data points — fundamentally underpowered; "
                    "stay on DH_only/DM_only (do not mix channels); use batch tracers"
                ),
            }
        )
    elif n_data < min_n_recommended:
        issues.append(
            {
                "issue": "n_data",
                "severity": "high",
                "message": (
                    f"n={n_data} is below recommended minimum n≥{min_n_recommended} "
                    "for periodicity / model comparison"
                ),
            }
        )

    if auto_calibrate_gamma and 0.9 <= cycles_possible <= 1.1:
        issues.append(
            {
                "issue": "s_space_cycles",
                "severity": "critical",
                "message": (
                    f"Δs/τ = {cycles_possible:.2f} with auto-calibrated γ — s-range spans "
                    "~1τ period by construction; Lomb 1/7 test on auto-γ coordinates is "
                    "circular (see fixed-γ diagnostic)"
                ),
            }
        )
    elif cycles_possible < 1.5:
        issues.append(
            {
                "issue": "s_space_cycles",
                "severity": "high",
                "message": (
                    f"Only {cycles_possible:.2f} full τ cycles in s-space; "
                    "periodicity tests have low resolving power"
                ),
            }
        )

    if tav_skipped:
        issues.append(
            {
                "issue": "tav_resonance",
                "severity": "high",
                "message": "tav_resonance periodogram skipped (requires n≥8)",
            }
        )

    if n_params is not None:
        dof = n_data - n_params
        if dof <= 0:
            issues.append(
                {
                    "issue": "model_complexity",
                    "severity": "critical",
                    "message": (
                        f"Model has {n_params} free parameters for n={n_data} "
                        f"(dof={dof}) — over-fitted or saturated"
                    ),
                }
            )
        elif dof < 3:
            issues.append(
                {
                    "issue": "model_complexity",
                    "severity": "medium",
                    "message": (
                        f"Only dof={dof} ({n_params} params, n={n_data}) — "
                        "uncertainties on oscillation amplitude are poorly constrained"
                    ),
                },
            )

    rank = {"critical": 3, "high": 2, "medium": 1, "ok": 0}
    overall = "ok"
    if issues:
        overall = max(issues, key=lambda row: rank[row["severity"]])["severity"]

    return {
        "issues": issues,
        "overall_severity": overall,
        "n_data": n_data,
        "cycles_possible": cycles_possible,
        "recommendation": (
            "Keep DATA_MODE=DH_only or DM_only (6 pts each); do not mix channels to "
            "inflate n. Auto-γ Lomb at ~1τ cycle is circular — trust fixed-γ "
            "diagnostics, legacy BOSS/eBOSS stack, and batch tracers. "
            "Run Model Compare separately."
            if overall in {"critical", "high"}
            else "Statistical power adequate for exploratory scan."
        ),
    }


def build_recommended_next_steps(
    *,
    power_assessment: dict[str, Any] | None,
    fit: dict[str, Any] | None = None,
    auto_diagnostics: dict[str, Any] | None = None,
    residual_diagnostics: dict[str, Any] | None = None,
    n_data: int = 0,
    include_legacy_bao: bool = False,
    fit_phase: bool = True,
    fit_hier: bool = True,
    min_n_target: int = 15,
) -> list[str]:
    """
    Actionable checklist after a DESI Tau-SB scan (Priorities 1 & 2).
    """
    steps: list[str] = []
    severity = (power_assessment or {}).get("overall_severity", "unknown")

    if n_data < min_n_target:
        if not include_legacy_bao:
            steps.append(
                f"Priority 1 — Increase n: enable include_legacy_bao=yes "
                f"(DESI+BOSS/eBOSS stack; current n={n_data}, target n≥{min_n_target})"
            )
        else:
            steps.append(
                f"Priority 1 — n={n_data} still below target {min_n_target}; "
                "add more legacy surveys or run batch tracers for per-tracer checks"
            )
        steps.append(
            "Priority 1 — Do not use Joint_DH_DM or quantity=all to inflate n "
            "(mixed channels break single-template fits)"
        )

    if fit and int(fit.get("dof", 0)) < 3:
        if fit_phase:
            steps.append(
                "Priority 1 — Reduce parameters: retry with fit_phase=no "
                f"(current dof={fit.get('dof')})"
            )
        if fit_hier and n_data < 12:
            steps.append(
                "Priority 1 — Reduce parameters: set fit_hier=no on small vectors "
                f"(n={n_data})"
            )

    if severity in {"critical", "high"}:
        steps.append(
            "Priority 1 — Trust fixed-γ non-circular p-values and γ jackknife; "
            "ignore auto-γ Lomb at ~1τ cycle (circular by construction)"
        )

    diag_done = bool(auto_diagnostics)
    inj_done = bool((auto_diagnostics or {}).get("inj_results"))
    resid_done = bool(residual_diagnostics)
    if not diag_done:
        steps.append(
            "Priority 2 — Run auto diagnostics: run_auto_diagnostics(..., z=meta['z']) "
            "with auto_calibrate_gamma=True"
        )
    elif not inj_done:
        steps.append(
            "Priority 2 — Run injection-recovery preview inside auto diagnostics "
            "(run_injection=True)"
        )
    if not resid_done:
        steps.append(
            "Priority 2 — Run full residual diagnostics after Tau-SB fit "
            "(run_full_residual_diagnostics on fit residuals)"
        )
    else:
        shapiro = (residual_diagnostics or {}).get("shapiro_wilk") or {}
        dw = float((residual_diagnostics or {}).get("durbin_watson_stat", float("nan")))
        if not shapiro.get("normal", shapiro.get("is_normal_at_5pct")):
            steps.append(
                "Priority 2 — Residuals fail normality (Shapiro-Wilk); "
                "inspect Q-Q plot and consider fewer free parameters"
            )
        if np.isfinite(dw) and (dw < 1.5 or dw > 2.5):
            steps.append(
                f"Priority 2 — Durbin-Watson={dw:.2f} suggests residual autocorrelation; "
                "check ACF / Breusch-Godfrey output"
            )

    if not steps:
        steps.append(
            "Power and core diagnostics look adequate — proceed to Model Compare "
            "and optional MCMC if publishing"
        )
    return steps