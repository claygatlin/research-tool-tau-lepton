"""
DESI Tau-SB analysis utilities (v2): covariance health and non-circular γ diagnostics.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy import stats
from scipy.optimize import minimize

COV_COND_THRESHOLD: float = 1e10
TIKHONOV_EPS: float = 1e-8


def check_covariance_health(cov: np.ndarray, *, name: str = "") -> dict[str, Any]:
    """
    Diagnose DESI Gaussian covariance matrix before χ² fits.

    Returns condition number, eigenvalue span, symmetry error, and warnings.
    """
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be square 2D, got shape {cov.shape}")

    n = int(cov.shape[0])
    sym_err = float(np.max(np.abs(cov - cov.T))) if n else 0.0
    trace = float(np.trace(cov))
    diag = np.diag(cov) if n else np.array([])
    min_diag = float(np.min(diag)) if n else float("nan")

    warnings: list[str] = []
    if sym_err > 1e-6 * max(np.max(np.abs(cov)), 1.0):
        warnings.append(f"asymmetry max|C-C.T|={sym_err:.2e}")

    if n == 0:
        return {
            "name": name,
            "n": 0,
            "trace": 0.0,
            "condition_number": float("inf"),
            "min_eigenvalue": float("nan"),
            "max_eigenvalue": float("nan"),
            "min_diagonal": float("nan"),
            "symmetry_error": sym_err,
            "is_positive_definite": False,
            "warnings": ["empty covariance matrix"],
            "healthy": False,
        }

    try:
        eigs = np.linalg.eigvalsh((cov + cov.T) * 0.5)
        min_eig = float(eigs.min())
        max_eig = float(eigs.max())
        cond = float(np.linalg.cond(cov))
    except np.linalg.LinAlgError as exc:
        warnings.append(f"eigen-analysis failed: {exc}")
        min_eig = float("nan")
        max_eig = float("nan")
        cond = float("inf")

    if min_diag <= 0:
        warnings.append(f"non-positive diagonal entry (min={min_diag:.2e})")
    if min_eig <= 0:
        warnings.append(f"not positive definite (λ_min={min_eig:.2e})")
    if cond > COV_COND_THRESHOLD:
        warnings.append(f"ill-conditioned (κ={cond:.2e} > {COV_COND_THRESHOLD:.0e})")

    healthy = not warnings
    return {
        "name": name,
        "n": n,
        "trace": trace,
        "condition_number": cond,
        "min_eigenvalue": min_eig,
        "max_eigenvalue": max_eig,
        "min_diagonal": min_diag,
        "symmetry_error": sym_err,
        "is_positive_definite": min_eig > 0 if np.isfinite(min_eig) else False,
        "warnings": warnings,
        "healthy": healthy,
    }


def apply_tikhonov_regularization(
    cov: np.ndarray,
    *,
    eps: float = TIKHONOV_EPS,
) -> tuple[np.ndarray, float]:
    """
    Light Tikhonov jitter: C_reg = C + ε (tr C / n) I.
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    if n == 0:
        return cov.copy(), 0.0
    reg_strength = float(eps * np.trace(cov) / n)
    cov_reg = cov + reg_strength * np.eye(n)
    return cov_reg, reg_strength


def prepare_covariance(
    cov: np.ndarray,
    *,
    name: str = "",
    cond_threshold: float = COV_COND_THRESHOLD,
    eps: float = TIKHONOV_EPS,
    verbose: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Check covariance health and apply Tikhonov regularization only when κ is high.
    """
    health = check_covariance_health(cov, name=name)
    cov_out = np.asarray(cov, dtype=float).copy()
    reg_applied = False
    reg_strength = 0.0

    cond = float(health.get("condition_number", 0.0))
    if cond > cond_threshold or not health.get("is_positive_definite", True):
        if verbose:
            print("Applying light Tikhonov regularization to covariance...")
        cov_out, reg_strength = apply_tikhonov_regularization(cov_out, eps=eps)
        reg_applied = True
        if verbose:
            print(f"Regularization strength: {reg_strength:.2e}")

    health["regularization_applied"] = reg_applied
    health["regularization_strength"] = reg_strength
    health["cond_threshold"] = float(cond_threshold)
    if reg_applied:
        post = check_covariance_health(cov_out, name=name)
        health["condition_number_after"] = post["condition_number"]
        health["min_eigenvalue_after"] = post["min_eigenvalue"]

    return cov_out, health


# Tav multi-domain drift + topological friction (session 2026-07-09)
AUGMENTED_COV_DRIFT_SCALE: float = 0.01
AUGMENTED_COV_FRICTION_SCALE: float = 0.005
AUGMENTED_COV_Z_KERNEL_SIGMA: float = 0.3
FRICTION_P_EXPONENT: float = -2.5
DELTA_GAMMA_DOMAIN: float = 1.12


def build_augmented_cov(
    base_cov: np.ndarray,
    z: np.ndarray,
    *,
    delta_gamma: float = DELTA_GAMMA_DOMAIN,
    friction_exponent: float = FRICTION_P_EXPONENT,
    drift_scale: float = AUGMENTED_COV_DRIFT_SCALE,
    friction_scale: float = AUGMENTED_COV_FRICTION_SCALE,
    z_kernel_sigma: float = AUGMENTED_COV_Z_KERNEL_SIGMA,
) -> np.ndarray:
    """
    Add theory-motivated drift + topological friction to the published BAO covariance.

    Implements the additive block from the 2026-07-09 improvement session:
    secular proper-time domain drift (Delta_gamma) and P(k) ~ k^friction_exponent.
    """
    cov = np.asarray(base_cov, dtype=float)
    z_arr = np.asarray(z, dtype=float).ravel()
    n = len(z_arr)
    if cov.shape != (n, n):
        raise ValueError(f"base_cov {cov.shape} incompatible with n={n} redshifts")
    extra = np.zeros((n, n), dtype=float)
    inv_sigma2 = 1.0 / max(z_kernel_sigma**2, 1e-12)
    for i in range(n):
        for j in range(n):
            dz = z_arr[i] - z_arr[j]
            drift = drift_scale * delta_gamma * np.exp(-0.5 * dz * dz * inv_sigma2)
            k_eff = 2 * np.pi / max(z_arr[i] + z_arr[j], 1e-6)
            friction = friction_scale * (k_eff**friction_exponent)
            extra[i, j] = drift + friction
    return cov + extra


def apply_augmented_covariance(
    cov: np.ndarray,
    z: np.ndarray,
    *,
    name: str = "",
    **aug_kw: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Augment covariance, re-run health checks, and return metadata."""
    augmented = build_augmented_cov(cov, z, **aug_kw)
    cov_out, health = prepare_covariance(augmented, name=name or "augmented_bao")
    meta = {
        "augmented": True,
        "delta_gamma": aug_kw.get("delta_gamma", DELTA_GAMMA_DOMAIN),
        "friction_exponent": aug_kw.get("friction_exponent", FRICTION_P_EXPONENT),
        "base_condition_number": health.get("condition_number"),
    }
    return cov_out, meta


def _call_model_func(
    model_func: Callable[..., np.ndarray],
    *,
    y: np.ndarray,
    z: np.ndarray | None,
    cov: np.ndarray | None,
    err: np.ndarray | None,
) -> np.ndarray:
    """Invoke ``model_func`` with the richest signature it accepts."""
    for kwargs in (
        {"y": y, "z": z, "cov": cov, "err": err},
        {"y": y, "cov": cov, "err": err},
        {"y": y, "cov": cov},
        {"y": y},
        {},
    ):
        try:
            out = model_func(**kwargs) if kwargs else model_func()
            return np.asarray(out, dtype=float)
        except TypeError:
            continue
    raise TypeError("model_func signature not supported; expected y= and optional z/cov/err")


def run_non_circular_gamma_diagnostic(
    *,
    y: np.ndarray,
    fixed_gamma: float = 10.0,
    period: float = 7.0,
    s: np.ndarray | None = None,
    z: np.ndarray | None = None,
    cov: np.ndarray | None = None,
    err: np.ndarray | None = None,
    model_func: Callable[..., np.ndarray] | None = None,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Lomb–Scargle 1/τ test on **fixed-γ** s coordinates (non-circular vs auto-γ).

    Parameters
    ----------
    y:
        Data vector (e.g. DESI BAO observables).
    s:
        Pre-computed s-grid; if omitted, built from ``z`` and ``fixed_gamma``.
    z:
        Redshift array (required when ``s`` is omitted or for poly baseline).
    cov, err:
        Passed through to ``model_func`` when fitting the subtraction model.
    model_func:
        Callable returning a model prediction vector (same length as ``y``).
        When omitted, a polynomial baseline in ``z`` is used.
    fixed_gamma:
        Cylinder stretch γ for s(z); default 10 avoids auto-γ circularity.
    """
    from menus.astronomical.desi.stats import bootstrap_lomb_scargle_pvalue, diagnose_frequency_grid

    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return {
            "fixed_gamma": float(fixed_gamma),
            "gamma": float(fixed_gamma),
            "period": float(period),
            "bootstrap_p_1_7": 1.0,
            "bootstrap_p_value": 1.0,
            "power_at_expected": 0.0,
            "cycles_possible": 0.0,
            "s_range": 0.0,
            "residual_source": "insufficient_data",
            "note": f"n={n} too few for Lomb–Scargle (need n≥3)",
            "n_data": n,
        }

    if s is None:
        if z is None:
            raise ValueError("run_non_circular_gamma_diagnostic requires s or z")
        from menus.astronomical.desi.scanner import s_from_z

        s_arr = s_from_z(np.asarray(z, dtype=float), gamma=fixed_gamma)
    else:
        s_arr = np.asarray(s, dtype=float)
        if len(s_arr) != n:
            raise ValueError(f"s length {len(s_arr)} != y length {n}")

    z_arr = np.asarray(z, dtype=float) if z is not None else None

    if model_func is not None:
        model_pred = _call_model_func(model_func, y=y, z=z_arr, cov=cov, err=err)
        if len(model_pred) != n:
            raise ValueError(f"model_func returned length {len(model_pred)}, expected {n}")
        residuals = y - model_pred
        residual_source = "model_func"
    else:
        if z_arr is None:
            raise ValueError("z required when model_func is None (poly baseline)")
        deg = min(3, max(1, n - 4))
        coeffs = np.polyfit(z_arr, y, deg=deg)
        model_pred = np.polyval(coeffs, z_arr)
        residuals = y - model_pred
        residual_source = "poly_baseline"

    expected_f = 1.0 / period
    grid = diagnose_frequency_grid(s_arr, period=period)
    boot = bootstrap_lomb_scargle_pvalue(
        s_arr,
        residuals,
        freq=expected_f,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    p_boot = float(boot["bootstrap_p_value"])

    return {
        "fixed_gamma": float(fixed_gamma),
        "gamma": float(fixed_gamma),
        "period": float(period),
        "expected_f": float(expected_f),
        "bootstrap_p_1_7": p_boot,
        "bootstrap_p_value": p_boot,
        "power_at_expected": float(boot["observed_power"]),
        "cycles_possible": float(grid["cycles_possible"]),
        "s_range": float(grid["s_range"]),
        "s_min": float(grid["s_min"]),
        "s_max": float(grid["s_max"]),
        "residual_source": residual_source,
        "n_bootstrap": int(boot.get("n_bootstrap", n_bootstrap)),
        "n_data": n,
        "note": "Independent of auto-γ calibration (non-circular periodicity check)",
    }


def make_default_gamma_model_func(
    z: np.ndarray,
    y: np.ndarray,
    *,
    period: float = 7.0,
) -> Callable[[np.ndarray, float, float], np.ndarray]:
    """Poly baseline + τ oscillation with γ-dependent s remap."""
    from menus.astronomical.desi.scanner import s_from_z

    z_arr = np.asarray(z, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    deg = min(3, max(1, len(z_arr) - 4))
    coeffs = np.polyfit(z_arr, y_arr, deg=deg)
    y_base = np.polyval(coeffs, z_arr)

    def model_func(_s: np.ndarray, a_osc: float, gamma: float) -> np.ndarray:
        ss = s_from_z(z_arr, gamma=gamma)
        return y_base + a_osc * np.sin(2 * np.pi * ss / period)

    return model_func


def diagnose_gamma_sensitivity(
    s: np.ndarray,
    y: np.ndarray,
    cov: np.ndarray,
    model_func: Callable[[np.ndarray, float, float], np.ndarray],
    *,
    gamma_range: np.ndarray | Sequence[float] | None = None,
    period: float = 7.0,
    verbose: bool = True,
    as_tuple: bool = False,
) -> dict[str, Any] | tuple[float, np.ndarray]:
    """
    Test how χ² changes with fixed γ values (amplitude refit at each γ).

    ``model_func(s, A_osc, gamma)`` must return a prediction vector matching ``y``.
    """
    from menus.astronomical.desi.scanner import gaussian_chi2

    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if gamma_range is None:
        gamma_range = np.linspace(5.0, 15.0, 21)
    gamma_arr = np.asarray(gamma_range, dtype=float)

    if verbose:
        print("\n=== GAMMA SENSITIVITY DIAGNOSTIC ===")

    chi2_list: list[float] = []
    for g in gamma_arr:
        g_val = float(g)

        def loss(a_vec: np.ndarray, _g: float = g_val) -> float:
            pred = model_func(s, float(a_vec[0]), _g)
            return gaussian_chi2(y, pred, cov=cov)

        res = minimize(loss, x0=np.array([0.0], dtype=float), method="Nelder-Mead")
        chi2_list.append(float(res.fun))

    chi2_arr = np.asarray(chi2_list, dtype=float)
    best_idx = int(np.argmin(chi2_arr))
    best_gamma = float(gamma_arr[best_idx])
    best_chi2 = float(chi2_arr[best_idx])

    if verbose:
        print(f"Best fixed γ: {best_gamma:.3f} → χ² = {best_chi2:.1f}")

    if as_tuple:
        return best_gamma, chi2_arr

    return {
        "gamma_range": gamma_arr,
        "chi2_list": chi2_arr,
        "best_gamma": best_gamma,
        "best_chi2": best_chi2,
        "best_index": best_idx,
        "period": float(period),
        "n_gamma": len(gamma_arr),
    }


def check_gamma_jackknife_variance(
    s: np.ndarray,
    y: np.ndarray,
    cov: np.ndarray,
    auto_calibrate_func: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Leave-one-out spread on an auto-calibration routine ``γ = f(s, y, cov)``.
    """
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = len(y)

    if verbose:
        print("\n=== GAMMA JACKKNIFE VARIANCE CHECK ===")

    jackknife_gammas: list[float] = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        try:
            g_jack = float(
                auto_calibrate_func(
                    s[mask],
                    y[mask],
                    cov[np.ix_(mask, mask)],
                )
            )
            jackknife_gammas.append(g_jack)
        except (ValueError, np.linalg.LinAlgError, TypeError):
            continue

    result: dict[str, Any] = {
        "jackknife_gammas": jackknife_gammas,
        "n_jackknife": len(jackknife_gammas),
        "gamma_std": float("nan"),
        "gamma_median": float("nan"),
        "warning": None,
    }

    if len(jackknife_gammas) > 1:
        g_arr = np.asarray(jackknife_gammas, dtype=float)
        gamma_std = float(np.std(g_arr, ddof=1))
        gamma_median = float(np.median(g_arr))
        result["gamma_std"] = gamma_std
        result["gamma_median"] = gamma_median
        if verbose:
            print(f"Jackknife γ std: {gamma_std:.6f}")
        if gamma_std < 1e-6:
            warn = (
                "Jackknife variance on γ is essentially zero. "
                "Possible bug in auto-calibration."
            )
            result["warning"] = warn
            if verbose:
                print(f"⚠️  WARNING: {warn}")
    elif verbose:
        print("Could not compute jackknife variance.")

    return result


def make_z_hierarchy_gamma_calibrator(
    z: np.ndarray,
    *,
    n_hier: float,
    delta_n: float = 0.0,
    period: float = 7.0,
    gamma_ref: float = 10.0,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], float]:
    """
    Build ``auto_calibrate_func(s, y, cov)`` that calibrates γ from redshift span.

    ``s`` rows are matched back to ``z`` via the reference s-grid at ``gamma_ref``.
    """
    from menus.astronomical.desi.scanner import calibrate_gamma_from_hierarchy, s_from_z

    z_full = np.asarray(z, dtype=float)
    s_full = s_from_z(z_full, gamma=gamma_ref)

    def _z_subset_from_s(s_sub: np.ndarray) -> np.ndarray:
        s_sub = np.asarray(s_sub, dtype=float)
        if len(s_sub) == len(z_full):
            return z_full
        if len(s_sub) == len(z_full) - 1:
            for i in range(len(z_full)):
                keep = np.ones(len(z_full), dtype=bool)
                keep[i] = False
                if np.allclose(s_full[keep], s_sub, rtol=1e-5, atol=1e-3):
                    return z_full[keep]
        raise ValueError("s subset does not match a leave-one-out jackknife slice")

    def auto_calibrate_func(
        s_sub: np.ndarray,
        _y_sub: np.ndarray,
        _cov_sub: np.ndarray,
    ) -> float:
        z_sub = _z_subset_from_s(s_sub)
        if len(z_sub) < 2:
            raise ValueError("need at least two redshift points to calibrate γ")
        return calibrate_gamma_from_hierarchy(
            z_min=float(np.min(z_sub)),
            z_max=float(np.max(z_sub)),
            n_hier=n_hier,
            delta_n=delta_n,
            period=period,
        )

    return auto_calibrate_func


def _save_gamma_diagnostic_figure(fig: Any, path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def _plot_gamma_sensitivity(
    gamma_range: np.ndarray,
    chi2_values: np.ndarray,
    best_gamma: float,
    *,
    save_path: str | None = None,
) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gamma_range, chi2_values, marker="o", color="tab:blue", label="χ²")
    ax.axvline(best_gamma, color="red", linestyle="--", label=f"Best γ = {best_gamma:.2f}")
    ax.set_xlabel("Fixed γ")
    ax.set_ylabel("χ²")
    ax.set_title("Gamma Sensitivity: χ² vs Fixed γ")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path is not None:
        _save_gamma_diagnostic_figure(fig, save_path)
    return fig


def _plot_gamma_jackknife(
    jackknife_gammas: Sequence[float],
    jack_std: float,
    *,
    save_path: str | None = None,
) -> Any:
    import matplotlib.pyplot as plt

    g_arr = np.asarray(jackknife_gammas, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(g_arr, bins=min(10, max(3, len(g_arr))), color="tab:orange", edgecolor="black", alpha=0.7)
    ax.axvline(
        float(np.mean(g_arr)),
        color="red",
        linestyle="--",
        label=f"Mean = {float(np.mean(g_arr)):.3f}",
    )
    ax.set_xlabel("Auto-calibrated γ (jackknife samples)")
    ax.set_ylabel("Count")
    ax.set_title(f"Jackknife Distribution of γ (std = {jack_std:.6f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path is not None:
        _save_gamma_diagnostic_figure(fig, save_path)
    return fig


def show_gamma_diagnostic_plots(
    results: dict[str, Any],
    *,
    block: bool = False,
) -> None:
    """
    Display γ diagnostic figures (non-blocking by default).

    Prefer ``plt.show(block=False)`` over per-figure ``.show()`` so both
    sensitivity and jackknife windows appear without halting the script.
    """
    import matplotlib.pyplot as plt

    has_plot = any(
        results.get(key) is not None
        for key in ("gamma_sensitivity_plot", "jackknife_plot", "rd_vs_gamma_plot")
    )
    if not has_plot:
        return
    plt.show(block=block)


def run_gamma_diagnostics(
    s: np.ndarray,
    y: np.ndarray,
    cov: np.ndarray,
    model_func: Callable[[np.ndarray, float, float], np.ndarray],
    auto_calibrate_func: Callable[[np.ndarray, np.ndarray, np.ndarray], float] | None = None,
    *,
    z: np.ndarray | None = None,
    gamma_range: np.ndarray | Sequence[float] | None = None,
    n_hier: float = 45.8,
    delta_n: float = 0.0,
    period: float = 7.0,
    print_summary: bool = True,
    make_plots: bool = False,
    save_plots: bool = False,
    plot_dir: str = "plots",
    show_plots: bool = False,
    show_plots_block: bool = False,
) -> dict[str, Any]:
    """
    One-stop γ sensitivity sweep + jackknife variance check.

    Parameters
    ----------
    s, y, cov:
        Data arrays on the analysis grid.
    model_func:
        ``model_func(s, A_osc, gamma)`` returning predictions matching ``y``.
    auto_calibrate_func:
        Optional ``γ = f(s_sub, y_sub, cov_sub)`` for row-wise jackknife.
    z:
        Optional redshifts; when provided, jackknife uses unique-z leave-one-out
        (more reliable than row-wise jackknife on duplicate-z BAO vectors).
    gamma_range:
        Fixed-γ values to scan (default ``linspace(5, 15, 21)``).
    print_summary:
        Print diagnostic banners and summary lines.
    make_plots:
        Build matplotlib figure objects and attach them to the result dict.
    save_plots:
        Save figures under ``plot_dir`` (implies ``make_plots=True``).
    plot_dir:
        Output directory when ``save_plots=True``.
    show_plots:
        Call ``plt.show(block=show_plots_block)`` after figures are built
        (implies ``make_plots=True``).
    show_plots_block:
        When ``show_plots=True``, block until plot windows are closed.

    Example
    -------
    After loading data::

        gamma_results = run_gamma_diagnostics(
            s=s, y=y, cov=cov,
            model_func=your_model_func,
            auto_calibrate_func=auto_calibrate_gamma,
            z=z,
            save_plots=True,
            plot_dir="gamma_diagnostics",
        )
        show_gamma_diagnostic_plots(gamma_results)  # plt.show(block=False)
    """
    from menus.astronomical.desi.scanner import calibrate_gamma_from_hierarchy
    from menus.astronomical.desi.stats import gamma_jackknife_std

    sens = diagnose_gamma_sensitivity(
        s,
        y,
        cov,
        model_func,
        gamma_range=gamma_range,
        period=period,
        verbose=print_summary,
        as_tuple=False,
    )
    if not isinstance(sens, dict):
        raise TypeError("diagnose_gamma_sensitivity returned unexpected type")

    results: dict[str, Any] = {
        "best_fixed_gamma": float(sens["best_gamma"]),
        "best_gamma": float(sens["best_gamma"]),
        "best_chi2": float(sens["best_chi2"]),
        "chi2_list": np.asarray(sens["chi2_list"], dtype=float).tolist(),
        "gamma_range": np.asarray(sens["gamma_range"], dtype=float).tolist(),
        "gamma_sensitivity": sens,
    }

    z_arr = np.asarray(z, dtype=float) if z is not None else None
    z_unique = np.unique(z_arr) if z_arr is not None else None

    if z_unique is not None and len(z_unique) >= 3:

        def _calibrate_z(zz: np.ndarray) -> float:
            return calibrate_gamma_from_hierarchy(
                z_min=float(np.min(zz)),
                z_max=float(np.max(zz)),
                n_hier=n_hier,
                delta_n=delta_n,
                period=period,
            )

        if auto_calibrate_func is None:
            auto_calibrate_func = make_z_hierarchy_gamma_calibrator(
                z_arr,
                n_hier=n_hier,
                delta_n=delta_n,
                period=period,
            )

        jack_vals = []
        for i in range(len(z_unique)):
            keep = np.ones(len(z_unique), dtype=bool)
            keep[i] = False
            jack_vals.append(float(_calibrate_z(z_unique[keep])))
        z_jk = gamma_jackknife_std(
            z_unique,
            n_hier=n_hier,
            delta_n=delta_n,
            period=period,
            calibrate_fn=_calibrate_z,
        )
        jack_std = float(z_jk.get("gamma_std", float("nan")))
        results["jackknife_std"] = jack_std
        results["jackknife_values"] = jack_vals
        results["jackknife"] = {
            "gamma_std": jack_std,
            "gamma_median": float(z_jk.get("gamma_median", float("nan"))),
            "n_jackknife": int(z_jk.get("n_jackknife", 0)),
            "method": "unique_z_leave_one_out",
            "warning": None,
        }
        if print_summary:
            print("\n=== GAMMA JACKKNIFE VARIANCE CHECK ===")
            print(f"Jackknife γ std: {jack_std:.8f} (unique z, n={len(z_unique)})")
        if jack_std < 1e-6:
            warn = (
                "Jackknife variance on γ is essentially zero. "
                "Auto-calibration may not be data-dependent."
            )
            results["jackknife"]["warning"] = warn
            if print_summary:
                print(f"⚠️  WARNING: {warn}")
    elif auto_calibrate_func is not None:
        jack = check_gamma_jackknife_variance(
            s,
            y,
            cov,
            auto_calibrate_func,
            verbose=print_summary,
        )
        jack_std = float(jack.get("gamma_std", float("nan")))
        jack_vals = list(jack.get("jackknife_gammas") or [])
        results["jackknife_std"] = jack_std
        results["jackknife_values"] = jack_vals
        results["jackknife"] = jack
        if print_summary and len(jack_vals) <= 1:
            print("Could not compute jackknife variance (function may not be working).")
    else:
        results["jackknife_std"] = float("nan")
        results["jackknife_values"] = []
        if print_summary:
            print("\n=== GAMMA JACKKNIFE VARIANCE CHECK ===")
            print("No auto_calibrate_func provided — skipping jackknife check.")

    results["gamma_jackknife_std"] = results.get("jackknife_std")
    results["gamma_jackknife_warning"] = (
        results.get("jackknife", {}) or {}
    ).get("warning")

    want_plots = make_plots or save_plots or show_plots
    if want_plots:
        gamma_arr = np.asarray(results["gamma_range"], dtype=float)
        chi2_arr = np.asarray(results["chi2_list"], dtype=float)
        sens_path = f"{plot_dir}/gamma_sensitivity.png" if save_plots else None
        results["gamma_sensitivity_plot"] = _plot_gamma_sensitivity(
            gamma_arr,
            chi2_arr,
            float(results["best_fixed_gamma"]),
            save_path=sens_path,
        )

        jack_vals = results.get("jackknife_values") or []
        jack_std = results.get("jackknife_std")
        if len(jack_vals) > 1 and jack_std is not None and np.isfinite(jack_std):
            jack_path = f"{plot_dir}/gamma_jackknife.png" if save_plots else None
            results["jackknife_plot"] = _plot_gamma_jackknife(
                jack_vals,
                float(jack_std),
                save_path=jack_path,
            )

    if show_plots and want_plots:
        show_gamma_diagnostic_plots(results, block=show_plots_block)

    best_gamma = results.get("best_gamma", results.get("best_fixed_gamma"))
    if best_gamma is not None and np.isfinite(float(best_gamma)):
        from menus.astronomical.desi.scanner import apply_tsb_sound_horizon, plot_rd_vs_gamma

        apply_tsb_sound_horizon(
            results,
            float(best_gamma),
            print_summary=print_summary,
        )
        if want_plots:
            rd_path = f"{plot_dir}/rd_vs_gamma.png" if save_plots else None
            results["rd_vs_gamma_plot"] = plot_rd_vs_gamma(
                best_gamma=float(best_gamma),
                save_plot=save_plots,
                plot_dir=plot_dir,
                save_path=rd_path,
            )

    return results


def run_post_load_gamma_diagnostics(
    s: np.ndarray,
    y: np.ndarray,
    cov: np.ndarray,
    z: np.ndarray,
    *,
    model_func: Callable[[np.ndarray, float, float], np.ndarray] | None = None,
    auto_calibrate_func: Callable[[np.ndarray, np.ndarray, np.ndarray], float] | None = None,
    n_hier: float = 45.8,
    delta_n: float = 0.0,
    period: float = 7.0,
    gamma_ref: float = 10.0,
    gamma_range: np.ndarray | Sequence[float] | None = None,
    verbose: bool = True,
    make_plots: bool = False,
    save_plots: bool = False,
    plot_dir: str = "plots",
    show_plots: bool = False,
    show_plots_block: bool = False,
) -> dict[str, Any]:
    """Run γ diagnostics immediately after data load (includes ``z`` for jackknife)."""
    if model_func is None:
        model_func = make_default_gamma_model_func(z, y, period=period)
    if auto_calibrate_func is None:
        auto_calibrate_func = make_z_hierarchy_gamma_calibrator(
            z,
            n_hier=n_hier,
            delta_n=delta_n,
            period=period,
            gamma_ref=gamma_ref,
        )
    return run_gamma_diagnostics(
        s,
        y,
        cov,
        model_func,
        auto_calibrate_func,
        z=z,
        gamma_range=gamma_range,
        n_hier=n_hier,
        delta_n=delta_n,
        period=period,
        print_summary=verbose,
        make_plots=make_plots,
        save_plots=save_plots,
        plot_dir=plot_dir,
        show_plots=show_plots,
        show_plots_block=show_plots_block,
    )


def run_auto_diagnostics(
    s: np.ndarray,
    y: np.ndarray,
    cov: np.ndarray,
    model_func: Callable[[np.ndarray, float, float], np.ndarray] | None = None,
    auto_calibrate_func: Callable[[np.ndarray, np.ndarray, np.ndarray], float] | None = None,
    *,
    z: np.ndarray | None = None,
    period: float = 7.0,
    n_hier: float = 45.8,
    delta_n: float = 0.0,
    gamma_range: np.ndarray | Sequence[float] | None = None,
    run_injection: bool = True,
    injection_seed: int = 0,
    print_summary: bool = True,
    make_plots: bool = False,
    save_plots: bool = False,
    plot_dir: str = "plots",
    show_plots: bool = False,
    show_plots_block: bool = False,
) -> dict[str, Any]:
    """
    Run γ diagnostics and injection-recovery in one block after data load.

    Example::

        y, cov, s, meta = get_data_by_mode()
        your_model_func = make_default_gamma_model_func(meta['z'], y)
        auto_calibrate_gamma = make_z_hierarchy_gamma_calibrator(meta['z'])
        results = run_auto_diagnostics(s, y, cov, your_model_func, auto_calibrate_gamma, z=meta['z'])
    """
    y_arr = np.asarray(y, dtype=float)
    s_arr = np.asarray(s, dtype=float)
    cov_arr = np.asarray(cov, dtype=float)
    z_arr = np.asarray(z, dtype=float) if z is not None else None

    if model_func is None:
        if z_arr is None:
            raise ValueError("run_auto_diagnostics requires model_func or z for default model")
        model_func = make_default_gamma_model_func(z_arr, y_arr, period=period)
    if auto_calibrate_func is None and z_arr is not None:
        auto_calibrate_func = make_z_hierarchy_gamma_calibrator(
            z_arr,
            n_hier=n_hier,
            delta_n=delta_n,
            period=period,
        )

    if print_summary:
        print("\n" + "=" * 60)
        print("RUNNING AUTOMATIC DIAGNOSTICS")
        print("=" * 60)

    gamma_results = run_gamma_diagnostics(
        s=s_arr,
        y=y_arr,
        cov=cov_arr,
        model_func=model_func,
        auto_calibrate_func=auto_calibrate_func,
        z=z_arr,
        gamma_range=gamma_range,
        n_hier=n_hier,
        delta_n=delta_n,
        period=period,
        print_summary=print_summary,
        make_plots=make_plots,
        save_plots=save_plots,
        plot_dir=plot_dir,
        show_plots=show_plots,
        show_plots_block=show_plots_block,
    )

    inj_results: list[dict[str, Any]] | None = None
    if run_injection:
        inj_results = run_injection_recovery_suite(
            s_arr,
            y_arr,
            cov_arr,
            model_func,
            z=z_arr,
            period=period,
            seed=injection_seed,
            verbose=print_summary,
        )

    if print_summary:
        print("\n=== DIAGNOSTICS COMPLETE ===")

    return {
        "gamma_results": gamma_results,
        "inj_results": inj_results,
        "gamma_diagnostics": gamma_results,
        "injection_recovery": inj_results,
    }


def run_gamma_stability_diagnostics(
    *,
    z: np.ndarray,
    y: np.ndarray,
    cov: np.ndarray,
    s: np.ndarray | None = None,
    model_func: Callable[[np.ndarray, float, float], np.ndarray] | None = None,
    auto_calibrate_func: Callable[[np.ndarray, np.ndarray, np.ndarray], float] | None = None,
    gamma_range: np.ndarray | Sequence[float] | None = None,
    n_hier: float = 45.8,
    delta_n: float = 0.0,
    period: float = 7.0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run γ sensitivity sweep and jackknife variance check together."""
    from menus.astronomical.desi.scanner import s_from_z

    z_arr = np.asarray(z, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    cov_arr = np.asarray(cov, dtype=float)
    s_arr = np.asarray(s, dtype=float) if s is not None else s_from_z(z_arr, gamma=10.0)

    if model_func is None:
        model_func = make_default_gamma_model_func(z_arr, y_arr, period=period)

    if auto_calibrate_func is None:
        auto_calibrate_func = make_z_hierarchy_gamma_calibrator(
            z_arr,
            n_hier=n_hier,
            delta_n=delta_n,
            period=period,
        )

    out = run_gamma_diagnostics(
        s_arr,
        y_arr,
        cov_arr,
        model_func,
        auto_calibrate_func,
        z=z_arr,
        gamma_range=gamma_range,
        n_hier=n_hier,
        delta_n=delta_n,
        period=period,
        print_summary=verbose,
    )
    return {
        "gamma_sensitivity": out.get("gamma_sensitivity"),
        "gamma_jackknife": out.get("jackknife"),
        **out,
    }


DEFAULT_INJECTION_AMPLITUDES: tuple[float, ...] = (0.01, 0.02, 0.03, 0.05, 0.08, 0.10)
DEFAULT_INJECTION_TRIALS: int = 100


def _default_injection_model(
    s: np.ndarray,
    amplitude_frac: float,
    phase_rad: float,
    *,
    y_baseline: np.ndarray,
    amplitude_scale: float,
    period: float,
) -> np.ndarray:
    """Baseline + fractional τ oscillation template."""
    s = np.asarray(s, dtype=float)
    return np.asarray(y_baseline, dtype=float) + amplitude_frac * amplitude_scale * np.sin(
        2 * np.pi * s / period + phase_rad
    )


def _recover_injected_oscillation(
    s: np.ndarray,
    y_injected: np.ndarray,
    y_baseline: np.ndarray,
    cov: np.ndarray,
    *,
    amp_scale: float,
    period: float,
    amp_guess_frac: float = 0.0,
    phase_starts: Sequence[float] | None = None,
) -> tuple[float, float]:
    """
    Fit fractional amplitude + phase on a fixed s-grid (γ implicit in ``s``).

    Multi-start over phase avoids the φ≡0 bias that broke the old (A, γ) fit.
    """
    from menus.astronomical.desi.scanner import gaussian_chi2, tau_sb_oscillatory_residual

    s_arr = np.asarray(s, dtype=float)
    y_inj = np.asarray(y_injected, dtype=float)
    y_base = np.asarray(y_baseline, dtype=float)
    if phase_starts is None:
        phase_starts = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)

    best_chi2 = float("inf")
    best_a_frac = float(amp_guess_frac)
    best_phase = 0.0

    for phase0 in phase_starts:
        x0 = np.array([amp_guess_frac, float(phase0)], dtype=float)

        def objective(theta: np.ndarray) -> float:
            pred = y_base + tau_sb_oscillatory_residual(
                s_arr,
                A=float(theta[0]),
                period=period,
                phase=float(theta[1]),
                amplitude_scale=amp_scale,
            )
            return gaussian_chi2(y_inj, pred, cov=cov)

        res = minimize(
            objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=[(-0.35, 0.35), (-np.pi, np.pi)],
        )
        chi2_val = float(res.fun)
        if chi2_val < best_chi2:
            best_chi2 = chi2_val
            best_a_frac = float(res.x[0])
            best_phase = float(res.x[1])

    return best_a_frac, best_phase


def _signed_amplitude_frac(
    a_frac: float,
    phase_rec: float,
    injected_signal: np.ndarray,
    s: np.ndarray,
    *,
    amp_scale: float,
    period: float,
) -> float:
    """
    Resolve (A, φ) ≡ (-A, φ+π) by aligning to the injected template.

    Uses the **signed** fitted oscillation (not |A| with raw φ) so positive
    injections map to positive recovered amplitudes.
    """
    from menus.astronomical.desi.scanner import tau_sb_oscillatory_residual

    inj = np.asarray(injected_signal, dtype=float)
    osc_fit = tau_sb_oscillatory_residual(
        s,
        A=float(a_frac),
        period=period,
        phase=float(phase_rec),
        amplitude_scale=amp_scale,
    )
    if float(np.dot(inj, osc_fit)) >= 0.0:
        return abs(float(a_frac))
    return -abs(float(a_frac))


def run_injection_recovery_suite(
    s: np.ndarray,
    y_obs: np.ndarray,
    cov: np.ndarray,
    model_func: Callable[[np.ndarray, float, float], np.ndarray] | None = None,
    *,
    z: np.ndarray | None = None,
    err: np.ndarray | None = None,
    amplitudes: Sequence[float] = DEFAULT_INJECTION_AMPLITUDES,
    n_injections: int = DEFAULT_INJECTION_TRIALS,
    noise_scale: float | None = None,
    period: float = 7.0,
    amplitude_scale: float | None = None,
    gamma_guess: float = 10.0,
    seed: int = 0,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Multi-amplitude injection-recovery with sign tracking and debiasing.

    Injects ``A_frac * scale * sin(2π s/τ + φ)`` on ``y_obs``, then fits
    **(A_frac, φ)** via χ² using the same τ template (γ fixed via the s-grid).

    The legacy 2-parameter ``(A, γ)`` fit with φ≡0 caused ~100% negative bias
    and ~50% sign accuracy when injections used random phases.

    ``amplitudes`` are fractional strengths relative to ``amplitude_scale``.
    """
    from menus.astronomical.desi.scanner import _oscillation_scale, gaussian_chi2

    _ = model_func  # retained for API compatibility; recovery uses τ template
    _ = gamma_guess

    s = np.asarray(s, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    cov = np.asarray(cov, dtype=float)
    z_arr = np.asarray(z, dtype=float) if z is not None else None
    n = len(y_obs)
    if len(s) != n:
        raise ValueError(f"s length {len(s)} != y_obs length {n}")
    if cov.shape != (n, n):
        raise ValueError(f"cov shape {cov.shape} incompatible with n={n}")

    rng = np.random.default_rng(seed)
    amp_scale = float(
        amplitude_scale if amplitude_scale is not None else _oscillation_scale(y_obs)
    )
    err_arr = np.asarray(err, dtype=float) if err is not None else np.sqrt(np.maximum(np.diag(cov), 0.0))
    if noise_scale is None:
        noise_scale = float(np.median(err_arr[err_arr > 0]) if np.any(err_arr > 0) else np.std(y_obs) * 0.1)

    deg = min(3, max(1, n - 4))
    if z_arr is not None and len(z_arr) == n:
        y_baseline = np.polyval(np.polyfit(z_arr, y_obs, deg=deg), z_arr)
    else:
        x_idx = np.linspace(0.0, 1.0, n)
        y_baseline = np.polyval(np.polyfit(x_idx, y_obs, deg=deg), x_idx)

    if verbose:
        print("\n=== IMPROVED INJECTION-RECOVERY SUITE ===")
        print("  Fit model: (A_frac, phase) on fixed s-grid — φ no longer pinned to 0")

    results: list[dict[str, Any]] = []
    for amp in amplitudes:
        amp_frac = float(amp)
        amp_true = amp_frac * amp_scale
        recovered_frac: list[float] = []
        recovered_phys: list[float] = []
        correct_sign = 0

        for _ in range(int(n_injections)):
            phase = float(rng.uniform(0, 2 * np.pi))
            injected_signal = amp_true * np.sin(2 * np.pi * s / period + phase)
            noise = rng.normal(0.0, 1.0, size=n) * err_arr
            # Inject on the poly baseline (not raw y_obs) so real-data misfit
            # does not masquerade as a second oscillation and bias recovery.
            y_injected = y_baseline + injected_signal + noise

            a_frac_rec, phase_rec = _recover_injected_oscillation(
                s,
                y_injected,
                y_baseline,
                cov,
                amp_scale=amp_scale,
                period=period,
                amp_guess_frac=amp_frac,
            )
            a_frac_signed = _signed_amplitude_frac(
                a_frac_rec,
                phase_rec,
                injected_signal,
                s,
                amp_scale=amp_scale,
                period=period,
            )
            a_phys_rec = a_frac_signed * amp_scale
            recovered_frac.append(a_frac_signed)
            recovered_phys.append(a_phys_rec)
            if np.sign(a_frac_signed) == np.sign(amp_frac):
                correct_sign += 1

        rec_frac = np.asarray(recovered_frac, dtype=float)
        rec = np.asarray(recovered_phys, dtype=float)
        mean_rec_frac = float(np.mean(rec_frac))
        mean_rec = float(np.mean(rec))
        median_rec = float(np.median(rec))
        bias = mean_rec - amp_true
        rel_bias = bias / amp_true if amp_true != 0 else float("nan")
        sign_acc = float(correct_sign / max(n_injections, 1))
        inflation = mean_rec / amp_true if abs(amp_true) > 1e-12 else float("nan")
        debiased_median = float(np.median(rec)) - bias

        row = {
            "true_amp": amp_frac,
            "true_amplitude": amp_frac,
            "A_true_frac": amp_frac,
            "A_true_physical": amp_true,
            "mean_recovered": mean_rec,
            "mean_recovered_frac": mean_rec_frac,
            "A_recovered_mean": mean_rec,
            "A_recovered_mean_frac": mean_rec_frac,
            "A_recovered_median": median_rec,
            "A_recovered_std": float(np.std(rec)),
            "bias": float(bias),
            "relative_bias": float(rel_bias),
            "sign_accuracy": sign_acc,
            "debiased_median": debiased_median,
            "debiased_median_frac": float(debiased_median / amp_scale) if amp_scale else float("nan"),
            "inflation_factor": float(inflation),
            "detection_fraction": sign_acc,
            "n_trials": int(n_injections),
            "rmse": float(np.sqrt(np.mean((rec - amp_true) ** 2))),
            "fit_method": "A_frac_phase_lbfgsb_multistart_signed_template",
        }
        results.append(row)

        if verbose:
            print(
                f"A_true={amp_frac:.2f} | Rec={mean_rec_frac:+.4f} (frac) | "
                f"Bias={rel_bias * 100:+.1f}% | SignAcc={sign_acc:.0%} | "
                f"Debiased≈{debiased_median / amp_scale:+.4f} (frac)"
            )

    return results


def run_multi_amplitude_injection_recovery(
    s: np.ndarray,
    y_true: np.ndarray,
    cov: np.ndarray,
    model_func: Callable[[np.ndarray, float, float], np.ndarray] | None = None,
    *,
    amplitudes: Sequence[float] = DEFAULT_INJECTION_AMPLITUDES,
    n_injections: int = DEFAULT_INJECTION_TRIALS,
    noise_scale: float | None = None,
    err: np.ndarray | None = None,
    period: float = 7.0,
    amplitude_scale: float | None = None,
    detection_threshold_frac: float = 0.5,
    seed: int = 0,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Injection–recovery across multiple fractional amplitudes to map fit bias.

    Injects ``A * sin(2π s/τ + φ)`` on top of ``y_true``, fits amplitude/phase
    with χ² using ``cov``, and records bias and detection rate per amplitude.

    Example::

        injection_results = run_multi_amplitude_injection_recovery(
            s=s_array,
            y_true=data_vector,
            cov=cov,
            model_func=your_model_func,
            amplitudes=[0.02, 0.05, 0.10],
            n_injections=100,
        )
    """
    from menus.astronomical.desi.scanner import _oscillation_scale, gaussian_chi2

    s = np.asarray(s, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = len(y_true)
    if len(s) != n:
        raise ValueError(f"s length {len(s)} != y_true length {n}")
    if cov.shape != (n, n):
        raise ValueError(f"cov shape {cov.shape} incompatible with n={n}")

    amp_scale = float(amplitude_scale if amplitude_scale is not None else _oscillation_scale(y_true))
    rng = np.random.default_rng(seed)
    y_base = y_true.copy()

    if model_func is None:

        def model_func(s_arr: np.ndarray, a_frac: float, phi: float) -> np.ndarray:
            return _default_injection_model(
                s_arr,
                a_frac,
                phi,
                y_baseline=y_base,
                amplitude_scale=amp_scale,
                period=period,
            )

    err_arr = np.asarray(err, dtype=float) if err is not None else None
    sigma = float(noise_scale if noise_scale is not None else (np.median(err_arr) if err_arr is not None else 0.01))

    results: list[dict[str, Any]] = []
    for amp in amplitudes:
        amp_frac = float(amp)
        recovered_amps: list[float] = []
        detected = 0
        correct_sign = 0

        for _ in range(int(n_injections)):
            phase_true = float(rng.uniform(0, 2 * np.pi))
            injected = amp_frac * amp_scale * np.sin(2 * np.pi * s / period + phase_true)
            if err_arr is not None:
                noise = rng.normal(0.0, 1.0, size=n) * err_arr
            else:
                noise = rng.normal(0.0, sigma, size=n)
            y_injected = y_true + injected + noise

            def objective(theta: np.ndarray, _mf=model_func) -> float:
                return gaussian_chi2(
                    y_injected,
                    _mf(s, float(theta[0]), float(theta[1])),
                    cov=cov,
                )

            res = minimize(
                objective,
                x0=np.array([0.0, 0.0]),
                method="L-BFGS-B",
                bounds=[(-0.25, 0.25), (-np.pi, np.pi)],
            )
            recovered = float(res.x[0])
            recovered_amps.append(recovered)
            if abs(recovered) > detection_threshold_frac * abs(amp_frac):
                detected += 1
            if np.sign(recovered) == np.sign(amp_frac):
                correct_sign += 1

        rec = np.asarray(recovered_amps, dtype=float)
        mean_recovered = float(np.mean(rec))
        bias = mean_recovered - amp_frac
        detection_rate = float(detected / max(n_injections, 1))
        sign_acc = float(correct_sign / max(n_injections, 1))
        inflation = mean_recovered / amp_frac if abs(amp_frac) > 1e-12 else float("nan")
        snr = amp_frac * amp_scale / max(sigma, 1e-12)

        row = {
            "true_amplitude": amp_frac,
            "A_true_frac": amp_frac,
            "mean_recovered": mean_recovered,
            "A_recovered_mean": mean_recovered,
            "A_recovered_std": float(np.std(rec)),
            "bias": float(bias),
            "relative_bias": float(bias / amp_frac) if amp_frac != 0 else 0.0,
            "inflation_factor": float(inflation),
            "detection_rate": detection_rate,
            "detection_fraction": detection_rate,
            "sign_accuracy": sign_acc,
            "recovery_fraction": float(np.mean(np.abs(rec - amp_frac) < max(0.35 * abs(amp_frac), 0.005))),
            "snr": float(snr),
            "n_trials": int(n_injections),
            "rmse": float(np.sqrt(np.mean((rec - amp_frac) ** 2))),
        }
        results.append(row)

        if verbose:
            print(
                f"A_true={amp_frac:.2f} | Recovered={mean_recovered:.4f} | "
                f"Bias={bias:+.4f} | Detection={detection_rate:.0%}"
            )

    return results


def _residual_significance_threshold(n: int) -> float:
    return 2.0 / np.sqrt(max(n, 1))


def _residual_autocorrelation_at_lag(values: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= len(values):
        return float("nan")
    a = values[:-lag]
    b = values[lag:]
    if len(a) < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _whiten_residuals(residuals: np.ndarray, cov: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    n = len(residuals)
    if cov.shape != (n, n):
        raise ValueError(f"cov shape {cov.shape} incompatible with n={n}")
    sym = (cov + cov.T) / 2.0
    reg = sym + TIKHONOV_EPS * np.eye(n)
    try:
        chol = np.linalg.cholesky(reg)
        return np.linalg.solve(chol, residuals)
    except np.linalg.LinAlgError:
        evals, evecs = np.linalg.eigh(reg)
        evals = np.maximum(evals, TIKHONOV_EPS)
        return evecs @ ((evecs.T @ residuals) / np.sqrt(evals))


def _plot_residual_acf(
    acf_values: Sequence[float],
    *,
    significance: float,
    save_path: str | None = None,
) -> Any:
    import matplotlib.pyplot as plt

    acf_arr = np.asarray(acf_values, dtype=float)
    lags = np.arange(len(acf_arr))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(lags, acf_arr, width=0.6, color="tab:blue", edgecolor="black", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(
        significance,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"±{significance:.2f} (approx 95%)",
    )
    ax.axhline(-significance, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Residual Autocorrelation Function (ACF)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)
    if save_path is not None:
        _save_gamma_diagnostic_figure(fig, save_path)
    return fig


def _durbin_watson_statistic(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    denom = float(np.sum(values**2))
    if denom <= 0.0:
        return float("nan")
    diff = np.diff(values)
    return float(np.sum(diff**2) / denom)


def _interpret_durbin_watson(dw: float) -> str:
    if not np.isfinite(dw):
        return "undefined"
    if dw < 1.5:
        return "Positive autocorrelation likely"
    if dw > 2.5:
        return "Negative autocorrelation likely"
    return "No strong first-order autocorrelation"


def breusch_godfrey_test(
    residuals: np.ndarray,
    lags: int = 4,
    *,
    print_summary: bool = True,
) -> dict[str, Any] | None:
    """
    Breusch-Godfrey LM test for serial correlation in ordered residuals.

    Uses ``statsmodels.stats.diagnostic.acorr_breusch_godfrey`` when available
    (intercept-only auxiliary regression). Falls back to a lagged-residual
    auxiliary regression with :func:`scipy.stats.chi2` tail probabilities.
    """
    res = np.asarray(residuals, dtype=float)
    n = len(res)
    test_lags = int(min(max(lags, 1), max(n - 2, 1)))
    if n < test_lags + 2:
        if print_summary:
            print(
                f"\n[Breusch-Godfrey Test] skipped "
                f"(need n > lags+1, got n={n}, lags={test_lags})"
            )
        return None

    result: dict[str, Any] | None = None

    try:
        from statsmodels.stats.diagnostic import acorr_lm

        lm_stat, p_value, f_stat, f_pvalue = acorr_lm(res, nlags=test_lags)
        result = {
            "lm_stat": float(lm_stat),
            "p_value": float(p_value),
            "f_stat": float(f_stat),
            "f_pvalue": float(f_pvalue),
            "lags": test_lags,
            "method": "statsmodels_acorr_lm",
            "significant_at_5pct": bool(p_value < 0.05),
        }
    except (ImportError, TypeError, ValueError, AttributeError):
        y = res[test_lags:]
        x_lags = [res[test_lags - lag : n - lag] for lag in range(1, test_lags + 1)]
        x = np.column_stack([np.ones(len(y), dtype=float), *x_lags])
        beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        y_hat = x @ beta
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
        lm_stat = len(y) * r2
        p_value = float(stats.chi2.sf(lm_stat, test_lags))
        result = {
            "lm_stat": float(lm_stat),
            "p_value": p_value,
            "lags": test_lags,
            "method": "manual_auxiliary_regression",
            "significant_at_5pct": bool(p_value < 0.05),
        }

    if print_summary and result is not None:
        print(f"\n[Breusch-Godfrey Test] (up to lag {test_lags})")
        print(f"  LM statistic: {result['lm_stat']:.4f}")
        print(f"  p-value     : {result['p_value']:.4f}")
        if result["significant_at_5pct"]:
            print("  ⚠️  Significant serial correlation detected")
        else:
            print("  → No significant serial correlation")

    return result


def _ljung_box_test(values: np.ndarray, max_lag: int) -> dict[str, Any] | None:
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
    except ImportError:
        return None

    n = len(values)
    lags = int(min(max_lag, max(n - 1, 1)))
    lb_df = acorr_ljungbox(np.asarray(values, dtype=float), lags=lags, return_df=True)
    pvals = lb_df["lb_pvalue"].to_numpy(dtype=float)
    return {
        "lags": [int(x) for x in lb_df.index.tolist()],
        "lb_stat": lb_df["lb_stat"].to_numpy(dtype=float).tolist(),
        "lb_pvalue": pvals.tolist(),
        "significant_lag_count": int(np.sum(pvals < 0.05)),
        "n_lags_tested": lags,
    }


def plot_residuals_vs_fitted(
    y_model: np.ndarray,
    residuals: np.ndarray,
    *,
    title: str = "Residuals vs Fitted Values",
    lowess_frac: float = 0.6,
    save_plot: bool = False,
    plot_dir: str = "residual_diagnostics",
) -> Any:
    """
    Residuals vs fitted scatter with LOWESS trend line.

    Returns the matplotlib figure. Uses ``statsmodels`` LOWESS when available;
    otherwise overlays a linear trend on sorted fitted values.
    """
    import matplotlib.pyplot as plt

    fitted = np.asarray(y_model, dtype=float)
    res = np.asarray(residuals, dtype=float)
    if fitted.shape != res.shape:
        raise ValueError("y_model and residuals must have the same shape")
    if len(fitted) < 2:
        raise ValueError("need at least 2 points for residuals vs fitted plot")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(fitted, res, alpha=0.6, edgecolor="k", s=40, label="Residuals")

    trend_label = "LOWESS trend"
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        smoothed = lowess(res, fitted, frac=lowess_frac)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="red", linewidth=2.5, label=trend_label)
    except ImportError:
        order = np.argsort(fitted)
        x_ord = fitted[order]
        y_ord = res[order]
        deg = int(min(1, len(fitted) - 1))
        coeffs = np.polyfit(x_ord, y_ord, deg=deg)
        x_line = np.linspace(float(np.min(fitted)), float(np.max(fitted)), 50)
        ax.plot(x_line, np.polyval(coeffs, x_line), color="red", linewidth=2.5, label="Linear trend")

    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Fitted Values (Model Prediction)")
    ax.set_ylabel("Residuals")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_plot:
        _save_gamma_diagnostic_figure(fig, f"{plot_dir}/residuals_vs_fitted.png")

    return fig


def plot_residual_qq(
    residuals: np.ndarray,
    *,
    title: str | None = None,
    save_plot: bool = False,
    plot_dir: str = "residual_diagnostics",
) -> tuple[Any, float]:
    """
    Normal Q-Q plot for residuals.

    Returns ``(figure, r_squared)`` where ``r_squared`` is the squared
    correlation coefficient from the probability plot fit.
    """
    import matplotlib.pyplot as plt

    res = np.asarray(residuals, dtype=float)
    if len(res) < 3:
        raise ValueError("need at least 3 points for a Q-Q plot")

    (osm, osr), (slope, intercept, r) = stats.probplot(res, dist="norm", plot=None)
    r_sq = float(r**2)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(osm, osr, alpha=0.7, edgecolor="k")
    minv = float(min(np.min(osm), np.min(osr)))
    maxv = float(max(np.max(osm), np.max(osr)))
    ax.plot([minv, maxv], [minv, maxv], "r--", linewidth=2)
    ax.set_title(title or f"Q-Q Plot (R² = {r_sq:.3f})")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.grid(True, alpha=0.3)

    if save_plot:
        _save_gamma_diagnostic_figure(fig, f"{plot_dir}/qq.png")

    return fig, r_sq


def show_residual_diagnostic_plots(
    results: dict[str, Any],
    *,
    block: bool = False,
) -> None:
    """Display residual diagnostic figures (non-blocking by default)."""
    import matplotlib.pyplot as plt

    figure_map = results.get("figures")
    if isinstance(figure_map, dict):
        has_plot = any(fig is not None for fig in figure_map.values())
    else:
        has_plot = any(
            results.get(key) is not None
            for key in ("acf_plot", "residuals_vs_fitted_plot", "qq_plot")
        )
    if has_plot:
        plt.show(block=block)


def analyze_residuals(
    s: np.ndarray,
    y_obs: np.ndarray,
    y_model: np.ndarray,
    cov: np.ndarray | None = None,
    *,
    max_lag: int = 8,
    print_summary: bool = True,
    make_plots: bool = True,
    save_plots: bool = False,
    plot_dir: str = "residual_diagnostics",
    show_plots: bool = False,
    show_plots_block: bool = False,
    breusch_godfrey_lags: int = 4,
) -> dict[str, Any]:
    """
    Full residual diagnostics:

    1. Autocorrelation at multiple lags
    2. Shapiro-Wilk normality test
    3. Autocorrelation plot (ACF)
    4. Durbin-Watson statistic
    5. Ljung-Box test (requires ``statsmodels``)
    6. Breusch-Godfrey LM test
    7. Residuals vs fitted plot (LOWESS trend)

    When ``cov`` is supplied, tests are evaluated on Mahalanobis-whitened
    residuals (more appropriate for correlated BAO data).

    Parameters
    ----------
    s:
        Analysis grid (retained for API symmetry; not used in computations).
    y_obs, y_model:
        Observed and model predictions.
    cov:
        Optional covariance for whitening.
    max_lag:
        Maximum lag for autocorrelation / ACF.
    print_summary:
        Print section banners and per-lag results.
    make_plots, save_plots, plot_dir, show_plots, show_plots_block:
        Plotting controls (same semantics as :func:`run_gamma_diagnostics`).

    Example
    -------
    After model predictions::

        y_model = your_model_func(s, A_osc, gamma)
        residual_results = analyze_residuals(
            s=s, y_obs=y, y_model=y_model, cov=cov,
            max_lag=6,
            save_plots=True,
            plot_dir="residual_diagnostics",
        )
        print("Shapiro-Wilk p-value:", residual_results["shapiro_wilk"]["p_value"])
        show_residual_diagnostic_plots(residual_results)  # plt.show(block=False)
    """
    _ = np.asarray(s, dtype=float)
    y_obs_arr = np.asarray(y_obs, dtype=float)
    y_model_arr = np.asarray(y_model, dtype=float)
    if y_obs_arr.shape != y_model_arr.shape:
        raise ValueError("y_obs and y_model must have the same shape")

    residuals = y_obs_arr - y_model_arr
    n = len(residuals)
    if n < 3:
        raise ValueError(f"need at least 3 points for residual analysis, got n={n}")

    analysis_vec = residuals
    whitened = False
    if cov is not None:
        analysis_vec = _whiten_residuals(residuals, cov)
        whitened = True

    results: dict[str, Any] = {
        "n": int(n),
        "residuals": residuals.tolist(),
        "whitened": whitened,
        "analysis_vector": np.asarray(analysis_vec, dtype=float).tolist(),
        "significance_threshold": float(_residual_significance_threshold(n)),
    }

    threshold = float(results["significance_threshold"])
    autocorr: dict[int, dict[str, float | bool]] = {}

    if print_summary:
        print("\n" + "=" * 60)
        print("RESIDUAL DIAGNOSTICS")
        print("=" * 60)
        print("\n[1] Residual Autocorrelation by Lag")
        if whitened:
            print("  (Mahalanobis-whitened residuals)")

    for lag in range(1, max_lag + 1):
        if lag >= n or (n - lag) < 2:
            continue
        r = _residual_autocorrelation_at_lag(analysis_vec, lag)
        if not np.isfinite(r):
            continue
        significant = bool(abs(r) > threshold)
        autocorr[lag] = {
            "r": float(r),
            "significant": significant,
        }
        if print_summary:
            status = "⚠️ Significant" if significant else ""
            print(f"  Lag {lag}: r = {r:+.4f} {status}")

    results["autocorrelation"] = autocorr

    if print_summary:
        print("\n[2] Shapiro-Wilk Normality Test")

    shapiro_stat, shapiro_p = stats.shapiro(np.asarray(analysis_vec, dtype=float))
    is_normal = bool(shapiro_p > 0.05)
    results["shapiro_wilk"] = {
        "statistic": float(shapiro_stat),
        "p_value": float(shapiro_p),
        "is_normal_at_5pct": is_normal,
        "normal_at_5pct": is_normal,
    }

    if print_summary:
        print(f"  Statistic: {shapiro_stat:.4f}")
        print(f"  p-value  : {shapiro_p:.4f}")
        print("  → Normal?", "Yes (p > 0.05)" if is_normal else "No (p < 0.05)")

    want_plots = make_plots or save_plots or show_plots
    if want_plots:
        acf_values = [1.0]
        for lag in range(1, max_lag + 1):
            if lag < n and (n - lag) >= 2:
                r_lag = _residual_autocorrelation_at_lag(analysis_vec, lag)
                acf_values.append(float(r_lag) if np.isfinite(r_lag) else 0.0)
            else:
                acf_values.append(0.0)

        results["acf_values"] = [float(v) for v in acf_values]
        acf_path = f"{plot_dir}/acf_plot.png" if save_plots else None
        results["acf_plot"] = _plot_residual_acf(
            acf_values,
            significance=threshold,
            save_path=acf_path,
        )
        if print_summary:
            print("\n[3] ACF plot generated")

        results["residuals_vs_fitted_plot"] = plot_residuals_vs_fitted(
            y_model_arr,
            residuals,
            save_plot=save_plots,
            plot_dir=plot_dir,
        )
        if print_summary:
            print("[3b] Residuals vs fitted plot generated")

    dw_stat = _durbin_watson_statistic(analysis_vec)
    dw_interp = _interpret_durbin_watson(dw_stat)
    results["durbin_watson"] = {
        "statistic": float(dw_stat),
        "interpretation": dw_interp,
    }

    if print_summary:
        print("\n[4] Durbin-Watson Statistic")
        print(f"  Durbin-Watson = {dw_stat:.4f}")
        print(f"  → {dw_interp}")

    lb_result = _ljung_box_test(analysis_vec, max_lag)
    results["ljung_box"] = lb_result

    if print_summary:
        print("\n[5] Ljung-Box Test (for autocorrelation at multiple lags)")
        if lb_result is None:
            print("  statsmodels not installed. Skipping Ljung-Box test.")
            print("  → Install with: pip install statsmodels")
        else:
            for lag_i, stat_i, p_i in zip(
                lb_result["lags"],
                lb_result["lb_stat"],
                lb_result["lb_pvalue"],
            ):
                flag = " ⚠️" if p_i < 0.05 else ""
                print(f"  Lag {lag_i}: LB = {stat_i:.4f}, p = {p_i:.4f}{flag}")
            sig_n = int(lb_result["significant_lag_count"])
            if sig_n > 0:
                print(f"  ⚠️  Autocorrelation detected at {sig_n} lag(s) (p < 0.05)")
            else:
                print("  → No significant autocorrelation detected (good)")

    bg_result = breusch_godfrey_test(
        analysis_vec,
        lags=breusch_godfrey_lags,
        print_summary=print_summary,
    )
    results["breusch_godfrey"] = bg_result

    if show_plots and want_plots:
        show_residual_diagnostic_plots(results, block=show_plots_block)

    return results


def run_residual_tests(
    residuals: np.ndarray,
    s: np.ndarray,
    cov: np.ndarray | None = None,
    *,
    max_lag: int = 8,
    print_summary: bool = True,
    make_plots: bool = False,
    save_plots: bool = False,
    plot_dir: str = "residual_diagnostics",
    breusch_godfrey_lags: int = 4,
) -> dict[str, Any]:
    """
    Standard residual diagnostics on a pre-computed residual vector.

    Thin wrapper around :func:`analyze_residuals` (``y_obs = residuals``,
    ``y_model = 0``).
    """
    resid = np.asarray(residuals, dtype=float)
    return analyze_residuals(
        s,
        resid,
        np.zeros_like(resid),
        cov,
        max_lag=max_lag,
        print_summary=print_summary,
        make_plots=make_plots,
        save_plots=save_plots,
        plot_dir=plot_dir,
        show_plots=False,
        breusch_godfrey_lags=breusch_godfrey_lags,
    )


def run_full_residual_diagnostics(
    s: np.ndarray,
    y_obs: np.ndarray,
    y_model: np.ndarray,
    cov: np.ndarray | None = None,
    *,
    max_lag: int = 8,
    save_plots: bool = False,
    plot_dir: str = "residual_diagnostics",
    verbose: bool = True,
    breusch_godfrey_lags: int = 4,
    show_plots: bool = False,
    show_plots_block: bool = False,
    best_gamma: float | None = None,
    banner: str | None = None,
    include_percolation: bool = False,
    include_whim_web: bool = False,
    include_binding_kit: bool = False,
    z: np.ndarray | None = None,
    n_hier: float | None = None,
) -> dict[str, Any]:
    """
    Master residual + TSB diagnostics: tests, summary stats, plots, and sound horizon.

    Runs :func:`analyze_residuals`, adds residual statistics, a Q-Q plot, TSB ``r_d``
    prediction via :func:`compute_tsb_rd`, and optional ``rd_vs_gamma`` figure.

    After a Tau-SB model fit::

        results = run_full_residual_diagnostics(
            s=s,
            y_obs=y,
            y_model=y_model,
            cov=cov,
            max_lag=8,
            save_plots=True,
            best_gamma=8.8511,  # auto-calibrated γ from the fit, or sensitivity best_gamma
        )

    When ``best_gamma`` is omitted, falls back to ``TSB_RD_GAMMA_FALLBACK`` (8.8511).

    Set ``include_percolation=True`` (or call :func:`run_master_diagnostics`) to add
    the simplified 2.5D Tau-cylinder percolation estimate.

    Set ``include_whim_web=True`` (or call :func:`run_master_diagnostics`) to attach
    Sonato WHIM / cosmological-web parameters and optional s-grid density modulation.
    """
    title = banner or "FULL RESIDUAL + TSB DIAGNOSTICS"
    if verbose:
        print("\n" + "=" * len(title))
        print(title)
        print("=" * len(title))

    core = analyze_residuals(
        s,
        y_obs,
        y_model,
        cov,
        max_lag=max_lag,
        print_summary=verbose,
        make_plots=True,
        save_plots=False,
        plot_dir=plot_dir,
        show_plots=False,
        breusch_godfrey_lags=breusch_godfrey_lags,
    )

    residuals = np.asarray(y_obs, dtype=float) - np.asarray(y_model, dtype=float)
    analysis_vec = np.asarray(core.get("analysis_vector", residuals.tolist()), dtype=float)

    results: dict[str, Any] = dict(core)
    results["mean_residual"] = float(np.mean(residuals))
    results["std_residual"] = float(np.std(residuals, ddof=0))

    shapiro = dict(results.get("shapiro_wilk") or {})
    shapiro["normal"] = bool(shapiro.get("is_normal_at_5pct", False))
    results["shapiro_wilk"] = shapiro

    dw = results.get("durbin_watson") or {}
    if isinstance(dw, dict):
        results["durbin_watson_stat"] = float(dw.get("statistic", float("nan")))
    else:
        results["durbin_watson_stat"] = float(dw)

    autocorr_flat: dict[int, float] = {}
    for lag, entry in (results.get("autocorrelation") or {}).items():
        if isinstance(entry, dict):
            autocorr_flat[int(lag)] = float(entry.get("r", float("nan")))
        else:
            autocorr_flat[int(lag)] = float(entry)
    results["autocorrelation_flat"] = autocorr_flat

    fig_qq, qq_r2 = plot_residual_qq(
        analysis_vec,
        save_plot=False,
        plot_dir=plot_dir,
    )
    results["qq_plot"] = fig_qq
    results["qq_r_squared"] = float(qq_r2)

    figures: dict[str, Any] = {
        "acf": results.get("acf_plot"),
        "qq": fig_qq,
        "residuals_vs_fitted": results.get("residuals_vs_fitted_plot"),
    }

    from menus.astronomical.desi.scanner import (
        TSB_RD_GAMMA_FALLBACK,
        compute_tsb_rd,
        plot_rd_vs_gamma,
        print_tsb_sound_horizon_summary,
    )

    gamma_used = float(best_gamma) if best_gamma is not None else float(TSB_RD_GAMMA_FALLBACK)
    if not np.isfinite(gamma_used) or gamma_used <= 0:
        gamma_used = float(TSB_RD_GAMMA_FALLBACK)
    results["best_gamma_used"] = gamma_used

    tsb_rd = compute_tsb_rd(gamma_used)
    results["tsb_sound_horizon"] = tsb_rd
    results["rd_residual"] = float(tsb_rd["rd_residual_fraction"])
    results["rd_residual_percent"] = float(tsb_rd["rd_residual_percent"])

    if verbose:
        print_tsb_sound_horizon_summary(tsb_rd)

    from menus.astronomical.desi.stats import search_higher_harmonics

    if verbose:
        print("\n[Higher harmonics — 1/7 ladder]")
    higher_harmonics = search_higher_harmonics(
        residuals,
        np.asarray(s, dtype=float),
        fundamental_freq=1.0 / 7.0,
        max_harmonic=5,
        n_bootstrap=2000,
        verbose=verbose,
    )
    results["higher_harmonics"] = higher_harmonics

    if include_binding_kit:
        from menus.astronomical.desi.harmonic_binding import (
            run_harmonic_binding_kit,
            search_harmonic_features,
        )
        from menus.astronomical.desi.scanner import N_HIER_BINDING

        z_arr = np.asarray(z, dtype=float) if z is not None else None
        if z_arr is None or len(z_arr) != len(residuals):
            z_proxy = np.linspace(0.1, 2.5, len(residuals))
            z_arr = z_proxy
            obs_proxy = np.asarray(y_model, dtype=float) + residuals
        else:
            obs_proxy = np.asarray(y_obs, dtype=float)
        features = search_harmonic_features(
            residuals,
            np.asarray(s, dtype=float),
            z=z_arr,
            verbose=verbose,
        )
        kit = run_harmonic_binding_kit(
            z_arr,
            obs_proxy,
            residuals,
            np.asarray(s, dtype=float),
            n_hier=float(n_hier if n_hier is not None else N_HIER_BINDING),
            gamma=gamma_used,
            verbose=verbose,
        )
        kit["harmonic_features"] = features
        results["harmonic_binding_kit"] = kit

    if include_percolation:
        from menus.astronomical.desi.stats import estimate_tsb_percolation

        percolation = estimate_tsb_percolation(len(residuals))
        results["percolation"] = percolation
        if verbose:
            print(
                f"\n[Percolation Estimate] Approx critical filling fraction "
                f"p_c ≈ {percolation['estimated_p_c']:.3f} "
                f"(n_sites ≈ {percolation['n_effective_sites']})"
            )

    if include_whim_web:
        from menus.astronomical.desi.scanner import (
            BEC_COHERENCE_LENGTH_MPC,
            compute_whim_web_context,
            print_whim_web_summary,
        )

        s_arr = np.asarray(s, dtype=float)
        s_span = float(np.max(s_arr) - np.min(s_arr)) if len(s_arr) else 0.0
        if s_span > 0:
            distance_proxy_mpc = BEC_COHERENCE_LENGTH_MPC * (s_arr - np.min(s_arr)) / s_span
        else:
            distance_proxy_mpc = np.zeros_like(s_arr)

        whim_parameters = compute_whim_web_context(
            distance_proxy_mpc,
            s=s_arr,
            gamma=gamma_used,
        )
        if verbose:
            print_whim_web_summary(whim_parameters)

        from menus.astronomical.desi.stats import search_whim_web_signatures

        if verbose:
            print("\n[WHIM / Web signature search]")
        whim_results = search_whim_web_signatures(
            residuals,
            s_arr,
            n_bootstrap=2000,
            gamma=gamma_used,
            verbose=verbose,
        )
        whim_results["parameters"] = whim_parameters
        results["whim_web"] = whim_results
        results["whim_web_signatures"] = whim_results
        results["whim_web_parameters"] = whim_parameters

    rd_fig = plot_rd_vs_gamma(
        best_gamma=gamma_used,
        save_plot=save_plots,
        plot_dir=plot_dir,
        save_path=f"{plot_dir}/rd_vs_gamma.png" if save_plots else None,
    )
    figures["rd_vs_gamma"] = rd_fig
    results["rd_vs_gamma_plot"] = rd_fig
    results["figures"] = figures

    if save_plots:
        for name, fig in figures.items():
            if fig is not None and name != "rd_vs_gamma":
                _save_gamma_diagnostic_figure(fig, f"{plot_dir}/{name}.png")
        if not verbose:
            import matplotlib.pyplot as plt

            for fig in figures.values():
                if fig is not None:
                    plt.close(fig)

    if verbose:
        print("\n[Summary]")
        print(f"  Mean residual      : {results['mean_residual']:.6f}")
        print(f"  Std residual       : {results['std_residual']:.6f}")
        print(f"  Durbin-Watson      : {results['durbin_watson_stat']:.3f}")
        print(f"  Shapiro-Wilk p     : {shapiro.get('p_value', float('nan')):.4f}")
        bg = results.get("breusch_godfrey")
        if bg:
            print(f"  Breusch-Godfrey p  : {bg.get('p_value', float('nan')):.4f}")
        print(f"  Q-Q R²             : {results['qq_r_squared']:.3f}")
        print(f"  TSB γ used         : {gamma_used:.3f}")
        print(f"  TSB r_d residual   : {results['rd_residual_percent']:+.2f}%")
        hh = results.get("higher_harmonics") or {}
        if hh.get("harmonics"):
            sig = [
                h
                for h, entry in hh["harmonics"].items()
                if entry.get("significant_at_5pct")
            ]
            print(
                f"  Higher harmonics   : "
                f"{'none significant' if not sig else f'h={sig} at 5% level'}"
            )
        perc = results.get("percolation") or {}
        if perc:
            print(f"  Percolation p_c    : {perc.get('estimated_p_c', float('nan')):.3f}")
        ww = results.get("whim_web") or {}
        ww_params = ww.get("parameters") or results.get("whim_web_parameters") or {}
        if ww_params:
            print(
                f"  WHIM ρ boost (mean) : {ww_params.get('modulation_mean', float('nan')):+.4f}"
                if ww_params.get("modulation_mean") is not None
                else f"  BEC L_coh (Mpc)     : {ww_params.get('bec_coherence_length_mpc', float('nan')):.1f}"
            )
        pearson = ww.get("pearson_s") or {}
        if pearson:
            print(
                f"  WHIM corr(s,resid)  : {pearson.get('correlation', float('nan')):+.4f} "
                f"(p={pearson.get('p_value', float('nan')):.3f})"
            )
        print(f"  Plots generated    : {list(figures.keys())}")

    if show_plots:
        show_residual_diagnostic_plots(results, block=show_plots_block)

    return results


def run_master_diagnostics(
    s: np.ndarray,
    y_obs: np.ndarray,
    y_model: np.ndarray,
    cov: np.ndarray | None = None,
    *,
    max_lag: int = 8,
    best_gamma: float = 8.8511,
    save_plots: bool = False,
    plot_dir: str = "diagnostics",
    verbose: bool = True,
    breusch_godfrey_lags: int = 4,
    show_plots: bool = False,
    show_plots_block: bool = False,
    include_binding_kit: bool = True,
    z: np.ndarray | None = None,
    n_hier: float | None = None,
) -> dict[str, Any]:
    """
    Master diagnostic function — one call for residual tests, TSB sound horizon,
    harmonic ladder search, percolation estimate, and optional plots.

    After a Tau-SB fit::

        results = run_master_diagnostics(
            s=s, y_obs=y, y_model=y_model, cov=cov,
            best_gamma=8.8511, save_plots=True,
        )
    """
    return run_full_residual_diagnostics(
        s,
        y_obs,
        y_model,
        cov,
        max_lag=max_lag,
        save_plots=save_plots,
        plot_dir=plot_dir,
        verbose=verbose,
        breusch_godfrey_lags=breusch_godfrey_lags,
        show_plots=show_plots,
        show_plots_block=show_plots_block,
        best_gamma=best_gamma,
        banner="MASTER TSB + RESIDUAL DIAGNOSTICS",
        include_percolation=True,
        include_whim_web=True,
        include_binding_kit=include_binding_kit,
        z=z,
        n_hier=n_hier,
    )