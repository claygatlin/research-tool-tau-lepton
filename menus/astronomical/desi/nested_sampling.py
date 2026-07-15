"""
dynesty nested sampling with geometric (Tau-SB) and physical (aDE) priors.

Computes log-evidence ln Z for model comparison per session 2026-07-09 roadmap.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from menus.astronomical.desi.prior_bounds import (
    ADE_AOSC_MAX_FRAC,
    ADE_OMEGA_MAX,
    ADE_OMEGA_MIN,
    ADE_ZSTAR_MAX,
    ADE_ZSTAR_MIN,
    GEOMETRIC_PRIOR_SIGMA_FRAC,
    SPARSE_N_THRESHOLD,
    ade_amplitude_bounds,
    ade_omega_bounds,
    ade_z_star_bounds,
    apply_evidence_corrections,
    apply_occam_evidence_correction,
    hier_frac_bounds,
    sparse_occam_penalty,
    tau_amplitude_frac_bounds,
)
from tav_shared.artifact_paths import TestSlug, artifact_path, artifact_timestamp, compose_dataset_slug
from menus.astronomical.desi.scanner import (
    GAMMA6_HEX,
    N_HIER_BINDING,
    R_TAU_MPC,
    TauSBScanner,
    _baseline_degree,
    _oscillation_scale,
    ade_dark_energy_modulation,
    gaussian_chi2,
    s_from_z,
    tau_sb_hierarchical_step,
    tau_sb_oscillatory_residual,
)

NESTED_NLIVE_DEFAULT: int = 200
NESTED_MAX_SAMPLES_DEFAULT: int = 2000


def _require_dynesty():
    try:
        import dynesty  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Install dynesty: pip install dynesty (see requirements-desi-production.txt)"
        ) from exc


def _loglikelihood_factory(
    *,
    model_builder: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    z: np.ndarray,
    obs: np.ndarray,
    err: np.ndarray | None,
    cov: np.ndarray | None,
    s: np.ndarray,
) -> Callable[[np.ndarray], float]:
    def loglikelihood(theta: np.ndarray) -> float:
        pred = model_builder(theta, z, s)
        chi2 = gaussian_chi2(obs, pred, err=err, cov=cov)
        return -0.5 * float(chi2)

    return loglikelihood


def _tau_sb_prior_transform(
    u: np.ndarray,
    *,
    deg: int,
    poly_center: np.ndarray,
    poly_spread: np.ndarray,
    a_center: float = 0.01,
    a_width: float | None = None,
    hier_center: float = 0.005,
    hier_width: float | None = None,
) -> np.ndarray:
    """Map unit cube → Tau-SB params with tight geometric oscillation priors."""
    a_lo, a_hi = tau_amplitude_frac_bounds(mode="nested")
    a_width = a_hi if a_width is None else a_width
    h_lo, h_hi = hier_frac_bounds(mode="nested", center=hier_center)
    hier_width = 0.5 * (h_hi - h_lo) if hier_width is None else hier_width
    u = np.asarray(u, dtype=float)
    n_base = deg + 1
    theta = np.empty(n_base + 3, dtype=float)
    for i in range(n_base):
        theta[i] = float(poly_center[i] + (u[i] - 0.5) * 2.0 * poly_spread[i])
    # A_frac: triangular peak at a_center, support [a_lo, a_hi]
    theta[n_base] = float(a_center + (u[n_base] - 0.5) * 2.0 * min(a_width, a_center * 2 + 0.02))
    theta[n_base] = float(np.clip(theta[n_base], a_lo, a_hi))
    # phase: uniform [-pi, pi]
    theta[n_base + 1] = float((u[n_base + 1] - 0.5) * 2.0 * np.pi)
    # hier_frac: tight geometric window
    theta[n_base + 2] = float(hier_center + (u[n_base + 2] - 0.5) * 2.0 * hier_width)
    return theta


def _ade_prior_transform(
    u: np.ndarray,
    *,
    deg: int,
    poly_center: np.ndarray,
    poly_spread: np.ndarray,
) -> np.ndarray:
    """Map unit cube → aDE params with physical priors only."""
    u = np.asarray(u, dtype=float)
    n_base = deg + 1
    theta = np.empty(n_base + 4, dtype=float)
    for i in range(n_base):
        theta[i] = float(poly_center[i] + (u[i] - 0.5) * 2.0 * poly_spread[i])
    a_lo, a_hi = ade_amplitude_bounds()
    z_lo, z_hi = ade_z_star_bounds()
    o_lo, o_hi = ade_omega_bounds()
    theta[n_base] = float(a_lo + u[n_base] * (a_hi - a_lo))
    theta[n_base + 1] = float(o_lo + u[n_base + 1] * (o_hi - o_lo))
    theta[n_base + 2] = float((u[n_base + 2] - 0.5) * 2.0 * np.pi)
    theta[n_base + 3] = float(z_lo + u[n_base + 3] * (z_hi - z_lo))
    return theta


def run_dynesty_tau_sb_geometric(
    scanner: TauSBScanner,
    z: np.ndarray,
    observable: np.ndarray,
    err: np.ndarray | None = None,
    cov: np.ndarray | None = None,
    *,
    nlive: int = NESTED_NLIVE_DEFAULT,
    max_samples: int = NESTED_MAX_SAMPLES_DEFAULT,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Nested sampling for Tau-SB with geometric priors on the oscillation sector.

    Baseline polynomial coeffs are centered on a quick polyfit; A_frac, phase,
    and hier_frac use tight theory-anchored windows (±1% scale on hier).
    """
    _require_dynesty()
    import dynesty

    z = np.asarray(z, dtype=float)
    obs = np.asarray(observable, dtype=float)
    gamma_used = scanner._gamma_for(z)
    s = s_from_z(z, gamma=gamma_used)
    amp_scale = _oscillation_scale(obs)
    deg = _baseline_degree(len(z), extra_params=3)
    ndim = deg + 4

    poly = np.polyfit(z, obs, deg)
    poly_spread = np.maximum(np.abs(poly) * 0.25, np.std(obs) * 0.05)

    def model(theta: np.ndarray, zz: np.ndarray, ss: np.ndarray) -> np.ndarray:
        base = np.polyval(theta[: deg + 1], zz)
        osc = tau_sb_oscillatory_residual(
            ss,
            A=float(theta[deg + 1]),
            period=scanner.period,
            phase=float(theta[deg + 2]),
            amplitude_scale=amp_scale,
        )
        hier = tau_sb_hierarchical_step(
            zz, amplitude=float(theta[deg + 3]) * amp_scale * 0.01
        )
        return base + osc + hier

    logl = _loglikelihood_factory(
        model_builder=model, z=z, obs=obs, err=err, cov=cov, s=s
    )

    def prior_transform(u: np.ndarray) -> np.ndarray:
        return _tau_sb_prior_transform(
            u,
            deg=deg,
            poly_center=poly,
            poly_spread=poly_spread,
        )

    sampler = dynesty.NestedSampler(
        logl,
        prior_transform,
        ndim,
        nlive=nlive,
        bound="multi",
        sample="rwalk",
        rstate=np.random.default_rng(int(seed)),
    )
    sampler.run_nested(maxiter=max_samples, print_progress=False)
    res = sampler.results
    logz = float(res.logz[-1])
    logz_err = float(res.logzerr[-1]) if len(res.logzerr) else float("nan")
    best = res.samples[np.argmax(res.logl)]

    out = {
        "model": "Tau-SB (geometric priors)",
        "log_evidence": logz,
        "log_evidence_err": logz_err,
        "n_live": nlive,
        "ndim": ndim,
        "gamma_used": gamma_used,
        "geometric_anchors": {
            "R_tau_mpc": R_TAU_MPC,
            "gamma6_hex": GAMMA6_HEX,
            "n_hier": N_HIER_BINDING,
            "prior_sigma_frac": GEOMETRIC_PRIOR_SIGMA_FRAC,
        },
        "best_fit_params": {
            "A_osc_frac": float(best[deg + 1]),
            "phase_rad": float(best[deg + 2]),
            "hier_frac": float(best[deg + 3]),
        },
        "method": "dynesty_nested_geometric",
    }
    scanner.results["nested_tau_sb"] = out
    return out


def run_dynesty_ade_physical(
    scanner: TauSBScanner,
    z: np.ndarray,
    observable: np.ndarray,
    err: np.ndarray | None = None,
    cov: np.ndarray | None = None,
    *,
    nlive: int = NESTED_NLIVE_DEFAULT,
    max_samples: int = NESTED_MAX_SAMPLES_DEFAULT,
    seed: int = 43,
) -> dict[str, Any]:
    """Nested sampling for aDE with physical priors (A≤0.05, z*>0)."""
    _require_dynesty()
    import dynesty

    z = np.asarray(z, dtype=float)
    obs = np.asarray(observable, dtype=float)
    gamma_used = scanner._gamma_for(z)
    s = s_from_z(z, gamma=gamma_used)
    amp_scale = _oscillation_scale(obs)
    deg = _baseline_degree(len(z), extra_params=5)
    ndim = deg + 5

    poly = np.polyfit(z, obs, deg)
    poly_spread = np.maximum(np.abs(poly) * 0.25, np.std(obs) * 0.05)

    def model(theta: np.ndarray, zz: np.ndarray, _ss: np.ndarray) -> np.ndarray:
        base = np.polyval(theta[: deg + 1], zz)
        ade = ade_dark_energy_modulation(
            zz,
            amplitude=float(theta[deg + 1]) * amp_scale * 0.01,
            omega=float(theta[deg + 2]),
            phase=float(theta[deg + 3]),
            z_star=max(float(theta[deg + 4]), ADE_ZSTAR_MIN),
        )
        return base + ade

    logl = _loglikelihood_factory(
        model_builder=model, z=z, obs=obs, err=err, cov=cov, s=s
    )

    def prior_transform(u: np.ndarray) -> np.ndarray:
        return _ade_prior_transform(
            u, deg=deg, poly_center=poly, poly_spread=poly_spread
        )

    sampler = dynesty.NestedSampler(
        logl,
        prior_transform,
        ndim,
        nlive=nlive,
        bound="multi",
        sample="rwalk",
        rstate=np.random.default_rng(int(seed)),
    )
    sampler.run_nested(maxiter=max_samples, print_progress=False)
    res = sampler.results
    logz = float(res.logz[-1])
    logz_err = float(res.logzerr[-1]) if len(res.logzerr) else float("nan")

    out = {
        "model": "aDE (physical priors)",
        "log_evidence": logz,
        "log_evidence_err": logz_err,
        "n_live": nlive,
        "ndim": ndim,
        "gamma_used": gamma_used,
        "priors": {
            "A_osc_max_frac": ADE_AOSC_MAX_FRAC,
            "z_star_min": ADE_ZSTAR_MIN,
        },
        "method": "dynesty_nested_ade_physical",
    }
    scanner.results["nested_ade"] = out
    return out


def run_nested_evidence_comparison(
    scanner: TauSBScanner,
    z: np.ndarray,
    observable: np.ndarray,
    err: np.ndarray | None = None,
    cov: np.ndarray | None = None,
    *,
    nlive: int = NESTED_NLIVE_DEFAULT,
    max_samples: int = NESTED_MAX_SAMPLES_DEFAULT,
    output_prefix: str = "tau_sb_nested",
    use_jax_geometric: bool = True,
    cov_already_augmented: bool = False,
    quantity: str = "DH_over_rs",
) -> dict[str, Any]:
    """
    Run Tau-SB (geometric) and aDE (physical) nested sampling; compare ln Z.

    When ``use_jax_geometric`` is True (default), uses the JAX 4-parameter cylinder
    likelihood from session 2026-07-09. Falls back to fit-aligned dynesty if JAX
    is not installed.
    """
    if use_jax_geometric:
        try:
            from menus.astronomical.desi.jax_likelihood import (
                JAX_AVAILABLE,
                NESTED_MAX_SAMPLES_JAX_DEFAULT,
                NESTED_NLIVE_JAX_DEFAULT,
                run_jax_nested_evidence_comparison,
            )

            if JAX_AVAILABLE:
                report = run_jax_nested_evidence_comparison(
                    z,
                    observable,
                    err=err,
                    cov=cov,
                    quantity=quantity,
                    cov_already_augmented=cov_already_augmented,
                    nlive=int(nlive or NESTED_NLIVE_JAX_DEFAULT),
                    max_samples=int(max_samples or NESTED_MAX_SAMPLES_JAX_DEFAULT),
                    output_prefix=output_prefix.replace("tau_sb_nested", "tau_sb_jax_nested")
                    if output_prefix == "tau_sb_nested"
                    else output_prefix,
                )
                scanner.results["nested_evidence"] = report
                scanner.results["nested_tau_sb"] = report["tau_sb"]
                scanner.results["nested_ade"] = report["ade"]
                return report
        except ImportError:
            print(
                "[Nested sampling] JAX not available; using fit-aligned dynesty "
                "(pip install jax jaxlib)"
            )

    print("\n[Nested sampling] Tau-SB geometric priors …")
    tau = run_dynesty_tau_sb_geometric(
        scanner,
        z,
        observable,
        err,
        cov,
        nlive=nlive,
        max_samples=max_samples,
        seed=42,
    )
    print(
        f"  ln Z(Tau-SB) = {tau['log_evidence']:.2f} "
        f"± {tau['log_evidence_err']:.2f}"
    )

    print("[Nested sampling] aDE physical priors …")
    ade = run_dynesty_ade_physical(
        scanner,
        z,
        observable,
        err,
        cov,
        nlive=nlive,
        max_samples=max_samples,
        seed=43,
    )
    print(
        f"  ln Z(aDE)    = {ade['log_evidence']:.2f} "
        f"± {ade['log_evidence_err']:.2f}"
    )

    delta = float(tau["log_evidence"] - ade["log_evidence"])
    favor = "Tau-SB" if delta > 0 else "aDE"
    n_data = len(np.asarray(z, dtype=float))
    ev = apply_evidence_corrections(
        log_evidence_tau=float(tau["log_evidence"]),
        log_evidence_ade=float(ade["log_evidence"]),
        n_data=n_data,
        ndim_tau=int(tau.get("ndim", 0)),
        ndim_ade=int(ade.get("ndim", 0)),
    )
    print(
        f"[Nested sampling] Δln Z = {delta:+.2f} → favors {favor} "
        f"(raw evidence)"
    )
    print(
        f"[Nested sampling] BIC-corrected Δln Z = "
        f"{ev['delta_log_evidence_bic']:+.2f} "
        f"(k_τ={ev['k_tau']}, k_aDE={ev['k_ade']}, n={n_data}) "
        f"→ favors {ev['favored_model_bic']}"
    )
    if ev["occam_penalty_tau_minus_ade"]:
        print(
            f"[Nested sampling] Sparse Occam Δln Z = "
            f"{ev['delta_log_evidence_occam_corrected']:+.2f} "
            f"(penalty={ev['occam_penalty_tau_minus_ade']:.2f}, n<{SPARSE_N_THRESHOLD}) "
            f"→ favors {ev['favored_model_sparse']}"
        )

    report = {
        "tau_sb": tau,
        "ade": ade,
        "delta_log_evidence_tau_minus_ade": delta,
        "favored_model": favor,
        "n_data": n_data,
        **ev,
        "timestamp": artifact_timestamp(),
    }
    out_path = artifact_path(
        TestSlug.NESTED_SAMPLING,
        compose_dataset_slug(output_prefix),
        "evidence",
        "json",
    )
    from menus.astronomical.desi.json_util import write_json

    write_json(out_path, report, indent=2, sort_keys=True)
    report["report_path"] = str(out_path)
    scanner.results["nested_evidence"] = report
    return report