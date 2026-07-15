"""
Test 2 — λ(z) redshift law from photon kinematics (Tav cylinder dispersion).

Compares the geometric stretch law

    λ(z)/λ₀ = exp(κ_leak ∫ (ρ_aeon R_τ / ⟨γ⟩₆ I) dt)

against standard (1+z) and fits Pantheon+ μ(z) + DESI BAO distance data.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy import optimize

from menus.astronomical.desi.fetcher import DEFAULT_COBAYA_ROOT
from menus.astronomical.desi.production import (
    PLANCK_RD_ERR_MPC,
    PLANCK_RD_MPC,
    load_pantheon_plus_subsample,
)
from tav_shared.artifact_paths import TestSlug, artifact_path, artifact_timestamp, compose_dataset_slug
from menus.astronomical.desi.scanner import (
    GAMMA6_HEX,
    H0_FIDUCIAL_KM_S_MPC,
    KAPPA_LEAK_DEFAULT,
    R_TAU_MPC,
    load_desi_from_cobaya_repo,
    s_from_z,
)

# Fiducial photon kinematics constants (session 2026-07-07)
RHO_AEON_FID: float = 1.0
I_STRUCT_FID: float = 1.0
KAPPA_LEAK_PRIOR: tuple[float, float] = (0.0, 0.05)
KAPPA_LEAK_FID: float = KAPPA_LEAK_DEFAULT
GAMMA6_FID: float = GAMMA6_HEX
R_TAU_FID: float = R_TAU_MPC


def dispersion_integrand(
    z: np.ndarray,
    *,
    rho_aeon: float = RHO_AEON_FID,
    r_tau: float = R_TAU_FID,
    gamma6: float = GAMMA6_FID,
    i_struct: float = I_STRUCT_FID,
    gamma_stretch: float = 10.0,
) -> np.ndarray:
    """
    Integrand (ρ_aeon R_τ / ⟨γ⟩₆ I) as a function of z mapped through s(z).
    """
    z = np.asarray(z, dtype=float)
    s = s_from_z(z, gamma=gamma_stretch)
    rate = (rho_aeon * r_tau) / (gamma6 * i_struct)
    return rate * np.ones_like(s)


def lambda_ratio_tsb(
    z: np.ndarray,
    *,
    kappa_leak: float,
    rho_aeon: float = RHO_AEON_FID,
    r_tau: float = R_TAU_FID,
    gamma6: float = GAMMA6_FID,
    i_struct: float = I_STRUCT_FID,
    gamma_stretch: float = 10.0,
) -> np.ndarray:
    """
    λ(z)/λ₀ from cumulative dispersion along the photon worldline.

    Uses trapezoid integration in s = −γ ln(1+z).
    """
    z = np.asarray(z, dtype=float)
    if len(z) == 0:
        return np.array([], dtype=float)
    z_sorted = np.sort(z)
    s = s_from_z(z_sorted, gamma=gamma_stretch)
    integrand = dispersion_integrand(
        z_sorted,
        rho_aeon=rho_aeon,
        r_tau=r_tau,
        gamma6=gamma6,
        i_struct=i_struct,
        gamma_stretch=gamma_stretch,
    )
    ds = np.diff(s, prepend=s[0])
    cumulative = np.cumsum(0.5 * (integrand + np.roll(integrand, 1)) * ds)
    cumulative[0] = 0.0
    stretch_sorted = np.exp(kappa_leak * cumulative)
    return np.interp(z, z_sorted, stretch_sorted)


def lambda_ratio_lcdm(z: np.ndarray) -> np.ndarray:
    """Standard cosmological stretch: λ(z)/λ₀ = 1 + z."""
    return 1.0 + np.asarray(z, dtype=float)


def mu_from_lambda_ratio(
    z: np.ndarray,
    stretch: np.ndarray,
    *,
    h0: float = H0_FIDUCIAL_KM_S_MPC,
) -> np.ndarray:
    """
    Distance modulus proxy: μ(z) ≈ 5 log₁₀(D_L/10 pc) with D_L ∝ stretch × D_L^ΛCDM.
    """
    z = np.asarray(z, dtype=float)
    stretch = np.asarray(stretch, dtype=float)
    # Base ΛCDM luminosity distance shape
    n = 64
    zg = np.linspace(0, float(np.max(z)) * 1.2 + 0.01, n)
    from menus.astronomical.desi.scanner import OMEGA_M_FIDUCIAL

    om = OMEGA_M_FIDUCIAL
    ez = np.sqrt(om * (1 + zg) ** 3 + (1 - om))
    integrand = 1.0 / ez
    dz = np.diff(zg, prepend=0)
    dc = (299792.458 / h0) * np.cumsum(0.5 * (integrand + np.roll(integrand, 1)) * dz)
    dc[0] = 0.0
    dc_z = np.interp(z, zg, dc)
    dl = stretch * dc_z * (1 + z)
    return 5.0 * np.log10(np.maximum(dl, 1e-6)) + 25.0


def fit_redshift_law(
    z_sn: np.ndarray,
    mu_sn: np.ndarray,
    mu_err: np.ndarray,
    z_bao: np.ndarray,
    dm_bao: np.ndarray,
    dm_err: np.ndarray,
    *,
    kappa_bounds: tuple[float, float] = KAPPA_LEAK_PRIOR,
) -> dict[str, Any]:
    """
    Fit κ_leak for TSB λ(z) vs fixed ΛCDM (1+z) on SN + BAO jointly.
    """
    z_sn = np.asarray(z_sn, dtype=float)
    mu_sn = np.asarray(mu_sn, dtype=float)
    mu_err = np.asarray(mu_err, dtype=float)
    z_bao = np.asarray(z_bao, dtype=float)
    dm_bao = np.asarray(dm_bao, dtype=float)
    dm_err = np.asarray(dm_err, dtype=float)

    def chi2_tsb(kappa: float) -> float:
        stretch_sn = lambda_ratio_tsb(z_sn, kappa_leak=kappa)
        mu_pred = mu_from_lambda_ratio(z_sn, stretch_sn)
        chi2_sn = float(np.sum(((mu_sn - mu_pred) / mu_err) ** 2))
        stretch_bao = lambda_ratio_tsb(z_bao, kappa_leak=kappa)
        # BAO D_M/r_d proxy scales with stretch
        dm_pred = dm_bao[0] * stretch_bao / stretch_bao[0] if len(z_bao) else stretch_bao
        chi2_bao = float(np.sum(((dm_bao - dm_pred) / dm_err) ** 2))
        return chi2_sn + chi2_bao

    def chi2_lcdm() -> float:
        stretch_sn = lambda_ratio_lcdm(z_sn)
        mu_pred = mu_from_lambda_ratio(z_sn, stretch_sn)
        chi2_sn = float(np.sum(((mu_sn - mu_pred) / mu_err) ** 2))
        stretch_bao = lambda_ratio_lcdm(z_bao)
        dm_pred = dm_bao[0] * stretch_bao / stretch_bao[0] if len(z_bao) else stretch_bao
        chi2_bao = float(np.sum(((dm_bao - dm_pred) / dm_err) ** 2))
        return chi2_sn + chi2_bao

    res = optimize.minimize_scalar(
        chi2_tsb,
        bounds=kappa_bounds,
        method="bounded",
    )
    kappa_best = float(res.x)
    chi2_tsb_best = float(res.fun)
    chi2_lcdm_val = chi2_lcdm()
    delta_chi2 = chi2_lcdm_val - chi2_tsb_best
    n_data = len(z_sn) + len(z_bao)

    return {
        "kappa_leak_best": kappa_best,
        "chi2_tsb": chi2_tsb_best,
        "chi2_lcdm": chi2_lcdm_val,
        "delta_chi2_lcdm_minus_tsb": float(delta_chi2),
        "n_data": int(n_data),
        "n_sne": len(z_sn),
        "n_bao": len(z_bao),
        "favored_model": "TSB_λ(z)" if delta_chi2 > 2.0 else "ΛCDM_(1+z)",
        "verdict": (
            "TSB REDSHIFT LAW PREFERRED"
            if delta_chi2 > 3.0
            else "INCONCLUSIVE"
            if abs(delta_chi2) <= 3.0
            else "ΛCDM PREFERRED"
        ),
        "geometric_constants": {
            "R_tau_mpc": R_TAU_FID,
            "gamma6": GAMMA6_FID,
            "rho_aeon": RHO_AEON_FID,
            "I_struct": I_STRUCT_FID,
        },
    }


def run_redshift_law_test(
    *,
    cobaya_path: str | Path | None = None,
    tracer: str = "ALL_GCcomb",
    max_sne: int = 80,
    output_prefix: str = "redshift_law_test2",
    plot: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Test 2 entry point — fit λ(z) on Pantheon+ + DESI BAO."""
    cobaya_path = cobaya_path or DEFAULT_COBAYA_ROOT
    pantheon = load_pantheon_plus_subsample(max_sne=max_sne)
    bao = load_desi_from_cobaya_repo(
        cobaya_path,
        tracer=tracer,
        quantity_filter="DM_over_rs",
    )

    fit = fit_redshift_law(
        pantheon["z"],
        pantheon["mu"],
        pantheon["mu_err"],
        np.asarray(bao["z"], dtype=float),
        np.asarray(bao["observable"], dtype=float),
        np.asarray(bao["err"], dtype=float),
    )

    z_plot = np.linspace(0.01, 1.5, 100)
    stretch_tsb = lambda_ratio_tsb(z_plot, kappa_leak=fit["kappa_leak_best"])
    stretch_lcdm = lambda_ratio_lcdm(z_plot)

    report = {
        "action": "Test 2 Redshift Law λ(z)",
        "fit": fit,
        "planck_rd_prior_mpc": PLANCK_RD_MPC,
        "planck_rd_err_mpc": PLANCK_RD_ERR_MPC,
        "pantheon_source": pantheon.get("source"),
        "bao_tracer": tracer,
        "z_samples": z_plot.tolist(),
        "stretch_tsb": stretch_tsb.tolist(),
        "stretch_lcdm": stretch_lcdm.tolist(),
        "timestamp": artifact_timestamp(),
    }

    if verbose:
        print("=" * 70)
        print("TEST 2 — λ(z) REDSHIFT LAW")
        print(f"  κ_leak best : {fit['kappa_leak_best']:.6f}")
        print(f"  χ² TSB      : {fit['chi2_tsb']:.2f}")
        print(f"  χ² ΛCDM     : {fit['chi2_lcdm']:.2f}")
        print(f"  Δχ²         : {fit['delta_chi2_lcdm_minus_tsb']:+.2f}")
        print(f"  Verdict     : {fit['verdict']}")
        print("=" * 70)

    from menus.astronomical.desi.json_util import write_json

    json_path = artifact_path(
        TestSlug.REDSHIFT_LAW,
        compose_dataset_slug(tracer, output_prefix),
        "report",
        "json",
    )
    write_json(json_path, report, indent=2, sort_keys=True)
    report["report_path"] = str(json_path)

    if plot:
        from menus.astronomical.desi.lss_visualization import plot_redshift_law

        report["plot_path"] = plot_redshift_law(report, prefix=output_prefix)

    return report