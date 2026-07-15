"""
Geometry-anchored theory vectors and Gaussian likelihood for Tau-SB DESI fits.

Implements Eq. (2) dispersion → BAO distance mapping and cylinder-derived
μ(θ) corrections (2026-07-09 pipeline + 2026-07-11 DR2 evaluation).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from menus.astronomical.desi.scanner import (
    GAMMA_REFERENCE,
    GAMMA6_HEX,
    H0_H_UNITS,
    KAPPA_LEAK_DEFAULT,
    LATE_UNIVERSE_DELTA_N,
    M0_MEV,
    N_HIER_BINDING,
    OMEGA_M_FIDUCIAL,
    RD_OBSERVED_MPC,
    R_TAU_MPC,
    TAU_RESONANCE_PERIOD,
    comoving_distance_hmpc,
    hubble_distance_hmpc,
    s_from_z,
)

# Fixed geometric anchors (Tav cylinder + hexametric Lorentz sector)
DELTA_GAMMA_DOMAIN: float = 1.12
FRICTION_P_EXPONENT: float = -2.5
KAPPA_LEAK: float = KAPPA_LEAK_DEFAULT


def bao_ratio_from_dispersion(
    z: np.ndarray | list[float],
    *,
    quantity: str = "DM_over_rs",
    kappa_leak: float = KAPPA_LEAK_DEFAULT,
    rd_mpc: float = RD_OBSERVED_MPC,
    rho_aeon: float = 1.0,
    r_tau: float = R_TAU_MPC,
    gamma6: float = GAMMA6_HEX,
    i_struct: float = 1.0,
    gamma_stretch: float = GAMMA_REFERENCE,
    om: float = OMEGA_M_FIDUCIAL,
    h0_h: float = H0_H_UNITS,
) -> np.ndarray:
    """
    Map Eq. (2) photon dispersion stretch to BAO distance ratios D/r_d.

    D_Tav(z) = D_ΛCDM(z) × [λ(z)/λ₀] with λ from cumulative κ_leak integral.
    """
    from menus.astronomical.desi.redshift_law import lambda_ratio_tsb

    z_arr = np.asarray(z, dtype=float)
    stretch = lambda_ratio_tsb(
        z_arr,
        kappa_leak=kappa_leak,
        rho_aeon=rho_aeon,
        r_tau=r_tau,
        gamma6=gamma6,
        i_struct=i_struct,
        gamma_stretch=gamma_stretch,
    )
    q = str(quantity).upper()
    if q.startswith("DH"):
        dist = hubble_distance_hmpc(z_arr, om=om, h0_h=h0_h)
    else:
        dist = comoving_distance_hmpc(z_arr, om=om, h0_h=h0_h)
    return dist * stretch / float(rd_mpc)


def tau_sb_mu(
    z: np.ndarray | list[float],
    *,
    observable_scale: float | None = None,
    quantity: str = "DH_over_rs",
    R_tau: float = R_TAU_MPC,
    gamma6: float = GAMMA6_HEX,
    n_hier: float = N_HIER_BINDING,
    gamma: float = GAMMA_REFERENCE,
    delta_gamma: float = DELTA_GAMMA_DOMAIN,
    delta_n: float = LATE_UNIVERSE_DELTA_N,
    kappa_leak: float = KAPPA_LEAK_DEFAULT,
    use_dispersion: bool = True,
    use_ekk_geometry: bool = False,
) -> np.ndarray:
    """
    Theory prediction vector μ_i for BAO ratios from Tav-cylinder geometry.

    Primary path: Eq. (2) dispersion → D_M/r_d or D_H/r_d, or full EKK chain when
    ``use_ekk_geometry=True`` (plasma r_d + exclusion + binding w_eff).
    Optional small 1/7 phase-slip and Δn hierarchical corrections on top.
    """
    z_arr = np.asarray(z, dtype=float)
    if use_ekk_geometry:
        from menus.astronomical.desi.bao_ekk_geometry import (
            EKKBAOParams,
            predict_bao_observables,
        )

        params = EKKBAOParams(gamma_stretch=gamma)
        quants = [quantity] * len(z_arr)
        pred = predict_bao_observables(z_arr, quants, params=params)
        base = np.asarray(pred["mu"], dtype=float)
        s = s_from_z(z_arr, gamma=gamma, n_hier=n_hier, delta_n=delta_n)
        phase_slip = 0.005 * np.sin(2 * np.pi * s / TAU_RESONANCE_PERIOD)
        hier_step = delta_n * 0.005 * np.sin(2 * np.pi * s / TAU_RESONANCE_PERIOD)
        return base * (1.0 + phase_slip + hier_step)
    if use_dispersion:
        base = bao_ratio_from_dispersion(
            z_arr,
            quantity=quantity,
            kappa_leak=kappa_leak,
            r_tau=R_tau,
            gamma6=gamma6,
            gamma_stretch=gamma,
        )
    else:
        if observable_scale is None:
            observable_scale = 18.0 if quantity.startswith("DM") else 15.0
        s = s_from_z(z_arr, gamma=gamma, n_hier=n_hier, delta_n=delta_n)
        disp = (
            kappa_leak
            * delta_gamma
            * np.log1p(z_arr)
            * (R_tau / R_TAU_MPC)
            / gamma6
        )
        phase_slip = 0.01 * np.sin(2 * np.pi * n_hier * np.log1p(z_arr) / TAU_RESONANCE_PERIOD)
        hier_step = delta_n * 0.01 * np.sin(2 * np.pi * s / TAU_RESONANCE_PERIOD)
        z_power = 0.35 if quantity.startswith("DM") else 0.25
        base = observable_scale * np.power(1.0 + z_arr, z_power)
        base = base * np.exp(disp) * (1.0 + phase_slip + hier_step)

    s = s_from_z(z_arr, gamma=gamma, n_hier=n_hier, delta_n=delta_n)
    phase_slip = 0.005 * np.sin(2 * np.pi * s / TAU_RESONANCE_PERIOD)
    hier_step = delta_n * 0.005 * np.sin(2 * np.pi * s / TAU_RESONANCE_PERIOD)
    return base * (1.0 + phase_slip + hier_step)


def generate_tau_sb_mock_bao(
    z_array: np.ndarray | list[float],
    *,
    quantity: str = "DH_over_rs",
    A_inject_frac: float = 0.02,
    rd_mpc: float = RD_OBSERVED_MPC,
    seed: int = 42,
    **mu_kw: Any,
) -> dict[str, Any]:
    """
    Forward-model synthetic BAO vector from cylinder geometry + optional 1/7 injection.

    Returns mean vector, diagonal errors, and full metadata for recovery tests.
    """
    rng = np.random.default_rng(seed)
    z_arr = np.asarray(z_array, dtype=float)
    mu = tau_sb_mu(z_arr, quantity=quantity, **mu_kw)
    s = s_from_z(z_arr, gamma=mu_kw.get("gamma", GAMMA_REFERENCE))
    injected = mu * (1.0 + A_inject_frac * np.sin(2 * np.pi * s / TAU_RESONANCE_PERIOD))
    err_frac = 0.02 + 0.005 * np.sqrt(np.maximum(z_arr, 0.0))
    err = err_frac * injected
    noise = rng.normal(0.0, err)
    obs = injected + noise
    return {
        "z": z_arr.tolist(),
        "observable": obs.tolist(),
        "err": err.tolist(),
        "mu_theory": mu.tolist(),
        "quantity": quantity,
        "rd_mpc": float(rd_mpc),
        "A_inject_frac": float(A_inject_frac),
        "seed": int(seed),
    }


def geometric_likelihood_summary(
    z: np.ndarray | list[float],
    observable: np.ndarray | list[float],
    err: np.ndarray | list[float] | None = None,
    cov: np.ndarray | None = None,
    *,
    quantity: str = "DH_over_rs",
    observable_scale: float | None = None,
) -> dict[str, Any]:
    """Compare data to fixed-geometry tau_sb_mu and return chi2 / logL diagnostics."""
    from menus.astronomical.desi.scanner import gaussian_chi2

    z_arr = np.asarray(z, dtype=float)
    obs = np.asarray(observable, dtype=float)
    scale = observable_scale or float(np.median(np.abs(obs))) or 1.0
    mu = tau_sb_mu(z_arr, observable_scale=scale, quantity=quantity)
    chi2 = float(gaussian_chi2(obs, mu, err=err, cov=cov))
    n = len(z_arr)
    return {
        "chi2": chi2,
        "ndof": max(0, n - 1),
        "reduced_chi2": chi2 / max(1, n - 1),
        "n_data": n,
        "quantity": quantity,
        "kappa_leak": KAPPA_LEAK_DEFAULT,
        "rd_mpc": RD_OBSERVED_MPC,
        "mu_preview": mu.tolist(),
    }