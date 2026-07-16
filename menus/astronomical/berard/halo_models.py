"""
Rotation-curve halo models: pISO, hybrid Berard–inertia + cored halo, baselines.

Unit handling follows tau-cosmology/tests/test_units.py (rho0_pc * 1e9).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from menus.astronomical.berard.constants import BERARD_CONSTANT

G_KPC = 4.30091e-6  # kpc (km/s)^2 / M_sun
LN7 = math.log(7.0)
MOND_A0_KMS2 = 1.2e-10 * 3.24078e-17 * (3.086e13) ** 2  # ~1.2e-10 m/s² → km/s² scale proxy


def v_quadrature(*components: np.ndarray | float) -> np.ndarray:
    """Combine velocity components in quadrature."""
    stack = np.stack([np.asarray(c, dtype=float) for c in components], axis=0)
    return np.sqrt(np.sum(stack**2, axis=0))


def v_piso_halo(
    r_kpc: np.ndarray,
    rho0_pc3: float,
    rc_kpc: float,
) -> np.ndarray:
    """Pseudo-isothermal (cored) halo circular speed with correct unit conversion."""
    r = np.maximum(np.asarray(r_kpc, dtype=float), 1e-6)
    rho0_kpc3 = float(rho0_pc3) * 1e9
    term = 1.0 - (float(rc_kpc) / r) * np.arctan(r / float(rc_kpc))
    term = np.maximum(term, 0.0)
    return np.sqrt(4.0 * np.pi * G_KPC * rho0_kpc3 * float(rc_kpc) ** 2 * term)


def v_burkert(
    r_kpc: np.ndarray,
    rho0_pc3: float,
    rc_kpc: float,
) -> np.ndarray:
    """Burkert cored profile (km/s)."""
    r = np.maximum(np.asarray(r_kpc, dtype=float), 1e-6)
    rho0 = float(rho0_pc3) * 1e9
    rc = float(rc_kpc)
    mass_enclosed_proxy = 4.0 * math.pi * rho0 * rc**3 * (
        0.5 * np.log(1.0 + (r / rc) ** 2) + np.arctan(r / rc) - np.arctan(1.0)
    )
    return np.sqrt(np.maximum(G_KPC * mass_enclosed_proxy / r, 0.0))


def v_nfw(
    r_kpc: np.ndarray,
    rho_s: float,
    r_s: float,
) -> np.ndarray:
    """NFW halo circular speed (km/s), rho_s in M_sun/kpc³."""
    r = np.maximum(np.asarray(r_kpc, dtype=float), 1e-6)
    rs = float(r_s)
    x = r / rs
    f = np.log(1.0 + x) - x / (1.0 + x)
    v2 = 4.0 * math.pi * G_KPC * float(rho_s) * rs**3 * f / r
    return np.sqrt(np.maximum(v2, 0.0))


def mond_simple_mu(g: np.ndarray, a0: float = 1.2e-10) -> np.ndarray:
    """Simple MOND interpolation μ(x)=x/(1+x), g in m/s²."""
    x = np.asarray(g, dtype=float) / float(a0)
    return x / (1.0 + x)


def v_baryons_only(
    r_kpc: np.ndarray,
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
    *,
    upsilon_disk: float = 0.5,
    upsilon_bulge: float = 0.5,
) -> np.ndarray:
    """Baryonic contribution only (no halo)."""
    return v_quadrature(
        np.asarray(v_gas, dtype=float),
        float(upsilon_disk) * np.asarray(v_disk, dtype=float),
        float(upsilon_bulge) * np.asarray(v_bulge, dtype=float),
    )


def v_berard_scaled_baryons(
    v_baryon: np.ndarray,
    *,
    bc: float = BERARD_CONSTANT,
    s0: float = 1.0,
    exponent: int = -2,
) -> np.ndarray:
    """
    Scale baryonic speeds for Berard modified inertia at homogeneous S₀.

    Manuscript convention (exponent=-2, S₀=1): m_eff = m BC² → v = v_N / BC.
    """
    _ = exponent
    denom = max(float(bc) * float(s0), 1e-9)
    return np.asarray(v_baryon, dtype=float) / denom


def v_hybrid_berard_piso(
    r_kpc: np.ndarray,
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
    *,
    rho0_pc3: float,
    rc_kpc: float,
    upsilon_disk: float = 0.5,
    upsilon_bulge: float = 0.5,
    bc: float = BERARD_CONSTANT,
) -> np.ndarray:
    """Hybrid: Berard-scaled baryons + pseudo-isothermal hidden-sector halo."""
    v_bar = v_baryons_only(
        r_kpc, v_gas, v_disk, v_bulge,
        upsilon_disk=upsilon_disk,
        upsilon_bulge=upsilon_bulge,
    )
    v_bar_berard = v_berard_scaled_baryons(v_bar, bc=bc)
    v_h = v_piso_halo(r_kpc, rho0_pc3, rc_kpc)
    return v_quadrature(v_bar_berard, v_h)


def v_plain_piso_total(
    r_kpc: np.ndarray,
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
    *,
    rho0_pc3: float,
    rc_kpc: float,
    upsilon_disk: float = 0.5,
    upsilon_bulge: float = 0.5,
) -> np.ndarray:
    """Plain pISO + baryons (no Berard inertia scaling)."""
    v_bar = v_baryons_only(
        r_kpc, v_gas, v_disk, v_bulge,
        upsilon_disk=upsilon_disk,
        upsilon_bulge=upsilon_bulge,
    )
    v_h = v_piso_halo(r_kpc, rho0_pc3, rc_kpc)
    return v_quadrature(v_bar, v_h)


@dataclass
class ModelSpec:
    name: str
    n_params: int
    kind: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "baryons_only": ModelSpec("baryons_only", 1, "baryons"),
    "plain_piso": ModelSpec("plain_piso", 3, "piso"),
    "berard_hybrid_piso": ModelSpec("berard_hybrid_piso", 3, "berard_piso"),
    "burkert": ModelSpec("burkert", 3, "burkert"),
    "nfw": ModelSpec("nfw", 3, "nfw"),
}


def model_velocity(
    kind: str,
    r_kpc: np.ndarray,
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
    params: dict[str, float],
) -> np.ndarray:
    """Dispatch model kind to circular speed prediction."""
    upsilon = float(params.get("upsilon_disk", 0.5))
    if kind == "baryons":
        return v_baryons_only(
            r_kpc, v_gas, v_disk, v_bulge, upsilon_disk=upsilon
        )
    if kind == "piso":
        return v_plain_piso_total(
            r_kpc, v_gas, v_disk, v_bulge,
            rho0_pc3=float(params["rho0_pc3"]),
            rc_kpc=float(params["rc_kpc"]),
            upsilon_disk=upsilon,
        )
    if kind == "berard_piso":
        return v_hybrid_berard_piso(
            r_kpc, v_gas, v_disk, v_bulge,
            rho0_pc3=float(params["rho0_pc3"]),
            rc_kpc=float(params["rc_kpc"]),
            upsilon_disk=upsilon,
            bc=float(params.get("bc", BERARD_CONSTANT)),
        )
    if kind == "burkert":
        v_bar = v_baryons_only(
            r_kpc, v_gas, v_disk, v_bulge, upsilon_disk=upsilon
        )
        v_h = v_burkert(r_kpc, float(params["rho0_pc3"]), float(params["rc_kpc"]))
        return v_quadrature(v_bar, v_h)
    if kind == "nfw":
        v_bar = v_baryons_only(
            r_kpc, v_gas, v_disk, v_bulge, upsilon_disk=upsilon
        )
        v_h = v_nfw(r_kpc, float(params["rho_s"]), float(params["r_s"]))
        return v_quadrature(v_bar, v_h)
    raise ValueError(f"Unknown model kind: {kind}")


def chi2_reduced(
    v_obs: np.ndarray,
    v_model: np.ndarray,
    e_vobs: np.ndarray,
) -> float:
    """Gaussian χ²_red with positive uncertainties only."""
    obs = np.asarray(v_obs, dtype=float)
    mod = np.asarray(v_model, dtype=float)
    err = np.maximum(np.asarray(e_vobs, dtype=float), 1.0)
    chi2 = float(np.sum(((obs - mod) / err) ** 2))
    return chi2 / max(obs.size, 1)


def information_criteria(
    chi2: float,
    n_data: int,
    n_params: int,
) -> dict[str, float]:
    """AIC and BIC from χ² (Gaussian likelihood, σ absorbed in χ²)."""
    k = int(n_params)
    n = int(n_data)
    aic = float(chi2 + 2 * k)
    bic = float(chi2 + k * math.log(max(n, 2)))
    return {"aic": aic, "bic": bic, "chi2": float(chi2)}


def bic_verdict(delta_bic: float) -> str:
    """Kass & Raftery style labels from pre-registration."""
    if delta_bic > 6.0:
        return "strong_favored"
    if delta_bic < -6.0:
        return "strong_disfavored"
    return "tie"