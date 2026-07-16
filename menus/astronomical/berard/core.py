"""
Berard Framework calculators — modified inertia, cosmology, galactic dynamics.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from menus.astronomical.berard.constants import (
    BERARD_CONSTANT,
    BERARD_EQUATIONS,
    DEFAULT_INERTIA_EXPONENT,
    FRAMEWORK_METADATA,
    H0_PLANCK_KM_S_MPC,
    H0_SHOES_KM_S_MPC,
    MOND_A0_M_S2,
    S0_PRESENT_EPOCH,
    STABILITY_WELL_F0_HZ,
)


def resonance_invariant(phi: float, phi_0: float) -> float:
    """Dimensionless Resonance Invariant S_0 = φ / φ_0."""
    if phi_0 == 0.0:
        raise ValueError("phi_0 must be non-zero")
    return float(phi / phi_0)


def effective_inertial_mass(
    bare_mass: float,
    *,
    s0: float = S0_PRESENT_EPOCH,
    bc: float = BERARD_CONSTANT,
    exponent: int = DEFAULT_INERTIA_EXPONENT,
) -> float:
    """m_eff = m · BC² · S_0^exponent (exponent -2 in manuscript fit)."""
    return float(bare_mass * (bc**2) * (s0**exponent))


def einstein_berard_energy(
    bare_mass: float,
    *,
    s0: float = S0_PRESENT_EPOCH,
    bc: float = BERARD_CONSTANT,
    exponent: int = DEFAULT_INERTIA_EXPONENT,
) -> float:
    """Rest energy E = m_eff in natural units."""
    return effective_inertial_mass(bare_mass, s0=s0, bc=bc, exponent=exponent)


def effective_density(rho: float, *, bc: float = BERARD_CONSTANT) -> float:
    """ρ_eff = ρ · BC² for homogeneous FRW background."""
    return float(rho * (bc**2))


def hubble_from_bare(h_bare_km_s_mpc: float, *, bc: float = BERARD_CONSTANT) -> float:
    """H_obs = BC · H_B."""
    return float(bc * h_bare_km_s_mpc)


def bare_hubble_from_observed(
    h_obs_km_s_mpc: float, *, bc: float = BERARD_CONSTANT
) -> float:
    """H_B = H_obs / BC."""
    return float(h_obs_km_s_mpc / bc)


def galactic_acceleration_scale(*, bc: float = BERARD_CONSTANT) -> float:
    """Dimensionless a_0 ≡ 1/BC² scaling for modified inertia."""
    return float(1.0 / (bc**2))


def observed_acceleration_from_newtonian(
    a_newton: float, *, bc: float = BERARD_CONSTANT
) -> float:
    """a_B = a_Newton / BC² (weaker effective inertia at fixed force)."""
    return float(a_newton / (bc**2))


def orbital_velocity_flat(
    r_kpc: float,
    enclosed_mass_msun: float,
    *,
    bc: float = BERARD_CONSTANT,
    g_msun_kpc_km2_s2: float = 4.302e-6,
) -> float:
    """
    Circular velocity (km/s) with Berard-modified inertia at radius r.

    v² = G M / r with effective m cancelling in F=ma when inertia scales uniformly.
    """
    r = float(r_kpc)
    if r <= 0.0:
        return 0.0
    v2 = g_msun_kpc_km2_s2 * float(enclosed_mass_msun) / r
    return float(math.sqrt(max(v2, 0.0)))


def stability_well_angular_frequency(*, f0_hz: float = STABILITY_WELL_F0_HZ) -> float:
    """m_φ = 2π f_0 in natural units with ℏ=1 convention (returns rad/s proxy)."""
    return float(2.0 * math.pi * f0_hz)


def scalar_potential_deviation(phi: float, phi_0: float, *, f0_hz: float = STABILITY_WELL_F0_HZ) -> float:
    """Quadratic V(φ) = ½ m_φ² (φ − φ_0)² with m_φ = 2π f_0."""
    m_phi = stability_well_angular_frequency(f0_hz=f0_hz)
    delta = float(phi - phi_0)
    return float(0.5 * (m_phi**2) * (delta**2))


def equation_registry_report() -> dict[str, Any]:
    """Serializable equation + constant summary for JSON artifacts."""
    return {
        "framework": FRAMEWORK_METADATA,
        "constants": {
            "BC": BERARD_CONSTANT,
            "f0_hz": STABILITY_WELL_F0_HZ,
            "S0_present": S0_PRESENT_EPOCH,
            "inertia_exponent_default": DEFAULT_INERTIA_EXPONENT,
            "BC_squared": BERARD_CONSTANT**2,
            "galactic_scale_a0": galactic_acceleration_scale(),
        },
        "equations": BERARD_EQUATIONS,
        "reference_hubble_km_s_mpc": {
            "planck": H0_PLANCK_KM_S_MPC,
            "shoes": H0_SHOES_KM_S_MPC,
        },
        "mond_a0_m_s2_reference": MOND_A0_M_S2,
    }


def demo_parameter_sweep(
    s0_values: np.ndarray | None = None,
    *,
    bare_mass: float = 1.0,
    bc: float = BERARD_CONSTANT,
) -> dict[str, Any]:
    """Sweep S_0 to show inertial rescaling away from present vacuum."""
    s0 = (
        np.asarray(s0_values, dtype=float)
        if s0_values is not None
        else np.linspace(0.85, 1.15, 13)
    )
    rows = []
    for val in s0.ravel():
        rows.append(
            {
                "S_0": float(val),
                "m_eff_inverse_exp": effective_inertial_mass(
                    bare_mass, s0=float(val), bc=bc, exponent=-2
                ),
                "m_eff_readme_exp": effective_inertial_mass(
                    bare_mass, s0=float(val), bc=bc, exponent=2
                ),
            }
        )
    return {"bare_mass": bare_mass, "BC": bc, "s0_sweep": rows}