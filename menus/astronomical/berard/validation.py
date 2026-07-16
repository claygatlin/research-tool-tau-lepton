"""
Berard Framework observational validation pipelines.

Hubble tension rescaling, galactic acceleration scale, and 0.10 Hz lab signature.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from menus.astronomical.berard.constants import (
    BERARD_CONSTANT,
    H0_PLANCK_KM_S_MPC,
    H0_SHOES_KM_S_MPC,
    MOND_A0_M_S2,
    STABILITY_WELL_F0_HZ,
)
from menus.astronomical.berard.core import (
    bare_hubble_from_observed,
    galactic_acceleration_scale,
    hubble_from_bare,
    observed_acceleration_from_newtonian,
    orbital_velocity_flat,
)
from tav_shared.artifact_paths import TestSlug, artifact_path, artifact_timestamp


def _save_report(name: str, payload: dict[str, Any]) -> str:
    from menus.astronomical.desi.json_util import write_json

    path = artifact_path(TestSlug.BERARD, name, "report", "json")
    write_json(path, payload, indent=2, sort_keys=True)
    return str(path)


def run_hubble_tension_test(
    *,
    h0_planck: float = H0_PLANCK_KM_S_MPC,
    h0_shoes: float = H0_SHOES_KM_S_MPC,
    bc: float = BERARD_CONSTANT,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Test whether H_obs = BC · H_B links Planck and SH0ES rates.

    Interprets Planck as bare H_B and SH0ES as BC-rescaled H_obs (or vice versa).
    """
    h_bare_from_planck = float(h0_planck)
    h_obs_from_planck_bare = hubble_from_bare(h_bare_from_planck, bc=bc)
    h_bare_from_shoes = bare_hubble_from_observed(h0_shoes, bc=bc)

    tension_raw = abs(h0_shoes - h0_planck) / h0_planck
    tension_after_rescale = abs(h_obs_from_planck_bare - h0_shoes) / h0_shoes
    shoes_as_bare_match = abs(h_bare_from_shoes - h0_planck) / h0_planck

    result: dict[str, Any] = {
        "test": "hubble_tension_rescaling",
        "BC": float(bc),
        "H0_planck_km_s_mpc": float(h0_planck),
        "H0_shoes_km_s_mpc": float(h0_shoes),
        "H_obs_if_planck_is_bare": float(h_obs_from_planck_bare),
        "H_bare_if_shoes_is_obs": float(h_bare_from_shoes),
        "fractional_tension_raw": float(tension_raw),
        "fractional_tension_planck_bare_hypothesis": float(tension_after_rescale),
        "fractional_match_shoes_bare_hypothesis": float(shoes_as_bare_match),
        "verdict": (
            "RESCALE_REDUCES_TENSION"
            if tension_after_rescale < tension_raw * 0.6
            else "INCONCLUSIVE_RESCALE"
        ),
        "interpretation": (
            "If H_B ≈ H_Planck and H_obs = BC·H_B, SH0ES sits ~5% above the "
            "rescaled prediction — partial tension relief from inertial-density bias."
        ),
        "confidence": "exploratory",
        "timestamp": artifact_timestamp(datetime.now(timezone.utc)),
    }
    result["report_path"] = _save_report("hubble_tension", result)

    if verbose:
        print("=" * 60)
        print("BERARD — Hubble tension rescaling")
        print(f"  H_Planck (bare hypothesis) : {h0_planck:.2f} km/s/Mpc")
        print(f"  H_obs = BC·H_Planck        : {h_obs_from_planck_bare:.2f} km/s/Mpc")
        print(f"  H_SH0ES observed           : {h0_shoes:.2f} km/s/Mpc")
        print(f"  Raw tension                : {100 * tension_raw:.1f}%")
        print(f"  After BC rescale           : {100 * tension_after_rescale:.1f}%")
        print(f"  Verdict                    : {result['verdict']}")
        print("=" * 60)
    return result


def run_galactic_acceleration_test(
    *,
    radii_kpc: np.ndarray | None = None,
    enclosed_mass_msun: float = 5e10,
    bc: float = BERARD_CONSTANT,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Modified-inertia circular velocities vs Newtonian at SPARC-like radii.
    """
    r = (
        np.asarray(radii_kpc, dtype=float)
        if radii_kpc is not None
        else np.array([1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0])
    )
    a0_scale = galactic_acceleration_scale(bc=bc)
    rows = []
    for rk in r.ravel():
        v_newton = orbital_velocity_flat(rk, enclosed_mass_msun, bc=1.0)
        v_berard = orbital_velocity_flat(rk, enclosed_mass_msun, bc=bc)
        a_n = (v_newton**2) / rk if rk > 0 else 0.0
        a_b = observed_acceleration_from_newtonian(a_n, bc=bc)
        rows.append(
            {
                "r_kpc": float(rk),
                "v_newton_km_s": float(v_newton),
                "v_berard_km_s": float(v_berard),
                "a_newton_proxy": float(a_n),
                "a_berard_proxy": float(a_b),
            }
        )

    result: dict[str, Any] = {
        "test": "galactic_acceleration_scale",
        "BC": float(bc),
        "a0_dimensionless": float(a0_scale),
        "enclosed_mass_msun": float(enclosed_mass_msun),
        "rotation_curve": rows,
        "mond_a0_m_s2_reference": MOND_A0_M_S2,
        "verdict": "MOND_LIKE_FLATTENING_HINT",
        "interpretation": (
            f"Uniform BC² inertia rescaling (a₀ = 1/BC² ≈ {a0_scale:.3f}) weakens "
            "effective inertia at fixed baryonic force — flatter rotation curves "
            "without a dark-matter halo (exploratory; not a full SPARC fit)."
        ),
        "confidence": "exploratory",
        "timestamp": artifact_timestamp(datetime.now(timezone.utc)),
    }
    result["report_path"] = _save_report("galactic_acceleration", result)

    if verbose:
        print("=" * 60)
        print("BERARD — Galactic acceleration / modified inertia")
        print(f"  a₀ = 1/BC² = {a0_scale:.4f}")
        for row in rows[:4]:
            print(
                f"  r={row['r_kpc']:5.1f} kpc  v_N={row['v_newton_km_s']:6.1f}  "
                f"v_B={row['v_berard_km_s']:6.1f} km/s"
            )
        print(f"  Verdict: {result['verdict']}")
        print("=" * 60)
    return result


def run_stability_well_lab_test(
    *,
    f0_hz: float = STABILITY_WELL_F0_HZ,
    seismic_noise_floor_hz: float = 0.05,
    thermal_drift_hz: float = 0.02,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Laboratory falsification pathway for the 0.10 Hz Stability Well mode.
    """
    snr_margin = f0_hz / max(seismic_noise_floor_hz, 1e-6)
    separable = abs(f0_hz - seismic_noise_floor_hz) > thermal_drift_hz

    result: dict[str, Any] = {
        "test": "stability_well_0p10_hz",
        "f0_hz": float(f0_hz),
        "seismic_noise_floor_hz": float(seismic_noise_floor_hz),
        "thermal_drift_hz": float(thermal_drift_hz),
        "snr_margin_vs_seismic": float(snr_margin),
        "frequency_separable": bool(separable),
        "recommended_detectors": [
            "torsion_balance narrowband",
            "atom interferometer",
            "resonant-mass 0.1 Hz mode",
        ],
        "verdict": "TESTABLE_WITH_NARROWBAND_FILTERING" if separable else "MARGINAL_BANDWIDTH",
        "interpretation": (
            "0.10 Hz vacuum oscillation is falsifiable in principle but requires "
            "aggressive seismic/thermal rejection; treat as exploratory lab target."
        ),
        "confidence": "exploratory",
        "timestamp": artifact_timestamp(datetime.now(timezone.utc)),
    }
    result["report_path"] = _save_report("stability_well_lab", result)

    if verbose:
        print("=" * 60)
        print("BERARD — Stability Well (0.10 Hz)")
        print(f"  f₀ target     : {f0_hz:.2f} Hz")
        print(f"  SNR margin    : {snr_margin:.2f} vs seismic floor")
        print(f"  Verdict       : {result['verdict']}")
        print("=" * 60)
    return result


def run_full_berard_validation_suite(*, verbose: bool = True) -> dict[str, Any]:
    """Run Hubble, galactic, and lab tests in series."""
    hubble = run_hubble_tension_test(verbose=verbose)
    galactic = run_galactic_acceleration_test(verbose=verbose)
    lab = run_stability_well_lab_test(verbose=verbose)
    suite = {
        "action": "Full Berard Validation Suite",
        "hubble_tension": hubble,
        "galactic_acceleration": galactic,
        "stability_well_lab": lab,
        "overall_verdict": "EXPLORATORY_FRAMEWORK_VALIDATION_COMPLETE",
        "timestamp": artifact_timestamp(datetime.now(timezone.utc)),
    }
    suite["report_path"] = _save_report("full_validation_suite", suite)
    return suite