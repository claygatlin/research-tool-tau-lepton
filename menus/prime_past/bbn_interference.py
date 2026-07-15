"""
Minimal BBN with Tav/Superblock cylinder interference on the expansion rate.

Integrates abundances vs ln(T) from T_START → T_FINAL (radiation-era cooling).
Used by Prime Past Harmonic → BBN Interference Scan in research_tool.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tav_shared.tav_project_paths import TAU_SUPERBLOCK_ROOT

ARTIFACTS_DIR = TAU_SUPERBLOCK_ROOT / "artifacts" / "prime_past"

# Standard BBN / cosmology anchors
ETA_BBN: float = 6.1e-10
T_START_MEV: float = 0.8
T_FINAL_MEV: float = 0.01
M_PL_MEV: float = 1.22e22
G_STAR: float = 10.75
Q_NP_MEV: float = 1.293
B_D_MEV: float = 2.22

DEFAULT_FRAMEWORK_PARAMS: dict[str, float] = {
    "A": 0.015,
    "delta_phi_cyl": 0.0,
    "delta_k_wind": 0.469,
    "xi": 8.0,
}

# Script-facing alias (minimal_bbn_tav.py, notebooks)
FRAMEWORK_PARAMS: dict[str, float] = DEFAULT_FRAMEWORK_PARAMS.copy()


@dataclass
class FrameworkParams:
    A: float = 0.015
    delta_phi_cyl: float = 0.0
    delta_k_wind: float = 0.469
    xi: float = 8.0

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> FrameworkParams:
        raw = raw or {}
        base = DEFAULT_FRAMEWORK_PARAMS.copy()
        for key in base:
            if key in raw and str(raw[key]).strip():
                try:
                    base[key] = float(raw[key])
                except ValueError:
                    pass
        return cls(**base)


def ensure_artifacts_dir() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


def tav_interference_delta(T: float, params: FrameworkParams) -> float:
    """Fractional Hubble perturbation from Tau cylinder interference."""
    T = max(float(T), 1e-4)
    s = np.log(T_START_MEV / T)
    phase = params.delta_phi_cyl + params.delta_k_wind
    envelope = np.exp(-s / max(params.xi, 0.1))
    return params.A * np.cos(phase) * envelope


def hubble_rate_MeV(T: float, params: FrameworkParams) -> float:
    """Radiation-dominated H(T) in MeV (natural units), with Tav interference."""
    T = max(float(T), 1e-4)
    H_std = 1.66 * np.sqrt(G_STAR) * (T**2) / M_PL_MEV
    return float(H_std * (1.0 + tav_interference_delta(T, params)))


def _y_n_equilibrium(T: float) -> float:
    return 1.0 / (1.0 + np.exp(Q_NP_MEV / max(T, 1e-4)))


def _temp_gate(T: float, T_on: float, *, width: float = 0.02) -> float:
    """Smooth turn-on as temperature drops below T_on (MeV)."""
    return float(0.5 * (1.0 + np.tanh((T_on - T) / width)))


def _normalize_abundances(y: np.ndarray) -> np.ndarray:
    """Keep mass fractions in [0, 1] with baryon budget ≤ 1."""
    Y_n, Y_D, Y_He4, Y_Li7 = [max(float(v), 0.0) for v in y]
    total = Y_n + 2.0 * Y_D + 4.0 * Y_He4 + 7.0 * Y_Li7
    if total > 0.999:
        scale = 0.999 / total
        Y_n *= scale
        Y_D *= scale
        Y_He4 *= scale
        Y_Li7 *= scale
    return np.array([Y_n, Y_D, Y_He4, Y_Li7], dtype=float)


def bbn_rhs_tau(tau: float, y: np.ndarray, params: FrameworkParams, *, T0: float) -> np.ndarray:
    """
    RHS for state [Y_n, Y_D, Y_He4, Y_Li7] with tau ≥ 0 and T = T0 * exp(-tau).

    As tau increases, temperature cools and nuclear yields build up.
    """
    T = float(T0 * np.exp(-tau))
    exp_mod = max(1.0 + tav_interference_delta(T, params), 0.05)

    Y_n, Y_D, Y_He4, Y_Li7 = _normalize_abundances(y)
    Y_p = max(1.0 - Y_n - 2.0 * Y_D - 4.0 * Y_He4 - 7.0 * Y_Li7, 1e-12)

    dY_n = 0.0
    dY_D = 0.0
    dY_He4 = 0.0
    dY_Li7 = 0.0

    # Weak freeze-out (approach n/p equilibrium on expansion timescale)
    Y_eq = _y_n_equilibrium(T)
    # Neutron fraction frozen at post-weak-freeze-out value (set at T_START)

    gate_he = _temp_gate(T, 0.18, width=0.025)
    if gate_he > 1e-4:
        # Effective 2n + 2p → He4 channel (post-D bottleneck, illustrative)
        rate = gate_he * 0.28 * Y_n * Y_p / exp_mod
        cap = max(0.26 - Y_He4, 0.0)
        take = min(rate, cap * 0.18, Y_n * 0.48, Y_p * 0.48)
        dY_He4 += take
        dY_n -= 2.0 * take

    gate_li = _temp_gate(T, 0.08, width=0.01)
    if gate_li > 1e-4 and Y_He4 > 1e-6:
        phase = params.delta_phi_cyl + params.delta_k_wind
        mod = 1.0 + params.A * 12.0 * np.cos(phase)
        form_li = gate_li * 1.5e-6 * Y_He4 * mod / exp_mod
        dY_Li7 += form_li

    deriv = np.array([dY_n, dY_D, dY_He4, dY_Li7], dtype=float)
    return np.clip(deriv, -0.08, 0.08)


def run_bbn(
    params: FrameworkParams | dict[str, Any] | None = None,
    *,
    label: str = "BBN run",
    verbose: bool = True,
) -> dict[str, Any]:
    """Evolve abundances from T_START to T_FINAL; return final yields."""
    fw = params if isinstance(params, FrameworkParams) else FrameworkParams.from_mapping(params)

    T0 = T_START_MEV
    # Post-weak-freeze-out n/p (~0.16 n at T ≈ 0.8 MeV)
    Y_n0 = 0.16
    y0 = np.array([Y_n0, 0.0, 0.0, 0.0], dtype=float)

    tau_end = float(np.log(T0 / T_FINAL_MEV))
    tau_grid = np.linspace(0.0, tau_end, 400)
    y = y0.copy()
    for idx in range(len(tau_grid) - 1):
        dtau = tau_grid[idx + 1] - tau_grid[idx]
        k1 = bbn_rhs_tau(tau_grid[idx], y, fw, T0=T0)
        y = _normalize_abundances(y + k1 * dtau)

    Y_n, Y_D, Y_He4, Y_Li7 = _normalize_abundances(y)
    T_final = float(T0 * np.exp(-tau_grid[-1]))
    Y_p = max(1.0 - Y_n - 2.0 * Y_D - 4.0 * Y_He4 - 7.0 * Y_Li7, 0.0)
    Y_p_mass = 4.0 * Y_He4

    result = {
        "label": label,
        "framework_params": {
            "A": fw.A,
            "delta_phi_cyl": fw.delta_phi_cyl,
            "delta_k_wind": fw.delta_k_wind,
            "xi": fw.xi,
        },
        "T_start_MeV": T0,
        "T_final_MeV": T_final,
        "Y_n": Y_n,
        "Y_p": Y_p,
        "Y_D": Y_D,
        "Y_He4": Y_He4,
        "Y_Li7": Y_Li7,
        "Y_p_mass": Y_p_mass,
        "Li7_H": Y_Li7,
        "Li_H": Y_Li7,
        "D_H": Y_D,
        "integration_success": True,
        "n_steps": int(len(tau_grid)),
        "integrator": "explicit_euler_tau",
    }

    if verbose:
        print(f"\n=== {label} ===")
        print(f"  T: {T0:.3f} → {T_final:.4f} MeV  ({result['n_steps']} steps)")
        print(f"  Y_n = {Y_n:.5f}   Y_p = {Y_p:.5f}")
        print(f"  Y_D = {Y_D:.2e}   Y_He4 = {Y_He4:.5f}")
        print(f"  ^4He mass fraction Y_p ≈ {Y_p_mass:.4f}")
        print(f"  ^7Li/H ≈ {Y_Li7:.2e}")

    return result


def run_bbn_comparison(
    params: FrameworkParams | dict[str, Any] | None = None,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """Standard (A=0) vs Tav-modified side-by-side."""
    fw = params if isinstance(params, FrameworkParams) else FrameworkParams.from_mapping(params)

    std = FrameworkParams(A=0.0, delta_phi_cyl=fw.delta_phi_cyl, delta_k_wind=fw.delta_k_wind, xi=fw.xi)
    std_run = run_bbn(std, label="Standard BBN (A=0)", verbose=verbose)
    mod_run = run_bbn(fw, label="Tav Framework Modified", verbose=verbose)

    shift_li = None
    shift_yp = None
    if std_run["Li7_H"] > 0:
        shift_li = (mod_run["Li7_H"] - std_run["Li7_H"]) / std_run["Li7_H"] * 100.0
    if std_run["Y_p_mass"] > 0:
        shift_yp = (mod_run["Y_p_mass"] - std_run["Y_p_mass"]) / std_run["Y_p_mass"] * 100.0

    comparison = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "standard": std_run,
        "modified": mod_run,
        "delta_percent": {
            "Li7_H": shift_li,
            "Y_p_mass": shift_yp,
        },
    }

    if verbose:
        print("\n" + "=" * 55)
        if shift_li is not None:
            print(f"^7Li/H shift (Tav vs standard): {shift_li:+.1f}%")
        if shift_yp is not None:
            print(f"^4He mass-fraction shift: {shift_yp:+.3f}%")
        print("Vary framework params in the entry form to explore phase/winding scenarios.")

    return comparison


def confront_empirical_abundances(
    comparison: dict[str, Any],
    *,
    plot: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Compare modified BBN yields to curated observations via empirical_tests."""
    from menus.empirical_tests.tests import (
        empirical_bbn_dataframe,
        fetch_empirical_data,
        test_bbn_abundances,
    )

    empirical = fetch_empirical_data(verbose=verbose)
    obs = empirical_bbn_dataframe(empirical)
    mod = comparison.get("modified") or {}
    std = comparison.get("standard") or {}
    result = test_bbn_abundances(mod, obs, standard_predictions=std, plot=plot)
    result["empirical_fallback"] = empirical
    comparison["empirical_confrontation"] = result

    if verbose:
        print("\n[Empirical confrontation — BBN abundances]")
        for row in result.get("tensions", []):
            print(
                f"  {row['observable']}: model={row['theory']:.4e}  "
                f"obs={row['exp']:.4e} ± {row['exp_unc']:.1e}  "
                f"({row['tension_sigma']:.1f}σ)  [{row.get('reference', '')}]"
            )
    return result


def save_bbn_report(comparison: dict[str, Any], *, prefix: str = "bbn_interference") -> str:
    """Write JSON report under artifacts/prime_past/."""
    ensure_artifacts_dir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    from tav_shared.artifact_paths import TestSlug, artifact_path, compose_dataset_slug

    path = artifact_path(TestSlug.PRIME_PAST, compose_dataset_slug(prefix), "report", "json")
    path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(f"[TAV ENGINE] BBN report saved: {path}")
    return str(path)