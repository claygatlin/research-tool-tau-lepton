"""
Cosmology falsification tests — Methods 3 and 5 (session 2026-07-10).

Method 3: Λ regularization  ρ_Λ,obs ≈ ρ_hadronic,vac / 7^{n_hier}
Method 5: H₀ derivation from vacuum − CMB pressure contrast
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from tav_shared.artifact_paths import TestSlug, artifact_path, artifact_timestamp
from menus.astronomical.desi.scanner import (
    H0_FIDUCIAL_KM_S_MPC,
    M0_MEV,
    N_HIER_BINDING,
)

# Reference scales (GeV⁴ and km/s/Mpc)
RHO_LAMBDA_OBS_GEV4: float = 2.5e-47  # order-of-magnitude dark-energy density
RHO_HADRONIC_VAC_GEV4: float = (0.2) ** 4  # ~QCD confinement scale proxy
G_NEWTON_SI: float = 6.67430e-11  # m³ kg⁻¹ s⁻²
C_LIGHT_KM_S: float = 299792.458
RHO_CMB_GEV4: float = 2.47e-47  # CMB energy density today (~0.26 ρ_crit)


def _save_report(name: str, payload: dict[str, Any]) -> str:
    from menus.astronomical.desi.json_util import write_json

    path = artifact_path(TestSlug.COSMOLOGY_FALSIFICATION, name, "report", "json")
    write_json(path, payload, indent=2, sort_keys=True)
    return str(path)


def run_lambda_regularization_test(
    *,
    n_hier: float = N_HIER_BINDING,
    rho_had_gev4: float = RHO_HADRONIC_VAC_GEV4,
    rho_lambda_obs_gev4: float = RHO_LAMBDA_OBS_GEV4,
    log_tolerance: float = 1.0,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Method 3 — test ρ_Λ,obs against ρ_had / 7^{n_hier}.

    Falsified if |log₁₀(ρ_obs/ρ_pred)| exceeds ``log_tolerance`` dex.
    """
    denom = float(7.0 ** n_hier)
    rho_pred = rho_had_gev4 / denom
    log_ratio = float(np.log10(rho_lambda_obs_gev4 / rho_pred))
    falsified = abs(log_ratio) > log_tolerance

    result = {
        "method": 3,
        "name": "Lambda regularization",
        "n_hier": float(n_hier),
        "rho_hadronic_vac_gev4": float(rho_had_gev4),
        "rho_lambda_predicted_gev4": float(rho_pred),
        "rho_lambda_observed_gev4": float(rho_lambda_obs_gev4),
        "seven_to_n_hier": float(denom),
        "log10_obs_over_pred": log_ratio,
        "log_tolerance_dex": float(log_tolerance),
        "falsified": falsified,
        "verdict": "POTENTIAL FALSIFICATION" if falsified else "REGULARIZATION CONSISTENT",
    }

    if verbose:
        print("=" * 70)
        print("METHOD 3 — Cosmological constant regularization")
        print(f"  ρ_Λ pred : {rho_pred:.3e} GeV⁴  (ρ_had / 7^{n_hier:.1f})")
        print(f"  ρ_Λ obs  : {rho_lambda_obs_gev4:.3e} GeV⁴")
        print(f"  log₁₀(obs/pred) : {log_ratio:+.2f} dex")
        print(f"  Verdict: {result['verdict']}")
        print("=" * 70)

    result["report_path"] = _save_report("method3_lambda", result)
    return result


def run_h0_derivation_test(
    *,
    h0_obs_km_s_mpc: float = H0_FIDUCIAL_KM_S_MPC,
    rho_vac_gev4: float = RHO_LAMBDA_OBS_GEV4,
    rho_cmb_gev4: float = RHO_CMB_GEV4,
    tolerance_frac: float = 0.25,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Method 5 — compare observed H₀ to √(8πG/3c²)(P_vac − P_CMB) scaling.

    Uses order-of-magnitude energy-density contrast; falsified if fractional
    mismatch exceeds ``tolerance_frac`` (default 25%).
    """
    # Convert GeV⁴ → kg/m³ via ℏc = 1.973e-16 GeV·m, c in m/s
    gev_to_j = 1.60218e-10
    hbar_c_gev_m = 1.97327e-16
    rho_vac_si = rho_vac_gev4 * gev_to_j / (hbar_c_gev_m**3)
    rho_cmb_si = rho_cmb_gev4 * gev_to_j / (hbar_c_gev_m**3)
    delta_rho = max(rho_vac_si - rho_cmb_si, 0.0)
    c_m_s = C_LIGHT_KM_S * 1000.0
    h0_pred_m_s_mpc = np.sqrt(8.0 * np.pi * G_NEWTON_SI / 3.0) * np.sqrt(delta_rho) / (c_m_s * 3.086e22)
    h0_pred = float(h0_pred_m_s_mpc * c_m_s / 1000.0)  # km/s/Mpc (crude unit bridge)
    frac_err = abs(h0_pred - h0_obs_km_s_mpc) / max(h0_obs_km_s_mpc, 1e-6)
    falsified = frac_err > tolerance_frac

    result = {
        "method": 5,
        "name": "H0 vacuum-pressure derivation",
        "h0_observed_km_s_mpc": float(h0_obs_km_s_mpc),
        "h0_predicted_km_s_mpc": h0_pred,
        "fractional_error": float(frac_err),
        "tolerance_fraction": float(tolerance_frac),
        "rho_vac_gev4": float(rho_vac_gev4),
        "rho_cmb_gev4": float(rho_cmb_gev4),
        "m0_mev_anchor": float(M0_MEV),
        "falsified": falsified,
        "verdict": "POTENTIAL FALSIFICATION" if falsified else "H0 DERIVATION CONSISTENT (order-of-mag)",
    }

    if verbose:
        print("=" * 70)
        print("METHOD 5 — Hubble parameter derivation")
        print(f"  H₀ obs  : {h0_obs_km_s_mpc:.2f} km/s/Mpc")
        print(f"  H₀ pred : {h0_pred:.2f} km/s/Mpc (vacuum − CMB pressure proxy)")
        print(f"  |ΔH₀|/H₀ : {frac_err:.1%}")
        print(f"  Verdict: {result['verdict']}")
        print("=" * 70)

    result["report_path"] = _save_report("method5_h0", result)
    return result


MANUSCRIPT_DPI: int = 300


def plot_cosmology_falsification(
    report: dict[str, Any],
    *,
    prefix: str = "cosmology_suite",
) -> str:
    """Two-panel figure: Method 3 Λ regularization and Method 5 H₀ derivation."""
    import matplotlib.pyplot as plt

    m3 = report.get("method_3_lambda", {})
    m5 = report.get("method_5_h0", {})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    log_ratio = float(m3.get("log10_obs_over_pred", 0.0))
    tol = float(m3.get("log_tolerance_dex", 1.0))
    colors = ["#55a868" if not m3.get("falsified") else "#c44e52"]
    ax.bar(["log₁₀(ρ_Λ obs/pred)"], [log_ratio], color=colors, alpha=0.85, width=0.5)
    ax.axhline(tol, color="#c44e52", ls="--", lw=1.0, label=f"+{tol:.1f} dex tol")
    ax.axhline(-tol, color="#c44e52", ls="--", lw=1.0)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("dex", fontsize=11)
    ax.set_title(f"Method 3 — {m3.get('verdict', '')}", fontsize=11)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    h0_obs = float(m5.get("h0_observed_km_s_mpc", 0.0))
    h0_pred = float(m5.get("h0_predicted_km_s_mpc", 0.0))
    ax.bar(
        ["H₀ obs", "H₀ pred"],
        [h0_obs, h0_pred],
        color=["#1f4e79", "#4c72b0"],
        alpha=0.85,
        width=0.45,
    )
    ax.set_ylabel("km/s/Mpc", fontsize=11)
    frac = float(m5.get("fractional_error", 0.0))
    ax.set_title(
        f"Method 5 — {m5.get('verdict', '')} (|Δ|/H₀={frac:.1%})",
        fontsize=11,
    )

    verdict = report.get("verdict", "")
    fig.suptitle(f"Cosmology Falsification — {verdict}", fontsize=13, y=1.02)
    fig.tight_layout()
    path = artifact_path(TestSlug.COSMOLOGY_FALSIFICATION, prefix, "plot", "png")
    fig.savefig(path, dpi=MANUSCRIPT_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def run_cosmology_falsification_suite(
    *,
    verbose: bool = True,
    output_prefix: str = "cosmology_suite",
    plot: bool = False,
) -> dict[str, Any]:
    """Run Methods 3 and 5 with a combined PASS / FALSIFICATION verdict."""
    m3 = run_lambda_regularization_test(verbose=verbose)
    m5 = run_h0_derivation_test(verbose=verbose)
    all_pass = not m3["falsified"] and not m5["falsified"]
    report = {
        "action": "Cosmology Falsification (Methods 3, 5)",
        "verdict": "SUITE PASS" if all_pass else "POTENTIAL FALSIFICATION",
        "all_pass": all_pass,
        "method_3_lambda": m3,
        "method_5_h0": m5,
        "timestamp": artifact_timestamp(),
    }
    report["report_path"] = _save_report(output_prefix, report)
    if verbose:
        print(f"[COSMOLOGY FALSIFICATION] Global verdict: {report['verdict']}")
    if plot:
        report["plot_path"] = plot_cosmology_falsification(report, prefix=output_prefix)
        if verbose:
            print(f"[COSMOLOGY FALSIFICATION] Plot: {report['plot_path']}")
    return report