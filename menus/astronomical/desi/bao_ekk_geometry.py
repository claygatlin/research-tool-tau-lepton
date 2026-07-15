"""
EKK + supersphere BAO re-analysis — falsifiable distance predictions vs DR2 covariances.

Maps Tav-cylinder geometry (Eq. 1–2), plasma-interval sound horizon, hierarchical
binding (effective w(z)), and Topological Exclusion floor (313.1 MeV) to BAO
observables D_M/r_d, D_H/r_d, and scaling parameters α_⊥, α_∥, α_AP.

This addresses the methodological gap noted in DESI_DR2_Evaluation (2026-07-11):
standard FLRW ruler calibration is replaced by a geometry-first prediction chain,
then confronted with the full Gaussian DR2 covariance (not diagonal-only).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np

from menus.astronomical.desi.fetcher import DEFAULT_COBAYA_ROOT
from menus.astronomical.desi.redshift_law import lambda_ratio_tsb
from tav_shared.artifact_paths import TestSlug, artifact_path, artifact_timestamp, compose_dataset_slug
from menus.astronomical.desi.scanner import (
    ARTIFACTS_DIR,
    DELTA_N_PLASMA,
    GAMMA6_HEX,
    GAMMA_REFERENCE,
    H0_H_UNITS,
    KAPPA_LEAK_DEFAULT,
    LATE_UNIVERSE_DELTA_N,
    M0_MEV,
    N_HIER_BINDING,
    OMEGA_M_FIDUCIAL,
    RD_OBSERVED_MPC,
    R_TAU_MPC,
    TAU_RESONANCE_PERIOD,
    TSB_RD_GAMMA_FALLBACK,
    comoving_distance_hmpc,
    compute_tsb_rd,
    gaussian_chi2,
    hubble_distance_hmpc,
    load_desi_from_cobaya_repo,
    s_from_z,
)

QuantityMode = Literal["DM_over_rs", "DH_over_rs", "joint_DH_DM", "all"]


@dataclass
class EKKBAOParams:
    """Fixed geometric parameters (minimal freedom — falsification-oriented)."""

    gamma_stretch: float = TSB_RD_GAMMA_FALLBACK
    kappa_leak: float = KAPPA_LEAK_DEFAULT
    r_tau_mpc: float = R_TAU_MPC
    gamma6: float = GAMMA6_HEX
    n_hier: float = N_HIER_BINDING
    delta_n: float = LATE_UNIVERSE_DELTA_N
    delta_n_plasma: float = DELTA_N_PLASMA
    m0_mev: float = M0_MEV
    rd_anchor_mpc: float = RD_OBSERVED_MPC
    om: float = OMEGA_M_FIDUCIAL
    h0_h: float = H0_H_UNITS
    # DESI-preferred CPL quadrant (fixed, not fitted — from binding-step emergence)
    w0_binding: float = -0.85
    wa_binding: float = -0.6


def predict_rd_plasma_interval(
    gamma: float,
    *,
    delta_n_plasma: float = DELTA_N_PLASMA,
    rd_anchor_mpc: float = RD_OBSERVED_MPC,
    gamma_anchor: float | None = None,
) -> dict[str, Any]:
    """
    Sound horizon from Tau-cylinder plasma axial advance (TSB sound-horizon derivations).

    r_d(γ) = Δn_plasma × r_d^anchor / γ with γ_anchor = Δn_plasma (exclusion-saturated
    plasma slice). At γ = Δn_plasma the ruler matches the twistor-projected anchor.
    """
    g = float(gamma)
    g_anchor = float(delta_n_plasma if gamma_anchor is None else gamma_anchor)
    rd_mpc = float(delta_n_plasma) * float(rd_anchor_mpc) / g
    rd_at_anchor = float(delta_n_plasma) * float(rd_anchor_mpc) / g_anchor
    return {
        "rd_predicted_mpc": rd_mpc,
        "rd_at_gamma_anchor_mpc": rd_at_anchor,
        "gamma_used": g,
        "gamma_anchor": g_anchor,
        "delta_n_plasma": float(delta_n_plasma),
        "formula": "r_d = Δn_plasma × r_anchor / γ",
        "legacy_compute_tsb_rd": compute_tsb_rd(g),
    }


def exclusion_phase_velocity(
    s: np.ndarray,
    *,
    m0_mev: float = M0_MEV,
    n_hier: float = N_HIER_BINDING,
) -> np.ndarray:
    """
    Topological Exclusion Principle → effective phase velocity along cylinder axis.

    Suppresses v_phase where exclusion saturation increases (fermionic occupancy
    limit on the 7-active-phase slots).
    """
    s = np.asarray(s, dtype=float)
    floor_ratio = float(m0_mev) / 313.1
    saturation = np.exp(-np.abs(s) / max(float(n_hier), 1.0))
    return floor_ratio * (1.0 - 0.08 * saturation)


def w_eff_binding(z: np.ndarray, params: EKKBAOParams) -> np.ndarray:
    """
    Effective equation of state from discrete binding levels (desi_mathematical_derivations §3).

    w_eff(a) ≈ −1 + Σ_j δw_j Θ(n(s) − n_j)  →  emergent w_a < 0 sector.
    """
    z = np.asarray(z, dtype=float)
    s = s_from_z(z, gamma=params.gamma_stretch, n_hier=params.n_hier, delta_n=params.delta_n)
    n_frac = np.clip(s / max(params.n_hier, 1.0), 0.0, 1.0)
    a = 1.0 / (1.0 + z)
    w_cpl = params.w0_binding + params.wa_binding * (1.0 - a)
    w_step = -1.0 + params.delta_n * 0.15 * n_frac
    return 0.5 * (w_cpl + w_step)


def _ez_modified(z: np.ndarray, params: EKKBAOParams) -> np.ndarray:
    """Modified expansion rate E(z) from binding + CPL sector."""
    z = np.asarray(z, dtype=float)
    ez_lcdm = np.sqrt(params.om * (1.0 + z) ** 3 + (1.0 - params.om))
    w = w_eff_binding(z, params)
    # Integral proxy for dark-energy history: multiplicative correction to E
    de_boost = 1.0 + 0.02 * (w + 1.0) * np.log1p(z)
    return ez_lcdm * np.clip(de_boost, 0.85, 1.25)


def tav_comoving_distance_hmpc(z: np.ndarray, params: EKKBAOParams) -> np.ndarray:
    """D_M(z) in h⁻¹ Mpc: ΛCDM backbone × Eq. (2) dispersion × exclusion velocity."""
    z = np.asarray(z, dtype=float)
    z_max = float(np.max(z)) if z.size else 0.0
    n = max(128, int(z_max * 64) + 1)
    zg = np.linspace(0.0, z_max * 1.05 + 1e-6, n)
    ez = _ez_modified(zg, params)
    s_g = s_from_z(zg, gamma=params.gamma_stretch, n_hier=params.n_hier, delta_n=params.delta_n)
    v_phase = exclusion_phase_velocity(s_g, m0_mev=params.m0_mev, n_hier=params.n_hier)
    integrand = v_phase / ez
    dz = np.diff(zg, prepend=0.0)
    dc = (299792.458 / params.h0_h) * np.cumsum(0.5 * (integrand + np.roll(integrand, 1)) * dz)
    dc[0] = 0.0
    stretch = lambda_ratio_tsb(
        z,
        kappa_leak=params.kappa_leak,
        r_tau=params.r_tau_mpc,
        gamma6=params.gamma6,
        gamma_stretch=params.gamma_stretch,
    )
    return np.interp(z, zg, dc) * stretch


def tav_hubble_distance_hmpc(z: np.ndarray, params: EKKBAOParams) -> np.ndarray:
    """D_H(z) in h⁻¹ Mpc with modified E(z) and dispersion stretch."""
    z = np.asarray(z, dtype=float)
    ez = _ez_modified(z, params)
    dh = 299792.458 / (params.h0_h * ez)
    stretch = lambda_ratio_tsb(
        z,
        kappa_leak=params.kappa_leak,
        r_tau=params.r_tau_mpc,
        gamma6=params.gamma6,
        gamma_stretch=params.gamma_stretch,
    )
    return dh * stretch


def lcdm_distances_hmpc(z: np.ndarray, params: EKKBAOParams) -> tuple[np.ndarray, np.ndarray]:
    """Fiducial ΛCDM D_M, D_H for α parameter ratios."""
    z = np.asarray(z, dtype=float)
    return comoving_distance_hmpc(z, om=params.om, h0_h=params.h0_h), hubble_distance_hmpc(
        z, om=params.om, h0_h=params.h0_h
    )


def bao_scaling_alphas(
    z: np.ndarray,
    *,
    dm_tav: np.ndarray,
    dh_tav: np.ndarray,
    params: EKKBAOParams,
) -> dict[str, Any]:
    """
    BAO distance scaling parameters (DESI / Eisenstein convention).

    α_⊥ = D_M^Tav / D_M^fid   (transverse / isotropic angular scale)
    α_∥   = D_H^Tav / D_H^fid (radial / redshift-space scale)
    α_AP  = (D_M/D_H)^Tav / (D_M/D_H)^fid  (Alcock–Paczynski)
    """
    z = np.asarray(z, dtype=float)
    dm_fid, dh_fid = lcdm_distances_hmpc(z, params)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha_perp = dm_tav / dm_fid
        alpha_par = dh_tav / dh_fid
        alpha_ap = (dm_tav / dh_tav) / (dm_fid / dh_fid)
    return {
        "z": z.tolist(),
        "alpha_perp": np.asarray(alpha_perp, dtype=float).tolist(),
        "alpha_parallel": np.asarray(alpha_par, dtype=float).tolist(),
        "alpha_ap": np.asarray(alpha_ap, dtype=float).tolist(),
        "alpha_perp_mean": float(np.nanmean(alpha_perp)),
        "alpha_parallel_mean": float(np.nanmean(alpha_par)),
        "alpha_ap_mean": float(np.nanmean(alpha_ap)),
    }


def predict_bao_vector(
    z: np.ndarray,
    quantities: list[str],
    *,
    params: EKKBAOParams | None = None,
    rd_mpc: float | None = None,
) -> np.ndarray:
    """Predict DESI-style observable vector (D_M/r_d, D_H/r_d, DV/r_d, …)."""
    params = params or EKKBAOParams()
    z = np.asarray(z, dtype=float)
    if rd_mpc is None:
        rd_mpc = predict_rd_plasma_interval(params.gamma_stretch)["rd_predicted_mpc"]
    dm = tav_comoving_distance_hmpc(z, params)
    dh = tav_hubble_distance_hmpc(z, params)
    rd = float(rd_mpc)
    out: list[float] = []
    for i, q in enumerate(quantities):
        qu = str(q).upper()
        if qu.startswith("DH"):
            out.append(float(dh[i] / rd))
        elif qu.startswith("DV"):
            dv = (z[i] * dh[i] * dm[i] ** 2) ** (1.0 / 3.0)
            out.append(float(dv / rd))
        else:
            out.append(float(dm[i] / rd))
    return np.asarray(out, dtype=float)


def predict_bao_observables(
    z: np.ndarray,
    quantities: list[str],
    *,
    params: EKKBAOParams | None = None,
    rd_mpc: float | None = None,
) -> dict[str, Any]:
    """Full EKK prediction package for one redshift/quantity list."""
    params = params or EKKBAOParams()
    z = np.asarray(z, dtype=float)
    rd_info = predict_rd_plasma_interval(params.gamma_stretch)
    if rd_mpc is None:
        rd_mpc = float(rd_info["rd_predicted_mpc"])
    dm = tav_comoving_distance_hmpc(z, params)
    dh = tav_hubble_distance_hmpc(z, params)
    alphas = bao_scaling_alphas(z, dm_tav=dm, dh_tav=dh, params=params)
    mu = predict_bao_vector(z, quantities, params=params, rd_mpc=rd_mpc)
    return {
        "mu": mu,
        "rd_mpc": float(rd_mpc),
        "rd_geometry": rd_info,
        "dm_hmpc": dm.tolist(),
        "dh_hmpc": dh.tolist(),
        "alphas": alphas,
        "params": {
            "gamma_stretch": params.gamma_stretch,
            "kappa_leak": params.kappa_leak,
            "r_tau_mpc": params.r_tau_mpc,
            "gamma6": params.gamma6,
            "n_hier": params.n_hier,
            "w0_binding": params.w0_binding,
            "wa_binding": params.wa_binding,
        },
    }


def _lcdm_bao_poly(z: np.ndarray, obs: np.ndarray) -> np.ndarray:
    deg = min(2, max(1, len(z) - 2))
    return np.polyval(np.polyfit(z, obs, deg), z)


def compare_ekk_to_dr2(
    bao_data: dict[str, Any],
    *,
    params: EKKBAOParams | None = None,
) -> dict[str, Any]:
    """
    Confront EKK geometry predictions with DR2 data + full covariance.

    Returns χ² for EKK (fixed geometry), ΛCDM polynomial, and residuals per bin.
    """
    params = params or EKKBAOParams()
    z = np.asarray(bao_data["z"], dtype=float)
    obs = np.asarray(bao_data["observable"], dtype=float)
    err_raw = bao_data.get("err")
    err = (
        np.asarray(err_raw, dtype=float)
        if err_raw is not None
        else np.ones_like(obs, dtype=float)
    )
    cov = bao_data.get("cov")
    q_raw = bao_data.get("quantity")
    if q_raw is None:
        quantities = ["DM_over_rs"] * len(z)
    else:
        quantities = list(np.asarray(q_raw, dtype=str))
    if len(quantities) != len(z):
        q_types = bao_data.get("quantity_types") or ["DM_over_rs"]
        quantities = []
        for i in range(len(z)):
            quantities.append(q_types[i % len(q_types)] if isinstance(q_types, list) else str(q_types))

    pred = predict_bao_observables(z, quantities, params=params)
    mu_ekk = np.asarray(pred["mu"], dtype=float)
    mu_lcdm = _lcdm_bao_poly(z, obs)

    chi2_ekk = float(gaussian_chi2(obs, mu_ekk, err=err, cov=cov))
    chi2_lcdm = float(gaussian_chi2(obs, mu_lcdm, err=err, cov=cov))
    n = len(z)
    delta = float(chi2_lcdm - chi2_ekk)

    residuals = {
        "z": z.tolist(),
        "quantities": quantities,
        "observed": obs.tolist(),
        "ekk_predicted": mu_ekk.tolist(),
        "lcdm_poly": mu_lcdm.tolist(),
        "residual_ekk": (obs - mu_ekk).tolist(),
        "residual_lcdm": (obs - mu_lcdm).tolist(),
        "sigma_err": err.tolist(),
    }

    verdict = (
        "EKK GEOMETRY FAVORED"
        if delta > 2.0
        else "ΛCDM FAVORED"
        if delta < -2.0
        else "INCONCLUSIVE"
    )

    return {
        "chi2_ekk": chi2_ekk,
        "chi2_lcdm": chi2_lcdm,
        "delta_chi2_lcdm_minus_ekk": delta,
        "n_data": int(n),
        "ndof": max(0, n - 1),
        "reduced_chi2_ekk": chi2_ekk / max(1, n - 1),
        "reduced_chi2_lcdm": chi2_lcdm / max(1, n - 1),
        "uses_full_covariance": cov is not None,
        "verdict": verdict,
        "prediction": pred,
        "residuals": residuals,
        "tracer": bao_data.get("tracer"),
        "quantity_filter": bao_data.get("quantity_filter"),
    }


MANUSCRIPT_DPI: int = 300


def plot_ekk_bao_reanalysis(
    report: dict[str, Any],
    *,
    prefix: str | None = None,
) -> str:
    """
    Multi-panel EKK BAO figure: D/r_d vs z, residuals, α scalings, χ² summary.
    """
    import matplotlib.pyplot as plt

    comp = report.get("comparison", report)
    resid = comp.get("residuals", {})
    pred = comp.get("prediction", {})
    alphas = pred.get("alphas", {})
    params = pred.get("params", {})

    z = np.asarray(resid.get("z", []), dtype=float)
    obs = np.asarray(resid.get("observed", []), dtype=float)
    ekk = np.asarray(resid.get("ekk_predicted", []), dtype=float)
    lcdm = np.asarray(resid.get("lcdm_poly", []), dtype=float)
    sig = np.asarray(resid.get("sigma_err", []), dtype=float)
    quantities = resid.get("quantities") or ["DM_over_rs"] * len(z)

    z_alpha = np.asarray(alphas.get("z", z), dtype=float)
    alpha_perp = np.asarray(alphas.get("alpha_perp", []), dtype=float)
    alpha_par = np.asarray(alphas.get("alpha_parallel", []), dtype=float)
    alpha_ap = np.asarray(alphas.get("alpha_ap", []), dtype=float)

    dm_mask = np.array(
        [not str(q).upper().startswith("DH") for q in quantities], dtype=bool
    )
    dh_mask = ~dm_mask

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    if dm_mask.any():
        ax.errorbar(
            z[dm_mask],
            obs[dm_mask],
            yerr=sig[dm_mask],
            fmt="o",
            color="#1f4e79",
            ms=6,
            capsize=3,
            label="DR2 D_M/r_d",
        )
        ax.plot(z[dm_mask], ekk[dm_mask], "s-", color="#4c72b0", ms=5, label="EKK D_M/r_d")
        ax.plot(
            z[dm_mask],
            lcdm[dm_mask],
            "--",
            color="#c44e52",
            lw=1.1,
            label="ΛCDM poly D_M/r_d",
        )
    if dh_mask.any():
        ax.errorbar(
            z[dh_mask],
            obs[dh_mask],
            yerr=sig[dh_mask],
            fmt="^",
            color="#55a868",
            ms=6,
            capsize=3,
            label="DR2 D_H/r_d",
        )
        ax.plot(z[dh_mask], ekk[dh_mask], "d-", color="#8172b2", ms=5, label="EKK D_H/r_d")
        ax.plot(
            z[dh_mask],
            lcdm[dh_mask],
            ":",
            color="#ccb974",
            lw=1.1,
            label="ΛCDM poly D_H/r_d",
        )
    ax.set_xlabel("z", fontsize=11)
    ax.set_ylabel("D / r_d", fontsize=11)
    ax.set_title("A. BAO distances vs redshift", fontsize=11)
    ax.legend(loc="best", fontsize=7, framealpha=0.9)

    ax = axes[0, 1]
    ax.axhline(0, color="k", lw=0.6)
    if len(z):
        ax.errorbar(
            z,
            resid.get("residual_ekk", obs - ekk),
            yerr=sig,
            fmt="o",
            color="#1f4e79",
            ms=5,
            capsize=2,
            label="obs − EKK",
        )
        ax.plot(z, resid.get("residual_lcdm", obs - lcdm), "x", color="#c44e52", ms=6, label="obs − ΛCDM")
    ax.set_xlabel("z", fontsize=11)
    ax.set_ylabel("Residual [D/r_d]", fontsize=11)
    ax.set_title("B. Residuals", fontsize=11)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 0]
    if alpha_perp.size:
        ax.plot(z_alpha, alpha_perp, "o-", color="#4c72b0", ms=4, label=r"$\alpha_\perp$")
    if alpha_par.size:
        ax.plot(z_alpha, alpha_par, "s-", color="#55a868", ms=4, label=r"$\alpha_\parallel$")
    if alpha_ap.size:
        ax.plot(z_alpha, alpha_ap, "^-", color="#c44e52", ms=4, label=r"$\alpha_{AP}$")
    ax.axhline(1.0, color="k", ls="--", lw=0.7, alpha=0.6)
    ax.set_xlabel("z", fontsize=11)
    ax.set_ylabel("α scaling", fontsize=11)
    ax.set_title("C. BAO scaling parameters", fontsize=11)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        f"Tracer: {report.get('tracer', 'n/a')}",
        f"Quantity: {comp.get('quantity_filter', report.get('quantity_filter', 'n/a'))}",
        f"γ stretch: {params.get('gamma_stretch', 'n/a')}",
        f"r_d pred: {pred.get('rd_mpc', 0):.2f} Mpc",
        f"χ² EKK: {comp.get('chi2_ekk', 0):.2f}",
        f"χ² ΛCDM: {comp.get('chi2_lcdm', 0):.2f}",
        f"Δχ² (Λ−EKK): {comp.get('delta_chi2_lcdm_minus_ekk', 0):+.2f}",
        f"Full cov: {comp.get('uses_full_covariance', False)}",
        f"Verdict: {comp.get('verdict', '')}",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), va="top", fontsize=10, family="monospace")
    ax.set_title("D. χ² summary", fontsize=11)

    verdict = comp.get("verdict", "")
    fig.suptitle(f"EKK BAO Re-analysis — {verdict}", fontsize=13, y=1.01)
    fig.tight_layout()
    dataset = prefix or compose_dataset_slug(
        report.get("tracer"),
        comp.get("quantity_filter", report.get("quantity_filter")),
    )
    path = artifact_path(TestSlug.EKK_BAO, dataset, "plot", "png")
    fig.savefig(path, dpi=MANUSCRIPT_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def run_ekk_bao_reanalysis(
    *,
    cobaya_path: str | None = None,
    tracer: str = "ALL_GCcomb",
    quantity_filter: QuantityMode | str = "joint_DH_DM",
    gamma: float | None = None,
    output_prefix: str = "ekk_bao_reanalysis",
    plot: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Menu entry: dedicated BAO re-analysis in EKK + supersphere geometry.

    Loads DR2 Gaussian tables, predicts D/r_d and α scaling parameters from fixed
    geometry, and evaluates χ² against the published covariance matrix.
    """
    cobaya_path = cobaya_path or str(DEFAULT_COBAYA_ROOT)
    qf = quantity_filter if quantity_filter != "all" else None
    bao = load_desi_from_cobaya_repo(cobaya_path, tracer=tracer, quantity_filter=qf)
    params = EKKBAOParams(gamma_stretch=float(gamma or TSB_RD_GAMMA_FALLBACK))
    comparison = compare_ekk_to_dr2(bao, params=params)

    if verbose:
        print("=" * 70)
        print("EKK + SUPERSPHERE BAO RE-ANALYSIS (DR2 covariance)")
        print(f"  Tracer   : {tracer}")
        print(f"  Quantity : {bao.get('quantity_filter', quantity_filter)}")
        print(f"  r_d pred : {comparison['prediction']['rd_mpc']:.2f} Mpc")
        print(f"  α_⊥ mean : {comparison['prediction']['alphas']['alpha_perp_mean']:.4f}")
        print(f"  α_∥ mean : {comparison['prediction']['alphas']['alpha_parallel_mean']:.4f}")
        print(f"  α_AP mean: {comparison['prediction']['alphas']['alpha_ap_mean']:.4f}")
        print(f"  χ² EKK   : {comparison['chi2_ekk']:.2f}")
        print(f"  χ² ΛCDM  : {comparison['chi2_lcdm']:.2f}")
        print(f"  Δχ²      : {comparison['delta_chi2_lcdm_minus_ekk']:+.2f}")
        print(f"  Verdict  : {comparison['verdict']}")
        print("=" * 70)

    from menus.astronomical.desi.json_util import write_json

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "action": "EKK BAO Re-analysis",
        "timestamp": artifact_timestamp(),
        "tracer": tracer,
        "quantity_filter": bao.get("quantity_filter", quantity_filter),
        "comparison": comparison,
        "framework_notes": (
            "Geometry-first chain: plasma-interval r_d, Eq.(2) dispersion, "
            "binding-step w_eff, exclusion v_phase; full DR2 cov χ²."
        ),
    }
    dataset = compose_dataset_slug(tracer, bao.get("quantity_filter", quantity_filter))
    path = artifact_path(TestSlug.EKK_BAO, dataset, "report", "json")
    write_json(path, report, indent=2, sort_keys=True)
    report["report_path"] = str(path)

    if plot:
        report["plot_path"] = plot_ekk_bao_reanalysis(
            report, prefix=compose_dataset_slug(tracer, bao.get("quantity_filter", quantity_filter))
        )
        if verbose:
            print(f"[EKK BAO] Plot: {report['plot_path']}")

    return report