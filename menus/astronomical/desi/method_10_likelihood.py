"""
Method 10 — joint LSS likelihood: comb P(k) + S(n) nodes + BAO vs ΛCDM.

Combines configuration-space (Methods 1, 4) and BAO distance likelihood into a
single χ² / evidence comparison between Tau-SB geometric and ΛCDM baselines.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from menus.astronomical.desi.fetcher import DEFAULT_COBAYA_ROOT
from menus.astronomical.desi.lss_falsification import (
    K_COMB_FUND,
    R_TAU,
    compute_power_spectrum_1d,
    compute_power_spectrum_3d,
    cross_correlate_S_n_nodes,
    search_for_comb_signature,
)
from tav_shared.artifact_paths import TestSlug, artifact_path, artifact_timestamp, compose_dataset_slug
from menus.astronomical.desi.scanner import load_desi_from_cobaya_repo

JAX_AVAILABLE = False
try:
    import jax  # noqa: F401

    JAX_AVAILABLE = True
except ImportError:
    pass

COMB_CHI2_WEIGHT: float = 12.0
NODE_CHI2_WEIGHT: float = 8.0
BAO_CHI2_WEIGHT: float = 1.0


def comb_log_likelihood_proxy(comb_result: dict[str, Any]) -> float:
    """
    Proxy log-likelihood from comb search significance.

    Higher when geometric peaks are detected at k = n/R_τ.
    """
    if comb_result.get("falsified", True):
        return -5.0
    sigs = comb_result.get("significance", [])
    if not sigs:
        return -3.0
    mean_sig = float(np.mean(np.abs(sigs)))
    n_hits = len(comb_result.get("detected_peaks", []))
    return float(n_hits * mean_sig - 0.5 * max(0, 3 - n_hits))


def node_log_likelihood_proxy(node_result: dict[str, Any]) -> float:
    """Proxy log-likelihood from S(n) node recovery fraction."""
    if node_result.get("falsified", True):
        return -4.0
    frac = float(node_result.get("recovery_fraction", 0.0))
    n_match = int(node_result.get("n_matched", 0))
    return float(2.0 * n_match + 5.0 * frac - 2.0)


def bao_chi2_lcdm(
    z: np.ndarray,
    observable: np.ndarray,
    err: np.ndarray,
    cov: np.ndarray | None = None,
) -> float:
    """ΛCDM polynomial baseline χ² on BAO observables."""
    from menus.astronomical.desi.scanner import gaussian_chi2

    z = np.asarray(z, dtype=float)
    obs = np.asarray(observable, dtype=float)
    deg = min(2, max(1, len(z) - 2))
    coeffs = np.polyfit(z, obs, deg)
    pred = np.polyval(coeffs, z)
    return float(gaussian_chi2(obs, pred, err=err, cov=cov))


def bao_chi2_tau_sb(
    z: np.ndarray,
    observable: np.ndarray,
    err: np.ndarray,
    cov: np.ndarray | None = None,
    *,
    quantity: str = "DM_over_rs",
    gamma: float = 10.0,
) -> float:
    """Tau-SB Eq. (2) dispersion BAO χ² on observables."""
    from menus.astronomical.desi.scanner import gaussian_chi2
    from menus.astronomical.desi.theory_likelihood import tau_sb_mu

    z = np.asarray(z, dtype=float)
    obs = np.asarray(observable, dtype=float)
    pred = tau_sb_mu(z, quantity=quantity, gamma=gamma, use_dispersion=True)
    return float(gaussian_chi2(obs, pred, err=err, cov=cov))


def joint_method10_scores(
    *,
    delta: np.ndarray,
    box_size_mpc: float,
    z_bao: np.ndarray,
    observable: np.ndarray,
    err: np.ndarray,
    cov: np.ndarray | None = None,
    quantity: str = "DM_over_rs",
    use_jax: bool = True,
    dimension: int = 3,
) -> dict[str, Any]:
    """
    Compute joint Method 10 score components from a δ field + BAO vectors.
    """
    use_jax = bool(use_jax and JAX_AVAILABLE)
    delta = np.asarray(delta, dtype=float)

    if dimension >= 3 and delta.ndim == 3:
        k, pk = compute_power_spectrum_3d(
            delta, box_size_mpc, use_jax=use_jax, n_k_bins=48
        )
        x = np.linspace(0.0, box_size_mpc, delta.shape[0])
        delta_1d = delta[delta.shape[0] // 2, delta.shape[1] // 2, :]
    else:
        if delta.ndim == 3:
            delta_1d = delta[delta.shape[0] // 2, delta.shape[1] // 2, :]
        else:
            delta_1d = delta.ravel()
        x = np.linspace(0.0, box_size_mpc, len(delta_1d))
        k, pk = compute_power_spectrum_1d(delta_1d, box_size_mpc, use_jax=use_jax)

    comb = search_for_comb_signature(
        k,
        pk,
        delta_for_bootstrap=delta if delta.ndim >= 2 else delta_1d,
        box_size_mpc=box_size_mpc,
        recovery_mode=False,
    )
    nodes = cross_correlate_S_n_nodes(x, delta_1d)

    ll_comb = comb_log_likelihood_proxy(comb)
    ll_nodes = node_log_likelihood_proxy(nodes)
    chi2_bao_lcdm = bao_chi2_lcdm(z_bao, observable, err, cov)
    chi2_bao_tsb = bao_chi2_tau_sb(z_bao, observable, err, cov, quantity=quantity)

    # Joint score (higher = better); map to pseudo-χ²
    score_lcdm = (
        -ll_comb * COMB_CHI2_WEIGHT
        - ll_nodes * NODE_CHI2_WEIGHT
        + chi2_bao_lcdm * BAO_CHI2_WEIGHT
    )
    score_tsb = (
        -ll_comb * COMB_CHI2_WEIGHT
        - ll_nodes * NODE_CHI2_WEIGHT
        + chi2_bao_tsb * BAO_CHI2_WEIGHT
    )

    delta_score = score_lcdm - score_tsb
    favored = "Tau-SB" if delta_score > 0 else "ΛCDM"

    return {
        "comb_search": comb,
        "node_cross_correlation": nodes,
        "loglike_comb_proxy": ll_comb,
        "loglike_nodes_proxy": ll_nodes,
        "chi2_bao_lcdm": chi2_bao_lcdm,
        "chi2_bao_tau_sb": chi2_bao_tsb,
        "joint_score_lcdm": score_lcdm,
        "joint_score_tau_sb": score_tsb,
        "delta_score_lcdm_minus_tsb": delta_score,
        "favored_model": favored,
        "verdict": (
            "METHOD 10 FAVORS TAU-SB"
            if delta_score > 2.0
            else "METHOD 10 FAVORS ΛCDM"
            if delta_score < -2.0
            else "METHOD 10 INCONCLUSIVE"
        ),
        "weights": {
            "comb": COMB_CHI2_WEIGHT,
            "nodes": NODE_CHI2_WEIGHT,
            "bao": BAO_CHI2_WEIGHT,
        },
        "k_comb_fundamental": K_COMB_FUND,
        "R_tau_mpc": R_TAU,
    }


def run_method10_joint_likelihood(
    *,
    delta: np.ndarray | None = None,
    delta_npz: str | None = None,
    cobaya_path: str | None = None,
    tracer: str = "ALL_GCcomb",
    quantity_filter: str = "DM_over_rs",
    box_size_mpc: float = 2000.0,
    use_jax: bool = True,
    dimension: int = 3,
    build_catalog_grid: bool = True,
    n_per_axis: int = 64,
    galaxies_per_shell: int = 600,
    seed: int = 42,
    output_prefix: str = "method10_joint",
    plot: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Method 10 menu entry — joint comb + S(n) + BAO likelihood comparison.
    """
    cobaya_path = cobaya_path or str(DEFAULT_COBAYA_ROOT)
    bao = load_desi_from_cobaya_repo(
        cobaya_path,
        tracer=tracer,
        quantity_filter=quantity_filter,
    )

    meta: dict[str, Any] = {}
    if delta is None and delta_npz:
        from menus.astronomical.desi.desi_catalog_to_grid import load_delta_grid

        delta, meta = load_delta_grid(delta_npz)
        box_size_mpc = float(meta.get("box_size_mpc", box_size_mpc))
    elif delta is None and build_catalog_grid:
        from menus.astronomical.desi.desi_catalog_to_grid import (
            catalog_to_delta_field,
            load_bao_shell_catalog,
        )

        catalog = load_bao_shell_catalog(
            cobaya_path=cobaya_path,
            tracer=tracer,
            galaxies_per_shell=galaxies_per_shell,
            seed=seed,
        )
        delta, meta = catalog_to_delta_field(
            catalog,
            n_per_axis=n_per_axis,
            box_size_mpc=box_size_mpc,
        )
    elif delta is None:
        raise ValueError("Provide delta, delta_npz, or set build_catalog_grid=True")

    cov = bao.get("cov")
    qty = str(bao.get("quantity_filter") or quantity_filter or "DM_over_rs")
    scores = joint_method10_scores(
        delta=delta,
        box_size_mpc=box_size_mpc,
        z_bao=np.asarray(bao["z"], dtype=float),
        observable=np.asarray(bao["observable"], dtype=float),
        err=np.asarray(bao["err"], dtype=float),
        cov=cov,
        quantity=qty,
        use_jax=use_jax,
        dimension=dimension,
    )

    report = {
        "action": "Method 10 Joint Likelihood",
        "method10": scores,
        "grid_metadata": meta,
        "bao_tracer": tracer,
        "n_bao": int(bao.get("n_data", len(bao["z"]))),
        "use_jax": bool(use_jax and JAX_AVAILABLE),
        "timestamp": artifact_timestamp(),
    }

    if verbose:
        print("=" * 70)
        print("METHOD 10 — JOINT LIKELIHOOD (comb + S(n) + BAO)")
        print(f"  Comb log-proxy  : {scores['loglike_comb_proxy']:.2f}")
        print(f"  Nodes log-proxy : {scores['loglike_nodes_proxy']:.2f}")
        print(f"  χ² BAO ΛCDM     : {scores['chi2_bao_lcdm']:.2f}")
        print(f"  χ² BAO Tau-SB   : {scores['chi2_bao_tau_sb']:.2f}")
        print(f"  Δscore (Λ−τ)    : {scores['delta_score_lcdm_minus_tsb']:+.2f}")
        print(f"  Verdict         : {scores['verdict']}")
        print("=" * 70)

    from menus.astronomical.desi.json_util import write_json

    json_path = artifact_path(
        TestSlug.METHOD10,
        compose_dataset_slug(tracer, quantity_filter, output_prefix),
        "report",
        "json",
    )
    write_json(json_path, report, indent=2, sort_keys=True)
    report["report_path"] = str(json_path)

    if plot:
        from menus.astronomical.desi.lss_visualization import plot_method10_summary

        report["plot_path"] = plot_method10_summary(report, prefix=output_prefix)

    return report