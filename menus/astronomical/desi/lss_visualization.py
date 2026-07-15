"""
Publication-quality LSS falsification figures (dpi=300).

Multi-panel suite: P(k) comb recovery, S(n) nodes on δ grids, Method 10 summary,
and Test 2 λ(z) comparison.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from tav_shared.artifact_paths import TestSlug, artifact_path, artifact_timestamp, compose_dataset_slug

MANUSCRIPT_DPI: int = 300
MANUSCRIPT_FIGSIZE: tuple[float, float] = (12, 8)


def plot_pk_comb_manuscript(
    k: np.ndarray,
    pk: np.ndarray,
    comb: dict[str, Any],
    *,
    prefix: str = "lss_pk",
    title: str | None = None,
) -> str:
    """High-resolution P(k) with geometric comb markers."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    mask = k > 0
    ax.loglog(k[mask], pk[mask] + 1e-30, color="#1f4e79", lw=1.2, label="P(k)")
    for n, k_exp in enumerate(comb.get("expected_k", [])[:10], start=1):
        ax.axvline(
            k_exp,
            color="#c44e52",
            ls="--",
            alpha=0.45,
            lw=0.9,
            label="k = n/R_τ" if n == 1 else "",
        )
    for k_meas, sig in comb.get("detected_peaks", []):
        ax.plot(
            k_meas,
            float(np.interp(k_meas, k, pk)),
            "o",
            color="#55a868",
            ms=7,
            label=f"hit σ={sig:.1f}",
        )
    ax.set_xlabel("k [h Mpc⁻¹]", fontsize=11)
    ax.set_ylabel("P(k)", fontsize=11)
    ax.set_title(title or f"LSS Comb — {comb.get('verdict', '')}", fontsize=12)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    path = artifact_path(TestSlug.LSS_PUBLICATION, prefix, "pk_plot", "png")
    fig.savefig(path, dpi=MANUSCRIPT_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_sn_nodes_manuscript(
    x: np.ndarray,
    delta: np.ndarray,
    node_result: dict[str, Any],
    *,
    prefix: str = "lss_nodes",
) -> str:
    """δ(x) with predicted S(n) nodes and observed peaks."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, delta, color="#1f4e79", alpha=0.75, lw=0.9, label="δ(x)")
    for mpc in node_result.get("predicted_nodes_mpc", [])[:10]:
        ax.axvline(mpc, color="#c44e52", ls="--", alpha=0.55, lw=0.9)
    for mpc in node_result.get("observed_peaks_mpc", [])[:10]:
        ax.axvline(mpc, color="#55a868", ls=":", alpha=0.7, lw=1.0)
    ax.set_xlabel("x [h⁻¹ Mpc]", fontsize=11)
    ax.set_ylabel("δ", fontsize=11)
    ax.set_title(
        f"S(n) Nodes — {node_result.get('verdict', '')}",
        fontsize=12,
    )
    from matplotlib.lines import Line2D

    legend_elems = [
        Line2D([0], [0], color="#1f4e79", label="δ(x)"),
        Line2D([0], [0], color="#c44e52", ls="--", label="predicted"),
        Line2D([0], [0], color="#55a868", ls=":", label="observed peaks"),
    ]
    ax.legend(handles=legend_elems, loc="best", fontsize=8)
    fig.tight_layout()
    path = artifact_path(TestSlug.LSS_PUBLICATION, prefix, "nodes_plot", "png")
    fig.savefig(path, dpi=MANUSCRIPT_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_method10_summary(report: dict[str, Any], *, prefix: str = "method10") -> str:
    """Bar chart of Method 10 joint score components."""
    import matplotlib.pyplot as plt

    m10 = report.get("method10", report)
    labels = ["comb log-proxy", "nodes log-proxy", "χ² BAO ΛCDM", "χ² BAO Tau-SB"]
    values = [
        m10.get("loglike_comb_proxy", 0),
        m10.get("loglike_nodes_proxy", 0),
        -m10.get("chi2_bao_lcdm", 0) / 10.0,
        -m10.get("chi2_bao_tau_sb", 0) / 10.0,
    ]
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color=colors, alpha=0.85)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Score component (normalized)", fontsize=11)
    ax.set_title(
        f"Method 10 — {m10.get('verdict', '')}",
        fontsize=12,
    )
    fig.autofmt_xdate(rotation=15)
    fig.tight_layout()
    path = artifact_path(TestSlug.METHOD10, prefix, "summary_plot", "png")
    fig.savefig(path, dpi=MANUSCRIPT_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_redshift_law(report: dict[str, Any], *, prefix: str = "redshift_law") -> str:
    """λ(z)/λ₀: TSB dispersion law vs ΛCDM (1+z)."""
    import matplotlib.pyplot as plt

    z = np.asarray(report.get("z_samples", []), dtype=float)
    stretch_tsb = np.asarray(report.get("stretch_tsb", []), dtype=float)
    stretch_lcdm = np.asarray(report.get("stretch_lcdm", []), dtype=float)
    fit = report.get("fit", {})
    kappa = fit.get("kappa_leak_best", 0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z, stretch_lcdm, "--", color="#c44e52", lw=1.2, label="ΛCDM (1+z)")
    ax.plot(
        z,
        stretch_tsb,
        "-",
        color="#1f4e79",
        lw=1.4,
        label=f"TSB λ(z), κ_leak={kappa:.4f}",
    )
    ax.set_xlabel("z", fontsize=11)
    ax.set_ylabel("λ(z) / λ₀", fontsize=11)
    ax.set_title(
        f"Test 2 Redshift Law — {fit.get('verdict', '')}",
        fontsize=12,
    )
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    path = artifact_path(TestSlug.REDSHIFT_LAW, prefix, "plot", "png")
    fig.savefig(path, dpi=MANUSCRIPT_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_manuscript_figure_suite(
    *,
    k: np.ndarray,
    pk: np.ndarray,
    comb: dict[str, Any],
    x: np.ndarray,
    delta: np.ndarray,
    node_result: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    prefix: str = "lss_manuscript",
) -> str:
    """
    Four-panel publication figure: P(k), δ nodes, injected k markers, metadata.
    """
    import matplotlib.pyplot as plt

    meta = metadata or {}
    fig, axes = plt.subplots(2, 2, figsize=MANUSCRIPT_FIGSIZE)

    # Panel A: P(k)
    ax = axes[0, 0]
    mask = k > 0
    ax.loglog(k[mask], pk[mask] + 1e-30, color="#1f4e79", lw=1.0)
    for k_exp in comb.get("expected_k", [])[:8]:
        ax.axvline(k_exp, color="#c44e52", ls="--", alpha=0.4, lw=0.8)
    ax.set_xlabel("k [h Mpc⁻¹]")
    ax.set_ylabel("P(k)")
    ax.set_title(f"A. Comb search — {comb.get('verdict', '')}")

    # Panel B: δ(x) nodes
    ax = axes[0, 1]
    ax.plot(x, delta, color="#1f4e79", alpha=0.7, lw=0.8)
    for mpc in node_result.get("predicted_nodes_mpc", [])[:8]:
        ax.axvline(mpc, color="#c44e52", ls="--", alpha=0.5)
    for mpc in node_result.get("observed_peaks_mpc", [])[:8]:
        ax.axvline(mpc, color="#55a868", ls=":", alpha=0.6)
    ax.set_xlabel("x [h⁻¹ Mpc]")
    ax.set_ylabel("δ")
    ax.set_title(f"B. S(n) nodes — {node_result.get('verdict', '')}")

    # Panel C: comb significance
    ax = axes[1, 0]
    sigs = comb.get("significance", [])
    if sigs:
        ax.bar(range(len(sigs)), sigs, color="#4c72b0", alpha=0.85)
    ax.set_xlabel("Comb harmonic index")
    ax.set_ylabel("Significance")
    ax.set_title("C. Comb peak significance")

    # Panel D: run metadata
    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        f"box = {meta.get('box_size_mpc', 'n/a')} h⁻¹ Mpc",
        f"grid = {meta.get('n_per_axis', meta.get('n_points', 'n/a'))}",
        f"δ std = {meta.get('delta_std', meta.get('noise_level', 'n/a'))}",
        f"R_τ = {meta.get('R_tau', meta.get('R_Tau', 7.0))} Mpc",
        f"matched nodes = {node_result.get('n_matched', 0)}/"
        f"{node_result.get('n_predicted', 0)}",
        f"comb hits = {len(comb.get('detected_peaks', []))}",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), va="top", fontsize=10, family="monospace")
    ax.set_title("D. Run metadata")

    fig.suptitle("LSS Falsification Manuscript Suite", fontsize=13, y=1.01)
    fig.tight_layout()
    path = artifact_path(TestSlug.LSS_PUBLICATION, prefix, "manuscript_suite", "png")
    fig.savefig(path, dpi=MANUSCRIPT_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def run_publication_figures(
    *,
    n_points: int = 4096,
    box_size_mpc: float = 3000.0,
    noise_level: float = 0.25,
    use_jax: bool = True,
    dimension: int = 1,
    seed: int = 42,
    output_prefix: str = "lss_manuscript",
    tracer: str = "ALL_GCcomb",
    quantity_filter: str = "DM_over_rs",
    verbose: bool = True,
) -> dict[str, Any]:
    """Generate full manuscript figure suite from mock injection + recovery."""
    from menus.astronomical.desi.lss_falsification import (
        compute_power_spectrum_1d,
        compute_power_spectrum_3d,
        cross_correlate_S_n_nodes,
        generate_synthetic_density_field_1d,
        generate_synthetic_density_field_3d,
        search_for_comb_signature,
    )

    if dimension >= 3:
        _, delta, meta = generate_synthetic_density_field_3d(
            n_per_axis=max(32, int(round(n_points ** (1 / 3)))),
            box_size_mpc=box_size_mpc / 3.0,
            noise_level=noise_level,
            seed=seed,
            inject_comb=True,
        )
        k, pk = compute_power_spectrum_3d(delta, box_size_mpc / 3.0, use_jax=use_jax)
        x = np.linspace(0.0, box_size_mpc / 3.0, delta.shape[0])
        delta_1d = delta[delta.shape[0] // 2, delta.shape[1] // 2, :]
    else:
        x, delta_1d, meta = generate_synthetic_density_field_1d(
            n_points=n_points,
            box_size_mpc=box_size_mpc,
            noise_level=noise_level,
            seed=seed,
            inject_comb=True,
            inject_sn_nodes=True,
        )
        k, pk = compute_power_spectrum_1d(delta_1d, box_size_mpc, use_jax=use_jax)

    comb = search_for_comb_signature(
        k,
        pk,
        delta_for_bootstrap=delta_1d,
        box_size_mpc=meta["box_size_mpc"],
        recovery_mode=True,
    )
    nodes = cross_correlate_S_n_nodes(x, delta_1d)

    suite_path = plot_manuscript_figure_suite(
        k=k,
        pk=pk,
        comb=comb,
        x=x,
        delta=delta_1d,
        node_result=nodes,
        metadata=meta,
        prefix=output_prefix,
    )
    pk_path = plot_pk_comb_manuscript(k, pk, comb, prefix=output_prefix)
    node_path = plot_sn_nodes_manuscript(x, delta_1d, nodes, prefix=output_prefix)

    bao_meta: dict[str, Any] = {}
    try:
        from menus.astronomical.desi.fetcher import DEFAULT_COBAYA_ROOT
        from menus.astronomical.desi.scanner import load_desi_from_cobaya_repo

        bao = load_desi_from_cobaya_repo(
            DEFAULT_COBAYA_ROOT,
            tracer=tracer,
            quantity_filter=quantity_filter,
        )
        bao_meta = {
            "tracer": tracer,
            "quantity_filter": quantity_filter,
            "n_bao": int(bao.get("n_data", len(bao.get("z", [])))),
            "z_eff": [float(v) for v in bao.get("z", [])],
            "observable": [float(v) for v in bao.get("observable", [])],
        }
    except Exception as exc:
        bao_meta = {"load_error": str(exc)}

    report = {
        "action": "LSS Publication Figures",
        "suite_path": suite_path,
        "pk_path": pk_path,
        "nodes_path": node_path,
        "comb_verdict": comb.get("verdict"),
        "node_verdict": nodes.get("verdict"),
        "bao_reference": bao_meta,
        "metadata": meta,
        "timestamp": artifact_timestamp(),
    }

    from menus.astronomical.desi.json_util import write_json

    json_path = artifact_path(
        TestSlug.LSS_PUBLICATION,
        compose_dataset_slug(tracer, quantity_filter, output_prefix),
        "report",
        "json",
    )
    write_json(json_path, report, indent=2, sort_keys=True)
    report["report_path"] = str(json_path)

    if verbose:
        print(f"[LSS VIZ] Manuscript suite: {suite_path}")
        print(f"[LSS VIZ] P(k) panel: {pk_path}")
        print(f"[LSS VIZ] Nodes panel: {node_path}")

    return report