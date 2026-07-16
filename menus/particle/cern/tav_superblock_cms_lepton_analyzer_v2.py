"""
Tav-Superblock CMS lepton analyzer v2 — chunked NanoAOD + Data/MC validation.

Built on ``scan_cms_nanoaod_chunked`` and ``tav_7fold_muon_analysis``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from menus.particle.cern.analyzer import scan_cms_nanoaod_chunked
from menus.particle.cern.fetcher import resolve_root_path
from menus.particle.cern.tav_superblock_cms_muon_analyzer import tav_7fold_muon_analysis
from tav_shared.artifact_paths import (
    TestSlug,
    artifact_path,
    artifact_run_dir,
    artifact_timestamp,
    compose_dataset_slug,
)
from tav_shared.chunked_results import (
    DEFAULT_MAX_CHUNK_MB,
    save_tav_results_chunked,
    should_chunk_tav_results,
)

LEPTON_BRANCH_SPECS: dict[str, dict[str, str]] = {
    "Muon": {"n_branch": "nMuon", "pt_branch": "Muon_pt"},
    "Electron": {"n_branch": "nElectron", "pt_branch": "Electron_pt"},
}


def _resolve_root(file_path: str | Path) -> Path:
    resolved = resolve_root_path(str(file_path))
    if resolved is not None:
        return Path(resolved)
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"ROOT file not found: {file_path}")
    return path


def discover_available_lepton_branches(
    file_path: str | Path,
    *,
    tree_name: str = "Events",
) -> dict[str, Any]:
    """Probe a NanoAOD ROOT file for muon/electron branches."""
    import uproot

    path = _resolve_root(file_path)
    with uproot.open(path) as handle:
        tree = handle[tree_name]
        available = {str(k) for k in tree.keys()}
        n_entries = int(tree.num_entries)

    leptons: dict[str, dict[str, Any]] = {}
    for name, spec in LEPTON_BRANCH_SPECS.items():
        n_ok = spec["n_branch"] in available
        pt_ok = spec["pt_branch"] in available
        leptons[name] = {
            "available": n_ok and pt_ok,
            "n_branch": spec["n_branch"] if n_ok else None,
            "pt_branch": spec["pt_branch"] if pt_ok else None,
        }

    return {
        "path": str(path),
        "tree": tree_name,
        "n_entries": n_entries,
        "branches": sorted(available),
        "leptons": leptons,
    }


def _enrich_lepton_result(
    tav: dict[str, Any],
    *,
    path: Path,
    lepton: str,
    scan: dict[str, Any],
    output_dir: str,
) -> dict[str, Any]:
    """Add validation-suite aliases and scan metadata to Tav output."""
    out = dict(tav)
    mult = out.get("multiplicity_7fold") or {}
    n_events = int(scan.get("n_events_processed") or mult.get("n_events") or 0)
    fracs = mult.get("mod7_fractions") or []
    alias_mult = dict(mult)
    if fracs and n_events > 0:
        alias_mult["mod7_histogram"] = [
            float(f) * n_events for f in fracs[:7]
        ]
    else:
        alias_mult.setdefault("mod7_histogram", [0.0] * 7)

    primary_raw = scan.get("primary_pt_hist")
    if primary_raw is None:
        primary_raw = scan["muon_pt_hist"]
    primary_hist = np.asarray(primary_raw)
    pt_block = out.get("muon_pt_7fold") or {}

    out.update(
        {
            "action": f"CMS Lepton Tav Analysis ({lepton})",
            "file_path": str(path),
            "lepton": lepton,
            "n_events_processed": n_events,
            "scan_summary": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in scan.items()
                if k
                not in (
                    "n_muon_per_event",
                    "n_electron_per_event",
                    "n_primary_per_event",
                )
            },
            "bin_edges": np.asarray(scan["bin_edges"]).tolist(),
            "muon_multiplicity_7fold": alias_mult,
            "muon_pt_histogram": primary_hist.tolist(),
        }
    )
    if lepton == "Electron":
        out["electron_pt_histogram"] = primary_hist.tolist()
        out["electron_multiplicity_7fold"] = alias_mult
        out["electron_pt_7fold"] = pt_block
        cross_raw = scan.get("crosscheck_pt_hist")
        if cross_raw is None:
            cross_raw = scan.get("muon_pt_hist")
        out["muon_pt_histogram"] = np.asarray(cross_raw if cross_raw is not None else []).tolist()
    out["output_dir"] = str(output_dir)
    return out


def tav_analyze_leptons_from_file(
    file_path: str | Path,
    *,
    output_dir: str = "",
    chunk_size: int = 500_000,
    entry_stop: int | None = None,
    lepton: str = "Muon",
    pt_bins: int = 20,
    pt_max_gev: float = 200.0,
    save_plots: bool = True,
    verbose: bool = True,
    use_curses_progress: bool = False,
    chunk_results: bool | None = None,
) -> dict[str, Any]:
    """
    Chunked full-file lepton analysis → ``tav_7fold_muon_analysis``.

    Supports **Muon** (``Muon_pt``, ``nMuon``) and **Electron**
    (``Electron_pt``, ``nElectron``) NanoAOD branches. When both exist, the
    non-primary channel is used as an optional cross-check histogram.
    """
    lepton_key = str(lepton or "Muon").strip().title()
    if lepton_key not in LEPTON_BRANCH_SPECS:
        raise ValueError(f"Unsupported lepton: {lepton!r}")

    path = _resolve_root(file_path)
    probe = discover_available_lepton_branches(path)
    if not probe["leptons"].get(lepton_key, {}).get("available"):
        avail = [k for k, v in probe["leptons"].items() if v.get("available")]
        raise KeyError(
            f"{lepton_key} branches missing in {path.name}. "
            f"Available leptons: {avail or 'none'}"
        )

    prefix = (output_dir or path.stem).strip() or path.stem
    dataset_slug = compose_dataset_slug(prefix, path.stem, lepton_key.lower())

    if verbose:
        print(f"[lepton v2] Scanning {path.name} ({probe['n_entries']:,} entries)…")

    from menus.particle.cern.analyzer import PT_REDUCTION_FLATTEN

    scan = scan_cms_nanoaod_chunked(
        path,
        chunk_size=int(chunk_size),
        entry_stop=entry_stop,
        pt_bins=int(pt_bins),
        pt_max_gev=float(pt_max_gev),
        primary_lepton=lepton_key,
        include_crosscheck=True,
        pt_reduction=PT_REDUCTION_FLATTEN,
        crosscheck_pt_reduction=PT_REDUCTION_FLATTEN,
        use_curses_progress=use_curses_progress,
        progress_title=f"CMS {lepton_key} Lepton Scan",
        verbose=verbose,
    )

    n_events = int(scan["n_events_processed"])
    auto_chunk = chunk_results
    if auto_chunk is None:
        auto_chunk = should_chunk_tav_results(n_events=n_events)

    tav = tav_7fold_muon_analysis(
        muon_pt_hist=scan["primary_pt_hist"],
        bin_edges=scan["bin_edges"],
        n_muon_per_event=scan["n_primary_per_event"],
        photon_pt_hist=scan.get("crosscheck_pt_hist"),
        output_dir=prefix,
        save_plots=save_plots,
        verbose=verbose,
        dataset_slug=dataset_slug,
        n_events_processed=n_events,
        chunk_results=auto_chunk,
    )

    return _enrich_lepton_result(
        tav,
        path=path,
        lepton=lepton_key,
        scan=scan,
        output_dir=prefix,
    )


def _pt_block(results: dict[str, Any], lepton: str) -> dict[str, Any]:
    key = "muon_pt_7fold" if lepton.title() == "Muon" else f"{lepton.lower()}_pt_7fold"
    return results.get(key) or results.get("muon_pt_7fold") or {}


def _mult_block(results: dict[str, Any], lepton: str) -> dict[str, Any]:
    if lepton.title() == "Electron":
        return (
            results.get("electron_multiplicity_7fold")
            or results.get("muon_multiplicity_7fold")
            or results.get("multiplicity_7fold")
            or {}
        )
    return (
        results.get("muon_multiplicity_7fold")
        or results.get("multiplicity_7fold")
        or {}
    )


def _pt_histogram(results: dict[str, Any], lepton: str) -> list[float]:
    if lepton.title() == "Electron":
        hist = results.get("electron_pt_histogram") or results.get("muon_pt_histogram")
    else:
        hist = results.get("muon_pt_histogram")
    return list(hist or [])


def _comparison_verdict(
    *,
    pt_delta: float,
    chi2_delta: float,
    data_sigma: float,
    mc_sigma: float,
    mc_role: str | None = None,
) -> str:
    if mc_role == "dedicated_signal":
        return "EXPLORATORY_PROCESS_MISMATCH (signal MC vs inclusive dimuon data)"
    if abs(pt_delta) > 1.5:
        return "EXPLORATORY_SHAPE_DIFFERENCE (pT subharmonic)"
    if abs(chi2_delta) > 10.0:
        return "EXPLORATORY_SHAPE_DIFFERENCE (mod-7 multiplicity)"
    if mc_sigma > 8.0 and data_sigma < 3.0:
        return "EXPLORATORY_ENGINE_SCORE_ASYMMETRY"
    if data_sigma > 8.0 and mc_sigma < 3.0:
        return "EXPLORATORY_ENGINE_SCORE_ASYMMETRY"
    if data_sigma > 5.0 and mc_sigma > 5.0:
        return "EXPLORATORY_BOTH_ELEVATED (engine scores — not HEP significance)"
    return "EXPLORATORY_CONSISTENT"


def _save_comparison_plot(
    *,
    data: dict[str, Any],
    mc: dict[str, Any],
    lepton: str,
    metrics: dict[str, Any],
    plot_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    data_pt = _pt_block(data, lepton)
    mc_pt = _pt_block(mc, lepton)
    data_mult = _mult_block(data, lepton)
    mc_mult = _mult_block(mc, lepton)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # A — normalized pT shape
    ax = axes[0, 0]
    d_hist = np.asarray(_pt_histogram(data, lepton), dtype=float)
    m_hist = np.asarray(_pt_histogram(mc, lepton), dtype=float)
    edges = np.asarray(data.get("bin_edges") or mc.get("bin_edges") or [], dtype=float)
    if d_hist.size and m_hist.size and edges.size == d_hist.size + 1:
        d_norm = d_hist / max(d_hist.sum(), 1.0)
        m_norm = m_hist / max(m_hist.sum(), 1.0)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, d_norm, "o-", label="Data", color="#1f4e79", lw=1.2)
        ax.plot(centers, m_norm, "s--", label="MC", color="#c44e52", lw=1.2)
        ax.set_xlabel(f"{lepton} pT [GeV]")
        ax.set_ylabel("Normalized counts")
    else:
        ax.text(0.1, 0.5, "pT histogram unavailable", transform=ax.transAxes)
    ax.set_title("A. pT spectrum (normalized)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # B — mod-7 fractions
    ax = axes[0, 1]
    d_frac = data_mult.get("mod7_fractions") or []
    m_frac = mc_mult.get("mod7_fractions") or []
    if d_frac and m_frac:
        x = np.arange(7)
        width = 0.35
        ax.bar(x - width / 2, d_frac[:7], width, label="Data", color="#55a868")
        ax.bar(x + width / 2, m_frac[:7], width, label="MC", color="#8172b2")
        ax.axhline(1 / 7.0, color="k", ls="--", lw=0.8)
        ax.set_xticks(x)
        ax.set_xlabel(f"{lepton} multiplicity mod 7")
        ax.set_ylabel("Fraction")
    else:
        ax.text(0.1, 0.5, "mod-7 data unavailable", transform=ax.transAxes)
    ax.set_title("B. Multiplicity mod-7")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # C — subharmonic + chi2 metrics
    ax = axes[1, 0]
    ax.axis("off")
    lines = [
        f"Δ subharmonic excess: {metrics.get('pt_delta', 0):.3f}",
        f"Data subharmonic:     {metrics.get('data_subharmonic_excess', 0):.3f}",
        f"MC subharmonic:       {metrics.get('mc_subharmonic_excess', 0):.3f}",
        f"Δ χ² mod-7:           {metrics.get('chi2_mod7_delta', 0):.3f}",
        f"Data χ² mod-7:        {metrics.get('data_chi2_mod7', 0):.3f}",
        f"MC χ² mod-7:          {metrics.get('mc_chi2_mod7', 0):.3f}",
        f"Data 7σ:              {metrics.get('data_significance_sigma', 0):.2f}",
        f"MC 7σ:                {metrics.get('mc_significance_sigma', 0):.2f}",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=9)
    ax.set_title("C. 7-fold metrics")

    # D — verdict
    ax = axes[1, 1]
    ax.axis("off")
    ax.text(
        0.05,
        0.7,
        metrics.get("verdict", ""),
        fontsize=12,
        fontweight="bold",
        color="#1f4e79",
    )
    ax.text(
        0.05,
        0.45,
        f"Lepton: {lepton}\n"
        f"Data events: {data.get('n_events_processed', '—'):,}\n"
        f"MC events:   {mc.get('n_events_processed', '—'):,}",
        family="monospace",
        fontsize=9,
    )
    ax.set_title("D. Validation summary")

    fig.suptitle(f"Tav Data vs MC — {lepton}", fontsize=12, y=1.01)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def tav_compare_data_mc(
    data_results: dict[str, Any],
    mc_results: dict[str, Any],
    *,
    output_dir: str = "tav_mc_validation",
    lepton: str = "Muon",
    save_plots: bool = True,
    verbose: bool = True,
    max_result_chunk_mb: float = DEFAULT_MAX_CHUNK_MB,
) -> dict[str, Any]:
    """
    Compare Data and MC Tav lepton analysis results; save report + overlay plot.
    """
    lepton_key = str(lepton or "Muon").strip().title()
    data_pt = _pt_block(data_results, lepton_key)
    mc_pt = _pt_block(mc_results, lepton_key)
    data_mult = _mult_block(data_results, lepton_key)
    mc_mult = _mult_block(mc_results, lepton_key)

    data_sp = data_results.get("seven_periodic") or {}
    mc_sp = mc_results.get("seven_periodic") or {}

    data_sub = float(data_pt.get("subharmonic_excess") or 0.0)
    mc_sub = float(mc_pt.get("subharmonic_excess") or 0.0)
    data_chi2 = float(data_mult.get("chi2_mod7_vs_uniform") or 0.0)
    mc_chi2 = float(mc_mult.get("chi2_mod7_vs_uniform") or 0.0)
    data_sigma = float(data_sp.get("significance_sigma") or 0.0)
    mc_sigma = float(mc_sp.get("significance_sigma") or 0.0)

    from menus.particle.cms.hep_statistics import (
        classify_mc_sample_label,
        mc_normalization_metadata,
        exploratory_classification_label,
    )

    mc_path = str(mc_results.get("file_path") or "")
    mc_meta = classify_mc_sample_label(mc_path)
    n_data = int(data_results.get("n_events") or data_results.get("n_events_processed") or 0)
    n_mc = int(mc_results.get("n_events") or mc_results.get("n_events_processed") or 0)
    norm_meta = mc_normalization_metadata(n_data, n_mc)

    pt_delta = data_sub - mc_sub
    chi2_delta = data_chi2 - mc_chi2

    from tav_shared.dataset_comparison.pipeline import (
        compare_aggregate_metrics,
        compare_histogram_pair,
    )
    from tav_shared.dataset_comparison.tolerance import tolerant_diff

    data_hist = np.asarray(_pt_histogram(data_results, lepton_key), dtype=float)
    mc_hist = np.asarray(_pt_histogram(mc_results, lepton_key), dtype=float)
    scan_data = data_results.get("scan_summary") or {}
    scan_mc = mc_results.get("scan_summary") or {}
    bin_edges_raw = scan_data.get("bin_edges") or scan_mc.get("bin_edges")
    bin_edges = np.asarray(bin_edges_raw, dtype=float) if bin_edges_raw is not None else None

    verdict = _comparison_verdict(
        pt_delta=pt_delta,
        chi2_delta=chi2_delta,
        data_sigma=data_sigma,
        mc_sigma=mc_sigma,
        mc_role=mc_meta.get("mc_role"),
    )

    prefix = (output_dir or "tav_mc_validation").strip()
    run_when = datetime.now(timezone.utc)
    slug = compose_dataset_slug(prefix, "data_mc_compare", lepton_key.lower())
    run_dir = artifact_run_dir(TestSlug.CERN, slug, when=run_when)

    comparison: dict[str, Any] = {
        "action": "Tav Data vs MC Comparison",
        "timestamp": artifact_timestamp(run_when),
        "lepton": lepton_key,
        "output_dir": prefix,
        "run_dir": str(run_dir),
        "data_file": data_results.get("file_path"),
        "mc_file": mc_results.get("file_path"),
        "data_subharmonic_excess": data_sub,
        "mc_subharmonic_excess": mc_sub,
        "pt_delta": pt_delta,
        "data_chi2_mod7": data_chi2,
        "mc_chi2_mod7": mc_chi2,
        "chi2_mod7_delta": chi2_delta,
        "data_significance_sigma": data_sigma,
        "mc_significance_sigma": mc_sigma,
        "data_engine_score_sigma_raw": (data_sp.get("engine_score_sigma_raw")),
        "mc_engine_score_sigma_raw": (mc_sp.get("engine_score_sigma_raw")),
        "engine_score_capped": bool(data_sp.get("engine_score_capped") or mc_sp.get("engine_score_capped")),
        "interpretation_label": data_sp.get("interpretation_label")
        or "internal_engine_score_not_hep_significance",
        "exploratory_classification": exploratory_classification_label(),
        "mc_sample_metadata": mc_meta,
        "normalization_metadata": norm_meta,
        "mc_strong_7fold": False,
        "data_strong_7fold": False,
        "verdict": verdict,
        "data_mod7_fractions": data_mult.get("mod7_fractions"),
        "mc_mod7_fractions": mc_mult.get("mod7_fractions"),
    }

    if data_hist.size and mc_hist.size:
        comparison["pt_histogram_comparison"] = compare_histogram_pair(
            data_hist,
            mc_hist,
            bin_edges=bin_edges,
        )
    comparison["tolerant_metric_comparison"] = compare_aggregate_metrics(
        {"subharmonic_excess": data_sub, "chi2_mod7": data_chi2},
        {"subharmonic_excess": mc_sub, "chi2_mod7": mc_chi2},
        tolerance_map={
            "subharmonic_excess": "amplitude",
            "chi2_mod7": "chi2",
        },
    ).as_dict()
    comparison["pt_delta_tolerant"] = tolerant_diff(data_sub, mc_sub, label="amplitude")
    comparison["chi2_delta_tolerant"] = tolerant_diff(data_chi2, mc_chi2, label="chi2")

    if save_plots:
        plot_path = artifact_path(
            TestSlug.CERN,
            compose_dataset_slug(slug, "compare"),
            "plot",
            "png",
            when=run_when,
            run_dir=run_dir,
        )
        _save_comparison_plot(
            data=data_results,
            mc=mc_results,
            lepton=lepton_key,
            metrics=comparison,
            plot_path=plot_path,
        )
        comparison["plot_path"] = str(plot_path)
        comparison["plot_paths"] = [str(plot_path)]

    report_base = artifact_path(
        TestSlug.CERN,
        compose_dataset_slug(slug, "compare"),
        "report",
        "json",
        when=run_when,
        run_dir=run_dir,
    ).stem

    created = save_tav_results_chunked(
        results=comparison,
        output_dir=run_dir,
        base_filename=report_base,
        max_size_mb=max_result_chunk_mb,
    )
    comparison["report_path"] = created[0]
    comparison["report_paths"] = created
    comparison["report_chunked"] = len(created) > 1

    if verbose:
        print("=== TAV DATA vs MC COMPARISON ===")
        print(f"  Lepton          : {lepton_key}")
        print(f"  Δ subharmonic   : {pt_delta:.3f}")
        print(f"  Δ χ² mod-7      : {chi2_delta:.3f}")
        print(
            f"  Engine score    : {data_sigma:.2f} / {mc_sigma:.2f} "
            f"(internal — not HEP significance)"
        )
        if mc_meta.get("warning"):
            print(f"  MC warning      : {mc_meta['warning']}")
        print(f"  Verdict         : {verdict}")
        if comparison.get("plot_path"):
            print(f"  Plot            : {comparison['plot_path']}")
        print(f"  Report          : {comparison['report_path']}")
        print("=" * 35)

    return comparison