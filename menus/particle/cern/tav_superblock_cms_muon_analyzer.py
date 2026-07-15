"""
Tav-Superblock 7-fold CMS muon analysis.

Searches education NanoAOD muon pT spectra and per-event multiplicity for
1/7 harmonic structure (octonionic clockwork falsification channel).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from menus.particle.cern.mod7_phase import mod7_phase_residues
from scipy import signal
from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi2_dist

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

TAV_HARMONIC_RATIO: float = 1.0 / 7.0
TAV_PERIOD: float = 7.0
M0_MEV_ANCHOR: float = 313.1
DEFAULT_PT_BINS: int = 28  # multiple of 7 for bin-phase tests

# Full-file CMS skims (61M+ events) blow up periodogram payloads without downsampling.
LARGE_PERIODogram_SERIES = 100_000
MAX_PERIODogram_POINTS = 8_192
MAX_HARMONIC_HITS = 12
MAX_PEAK_INDICES = 12


def _as_1d_float(arr: Any) -> np.ndarray:
    out = np.asarray(arr, dtype=float).ravel()
    return out[np.isfinite(out)]


def _bin_centers(edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=float)
    if edges.size < 2:
        return np.array([], dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])


def _exponential_detrend(
    x: np.ndarray,
    hist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Exponential detrend: ``h(x) ≈ A·exp(-b·x) + c``."""

    def model(xin: np.ndarray, amp: float, slope: float, offset: float) -> np.ndarray:
        return amp * np.exp(-slope * xin) + offset

    xx = _as_1d_float(x)
    hh = _as_1d_float(hist)
    n = min(xx.size, hh.size)
    if n < 3:
        return hh, np.zeros_like(hh), {"A": 0.0, "b": 0.0, "c": 0.0}
    xx, hh = xx[:n], hh[:n]
    p0 = [float(np.max(hh)), 0.5, 0.0]
    try:
        popt, _ = curve_fit(model, xx, hh, p0=p0, maxfev=10000)
    except (RuntimeError, ValueError):
        popt = p0
    trend = model(xx, *popt)
    residuals = hh - trend
    return trend, residuals, {"A": float(popt[0]), "b": float(popt[1]), "c": float(popt[2])}


def _fit_7periodic(
    x: np.ndarray,
    residuals: np.ndarray,
    *,
    period: float = TAV_PERIOD,
) -> tuple[float, float, float]:
    """Fit ``A·sin(2π x / period + φ) + c``; returns ``(A7, phi, amplitude)``."""

    def model(xin: np.ndarray, amp: float, phi: float, offset: float) -> np.ndarray:
        return amp * np.sin(2.0 * np.pi * xin / period + phi) + offset

    xx = _as_1d_float(x)
    yy = _as_1d_float(residuals)
    n = min(xx.size, yy.size)
    if n < 4:
        return 0.0, 0.0, 0.0
    xx, yy = xx[:n], yy[:n]
    p0 = [float(np.std(yy)), 0.0, float(np.mean(yy))]
    try:
        popt, _ = curve_fit(model, xx, yy, p0=p0, maxfev=8000)
        amp, phi, _offset = (float(popt[0]), float(popt[1]), float(popt[2]))
        return amp, phi, abs(amp)
    except (RuntimeError, ValueError):
        return 0.0, 0.0, float(np.std(yy))


def _downsample_series(series: np.ndarray, max_points: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Stride-downsample long event series before FFT periodogram."""
    n = int(series.size)
    if n <= max_points:
        return series, {"downsampled": False, "n_original": n, "n_analyzed": n}
    stride = int(np.ceil(n / max_points))
    down = series[::stride]
    return down, {
        "downsampled": True,
        "n_original": n,
        "n_analyzed": int(down.size),
        "stride": stride,
    }


def _trim_periodogram_payload(spec: dict[str, Any]) -> dict[str, Any]:
    """Cap harmonic hit lists so large-run JSON stays bounded."""
    out = dict(spec)
    hits = list(out.get("harmonic_hits") or [])
    if len(hits) > MAX_HARMONIC_HITS:
        hits.sort(key=lambda h: float(h.get("power") or 0.0), reverse=True)
        out["harmonic_hits"] = hits[:MAX_HARMONIC_HITS]
        out["harmonic_hits_trimmed"] = True
        out["harmonic_hits_total"] = len(hits)
    peaks = list(out.get("peak_indices") or [])
    if len(peaks) > MAX_PEAK_INDICES:
        out["peak_indices"] = peaks[:MAX_PEAK_INDICES]
        out["peak_indices_trimmed"] = True
    return out


def _periodogram_7fold(series: np.ndarray) -> dict[str, Any]:
    """Periodogram phase test — peaks near k/7 in normalized frequency."""
    clean = _as_1d_float(series)
    if clean.size < 8:
        return {
            "n_points": int(clean.size),
            "detected": False,
            "harmonic_hits": [],
            "verdict": "UNDERPOWERED",
        }

    analyzed, ds_meta = _downsample_series(clean, MAX_PERIODogram_POINTS)
    detrended = signal.detrend(analyzed, type="linear")
    freqs, power = signal.periodogram(detrended, detrend=False)
    if power.size < 3:
        return {
            "n_points": int(clean.size),
            "detected": False,
            "harmonic_hits": [],
            "verdict": "UNDERPOWERED",
            **ds_meta,
        }

    peak_idx, _props = signal.find_peaks(power, height=np.percentile(power, 85), distance=1)
    hits: list[dict[str, Any]] = []
    n = len(power)
    for idx in peak_idx:
        phase = float(idx) / max(n - 1, 1)
        nearest = round(phase / TAV_HARMONIC_RATIO) * TAV_HARMONIC_RATIO
        delta = min(
            abs(phase - nearest),
            abs(phase - nearest - 1.0),
            abs(phase - nearest + 1.0),
        )
        if delta < 0.05:
            hits.append(
                {
                    "bin_index": int(idx),
                    "normalized_phase": phase,
                    "nearest_seventh": float(nearest),
                    "phase_delta": float(delta),
                    "power": float(power[idx]),
                }
            )

    baseline = float(np.median(power)) if power.size else 0.0
    sub_idx = max(1, int(round((TAV_HARMONIC_RATIO) * (n - 1))))
    sub_power = float(power[sub_idx]) if sub_idx < power.size else 0.0
    sub_excess = (sub_power / baseline) if baseline > 0 else 0.0

    detected = len(hits) >= 2 or sub_excess >= 2.5
    verdict = (
        "7-FOLD SIGNATURE"
        if detected and len(hits) >= 2
        else "SUBHARMONIC HINT"
        if sub_excess >= 2.0
        else "INCONCLUSIVE"
        if hits
        else "NO 7-FOLD STRUCTURE"
    )

    return _trim_periodogram_payload(
        {
            "n_points": int(clean.size),
            "detected": bool(detected),
            "harmonic_hits": hits,
            "subharmonic_excess": sub_excess,
            "verdict": verdict,
            "peak_indices": [int(i) for i in peak_idx],
            **ds_meta,
        }
    )


def _multiplicity_7fold(n_muon: np.ndarray) -> dict[str, Any]:
    """Per-event nMuon — mod-7 occupancy and multiplicity periodogram."""
    counts = _as_1d_float(n_muon)
    if counts.size < 8:
        return {"n_events": int(counts.size), "verdict": "UNDERPOWERED", "detected": False}

    mod_residues = mod7_phase_residues(counts, clip_max=32)
    mod_hist = np.bincount(mod_residues, minlength=7).astype(float)
    mod_hist /= max(mod_hist.sum(), 1.0)

    # Deviation from uniform mod-7 (null for Poisson-like multiplicity)
    uniform = 1.0 / 7.0
    chi2_mod7 = float(np.sum((mod_hist - uniform) ** 2 / uniform))

    # Cumulative multiplicity series (length n_events) for periodogram.
    # Downsample inside _periodogram_7fold for multi-million-event skims.
    int_counts = np.clip(np.rint(counts), 0, 32).astype(np.int64)
    cum_mult = np.cumsum(int_counts.astype(float))
    ds_meta: dict[str, Any] = {
        "downsampled": False,
        "n_original": int(counts.size),
        "n_analyzed": int(counts.size),
    }
    if counts.size > LARGE_PERIODogram_SERIES:
        cum_mult, ds_meta = _downsample_series(cum_mult, MAX_PERIODogram_POINTS)
    spec = _periodogram_7fold(cum_mult)
    spec["n_events_original"] = int(counts.size)
    if ds_meta.get("downsampled"):
        spec.update(ds_meta)
    detected = chi2_mod7 > 14.0 or spec.get("detected", False)
    verdict = (
        "MULTIPLICITY 7-FOLD"
        if detected and chi2_mod7 > 14.0
        else "MULTIPLICITY PERIODOGRAM HINT"
        if spec.get("detected")
        else "INCONCLUSIVE"
        if chi2_mod7 > 7.0
        else "NO 7-FOLD MULTIPLICITY"
    )

    return {
        "n_events": int(counts.size),
        "mean_n_muon": float(np.mean(counts)),
        "max_n_muon": float(np.max(counts)),
        "mod7_residues": mod_residues.tolist(),
        "mod7_fractions": mod_hist.tolist(),
        "chi2_mod7_vs_uniform": chi2_mod7,
        "cumulative_periodogram": spec,
        "detected": bool(detected),
        "verdict": verdict,
    }


def _pt_spectrum_7fold(hist: np.ndarray, bin_edges: np.ndarray | None) -> dict[str, Any]:
    counts = _as_1d_float(hist)
    if counts.size < 8:
        return {"verdict": "UNDERPOWERED", "detected": False}

    if bin_edges is not None and len(bin_edges) == len(counts) + 1:
        centers = _bin_centers(np.asarray(bin_edges, dtype=float))
        log_centers = np.log1p(np.maximum(centers, 0.0))
        if log_centers.size == counts.size:
            # Resample to uniform log-pT grid for FFT
            grid = np.linspace(log_centers.min(), log_centers.max(), max(28, counts.size))
            resampled = np.interp(grid, log_centers, counts)
            spec = _periodogram_7fold(resampled)
            spec["bin_edges_gev"] = [float(x) for x in bin_edges.tolist()]
            spec["pt_centers_gev"] = [float(x) for x in centers.tolist()]
            return spec

    return _periodogram_7fold(counts)


def _seven_periodic_summary(
    muon_pt: dict[str, Any],
    multiplicity: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Combined 7-periodic significance (σ) for menu / publication reporting.

    Merges pT subharmonic excess and multiplicity mod-7 χ² into one scalar.
    """
    sigmas: list[float] = []

    sub_excess = float(muon_pt.get("subharmonic_excess") or 0.0)
    if sub_excess > 1.0:
        sigmas.append(float(min(12.0, (sub_excess - 1.0) * 1.8)))

    n_hits = len(muon_pt.get("harmonic_hits") or [])
    if n_hits >= 2:
        sigmas.append(float(min(12.0, 2.0 + 0.75 * n_hits)))

    chi2_mod7 = None
    if multiplicity:
        chi2_mod7 = float(multiplicity.get("chi2_mod7_vs_uniform") or 0.0)
        if chi2_mod7 > 7.0:
            # 6 dof uniform-mod-7 null; one-sided excess p-value → Gaussian σ
            p_tail = float(1.0 - chi2_dist.cdf(chi2_mod7, df=6))
            p_tail = max(p_tail, 1e-300)
            from scipy.stats import norm

            sigmas.append(float(min(12.0, norm.isf(p_tail))))

    significance = float(max(sigmas)) if sigmas else 0.0
    return {
        "significance_sigma": significance,
        "pt_subharmonic_excess": sub_excess,
        "pt_harmonic_hit_count": n_hits,
        "multiplicity_chi2_mod7": chi2_mod7,
        "components_sigma": sigmas,
    }


def _photon_muon_crosscheck(
    muon_hist: np.ndarray,
    photon_hist: np.ndarray,
) -> dict[str, Any]:
    m = _as_1d_float(muon_hist)
    p = _as_1d_float(photon_hist)
    n = min(m.size, p.size)
    if n < 8:
        return {"verdict": "UNDERPOWERED", "detected": False}
    m, p = m[:n], p[:n]
    if np.std(m) > 0 and np.std(p) > 0:
        corr_arr = np.corrcoef(m, p)[0, 1]
        corr = float(np.asarray(corr_arr).reshape(-1)[0])
    else:
        corr = 0.0
    ratio = np.divide(m, p + 1e-9)
    ratio_spec = _periodogram_7fold(ratio)
    return {
        "muon_photon_correlation": corr,
        "ratio_spectrum_7fold": ratio_spec,
        "detected": ratio_spec.get("detected", False),
        "verdict": ratio_spec.get("verdict", "INCONCLUSIVE"),
    }


def _save_plots(
    result: dict[str, Any],
    *,
    muon_pt_hist: np.ndarray,
    bin_edges: np.ndarray | None,
    n_muon_per_event: np.ndarray | None,
    photon_pt_hist: np.ndarray | None,
    output_dir: Path,
    dataset_slug: str,
    run_dir: Path,
    run_when: datetime,
) -> list[str]:
    import matplotlib.pyplot as plt

    paths: list[str] = []
    counts = _as_1d_float(muon_pt_hist)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    if bin_edges is not None and len(bin_edges) == len(counts) + 1:
        ax.stairs(counts, bin_edges, color="#1f4e79", fill=True, alpha=0.85)
        ax.set_xlabel("Muon pT [GeV]")
    else:
        ax.bar(np.arange(len(counts)), counts, color="#1f4e79", alpha=0.85)
        ax.set_xlabel("pT bin index")
    ax.set_ylabel("Counts")
    ax.set_title("A. Muon pT spectrum")

    ax = axes[0, 1]
    detrended = signal.detrend(counts, type="linear") if counts.size >= 4 else counts
    freqs, power = signal.periodogram(detrended) if detrended.size >= 8 else ([], [])
    if len(power):
        ax.semilogy(power, color="#4c72b0", lw=1.0)
        for hit in result.get("muon_pt_7fold", {}).get("harmonic_hits", []):
            ax.axvline(hit["bin_index"], color="#c44e52", ls=":", alpha=0.7)
    ax.set_xlabel("Periodogram bin")
    ax.set_ylabel("Power")
    ax.set_title(f"B. pT 7-fold — {result.get('muon_pt_7fold', {}).get('verdict', '')}")

    ax = axes[1, 0]
    mult = result.get("multiplicity_7fold") or {}
    fracs = mult.get("mod7_fractions")
    if fracs:
        ax.bar(range(7), fracs, color="#55a868", alpha=0.85)
        ax.axhline(1.0 / 7.0, color="k", ls="--", lw=0.8)
        ax.set_xlabel("nMuon mod 7")
        ax.set_ylabel("Fraction")
        ax.set_title(f"C. Multiplicity — {mult.get('verdict', 'n/a')}")
    else:
        ax.axis("off")
        ax.text(0.1, 0.5, "nMuon per event not provided", fontsize=10)

    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        f"Global: {result.get('verdict', '')}",
        f"pT 7-fold: {result.get('muon_pt_7fold', {}).get('verdict', '')}",
        f"Subharmonic excess: {result.get('muon_pt_7fold', {}).get('subharmonic_excess', 0):.2f}×",
        f"Multiplicity: {mult.get('verdict', 'n/a')}",
        f"M₀ anchor: {M0_MEV_ANCHOR} MeV",
        f"Photon cross-check: {(result.get('photon_crosscheck') or {}).get('verdict', 'n/a')}",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=9)
    ax.set_title("D. Summary")

    fig.suptitle("Tav-Superblock CMS 7-Fold Muon Analysis", fontsize=12, y=1.01)
    fig.tight_layout()

    plot_path = artifact_path(
        TestSlug.CERN,
        compose_dataset_slug(dataset_slug, "tav_7fold_muon"),
        "plot",
        "png",
        when=run_when,
        run_dir=run_dir,
    )
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(plot_path))
    return paths


def tav_7fold_muon_analysis(
    muon_pt_hist: np.ndarray | list,
    *,
    bin_edges: np.ndarray | list | None = None,
    n_muon_per_event: np.ndarray | list | None = None,
    photon_pt_hist: np.ndarray | list | None = None,
    output_dir: str | Path = "tav_cms_output",
    save_plots: bool = True,
    verbose: bool = True,
    dataset_slug: str | None = None,
    n_events_processed: int | None = None,
    chunk_results: bool | None = None,
    max_result_chunk_mb: float = DEFAULT_MAX_CHUNK_MB,
) -> dict[str, Any]:
    """
    Search CMS muon observables for Tav 1/7 harmonic signatures.

    Parameters
    ----------
    muon_pt_hist
        Binned muon pT counts from NanoAOD scan.
    bin_edges
        pT bin edges in GeV (strongly recommended for physical 7-fold test).
    n_muon_per_event
        Per-event ``nMuon`` array — unlocks multiplicity mod-7 test.
    photon_pt_hist
        Optional photon/electron pT histogram for cross-channel check.
    output_dir
        Dataset label for artifact subdirectory (canonical paths under
        ``artifacts/cern_analysis/``).
    """
    hist = _as_1d_float(muon_pt_hist)
    edges = np.asarray(bin_edges, dtype=float) if bin_edges is not None else None
    slug = dataset_slug or Path(str(output_dir)).name or "tav_cms_output"
    run_when = datetime.now(timezone.utc)
    run_dir = artifact_run_dir(TestSlug.CERN, slug, when=run_when)

    muon_pt = _pt_spectrum_7fold(hist, edges)
    multiplicity: dict[str, Any] | None = None
    if n_muon_per_event is not None:
        multiplicity = _multiplicity_7fold(np.asarray(n_muon_per_event))

    photon_cross: dict[str, Any] | None = None
    if photon_pt_hist is not None:
        photon_cross = _photon_muon_crosscheck(hist, np.asarray(photon_pt_hist))

    hits = int(muon_pt.get("detected", False))
    mult_hit = bool(multiplicity and multiplicity.get("detected"))
    photon_hit = bool(photon_cross and photon_cross.get("detected"))

    if hits and mult_hit:
        verdict = "STRONG 7-FOLD (pT + multiplicity)"
    elif hits or mult_hit:
        verdict = "PARTIAL 7-FOLD SIGNATURE"
    elif photon_hit:
        verdict = "PHOTON RATIO 7-FOLD HINT"
    elif muon_pt.get("verdict") == "SUBHARMONIC HINT":
        verdict = "WEAK SUBHARMONIC HINT"
    else:
        verdict = "NO CLEAR 7-FOLD SIGNATURE"

    seven_periodic = _seven_periodic_summary(muon_pt, multiplicity)

    result: dict[str, Any] = {
        "action": "CMS 7-Fold Muon Tav Analysis",
        "timestamp": artifact_timestamp(),
        "m0_mev_anchor": M0_MEV_ANCHOR,
        "tav_harmonic_ratio": TAV_HARMONIC_RATIO,
        "muon_pt_7fold": muon_pt,
        "multiplicity_7fold": multiplicity,
        "photon_crosscheck": photon_cross,
        "seven_periodic": seven_periodic,
        "verdict": verdict,
        "n_pt_bins": int(hist.size),
        "total_muons_binned": int(hist.sum()),
        "output_dir": str(output_dir),
    }

    if save_plots:
        out_path = Path(output_dir)
        result["plot_paths"] = _save_plots(
            result,
            muon_pt_hist=hist,
            bin_edges=edges,
            n_muon_per_event=(
                np.asarray(n_muon_per_event) if n_muon_per_event is not None else None
            ),
            photon_pt_hist=(
                np.asarray(photon_pt_hist) if photon_pt_hist is not None else None
            ),
            output_dir=out_path,
            dataset_slug=slug,
            run_dir=run_dir,
            run_when=run_when,
        )

    report_path = artifact_path(
        TestSlug.CERN,
        compose_dataset_slug(slug, "tav_7fold_muon"),
        "report",
        "json",
        when=run_when,
        run_dir=run_dir,
    )
    result["run_dir"] = str(run_dir)
    n_events = n_events_processed
    if n_events is None and multiplicity:
        n_events = int(multiplicity.get("n_events") or 0)

    use_chunked = should_chunk_tav_results(
        n_events=int(n_events or 0),
        chunk_results=chunk_results,
    )

    if use_chunked:
        created_files = save_tav_results_chunked(
            results=result,
            output_dir=report_path.parent,
            base_filename=report_path.stem,
            max_size_mb=max_result_chunk_mb,
        )
        result["report_paths"] = created_files
        result["report_chunked"] = len(created_files) > 1
        result["report_path"] = created_files[0]
        result["max_result_chunk_mb"] = float(max_result_chunk_mb)
    else:
        from menus.astronomical.desi.json_util import write_json

        write_json(report_path, result, indent=2, sort_keys=True)
        result["report_path"] = str(report_path)
        result["report_paths"] = [str(report_path)]
        result["report_chunked"] = False

    if verbose:
        print("=" * 70)
        print("TAV-SUPERBLOCK CMS 7-FOLD MUON ANALYSIS")
        print(f"  pT verdict      : {muon_pt.get('verdict')}")
        if multiplicity:
            print(f"  Multiplicity    : {multiplicity.get('verdict')}")
        print(f"  Global verdict  : {verdict}")
        print(f"  7-periodic σ    : {seven_periodic.get('significance_sigma', 0):.2f}")
        if result.get("report_chunked"):
            print(
                f"  Report (chunked): {result['report_path']} "
                f"({len(result['report_paths']) - 1} parts, "
                f"≤{max_result_chunk_mb:.1f} MB each)"
            )
        else:
            print(f"  Report          : {result['report_path']}")
        if result.get("plot_paths"):
            for p in result["plot_paths"]:
                print(f"  Plot            : {p}")
        print("=" * 70)

    return result