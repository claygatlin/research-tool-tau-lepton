"""
HEP-standard statistics upgrades for CMS dimuon validation.

Implements critique recommendations (2026-07-15): trigger-aware mod-7 nulls,
labeled engine scores (not particle-physics significance), MC role warnings,
fine q_T binning, and ordering/partition null tests.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.stats import chi2 as chi2_dist
from scipy.stats import norm

GEOMETRIC_FLOOR_GEV: float = 0.3131
ENGINE_SCORE_CAP: float = 12.0
COARSE_PT_BIN_WIDTH_GEV: float = 10.0
FINE_QT_BIN_WIDTH_GEV: float = 0.01
PREREG_QT_SIGNAL_WINDOW: tuple[float, float] = (0.25, 0.40)
PREREG_QT_SIDEBAND_LO: tuple[float, float] = (0.05, 0.20)
PREREG_QT_SIDEBAND_HI: tuple[float, float] = (0.45, 0.80)

SIGNAL_MC_HINTS = ("higgs", "smhiggs", "signal", "zzto4l", "zzto4l")
BACKGROUND_MC_HINTS = ("dy", "dyjets", "ttbar", "tt_", "qcd", "wjets", "diboson")


def trigger_aware_mod7_expected_fractions(n_muon: np.ndarray) -> np.ndarray:
    """
    Map empirical P(n_μ) to expected mod-7 residue fractions.

    For DoubleMu-triggered data, n_μ = 2 ⟹ r = 2 (mod 7), so residue-2 excess
    under a uniform null is a selection artifact, not new physics.
    """
    counts = np.asarray(n_muon, dtype=np.int64).ravel()
    counts = counts[counts >= 0]
    if counts.size == 0:
        return np.full(7, 1.0 / 7.0)
    max_n = int(np.max(counts)) + 1
    hist_n = np.bincount(counts, minlength=max_n).astype(float)
    prob = hist_n / max(hist_n.sum(), 1.0)
    expected = np.zeros(7, dtype=float)
    for n_val, p_n in enumerate(prob):
        if p_n <= 0.0:
            continue
        expected[n_val % 7] += p_n
    total = expected.sum()
    if total <= 0.0:
        return np.full(7, 1.0 / 7.0)
    return expected / total


def pearson_chi2_counts(
    observed: np.ndarray,
    expected: np.ndarray,
    *,
    min_exp: float = 1e-9,
) -> dict[str, float]:
    """Pearson χ² on count bins with full metadata."""
    obs = np.asarray(observed, dtype=float).ravel()
    exp = np.asarray(expected, dtype=float).ravel()
    if obs.size != exp.size:
        raise ValueError("observed and expected must have equal length")
    chi2 = float(np.sum((obs - exp) ** 2 / np.maximum(exp, min_exp)))
    df = max(int(obs.size) - 1, 1)
    p_value = float(1.0 - chi2_dist.cdf(chi2, df=df))
    return {
        "chi2": chi2,
        "df": float(df),
        "p_value": p_value,
        "reduced_chi2": chi2 / df,
    }


def z_from_p_one_sided(p_value: float) -> float:
    p = float(np.clip(p_value, 1e-300, 1.0))
    return float(norm.isf(p))


def engine_score_from_components(
    components: list[float],
    *,
    cap: float = ENGINE_SCORE_CAP,
) -> dict[str, Any]:
    """
    Internal engine score — explicitly NOT particle-physics significance.

    Legacy pipeline capped at 12σ; this reports raw and capped values.
    """
    raw = float(max(components)) if components else 0.0
    capped = float(min(raw, cap))
    return {
        "significance_sigma": capped,
        "engine_score_sigma": capped,
        "engine_score_sigma_raw": raw,
        "engine_score_capped": raw > cap + 1e-9,
        "engine_score_cap": cap,
        "interpretation_label": (
            "internal_engine_score_not_hep_significance"
            if raw > 0.0
            else "no_excess_detected"
        ),
        "hep_significance_note": (
            "Not a Cowan et al. likelihood-ratio significance. "
            "Use preregistered q_T study for HEP-grade inference."
        ),
        "components_sigma": [float(c) for c in components],
    }


def classify_mc_sample_label(path_or_name: str) -> dict[str, Any]:
    """Label MC as dedicated signal vs background for Data-MC comparisons."""
    text = str(path_or_name).lower().replace("\\", "/")
    if any(hint in text for hint in SIGNAL_MC_HINTS):
        return {
            "mc_role": "dedicated_signal",
            "valid_as_sm_background": False,
            "warning": (
                "SMHiggsToZZTo4L is an H→ZZ→4ℓ signal sample — not a universal SM "
                "background for inclusive DoubleMuParked. Prefer DYJetsToLL + HF + "
                "tt̄ stack (Option 3 MC stack)."
            ),
        }
    if any(hint in text for hint in BACKGROUND_MC_HINTS):
        return {
            "mc_role": "background_process",
            "valid_as_sm_background": True,
            "warning": None,
        }
    return {
        "mc_role": "unknown",
        "valid_as_sm_background": False,
        "warning": (
            "MC process role unknown — shape comparisons are exploratory only."
        ),
    }


def mc_normalization_metadata(n_data: int, n_mc: int) -> dict[str, Any]:
    ratio = float(n_data) / max(float(n_mc), 1.0)
    return {
        "n_data": int(n_data),
        "n_mc": int(n_mc),
        "event_count_ratio": ratio,
        "note": (
            "Raw histogram amplitude ratios reflect event-count disparity and "
            "process mismatch; compare per-event shapes or luminosity-weighted MC."
        ),
    }


def compute_acoplanarity(delta_phi: np.ndarray) -> np.ndarray:
    """α = π − |Δφ| (radians); α = 0 for back-to-back dimuons."""
    return np.pi - np.asarray(delta_phi, dtype=float)


def compute_pt_balance(leading_pt: np.ndarray, subleading_pt: np.ndarray) -> np.ndarray:
    """|pT1 − pT2| / (pT1 + pT2) balance metric."""
    p1 = np.asarray(leading_pt, dtype=float)
    p2 = np.asarray(subleading_pt, dtype=float)
    denom = np.maximum(p1 + p2, 1e-9)
    return np.abs(p1 - p2) / denom


def fine_qt_histogram(
    qt_gev: np.ndarray,
    *,
    q_max: float = 2.0,
    bin_width: float = FINE_QT_BIN_WIDTH_GEV,
) -> dict[str, Any]:
    """Fine-binned q_T histogram suitable for 313.1 MeV-scale searches."""
    qt = np.asarray(qt_gev, dtype=float).ravel()
    qt = qt[np.isfinite(qt) & (qt >= 0.0) & (qt <= q_max)]
    edges = np.arange(0.0, q_max + bin_width, bin_width)
    hist, edges_out = np.histogram(qt, bins=edges)
    centers = 0.5 * (edges_out[:-1] + edges_out[1:])
    sig_lo, sig_hi = PREREG_QT_SIGNAL_WINDOW
    lo_lo, lo_hi = PREREG_QT_SIDEBAND_LO
    hi_lo, hi_hi = PREREG_QT_SIDEBAND_HI
    signal_mask = (centers >= sig_lo) & (centers < sig_hi)
    side_lo_mask = (centers >= lo_lo) & (centers < lo_hi)
    side_hi_mask = (centers >= hi_lo) & (centers < hi_hi)
    q0 = GEOMETRIC_FLOOR_GEV
    return {
        "observable": "q_T = |pT1_vec + pT2_vec|",
        "bin_width_gev": bin_width,
        "bin_edges_gev": edges_out.tolist(),
        "counts": hist.tolist(),
        "n_events": int(qt.size),
        "mean_qt_gev": float(qt.mean()) if qt.size else 0.0,
        "median_qt_gev": float(np.median(qt)) if qt.size else 0.0,
        "q0_gev": q0,
        "q0_bin_index": int(round(q0 / bin_width)),
        "signal_window_gev": {"lo": sig_lo, "hi": sig_hi},
        "signal_window_counts": int(hist[signal_mask].sum()),
        "sideband_lo_counts": int(hist[side_lo_mask].sum()),
        "sideband_hi_counts": int(hist[side_hi_mask].sum()),
        "coarse_pt_bin_width_gev": COARSE_PT_BIN_WIDTH_GEV,
        "coarse_bins_span_q0": q0 / COARSE_PT_BIN_WIDTH_GEV,
        "coarse_pt_bins_resolve_q0": False,
        "note": (
            f"10 GeV leading-muon pT bins span only {q0 / COARSE_PT_BIN_WIDTH_GEV:.4f} "
            f"of one bin at q₀ = {q0} GeV — use this fine q_T histogram for recoil searches."
        ),
    }


def mod7_analysis_package(
    n_muon: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Mod-7 occupancy with trigger-aware and uniform null references."""
    from menus.particle.cern.mod7_phase import mod7_phase_residues, weighted_mod7_histogram

    arr = np.asarray(n_muon, dtype=np.int64).ravel()
    if arr.size == 0:
        return {"verdict": "UNDERPOWERED", "n_events": 0}

    if weights is not None and np.asarray(weights).size == arr.size:
        w = np.asarray(weights, dtype=float)
        hist = weighted_mod7_histogram(arr.astype(float), w)
    else:
        residues = mod7_phase_residues(arr)
        hist = np.bincount(residues, minlength=7).astype(float)

    total = float(hist.sum())
    fracs = (hist / total).tolist() if total > 0 else [0.0] * 7
    expected_trigger = trigger_aware_mod7_expected_fractions(arr)
    expected_uniform = np.full(7, 1.0 / 7.0)

    chi2_trigger = pearson_chi2_counts(hist, expected_trigger * total)
    chi2_uniform = pearson_chi2_counts(hist, expected_uniform * total)

    return {
        "n_events": int(arr.size),
        "mean_n_muon": float(np.mean(arr)),
        "mod7_fractions": fracs,
        "mod7_histogram": hist.tolist(),
        "trigger_aware_expected_fractions": expected_trigger.tolist(),
        "fraction_residue_2": float(fracs[2]) if len(fracs) > 2 else 0.0,
        "chi2_mod7_vs_trigger_aware": chi2_trigger["chi2"],
        "chi2_mod7_vs_trigger_aware_pvalue": chi2_trigger["p_value"],
        "chi2_mod7_vs_trigger_aware_z": z_from_p_one_sided(chi2_trigger["p_value"]),
        "chi2_mod7_vs_uniform": chi2_uniform["chi2"],
        "chi2_mod7_vs_uniform_pvalue": chi2_uniform["p_value"],
        "chi2_mod7_vs_uniform_z": z_from_p_one_sided(chi2_uniform["p_value"]),
        "null_hypothesis_warning": (
            "Uniform 1/7 null is incorrect for DoubleMu-triggered samples; "
            "residue-2 dominance follows from n_μ ≈ 2 under dimuon trigger."
        ),
        "recommended_null": "trigger_aware_from_empirical_P_n_muon",
        "exploratory_verdict": (
            "TRIGGER_BIAS_EXPECTED"
            if float(fracs[2]) > 0.35
            else "REVIEW_TRIGGER_MODEL"
        ),
    }


def shuffle_order_null_test(
    series: np.ndarray,
    periodogram_fn: Callable[[np.ndarray], dict[str, Any]],
    *,
    n_toys: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    """Shuffle event order; excess that vanishes indicates ordering artifact."""
    clean = np.asarray(series, dtype=float).ravel()
    if clean.size < 16:
        return {"available": False, "reason": "series too short"}

    rng = np.random.default_rng(seed)
    base = periodogram_fn(clean)
    base_excess = float(base.get("subharmonic_excess") or 0.0)
    toy_excesses: list[float] = []
    for _ in range(int(n_toys)):
        shuffled = clean.copy()
        rng.shuffle(shuffled)
        toy_excesses.append(
            float(periodogram_fn(shuffled).get("subharmonic_excess") or 0.0)
        )
    arr = np.asarray(toy_excesses, dtype=float)
    std = float(max(arr.std(), 0.05))
    ordering_sensitive = base_excess > float(arr.mean()) + 2.5 * std
    return {
        "available": True,
        "baseline_subharmonic_excess": base_excess,
        "shuffled_mean_excess": float(arr.mean()),
        "shuffled_std_excess": float(arr.std()),
        "n_toys": int(n_toys),
        "ordering_sensitive": bool(ordering_sensitive),
        "verdict": (
            "ORDERING_ARTIFACT_SUSPECTED"
            if ordering_sensitive
            else "STABLE_UNDER_SHUFFLE"
        ),
    }


def chunk_stride_sensitivity(
    series: np.ndarray,
    periodogram_fn: Callable[[np.ndarray], dict[str, Any]],
    *,
    chunk_sizes: tuple[int, ...] = (1024, 7513, 16384),
    strides: tuple[int, ...] = (1, 7, 13),
) -> dict[str, Any]:
    """Repeat periodogram on alternate chunk/stride partitions."""
    clean = np.asarray(series, dtype=float).ravel()
    if clean.size < 32:
        return {"available": False, "reason": "series too short"}

    base = periodogram_fn(clean)
    base_excess = float(base.get("subharmonic_excess") or 0.0)
    rows: list[dict[str, Any]] = []
    for chunk_size in chunk_sizes:
        for stride in strides:
            if stride >= chunk_size:
                continue
            segment = clean[:chunk_size:stride]
            if segment.size < 8:
                continue
            spec = periodogram_fn(segment)
            rows.append(
                {
                    "chunk_size": int(chunk_size),
                    "stride": int(stride),
                    "n_points": int(segment.size),
                    "subharmonic_excess": float(spec.get("subharmonic_excess") or 0.0),
                }
            )
    stable = (
        all(abs(r["subharmonic_excess"] - base_excess) < 1.5 for r in rows)
        if rows
        else None
    )
    return {
        "available": True,
        "baseline_subharmonic_excess": base_excess,
        "variants": rows,
        "partition_stable": stable,
        "verdict": (
            "PARTITION_STABLE"
            if stable
            else "PARTITION_SENSITIVE"
            if stable is False
            else "INSUFFICIENT_VARIANTS"
        ),
    }


def split_sample_stability(
    values: np.ndarray,
    *,
    metric_fn: Callable[[np.ndarray], float],
) -> dict[str, Any]:
    """First-half vs second-half stability check."""
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size < 20:
        return {"available": False, "reason": "too few events"}
    mid = arr.size // 2
    first = metric_fn(arr[:mid])
    second = metric_fn(arr[mid:])
    rel_diff = abs(first - second) / max(abs(first), abs(second), 1e-9)
    return {
        "available": True,
        "first_half_metric": float(first),
        "second_half_metric": float(second),
        "relative_difference": float(rel_diff),
        "stable": rel_diff < 0.25,
        "verdict": "STABLE" if rel_diff < 0.25 else "UNSTABLE_ACROSS_SPLIT",
    }


def exploratory_classification_label() -> str:
    return "EXPLORATORY_PIPELINE_ANOMALY — not a discovery claim"