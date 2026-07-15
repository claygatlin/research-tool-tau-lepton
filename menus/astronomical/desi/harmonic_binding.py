"""
DESI Harmonic Binding Level Detection Kit.

Maps hierarchical binding levels (n_hier, Δn) to redshift/comoving scales,
detects discrete transitions in BAO residuals, extends harmonic search, and
provides injection tools for recovery tests (Injection/Recovery menu only).

Related: desi_harmonic_binding_levels.md, sound_horizon_tsb_integration.md
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from menus.astronomical.desi.scanner import (
    LATE_UNIVERSE_DELTA_N,
    N_HIER_BINDING,
    TAU_RESONANCE_PERIOD,
    s_from_z,
)

# Five-scale binding hierarchy (sub-quantum → gravitic)
BINDING_LEVEL_NAMES: tuple[str, ...] = (
    "sub_quantum",
    "particle",
    "atomic_em",
    "cosmic_web",
    "gravitic",
)

# DESI DR2 tracer redshift windows for overlap tests
DESI_TRACER_Z_WINDOWS: dict[str, tuple[float, float]] = {
    "BGS_BRIGHT-21.35_GCcomb": (0.1, 0.4),
    "LRG_GCcomb_z0.4-0.6": (0.4, 0.6),
    "LRG_GCcomb_z0.6-0.8": (0.6, 0.8),
    "LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1": (0.8, 1.1),
    "ELG_LOPnotqso_GCcomb_z1.1-1.6": (1.1, 1.6),
    "QSO_GCcomb": (0.8, 2.1),
    "Lya_GCcomb": (2.0, 3.5),
    "ALL_GCcomb": (0.1, 3.5),
}

CYCLE_142857: float = 142857.0


def _n_fraction_to_redshift(
    n_frac: float,
    *,
    n_hier: float,
    delta_n: float,
    z_max: float,
) -> float:
    """Map conformal stretch fraction to approximate redshift (production calibration)."""
    if n_frac <= 0:
        return 0.0
    ratio = float(n_frac / n_hier)
    return float(z_max * ratio ** delta_n)


def map_binding_levels_to_redshift(
    *,
    n_hier: float = N_HIER_BINDING,
    delta_n: float = LATE_UNIVERSE_DELTA_N,
    gamma: float = 10.0,
    z_max: float = 3.5,
    desi_tracers: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Convert n_hier binding levels and Δn steps to redshift / s intervals.

    Returns level boundaries, Δn transition redshifts, comoving s-ranges, and
    DESI tracer overlap when ``desi_tracers`` is supplied.
    """
    n_levels = len(BINDING_LEVEL_NAMES)
    level_entries: list[dict[str, Any]] = []
    for idx, name in enumerate(BINDING_LEVEL_NAMES):
        n_lo = idx * (n_hier / n_levels)
        n_hi = (idx + 1) * (n_hier / n_levels)
        z_lo = _n_fraction_to_redshift(n_lo, n_hier=n_hier, delta_n=delta_n, z_max=z_max)
        z_hi = _n_fraction_to_redshift(n_hi, n_hier=n_hier, delta_n=delta_n, z_max=z_max)
        z_mid = 0.5 * (z_lo + z_hi)
        s_pair = s_from_z(np.array([z_lo, z_hi]), gamma=gamma)
        s_lo, s_hi = float(s_pair[0]), float(s_pair[1])
        level_entries.append(
            {
                "level_index": idx,
                "name": name,
                "n_hier_lo": float(n_lo),
                "n_hier_hi": float(n_hi),
                "z_lo": float(z_lo),
                "z_hi": float(z_hi),
                "z_mid": float(z_mid),
                "s_lo": float(min(s_lo, s_hi)),
                "s_hi": float(max(s_lo, s_hi)),
            }
        )

    # Δn step transitions (local temporal flow between binding wells)
    n_steps = max(1, int(round(n_hier / delta_n)))
    delta_n_transitions: list[dict[str, float]] = []
    for k in range(1, n_steps + 1):
        n_frac = k * delta_n
        if n_frac > n_hier:
            break
        z_t = _n_fraction_to_redshift(n_frac, n_hier=n_hier, delta_n=delta_n, z_max=z_max)
        delta_n_transitions.append({"step_k": k, "n_hier": float(n_frac), "z_transition": float(z_t)})

    # Legacy reference transitions (production.py calibration)
    reference_transition_zs = [
        float(0.3 * (n_hier / 45.8) ** delta_n),
        1.0,
        float(2.0 * (1.0 + delta_n)),
    ]

    tracer_overlap: dict[str, list[str]] = {}
    tracers = list(desi_tracers or DESI_TRACER_Z_WINDOWS.keys())
    for level in level_entries:
        hits: list[str] = []
        for tracer in tracers:
            window = DESI_TRACER_Z_WINDOWS.get(tracer)
            if window is None:
                continue
            t_lo, t_hi = window
            if t_hi >= level["z_lo"] and t_lo <= level["z_hi"]:
                hits.append(tracer)
        tracer_overlap[level["name"]] = hits

    return {
        "n_hier": float(n_hier),
        "delta_n": float(delta_n),
        "gamma": float(gamma),
        "z_max": float(z_max),
        "binding_levels": level_entries,
        "delta_n_transitions": delta_n_transitions,
        "reference_transition_zs": reference_transition_zs,
        "desi_tracer_overlap": tracer_overlap,
        "n_levels": n_levels,
    }


def detect_binding_transitions(
    z: np.ndarray,
    observable: np.ndarray,
    residuals: np.ndarray | None = None,
    *,
    n_hier: float = N_HIER_BINDING,
    delta_n: float = LATE_UNIVERSE_DELTA_N,
    gamma: float = 10.0,
    penalty: float = 3.0,
    min_size: int = 2,
    algorithm: str = "pelt",
    prediction_tolerance: float = 0.15,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Targeted step/changepoint detection aligned to predicted binding transitions.

    Runs ruptures PELT (or BinSeg) on sorted residuals, then scores alignment
    with :func:`map_binding_levels_to_redshift` predictions.
    """
    z_arr = np.asarray(z, dtype=float)
    obs_arr = np.asarray(observable, dtype=float)
    order = np.argsort(z_arr)
    z_sorted = z_arr[order]
    obs_sorted = obs_arr[order]

    if residuals is None:
        deg = min(2, max(1, len(z_sorted) - 1))
        baseline = np.polyval(np.polyfit(z_sorted, obs_sorted, deg=deg), z_sorted)
        resid = obs_sorted - baseline
    else:
        resid = np.asarray(residuals, dtype=float)[order]

    mapping = map_binding_levels_to_redshift(n_hier=n_hier, delta_n=delta_n, gamma=gamma)
    predicted_zs = list(mapping["reference_transition_zs"])
    predicted_zs.extend(t["z_transition"] for t in mapping["delta_n_transitions"][:6])
    predicted_zs = sorted(set(round(z, 4) for z in predicted_zs if z > 0))

    n_pts = len(z_sorted)
    if n_pts < 2:
        return {
            "skipped": True,
            "skip_reason": f"insufficient data (n={n_pts})",
            "mapping": mapping,
            "predicted_transition_zs": predicted_zs,
            "detected_transition_zs": [],
            "alignment": {},
        }

    detected_zs: list[float] = []
    break_indices: list[int] = []
    algo_used = algorithm

    try:
        import ruptures as rpt
    except ImportError:
        # Fallback: local step test at predicted z only
        algo_used = "predicted_window_test"
        for z_pred in predicted_zs:
            mask = np.abs(z_sorted - z_pred) <= prediction_tolerance
            if np.any(mask):
                detected_zs.append(float(z_pred))
        result_alignment = _score_transition_alignment(
            detected_zs, predicted_zs, tolerance=prediction_tolerance
        )
        if verbose:
            _print_binding_transition_report(mapping, detected_zs, predicted_zs, result_alignment)
        return {
            "algorithm": algo_used,
            "skipped": False,
            "mapping": mapping,
            "predicted_transition_zs": predicted_zs,
            "detected_transition_zs": detected_zs,
            "break_indices": [],
            "alignment": result_alignment,
            "penalty": penalty,
        }

    y = resid.reshape(-1, 1)
    if algorithm == "binseg":
        algo = rpt.Binseg(model="l2", min_size=min_size).fit(y)
        bkps = algo.predict(n_bkps=min(3, max(1, n_pts // 3)))
    else:
        algo = rpt.Pelt(model="l2", min_size=min_size).fit(y)
        bkps = algo.predict(pen=penalty)

    break_indices = [int(b) for b in bkps if b < n_pts]
    detected_zs = [float(z_sorted[idx - 1]) for idx in break_indices if idx > 0]

    result_alignment = _score_transition_alignment(
        detected_zs, predicted_zs, tolerance=prediction_tolerance
    )

    if verbose:
        _print_binding_transition_report(mapping, detected_zs, predicted_zs, result_alignment)

    return {
        "algorithm": algo_used,
        "penalty": float(penalty),
        "skipped": False,
        "mapping": mapping,
        "predicted_transition_zs": predicted_zs,
        "detected_transition_zs": detected_zs,
        "break_indices": break_indices,
        "n_breaks": len(detected_zs),
        "alignment": result_alignment,
        "residuals": resid.tolist(),
        "z_sorted": z_sorted.tolist(),
    }


def _score_transition_alignment(
    detected: Sequence[float],
    predicted: Sequence[float],
    *,
    tolerance: float,
) -> dict[str, Any]:
    matches: list[dict[str, float]] = []
    for z_det in detected:
        best_pred = None
        best_dist = float("inf")
        for z_pred in predicted:
            dist = abs(z_det - z_pred)
            if dist < best_dist:
                best_dist = dist
                best_pred = z_pred
        matches.append(
            {
                "detected_z": float(z_det),
                "nearest_predicted_z": float(best_pred) if best_pred is not None else float("nan"),
                "delta_z": float(best_dist),
                "aligned": bool(best_dist <= tolerance),
            }
        )
    n_aligned = sum(1 for m in matches if m["aligned"])
    return {
        "tolerance": float(tolerance),
        "n_detected": len(detected),
        "n_predicted": len(predicted),
        "n_aligned": n_aligned,
        "alignment_fraction": float(n_aligned / max(1, len(detected))),
        "matches": matches,
    }


def _print_binding_transition_report(
    mapping: dict[str, Any],
    detected: Sequence[float],
    predicted: Sequence[float],
    alignment: dict[str, Any],
) -> None:
    print("\n[Harmonic Binding — level mapping]")
    for level in mapping["binding_levels"]:
        tracers = mapping["desi_tracer_overlap"].get(level["name"], [])
        tracer_note = f" tracers={tracers[:3]}" if tracers else ""
        print(
            f"  {level['name']:12s}  z=[{level['z_lo']:.2f}, {level['z_hi']:.2f}]  "
            f"n_hier=[{level['n_hier_lo']:.1f}, {level['n_hier_hi']:.1f}]{tracer_note}"
        )
    print(f"  Δn transitions (first 4): {[round(t['z_transition'], 2) for t in mapping['delta_n_transitions'][:4]]}")
    print(f"  Reference z: {mapping['reference_transition_zs']}")
    print("\n[Harmonic Binding — transition detection]")
    print(f"  Detected z:   {[round(z, 3) for z in detected]}")
    print(f"  Predicted z:  {[round(z, 3) for z in predicted[:8]]}")
    print(
        f"  Aligned: {alignment.get('n_aligned', 0)}/{alignment.get('n_detected', 0)} "
        f"(tol={alignment.get('tolerance', 0.15):.2f})"
    )


def search_harmonic_features(
    residuals: np.ndarray,
    s: np.ndarray,
    z: np.ndarray | None = None,
    *,
    fundamental_freq: float = 1.0 / TAU_RESONANCE_PERIOD,
    max_harmonic: int = 5,
    n_bootstrap: int = 2000,
    k_bins: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Extended harmonic search: 1/7 ladder + z-binned residual spectra + 142857-scale probe.

    Wraps :func:`search_higher_harmonics` and adds full-shape proxies on binned data.
    """
    from menus.astronomical.desi.stats import search_higher_harmonics

    s_arr = np.asarray(s, dtype=float)
    resid = np.asarray(residuals, dtype=float)
    ladder = search_higher_harmonics(
        resid,
        s_arr,
        fundamental_freq=fundamental_freq,
        max_harmonic=max_harmonic,
        n_bootstrap=n_bootstrap,
        verbose=verbose,
    )

    z_binned: dict[str, Any] = {}
    if z is not None and len(z) == len(resid):
        z_arr = np.asarray(z, dtype=float)
        n_bins = k_bins or min(5, max(2, len(resid) // 2))
        edges = np.linspace(float(np.min(z_arr)), float(np.max(z_arr)), n_bins + 1)
        bin_features: list[dict[str, Any]] = []
        for b in range(n_bins):
            mask = (z_arr >= edges[b]) & (z_arr < edges[b + 1] if b < n_bins - 1 else z_arr <= edges[b + 1])
            if np.sum(mask) < 2:
                continue
            sub = search_higher_harmonics(
                resid[mask],
                s_arr[mask],
                fundamental_freq=fundamental_freq,
                max_harmonic=min(3, max_harmonic),
                n_bootstrap=max(500, n_bootstrap // 4),
                verbose=False,
            )
            bin_features.append(
                {
                    "z_lo": float(edges[b]),
                    "z_hi": float(edges[b + 1]),
                    "n_points": int(np.sum(mask)),
                    "h1_pvalue": float(
                        sub.get("harmonics", {}).get(1, {}).get("bootstrap_p_value", float("nan"))
                    ),
                    "any_significant": bool(sub.get("any_significant_at_5pct", False)),
                }
            )
        z_binned = {"n_bins": n_bins, "bins": bin_features}

    # 142857-cycle sub-harmonic probe (scaled to s-span)
    cycle_probe: dict[str, Any] = {}
    if len(s_arr) >= 3:
        s_span = float(np.max(s_arr) - np.min(s_arr))
        if s_span > 0:
            f_142857 = fundamental_freq / CYCLE_142857
            from menus.astronomical.desi.stats import _lomb_scargle_power_at_frequency

            power, backend = _lomb_scargle_power_at_frequency(s_arr, resid, f_142857)
            cycle_probe = {
                "frequency": f_142857,
                "observed_power": float(power),
                "backend": backend,
                "note": "142857-cycle sub-harmonic scaled to s-grid span",
            }
            if verbose:
                print(f"[142857-cycle probe] f={f_142857:.2e}, power={power:.2e} ({backend})")

    return {
        "ladder": ladder,
        "z_binned_harmonics": z_binned,
        "cycle_142857_probe": cycle_probe,
        "fundamental_freq": float(fundamental_freq),
    }


def inject_binding_modulation(
    z: np.ndarray,
    observable: np.ndarray | None = None,
    *,
    baseline: float | None = None,
    amplitude_frac: float = 0.02,
    binding_steps: bool = True,
    harmonic_period: float = TAU_RESONANCE_PERIOD,
    gamma: float = 10.0,
    n_hier: float = N_HIER_BINDING,
    delta_n: float = LATE_UNIVERSE_DELTA_N,
    noise_sigma: float = 0.008,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Inject 1/7 harmonic + binding-level steps into observables (recovery tests only).

    For use under Injection/Recovery Tests — not a substitute for live DESI data.
    """
    from menus.astronomical.desi.scanner import (
        _oscillation_scale,
        tau_sb_hierarchical_step,
        tau_sb_oscillatory_residual,
    )

    rng = np.random.default_rng(seed)
    z_arr = np.asarray(z, dtype=float)
    base = float(baseline if baseline is not None else (np.median(observable) if observable is not None else 1.0))
    s_arr = s_from_z(z_arr, gamma=gamma)
    amp_scale = _oscillation_scale(np.full(len(z_arr), base))

    harmonic = tau_sb_oscillatory_residual(
        s_arr, A=amplitude_frac, period=harmonic_period, amplitude_scale=amp_scale
    )
    steps = np.zeros_like(z_arr, dtype=float)
    if binding_steps:
        mapping = map_binding_levels_to_redshift(n_hier=n_hier, delta_n=delta_n, gamma=gamma)
        transition_zs = mapping["reference_transition_zs"]
        steps = tau_sb_hierarchical_step(z_arr, transition_zs=transition_zs, amplitude=0.005 * amp_scale)

    true_model = base + harmonic + steps
    noise = rng.normal(0.0, noise_sigma, size=len(z_arr))
    injected = true_model + noise
    err = np.full(len(z_arr), noise_sigma)
    return {
        "z": z_arr,
        "observable": injected,
        "true_model": true_model,
        "harmonic_component": harmonic,
        "binding_steps": steps,
        "err": err,
        "cov": np.diag(err**2),
        "s": s_arr,
        "amplitude_frac": float(amplitude_frac),
        "binding_mapping": mapping if binding_steps else None,
        "label": "binding_modulation_injected",
        "note": "Synthetic injection for recovery tests only",
    }


def run_harmonic_binding_kit(
    z: np.ndarray,
    observable: np.ndarray,
    residuals: np.ndarray,
    s: np.ndarray,
    *,
    n_hier: float = N_HIER_BINDING,
    delta_n: float = LATE_UNIVERSE_DELTA_N,
    gamma: float = 10.0,
    desi_tracers: Sequence[str] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    One-call harmonic binding detection kit for DESI workflow integration.

    Runs level mapping, transition detection, and extended harmonic features.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("DESI HARMONIC BINDING LEVEL DETECTION KIT")
        print("=" * 60)

    mapping = map_binding_levels_to_redshift(
        n_hier=n_hier,
        delta_n=delta_n,
        gamma=gamma,
        desi_tracers=desi_tracers,
    )
    transitions = detect_binding_transitions(
        z,
        observable,
        residuals,
        n_hier=n_hier,
        delta_n=delta_n,
        gamma=gamma,
        verbose=verbose,
    )
    harmonics = search_harmonic_features(
        residuals,
        s,
        z=z,
        verbose=verbose,
    )

    kit = {
        "kit_version": "2026-06-30",
        "binding_level_mapping": mapping,
        "binding_transitions": transitions,
        "harmonic_features": harmonics,
        "summary": {
            "n_binding_levels": mapping["n_levels"],
            "n_detected_transitions": transitions.get("n_breaks", len(transitions.get("detected_transition_zs", []))),
            "harmonic_ladder_significant": bool(
                harmonics.get("ladder", {}).get("any_significant_at_5pct", False)
            ),
            "alignment_fraction": float(transitions.get("alignment", {}).get("alignment_fraction", 0.0)),
        },
    }

    if verbose:
        print("\n[Kit summary]")
        sm = kit["summary"]
        print(f"  Binding levels mapped : {sm['n_binding_levels']}")
        print(f"  Transitions detected  : {sm['n_detected_transitions']}")
        print(f"  Prediction alignment  : {sm['alignment_fraction']:.2f}")
        print(f"  1/7 ladder significant: {sm['harmonic_ladder_significant']}")

    return kit