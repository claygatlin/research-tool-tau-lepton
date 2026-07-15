"""
Option A — relaxed recoil isolation window (recommended Phase 4 variant).

Uses a non-contradictory validation gate (ΔR ≤ 1.5, recoil pT ceiling) without
simultaneous Δφ_min cuts that collapse back-to-back dimuon event loops.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tav_superblock_cms_muon_analyzer import M0_MEV_ANCHOR

from menus.particle.cms.tav_enhanced_validation_v2 import (
    DEFAULT_PT_BIN_COUNT,
    DEFAULT_PT_MAX_GEV,
    MIN_EVENTS_GEOMETRIC_FLOOR,
    RECOIL_VALIDATION_DELTA_R_MAX,
    _geometric_floor_proxy,
    _isolation_quality_label,
    _passes_recoil_angular_cuts,
    _sevenfold_metrics_from_pt_histogram,
    extract_dimuon_kinematics,
    geometric_floor_scan_failure,
    normalize_dimuon_events,
    option2_pt_thresholds,
    recoil_validation_gate_cuts,
    scan_pt_thresholds_with_recoil_cuts,
    vector_winding_rate_from_mod_excess,
)
from tav_shared.artifact_paths import (
    TestSlug,
    artifact_path,
    artifact_timestamp,
    compose_dataset_slug,
)

OPTION_A_LABEL = "Option A — Relaxed Recoil Isolation Window"


def option_a_recoil_pt_sweep() -> list[float]:
    """Stepped recoil system-pT ceiling (GeV): 15 → 30 → 45."""
    return [15.0, 30.0, 45.0]


def option_a_delta_phi_sweep() -> list[float]:
    """Diagnostic Δφ reference envelope (not applied as a joint cut)."""
    return [float(x) for x in np.linspace(1.8, 2.5, 4)]


def option_a_default_delta_r_max() -> float:
    return RECOIL_VALIDATION_DELTA_R_MAX


def scan_recoil_isolation_window(
    data_events: Any,
    *,
    recoil_pt_max_values: Sequence[float] | None = None,
    delta_phi_min_values: Sequence[float] | None = None,
    delta_r_max: float | None = None,
    pt_bins: int = DEFAULT_PT_BIN_COUNT,
    pt_max_gev: float = DEFAULT_PT_MAX_GEV,
    min_events: int = MIN_EVENTS_GEOMETRIC_FLOOR,
    track_winding_rate: bool = True,
    winding_rate_params: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Grid scan over recoil_pt_max with a simple ΔR validation gate.

    ``delta_phi_min_values`` is retained for reporting only — it is not combined
    with ``delta_r_max`` in the selection mask (that joint filter collapses to
    zero events on back-to-back dimuon topologies).
    """
    _ = delta_phi_min_values  # legacy API; diagnostic envelope only
    events = normalize_dimuon_events(data_events)
    n_input = int(events["leading_pt"].size)
    if n_input < min_events:
        return {
            "option": OPTION_A_LABEL,
            "verdict": "UNDERPOWERED",
            "reason": f"fewer than {min_events} dimuon events",
            "n_dimuon_events_input": n_input,
        }

    recoil_values = list(recoil_pt_max_values or option_a_recoil_pt_sweep())
    dr_max = float(delta_r_max if delta_r_max is not None else option_a_default_delta_r_max())
    dphi_reference = option_a_delta_phi_sweep()
    bin_edges = np.linspace(0.0, float(pt_max_gev), int(pt_bins) + 1)

    unisolated_hist, _ = np.histogram(events["leading_pt"], bins=bin_edges)
    unisolated_metrics = _sevenfold_metrics_from_pt_histogram(
        unisolated_hist, bin_edges
    )
    vector_winding = vector_winding_rate_from_mod_excess(
        subharmonic_excess=unisolated_metrics["subharmonic_excess"],
        winding_rate_params=winding_rate_params,
    )

    rows: list[dict[str, Any]] = []
    for recoil_max in recoil_values:
        cuts = recoil_validation_gate_cuts(
            delta_r_max=dr_max,
            recoil_pt_max=float(recoil_max),
        )
        mask = _passes_recoil_angular_cuts(events, cuts, require_delta_phi=False)
        surviving = int(mask.sum())
        pass_fraction = float(surviving / max(n_input, 1))
        dphi_median = (
            float(np.median(events["delta_phi"][mask])) if surviving else None
        )
        row: dict[str, Any] = {
            "recoil_pt_max_gev": float(recoil_max),
            "delta_phi_min": None,
            "delta_phi_median_survivors": dphi_median,
            "delta_r_max": dr_max,
            "events_after_cuts": surviving,
            "event_pass_fraction": pass_fraction,
            "sevenfold_amplitude": 0.0,
            "subharmonic_excess": 0.0,
            "significance_proxy": 0.0,
            "winding_rate_proxy": (
                vector_winding.get("winding_rate_proxy") if track_winding_rate else None
            ),
            "geometric_floor_proxy_mev": float(M0_MEV_ANCHOR),
            "isolation_quality": "underpowered",
        }
        if surviving >= min_events:
            leading_pts = events["leading_pt"][mask]
            hist, _ = np.histogram(leading_pts, bins=bin_edges)
            metrics = _sevenfold_metrics_from_pt_histogram(hist, bin_edges)
            row.update(
                {
                    "sevenfold_amplitude": metrics["sevenfold_amplitude"],
                    "subharmonic_excess": metrics["subharmonic_excess"],
                    "significance_proxy": metrics["significance_proxy"],
                    "geometric_floor_proxy_mev": _geometric_floor_proxy(
                        subharmonic_excess=metrics["subharmonic_excess"],
                        significance_proxy=metrics["significance_proxy"],
                    ),
                    "isolation_quality": _isolation_quality_label(
                        subharmonic_excess=metrics["subharmonic_excess"],
                        significance_proxy=metrics["significance_proxy"],
                        events_after_cuts=surviving,
                    ),
                }
            )
        rows.append(row)

    powered = [r for r in rows if r["isolation_quality"] != "underpowered"]
    best_window: dict[str, Any] | None = None
    if powered:
        best_window = max(
            powered,
            key=lambda r: (
                float(r.get("subharmonic_excess") or 0.0),
                float(r.get("significance_proxy") or 0.0),
                float(r.get("event_pass_fraction") or 0.0),
                int(r.get("events_after_cuts") or 0),
            ),
        )
    elif rows:
        best_window = max(rows, key=lambda r: int(r.get("events_after_cuts") or 0))

    return {
        "option": OPTION_A_LABEL,
        "action": "Relaxed recoil isolation window sweep (313.1 MeV floor)",
        "n_dimuon_events_input": n_input,
        "recoil_pt_max_sweep_gev": recoil_values,
        "delta_phi_min_sweep": dphi_reference,
        "delta_phi_cut_applied": False,
        "delta_r_max": dr_max,
        "validation_gate": f"ΔR≤{dr_max} + recoil pT ceiling (no Δφ_min joint cut)",
        "vector_winding_baseline": vector_winding,
        "unisolated_subharmonic_excess": unisolated_metrics["subharmonic_excess"],
        "isolation_window_scan": rows,
        "best_isolation_window": best_window,
        "interpretation": (
            "Stepped recoil (15/30/45 GeV) with a simple ΔR≤1.5 validation gate "
            "keeps a finite event fraction through the loop. Winding-rate proxy is "
            "anchored to pileup-invariant mod-2/mod-7 excess from un-isolated spectra."
        ),
    }


def run_option_a_relaxed_recoil_isolation(
    data_file: str | Path,
    *,
    recoil_pt_max_values: Sequence[float] | None = None,
    delta_phi_min_values: Sequence[float] | None = None,
    delta_r_max: float | None = None,
    run_pt_threshold_scan: bool = True,
    pt_thresholds: np.ndarray | None = None,
    winding_rate_params: dict[str, float] | None = None,
    track_winding_rate: bool = True,
    chunk_size: int = 500_000,
    entry_stop: int | None = None,
    output_dir: str | Path = "option_a_recoil_isolation",
    run_when: datetime | None = None,
    dataset_slug: str = "option_a_recoil",
    save_report: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Option A entry: isolation-window grid + optional pT scan at best window."""
    dimuon_kin = extract_dimuon_kinematics(
        data_file,
        chunk_size=chunk_size,
        entry_stop=entry_stop,
        verbose=verbose,
    )
    if dimuon_kin is None:
        return geometric_floor_scan_failure(
            "Muon_eta and/or Muon_phi unavailable",
            missing_branches=["Muon_eta", "Muon_phi"],
        )

    window_report = scan_recoil_isolation_window(
        dimuon_kin,
        recoil_pt_max_values=recoil_pt_max_values,
        delta_phi_min_values=delta_phi_min_values,
        delta_r_max=delta_r_max,
        track_winding_rate=track_winding_rate,
        winding_rate_params=winding_rate_params,
    )
    if window_report.get("verdict") == "UNDERPOWERED":
        window_report["status"] = "OPTION_A_UNDERPOWERED"
        return window_report

    best = window_report.get("best_isolation_window")
    pt_scan: dict[str, Any] | None = None
    if run_pt_threshold_scan and best:
        angular_cuts = recoil_validation_gate_cuts(
            delta_r_max=float(best.get("delta_r_max") or option_a_default_delta_r_max()),
            recoil_pt_max=float(best["recoil_pt_max_gev"]),
        )
        thresholds = (
            np.asarray(pt_thresholds, dtype=float)
            if pt_thresholds is not None
            else option2_pt_thresholds()
        )
        pt_scan = scan_pt_thresholds_with_recoil_cuts(
            dimuon_kin,
            thresholds,
            angular_cuts,
            require_delta_phi=False,
            winding_rate_params=winding_rate_params,
            track_winding_rate=track_winding_rate,
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_when = run_when or datetime.now(timezone.utc)

    report: dict[str, Any] = {
        **window_report,
        "source_file": str(dimuon_kin.get("path") or data_file),
        "n_events_scanned": int(dimuon_kin.get("n_events_scanned") or 0),
        "entry_stop": int(dimuon_kin.get("entry_stop") or 0),
        "pt_threshold_scan": pt_scan,
        "best_isolation": (pt_scan or {}).get("best_isolation") or best,
        "angular_cuts_applied": (
            recoil_validation_gate_cuts(
                delta_r_max=float(best.get("delta_r_max") or option_a_default_delta_r_max()),
                recoil_pt_max=float(best["recoil_pt_max_gev"]),
            )
            if best
            else None
        ),
        "output_dir": str(out_dir),
        "timestamp": artifact_timestamp(run_when),
    }

    if save_report:
        path = artifact_path(
            TestSlug.CERN,
            compose_dataset_slug(dataset_slug, "option_a_recoil"),
            "report",
            "json",
            when=run_when,
            run_dir=out_dir,
        )
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=float)
        report["report_path"] = str(path)

    if verbose:
        print_option_a_recoil_summary(report)
    return report


def print_option_a_recoil_summary(report: dict[str, Any]) -> None:
    """Stdout banner for Option A relaxed recoil isolation sweep."""
    print("=== TAV OPTION A: RELAXED RECOIL ISOLATION ===")
    if report.get("status") in {
        "GEOMETRIC_FLOOR_SCAN_FAILED",
        "OPTION_A_UNDERPOWERED",
    } or report.get("verdict") == "UNDERPOWERED":
        print(f"  Status            : {report.get('status') or report.get('verdict')}")
        print(f"  Reason            : {report.get('reason', 'n/a')}")
        print("=" * 46)
        return

    print(f"  Dimuon events     : {report.get('n_dimuon_events_input', 0):,}")
    print(f"  Validation gate   : {report.get('validation_gate', 'n/a')}")
    print(
        f"  Recoil sweep      : {report.get('recoil_pt_max_sweep_gev', [])} GeV"
    )
    baseline = report.get("vector_winding_baseline") or {}
    if baseline.get("winding_rate_proxy") is not None:
        print(
            f"  Winding (un-iso.) : {baseline.get('winding_rate_proxy', 0):.2e} "
            f"(mod2 excess {baseline.get('mod2_excess', 0):+.4f})"
        )

    best_win = report.get("best_isolation_window") or {}
    if best_win:
        print(
            f"  Best window       : recoil≤{best_win.get('recoil_pt_max_gev', 0):.0f} GeV, "
            f"ΔR≤{best_win.get('delta_r_max', 0):.1f} "
            f"({best_win.get('events_after_cuts', 0):,} events, "
            f"{100 * best_win.get('event_pass_fraction', 0):.1f}% pass)"
        )
        print(
            f"  Subharmonic exces : {best_win.get('subharmonic_excess', 0):.2f}×"
        )

    best = report.get("best_isolation") or {}
    if best.get("pt_threshold_gev") is not None:
        print(
            f"  Best pT cut       : {best.get('pt_threshold_gev', 0):.1f} GeV "
            f"({best.get('events_after_cuts', 0):,} events)"
        )
        print(
            f"  7-fold amplitude  : {best.get('sevenfold_amplitude', 0):.3f}"
        )
        if best.get("geometric_floor_proxy_mev") is not None:
            print(
                f"  Floor proxy       : {best.get('geometric_floor_proxy_mev', 0):.1f} MeV"
            )
        print(f"  Isolation quality : {best.get('isolation_quality', 'n/a')}")
    if report.get("report_path"):
        print(f"  Report            : {report['report_path']}")
    print("=" * 46)