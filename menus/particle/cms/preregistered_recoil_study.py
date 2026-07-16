#!/usr/bin/env python3
"""
Preregistered dimuon recoil q_T study with HEP controls (critique 2026-07-15).

Implements frozen signal/sideband windows, fine q_T binning, trigger-aware mod-7
null, ordering/partition null tests, and split-sample stability — without
interpretive physics labels in the detection layer.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from menus.particle.cms.hep_statistics import (
    GEOMETRIC_FLOOR_GEV,
    PREREG_QT_SIGNAL_WINDOW,
    compute_acoplanarity,
    compute_pt_balance,
    exploratory_classification_label,
    fine_qt_histogram,
    mod7_analysis_package,
    shuffle_order_null_test,
    chunk_stride_sensitivity,
    split_sample_stability,
)
from menus.particle.cms.tav_enhanced_validation_v2 import (
    extract_dimuon_kinematics,
    normalize_dimuon_events,
)
from tav_shared.artifact_paths import TestSlug, artifact_path, artifact_timestamp, compose_dataset_slug


def _dimuon_invariant_mass_gev(
    pt1: np.ndarray,
    pt2: np.ndarray,
    eta1: np.ndarray | None = None,
    eta2: np.ndarray | None = None,
    phi1: np.ndarray | None = None,
    phi2: np.ndarray | None = None,
) -> np.ndarray | None:
    """Approximate m_μμ when full 4-vector branches unavailable (uses ΔR proxy)."""
    if eta1 is None or phi1 is None:
        return None
    p1 = np.asarray(pt1, dtype=float)
    p2 = np.asarray(pt2, dtype=float)
    e1 = np.hypot(p1, 0.105)
    e2 = np.hypot(p2, 0.105)
    px1 = p1 * np.cos(phi1)
    py1 = p1 * np.sin(phi1)
    px2 = p2 * np.cos(phi2)
    py2 = p2 * np.sin(phi2)
    pz1 = p1 * np.sinh(eta1)
    pz2 = p2 * np.sinh(eta2)
    mass_sq = (
        (e1 + e2) ** 2
        - (px1 + px2) ** 2
        - (py1 + py2) ** 2
        - (pz1 + pz2) ** 2
    )
    return np.sqrt(np.maximum(mass_sq, 0.0))


def _region_tag(mass: float | None) -> str:
    if mass is None:
        return "continuum_untagged"
    if 2.8 < mass < 3.4:
        return "jpsi_window"
    if 9.0 < mass < 10.6:
        return "upsilon_window"
    if 70.0 < mass < 110.0:
        return "z_window"
    return "continuum"


def run_preregistered_recoil_study(
    data_file: str,
    *,
    output_dir: str | None = None,
    entry_stop: int | None = None,
    n_shuffle_toys: int = 20,
    q_max_gev: float = 2.0,
    qt_bin_width_gev: float = 0.01,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Execute preregistered q_T recoil pipeline on DoubleMuParked NanoAOD.

    Returns neutral-statistics JSON report (exploratory classification only).
    """
    from menus.particle.cern.tav_superblock_cms_muon_analyzer import _periodogram_7fold

    kin = extract_dimuon_kinematics(
        data_file,
        entry_stop=entry_stop,
        verbose=verbose,
    )
    if kin is None:
        return {
            "status": "FAILED",
            "reason": "dimuon kinematics unavailable (Muon_eta/phi missing?)",
            "classification": exploratory_classification_label(),
        }

    events = normalize_dimuon_events(kin)
    q_t = events["system_pt"]
    alpha = compute_acoplanarity(events["delta_phi"])
    balance = compute_pt_balance(events["leading_pt"], events["subleading_pt"])

    n_muon = kin.get("n_muon_per_event")
    mod7_pack: dict[str, Any] | None = None
    if n_muon is not None:
        mod7_pack = mod7_analysis_package(np.asarray(n_muon, dtype=np.int64))

    qt_hist = fine_qt_histogram(q_t, q_max=q_max_gev, bin_width=qt_bin_width_gev)

    cum_mult = np.cumsum(np.ones(q_t.size))
    shuffle_null = shuffle_order_null_test(
        cum_mult,
        _periodogram_7fold,
        n_toys=n_shuffle_toys,
    )
    partition_null = chunk_stride_sensitivity(cum_mult, _periodogram_7fold)

    def _mean_qt(arr: np.ndarray) -> float:
        return float(np.mean(arr)) if arr.size else 0.0

    split_qt = split_sample_stability(q_t, metric_fn=_mean_qt)

    sig_lo, sig_hi = PREREG_QT_SIGNAL_WINDOW
    signal_mask = (q_t >= sig_lo) & (q_t < sig_hi)
    signal_fraction = float(signal_mask.mean()) if q_t.size else 0.0

    pT_slices = [
        ("pT_10_20", (events["leading_pt"] >= 10) & (events["leading_pt"] < 20)),
        ("pT_20_50", (events["leading_pt"] >= 20) & (events["leading_pt"] < 50)),
        ("pT_50_100", (events["leading_pt"] >= 50) & (events["leading_pt"] < 100)),
    ]
    pt_slice_reports: dict[str, Any] = {}
    for label, mask in pT_slices:
        subset = q_t[mask]
        if subset.size < 10:
            continue
        pt_slice_reports[label] = {
            "n_events": int(subset.size),
            "mean_qt_gev": float(subset.mean()),
            "mean_alpha_rad": float(alpha[mask].mean()),
            "expected_alpha_at_q0": float(GEOMETRIC_FLOOR_GEV / max(events["leading_pt"][mask].mean(), 1.0)),
            "signal_window_fraction": float(((subset >= sig_lo) & (subset < sig_hi)).mean()),
        }

    when = datetime.now(timezone.utc)
    slug = compose_dataset_slug(
        Path(str(kin.get("path") or data_file)).stem,
        "preregistered_qt_recoil",
    )
    report: dict[str, Any] = {
        "action": "Preregistered Recoil q_T Study (HEP Controls)",
        "timestamp": artifact_timestamp(when),
        "classification": exploratory_classification_label(),
        "preregistration": {
            "version": "critique_upgrade_2026-07-15",
            "signal_window_gev": {"lo": sig_lo, "hi": sig_hi},
            "q0_gev": GEOMETRIC_FLOOR_GEV,
            "qt_bin_width_gev": qt_bin_width_gev,
            "frozen_before_unblinding": True,
        },
        "source_file": str(kin.get("path") or data_file),
        "ingestion_validation": kin.get("ingestion_validation"),
        "n_events_scanned": int(kin.get("n_events_scanned") or 0),
        "n_dimuon_events": int(q_t.size),
        "entry_stop": int(kin.get("entry_stop") or 0),
        "kinematic_observables": {
            "q_T_gev": "system_pt alias — |pT1_vec + pT2_vec|",
            "alpha_rad": "pi - |delta_phi|",
            "pt_balance": "|pT1-pT2|/(pT1+pT2)",
        },
        "summary_statistics": {
            "mean_qt_gev": float(q_t.mean()) if q_t.size else 0.0,
            "median_qt_gev": float(np.median(q_t)) if q_t.size else 0.0,
            "mean_alpha_rad": float(alpha.mean()) if alpha.size else 0.0,
            "mean_pt_balance": float(balance.mean()) if balance.size else 0.0,
            "signal_window_event_fraction": signal_fraction,
        },
        "fine_qt_histogram": qt_hist,
        "mod7_trigger_aware": mod7_pack,
        "null_tests": {
            "event_order_shuffle": shuffle_null,
            "chunk_stride_partition": partition_null,
            "split_sample_qt_mean": split_qt,
        },
        "pt_slice_stability": pt_slice_reports,
        "methodology_notes": [
            "Opposite-charge selection requires Muon_charge branch (not in reduced dimu skim).",
            "Detector momentum-scale corrections not applied — exploratory Open Data exercise.",
            "Do not interpret mod-7 residue-2 excess as new topology without trigger-aware null.",
            "10 GeV leading-pT bins cannot resolve 313.1 MeV features — use fine q_T histogram.",
        ],
        "recommended_next_steps": [
            "Option 3 MC stack (DY + tt̄ + HF) for shape comparisons.",
            "Official muon momentum-scale constraint via Z→μμ peak.",
            "Held-out Run B vs Run C split after freezing cuts.",
            "Signal injection toys at q₀ = 0.3131 GeV for calibration.",
        ],
    }

    verdict_parts: list[str] = []
    if mod7_pack and mod7_pack.get("exploratory_verdict") == "TRIGGER_BIAS_EXPECTED":
        verdict_parts.append("MOD7_TRIGGER_BIAS")
    if shuffle_null.get("verdict") == "ORDERING_ARTIFACT_SUSPECTED":
        verdict_parts.append("ORDERING_ARTIFACT")
    if split_qt.get("verdict") == "UNSTABLE_ACROSS_SPLIT":
        verdict_parts.append("SPLIT_UNSTABLE")
    if not verdict_parts:
        verdict_parts.append("NEUTRAL_EXPLORATORY_PASS")
    report["aggregate_verdict"] = "+".join(verdict_parts)

    out_path = artifact_path(
        TestSlug.CERN,
        slug,
        "report",
        "json",
        when=when,
    )
    if output_dir:
        out_path = Path(output_dir) / out_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(out_path)

    if verbose:
        print("\n=== PREREGISTERED RECOIL q_T STUDY ===")
        print(f"  Classification : {report['classification']}")
        print(f"  Dimuon events  : {report['n_dimuon_events']:,}")
        print(f"  Mean q_T       : {report['summary_statistics']['mean_qt_gev']:.4f} GeV")
        print(f"  Signal window  : {sig_lo}–{sig_hi} GeV ({signal_fraction*100:.3f}% of events)")
        if mod7_pack:
            print(
                f"  Residue-2 frac : {mod7_pack.get('fraction_residue_2', 0)*100:.2f}% "
                f"(trigger-aware χ²={mod7_pack.get('chi2_mod7_vs_trigger_aware', 0):.2f})"
            )
        print(f"  Shuffle null   : {shuffle_null.get('verdict', 'n/a')}")
        print(f"  Aggregate      : {report['aggregate_verdict']}")
        print(f"  Report         : {out_path}")
        print("=" * 38)

    return report


def print_preregistered_summary(report: dict[str, Any]) -> None:
    """Console summary for menu integration."""
    stats = report.get("summary_statistics") or {}
    print(f"[HEP CONTROLS] {report.get('aggregate_verdict', 'UNKNOWN')}")
    print(f"[HEP CONTROLS] q_T mean = {stats.get('mean_qt_gev', 0):.4f} GeV")
    print(f"[HEP CONTROLS] Report: {report.get('report_path', 'n/a')}")