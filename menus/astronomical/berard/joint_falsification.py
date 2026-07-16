"""
Joint falsification across cosmological, galactic (NGC 3198), and lab channels.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from menus.astronomical.berard.ngc3198_benchmark import run_ngc3198_full_benchmark
from menus.astronomical.berard.plotting import import_reference_plots
from menus.astronomical.berard.validation import (
    run_galactic_acceleration_test,
    run_hubble_tension_test,
    run_stability_well_lab_test,
)
from tav_shared.artifact_paths import TestSlug, artifact_path, artifact_timestamp


def _channel_score(verdict: str, positive: set[str]) -> int:
    return 1 if verdict in positive else 0


def run_joint_falsification(
    *,
    include_ngc3198: bool = True,
    import_plots: bool = True,
    verbose: bool = True,
    save_report: bool = True,
) -> dict[str, Any]:
    """
    Run cosmology + NGC 3198 benchmark + lab channels; aggregate falsification summary.
    """
    from menus.astronomical.desi.json_util import write_json

    hubble = run_hubble_tension_test(verbose=verbose)
    galactic_demo = run_galactic_acceleration_test(verbose=verbose)
    lab = run_stability_well_lab_test(verbose=verbose)

    ngc3198: dict[str, Any] | None = None
    if include_ngc3198:
        ngc3198 = run_ngc3198_full_benchmark(verbose=verbose, save_report=False)

    ref_plots: list[str] = []
    if import_plots:
        ref_plots = import_reference_plots()

    cosmology_ok = hubble.get("verdict") == "RESCALE_REDUCES_TENSION"
    lab_ok = lab.get("verdict") == "TESTABLE_WITH_NARROWBAND_FILTERING"

    phase1 = (ngc3198 or {}).get("phase1_verdict", "SKIPPED")
    phase2 = ((ngc3198 or {}).get("phase2_residual") or {}).get("verdict", "SKIPPED")

    channels = {
        "cosmology": {
            "test": "hubble_tension_rescaling",
            "verdict": hubble.get("verdict"),
            "supports_berard": cosmology_ok,
        },
        "galaxy_ngc3198_phase1": {
            "test": "cored_halo_benchmark",
            "verdict": phase1,
            "supports_berard": phase1 in {"CORED_MODEL_FAVORED_OVER_BARYONS", "HONEST_TIE_EXPECTED"},
        },
        "galaxy_ngc3198_phase2": {
            "test": "log_periodic_residual_1_over_ln7",
            "verdict": phase2,
            "supports_berard": phase2 in {"PHASE2_DETECTION_CANDIDATE", "PHASE2_MARGINAL_HINT"},
        },
        "laboratory": {
            "test": "stability_well_0p10_hz",
            "verdict": lab.get("verdict"),
            "supports_berard": lab_ok,
        },
    }

    n_support = sum(1 for ch in channels.values() if ch["supports_berard"])
    if phase2 == "PHASE2_DETECTION_CANDIDATE":
        overall = "JOINT_HINT_MULTI_CHANNEL"
    elif n_support >= 2:
        overall = "JOINT_INCONCLUSIVE_MULTI_CHANNEL"
    elif n_support == 0:
        overall = "JOINT_NULL_OR_CONSTRAINT"
    else:
        overall = "JOINT_PARTIAL_SUPPORT"

    report: dict[str, Any] = {
        "action": "Joint Berard falsification (cosmology + galaxy + lab)",
        "timestamp": artifact_timestamp(datetime.now(timezone.utc)),
        "channels": channels,
        "hubble_tension": hubble,
        "galactic_acceleration_demo": galactic_demo,
        "stability_well_lab": lab,
        "ngc3198_benchmark": ngc3198,
        "imported_reference_plots": ref_plots,
        "overall_verdict": overall,
        "n_channels_supporting": n_support,
        "confidence": "exploratory",
        "interpretation": (
            "Cosmology and lab channels test Berard scalar parameters; NGC 3198 "
            "benchmark tests hybrid inertia + cored halo with pre-registered Phase 2. "
            "No single-channel detection confirms the framework."
        ),
    }

    if save_report:
        path = artifact_path(TestSlug.BERARD, "joint_falsification", "report", "json")
        write_json(path, report, indent=2, sort_keys=True)
        report["report_path"] = str(path)

    if verbose:
        print("=" * 60)
        print("JOINT BERARD FALSIFICATION")
        for key, ch in channels.items():
            print(f"  {key:28s} {ch['verdict']}")
        print(f"  Overall                    : {overall}")
        if report.get("report_path"):
            print(f"  Report                     : {report['report_path']}")
        print("=" * 60)
    return report