"""
Berard Framework ↔ Tau-Superblock research tool.

Integrates tau-cosmology-berard-framework (resonance-field cosmology + Tiny_Tau
quantum counts) as an exploratory astronomical submenu.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from menus.astronomical.berard.core import demo_parameter_sweep, equation_registry_report
from menus.astronomical.berard.fetcher import (
    DATASETS_DIR,
    knowledge_index,
    list_cached_quantum_counts,
    sync_berard_assets,
)
from menus.astronomical.berard.quantum_counts import run_quantum_counts_analysis
from menus.astronomical.berard.joint_falsification import run_joint_falsification
from menus.astronomical.berard.ngc3198_benchmark import run_ngc3198_full_benchmark
from menus.astronomical.berard.ngc3198_data import ingest_ngc3198_from_sparc
from menus.astronomical.berard.plotting import import_reference_plots
from menus.astronomical.berard.validation import (
    run_full_berard_validation_suite,
    run_galactic_acceleration_test,
    run_hubble_tension_test,
    run_stability_well_lab_test,
)
from tav_shared.artifact_paths import TestSlug, artifact_path
from tav_shared.llm_analysis import analyze_run_output as query_llm_analysis

MODULE_TAG = "BERARD_FRAMEWORK"
SUBMENU_TITLE = "Berard Resonance Cosmology"

MENU_ACTIONS = [
    "Sync Berard & Tau-Cosmology Assets",
    "Freeze NGC 3198 SPARC Data",
    "NGC 3198 Hybrid Benchmark (Phase 1+2)",
    "Joint Falsification (Cosmo + Galaxy + Lab)",
    "Berard Equations & Constants Report",
    "Hubble Tension Rescaling Test",
    "Galactic Acceleration (Modified Inertia)",
    "Stability Well 0.10 Hz Lab Signature",
    "Analyze Tiny_Tau Quantum Counts",
    "Import Reference Plots & Figures",
    "Full Berard Validation Suite",
]


def berard_entry_instructions(action: str) -> list[str]:
    instructions: dict[str, list[str]] = {
        "Sync Berard & Tau-Cosmology Assets": [
            "Pull berard-framework + tau-cosmology (PREREG, NGC 3198 benchmark, counts).",
            f"Caches under {DATASETS_DIR}/ and Personal_Files/berard-framework/.",
            "Includes PREREGISTRATION.md and reproducible_audit_script from tau-cosmology.",
        ],
        "Freeze NGC 3198 SPARC Data": [
            "Ingest SPARC MassModels into frozen data/ngc3198_rotation_curve.csv.",
            "SHA256 provenance; no residual-based outlier rejection (pre-reg quality cut).",
        ],
        "NGC 3198 Hybrid Benchmark (Phase 1+2)": [
            "Phase 1: baryons, plain pISO, Berard hybrid, Burkert, NFW with ΔBIC table.",
            "Phase 2: pre-registered 1/ln(7) log-periodic residual on best smooth model.",
            "Generates Berard-style rotation/residual/comparison plot.",
        ],
        "Joint Falsification (Cosmo + Galaxy + Lab)": [
            "Hubble rescaling + NGC 3198 benchmark + 0.10 Hz lab + reference plot import.",
            "Aggregated multi-channel verdict with exploratory confidence metadata.",
        ],
        "Import Reference Plots & Figures": [
            "Copies tau-cosmology NGC 3198 PNGs and berard-framework APS PDF figures.",
            f"Destination: {DATASETS_DIR}/figures/",
        ],
        "Berard Equations & Constants Report": [
            "Exports BC=1.054, f₀=0.10 Hz, equation registry, and S₀ sweep.",
            "Marked exploratory / not peer-reviewed in artifact metadata.",
            "Report: artifacts/berard_framework/…",
        ],
        "Hubble Tension Rescaling Test": [
            "Tests H_obs = BC·H_B against Planck (67.4) vs SH0ES (73.0) km/s/Mpc.",
            "Exploratory tension relief metric — not a full MCMC cosmology fit.",
        ],
        "Galactic Acceleration (Modified Inertia)": [
            "a_B = a_Newton/BC² rotation-curve demo at SPARC-like radii.",
            "Modified inertia sector only (WEP preserved locally at S₀≈1).",
        ],
        "Stability Well 0.10 Hz Lab Signature": [
            "Laboratory falsification pathway for the vacuum resonance mode.",
            "Flags seismic/thermal bandwidth challenges for 0.10 Hz detection.",
        ],
        "Analyze Tiny_Tau Quantum Counts": [
            "Parses sparse Qiskit Aer bitstring histograms (reps3/neg_reps3 1054 runs).",
            "Optional counts_file path; blank uses cached datasets/berard_framework/quantum_counts/.",
        ],
        "Full Berard Validation Suite": [
            "Runs Hubble + galactic + lab tests in series with JSON summary.",
            "Suitable entry point after Sync Berard Framework Assets.",
        ],
    }
    return instructions.get(
        action,
        [
            "Berard Framework: resonance scalar φ, invariant S₀, BC=1.054, f₀=0.10 Hz.",
            "Exploratory module — present results with confidence metadata.",
            f"Dataset cache: {DATASETS_DIR}/",
        ],
    )


def berard_entry_fields(action: str) -> list[dict]:
    fields: list[dict] = []
    if action in {"Sync Berard & Tau-Cosmology Assets", "Sync Berard Framework Assets", "Freeze NGC 3198 SPARC Data"}:
        fields.extend(
            [
                {
                    "key": "force_refresh",
                    "label": "Force re-sync",
                    "default": "no",
                    "choices": ["no", "yes"],
                },
                {
                    "key": "include_large",
                    "label": "Include large periodic JSON (3.6 MB)",
                    "default": "no",
                    "choices": ["no", "yes"],
                },
            ]
        )
    if action == "Analyze Tiny_Tau Quantum Counts":
        fields.append(
            {
                "key": "counts_file",
                "label": "Counts JSON path (optional)",
                "default": "",
                "required": False,
                "hint": "Blank = all cached *_counts.json in berard_framework/quantum_counts/",
            }
        )
    if action == "Hubble Tension Rescaling Test":
        fields.extend(
            [
                {
                    "key": "h0_planck",
                    "label": "H₀ Planck (km/s/Mpc)",
                    "default": "67.4",
                    "required": False,
                },
                {
                    "key": "h0_shoes",
                    "label": "H₀ SH0ES (km/s/Mpc)",
                    "default": "73.0",
                    "required": False,
                },
                {
                    "key": "bc_override",
                    "label": "BC override (optional)",
                    "default": "",
                    "required": False,
                },
            ]
        )
    return fields


entry_fields = berard_entry_fields
entry_instructions = berard_entry_instructions


def _yes(value: str) -> bool:
    return str(value or "").strip().lower() in {"yes", "y", "true", "1"}


def _float_opt(raw: str, default: float) -> float:
    text = (raw or "").strip()
    if not text:
        return default
    return float(text)


def _save_equation_report(payload: dict[str, Any]) -> str:
    from menus.astronomical.desi.json_util import write_json

    path = artifact_path(TestSlug.BERARD, "equations_constants", "report", "json")
    write_json(path, payload, indent=2, sort_keys=True)
    return str(path)


def run_action(selection: str, show_plots: bool = True, options: dict | None = None) -> str | None:
    options = options or {}

    if selection in {"Sync Berard & Tau-Cosmology Assets", "Sync Berard Framework Assets"}:
        summary = sync_berard_assets(
            force_refresh=_yes(options.get("force_refresh")),
            include_large=_yes(options.get("include_large")),
        )
        print(f"[BERARD] Source repo     : {summary.source}")
        print(f"[BERARD] Tau-cosmology   : {summary.tau_cosmology_source}")
        print(f"[BERARD] Text assets     : {len(summary.text_files)}")
        print(f"[BERARD] Quantum counts  : {len(summary.quantum_files)}")
        print(f"[BERARD] Tau assets      : {len(summary.tau_cosmology_files)}")
        print(f"[BERARD] Figures         : {len(summary.figure_files)}")
        print(f"[BERARD] Cache root      : {DATASETS_DIR}")
        return str(DATASETS_DIR)

    if selection == "Freeze NGC 3198 SPARC Data":
        meta = ingest_ngc3198_from_sparc(force=_yes(options.get("force_refresh")))
        print(f"[BERARD] Frozen points   : {meta.get('n_points')}")
        print(f"[BERARD] SHA256          : {meta.get('sha256')}")
        return meta.get("csv_path")

    if selection == "NGC 3198 Hybrid Benchmark (Phase 1+2)":
        report = run_ngc3198_full_benchmark(verbose=True)
        return report.get("report_path")

    if selection == "Joint Falsification (Cosmo + Galaxy + Lab)":
        report = run_joint_falsification(verbose=True)
        return report.get("report_path")

    if selection == "Import Reference Plots & Figures":
        paths = import_reference_plots()
        print(f"[BERARD] Imported {len(paths)} reference plot(s)")
        for p in paths[:8]:
            print(f"  {p}")
        return paths[0] if paths else None

    if selection == "Berard Equations & Constants Report":
        report = equation_registry_report()
        report["s0_demo_sweep"] = demo_parameter_sweep()
        report["knowledge_index"] = knowledge_index()
        report["report_path"] = _save_equation_report(report)
        print(json.dumps(report["constants"], indent=2))
        print(f"[BERARD] Report: {report['report_path']}")
        return report["report_path"]

    if selection == "Hubble Tension Rescaling Test":
        bc = _float_opt(options.get("bc_override"), 1.054)
        result = run_hubble_tension_test(
            h0_planck=_float_opt(options.get("h0_planck"), 67.4),
            h0_shoes=_float_opt(options.get("h0_shoes"), 73.0),
            bc=bc,
            verbose=True,
        )
        return result.get("report_path")

    if selection == "Galactic Acceleration (Modified Inertia)":
        result = run_galactic_acceleration_test(verbose=True)
        return result.get("report_path")

    if selection == "Stability Well 0.10 Hz Lab Signature":
        result = run_stability_well_lab_test(verbose=True)
        return result.get("report_path")

    if selection == "Analyze Tiny_Tau Quantum Counts":
        raw = (options.get("counts_file") or "").strip()
        if raw:
            paths = [Path(raw).expanduser()]
        else:
            paths = list_cached_quantum_counts()
            if not paths:
                sync_berard_assets()
                paths = list_cached_quantum_counts()
        if not paths:
            print("[BERARD] No counts JSON found — run Sync Berard Framework Assets first.")
            return None
        result = run_quantum_counts_analysis(paths, verbose=True)
        return result.get("report_path")

    if selection == "Full Berard Validation Suite":
        sync_berard_assets()
        suite = run_full_berard_validation_suite(verbose=True)
        counts = list_cached_quantum_counts()
        if counts:
            suite["quantum_counts"] = run_quantum_counts_analysis(
                counts[:2], verbose=False
            )
        print(f"[BERARD] Suite report: {suite.get('report_path')}")
        return suite.get("report_path")

    print(f"[BERARD] Unknown action: {selection}")
    return None


def is_module_selection(repo: str | None) -> bool:
    return repo == MODULE_TAG