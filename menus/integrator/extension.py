"""
Tav Framework Integrator ↔ research_tool.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt

from menus.integrator.fetcher import DATASETS_DIR, FINISHED2_DIR, PLANCK_MAP_CATALOG
from tav_shared.llm_analysis import analyze_run_output as query_llm_analysis
from tav_shared.run_output import parse_show_graphics
from menus.integrator.integrator import DEFAULT_CONFIG, TavFrameworkIntegrator

MODULE_TAG = "TAV_DATA_INTEGRATOR"
SUBMENU_TITLE = "Tav Framework Integrator"

MENU_ACTIONS = [
    "Pull Integrator Datasets",
    "Run Planck+SPARC Correlation",
    "Full Correlation Pipeline",
    "Lock Analytic α Calibration",
    "Generate Summary Report",
]

_ACTION_CONFIG = {
    "Pull Integrator Datasets": {"pull_only": True},
    "Run Planck+SPARC Correlation": {"pipeline": True, "report": False},
    "Full Correlation Pipeline": {"pipeline": True, "report": True},
    "Lock Analytic α Calibration": {"calibration_only": True},
    "Generate Summary Report": {"report_only": True},
}


def is_module_selection(repo: str | None) -> bool:
    return repo == MODULE_TAG


def _build_config(options: dict) -> dict:
    config = DEFAULT_CONFIG.copy()
    config = {
        "planck": dict(DEFAULT_CONFIG["planck"]),
        "sparc": dict(DEFAULT_CONFIG["sparc"]),
        "output": dict(DEFAULT_CONFIG["output"]),
        "tav": dict(DEFAULT_CONFIG["tav"]),
        "analytic": dict(DEFAULT_CONFIG["analytic"]),
    }
    config["planck"]["map_paths"] = dict(DEFAULT_CONFIG["planck"]["map_paths"])
    config["planck"]["mask_path"] = (options.get("mask_path") or config["planck"]["mask_path"]).strip()
    config["sparc"]["master_table"] = (
        options.get("sparc_master") or config["sparc"]["master_table"]
    ).strip()
    config["output"]["artifacts_dir"] = (
        options.get("artifacts_dir") or config["output"]["artifacts_dir"]
    ).strip()
    config["output"]["demo_mode"] = str(options.get("demo_mode", "no")).lower() in {
        "yes",
        "y",
        "true",
        "1",
    }
    try:
        config["planck"]["lmax"] = int(options.get("lmax") or config["planck"]["lmax"])
    except ValueError:
        pass
    try:
        config["sparc"]["batch_limit"] = int(
            options.get("batch_limit") or config["sparc"]["batch_limit"]
        )
    except ValueError:
        pass
    config["sparc"]["force_rescan"] = str(options.get("force_rescan", "no")).lower() in {
        "yes",
        "y",
        "true",
        "1",
    }

    keys_raw = (options.get("planck_keys") or "sevem_hm1,nilc_hm2").strip()
    return config, [k.strip() for k in keys_raw.split(",") if k.strip()]


def _show_saved_plots(artifacts_dir: Path) -> None:
    from pathlib import Path as _Path

    search_dirs = [artifacts_dir, _Path(__file__).resolve().parent / "artifacts"]
    shown = False
    plot_paths = []
    for folder in search_dirs:
        if folder.is_dir():
            plot_paths.extend(sorted(folder.glob("tav_integrator_*_*.png")))
    for plot_path in plot_paths:
        if "_spectrum" not in plot_path.name and "_fft" not in plot_path.name:
            continue
        image = plt.imread(plot_path)
        plt.figure(figsize=(10, 6))
        plt.imshow(image)
        plt.axis("off")
        plt.title(os.path.basename(plot_path))
        plt.tight_layout()
        shown = True
    if shown:
        plt.show()


# =============================================================================
# BLOCK: Entry form — fields
# =============================================================================
def entry_fields(action: str) -> list[dict]:
    """Curses entry-form field specs for Tav Framework Integrator actions."""
    if action == "Pull Integrator Datasets":
        return [
            {
                "key": "planck_keys",
                "label": "Planck map keys (comma-separated)",
                "default": "sevem_hm1,nilc_hm2",
                "required": False,
                "hint": "e.g. sevem_hm1,nilc_hm2,commander_full",
            },
            {
                "key": "batch_limit",
                "label": "Pull batch limit (0 = all pending)",
                "default": "0",
                "required": False,
                "hint": "Resumes via datasets/fits/done.txt",
            },
            {
                "key": "force_refresh",
                "label": "Force re-download",
                "default": "no",
                "required": False,
                "hint": "Overwrite cached FITS in datasets/fits/",
                "choices": ["no", "yes"],
            },
            {
                "key": "force_rescan",
                "label": "Force re-pull (ignore done.txt)",
                "default": "no",
                "required": False,
                "hint": "Re-fetch from the first catalog keys",
                "choices": ["no", "yes"],
            },
        ]
    if action == "Lock Analytic α Calibration":
        return []
    return [
        {
            "key": "planck_keys",
            "label": "Planck map keys",
            "default": "sevem_hm1,nilc_hm2",
            "required": False,
            "hint": "Comma-separated keys from integrator catalog",
        },
        {
            "key": "sparc_master",
            "label": "SPARC master table (optional)",
            "default": "",
            "required": False,
            "hint": "Blank = datasets/sparc/SPARC_master_table.csv",
        },
        {
            "key": "batch_limit",
            "label": "SPARC galaxies to analyze",
            "default": "10",
            "required": False,
            "hint": "Pending galaxies per run; resumes via datasets/sparc/batch_done.txt",
        },
        {
            "key": "force_rescan",
            "label": "Force re-analyze (ignore batch_done.txt)",
            "default": "no",
            "required": False,
            "hint": "Restart SPARC analysis from the first galaxies",
            "choices": ["no", "yes"],
        },
        {
            "key": "lmax",
            "label": "Planck lmax",
            "default": "2000",
            "required": False,
            "hint": "Multipole cap for anafast",
        },
        {
            "key": "artifacts_dir",
            "label": "Report output directory",
            "default": "artifacts/tav_integrator",
            "required": False,
            "hint": "JSON/TXT summary reports",
        },
        {
            "key": "demo_mode",
            "label": "Demo mode",
            "default": "no",
            "required": False,
            "hint": "Use no for real FITS correlation",
            "choices": ["no", "yes"],
        },
        {
            "key": "show_graphics",
            "label": "Graphics mode",
            "default": "artifacts",
            "required": False,
            "hint": "popup = Tk windows | artifacts = PNG only",
            "choices": ["artifacts", "popup"],
        },
    ]


# =============================================================================
# BLOCK: Entry form — instructions
# =============================================================================
def entry_instructions(action: str) -> list[str]:
    """Short help bullets shown above the integrator entry form."""
    if action == "Pull Integrator Datasets":
        return [
            "Fetches Planck CMB FITS into datasets/fits/ (IRSA wget).",
            "Builds SPARC master CSV in datasets/sparc/.",
            f"Processed FITS move to {FINISHED2_DIR}/ after pipeline runs.",
        ]
    if action == "Lock Analytic α Calibration":
        return [
            "Locks optimized α = 41.341 (June 25, 2026) into integrator config.",
            "Rebuilds analytic_tav_ladder and void-inflation simulation peaks.",
            "Reports mean residual vs reference Planck multipoles.",
            "Also runs automatically at end of Full Correlation Pipeline.",
        ]
    return [
        "Cross-correlates Planck Tav harmonics with SPARC shadow ratios.",
        f"Incoming FITS cache: {DATASETS_DIR}/",
        "Default Planck keys: sevem_hm1, nilc_hm2.",
        f"Reports: artifacts/tav_integrator/ | archive: {FINISHED2_DIR}/",
        "demo_mode=no required for real-data runs.",
    ]


def run_action(selection: str, show_plots: bool = True, options: dict | None = None) -> None:
    options = options or {}
    config_entry = _ACTION_CONFIG.get(selection)
    if config_entry is None:
        print(f"[TAV ENGINE] Unknown integrator action: {selection}")
        return

    if config_entry.get("pull_only"):
        from menus.integrator.fetcher import pull_integrator_datasets

        force = str(options.get("force_refresh", "no")).lower() in {"yes", "y", "true", "1"}
        config, planck_keys = _build_config(options)
        try:
            batch_limit = int(options.get("batch_limit") or 0)
        except ValueError:
            batch_limit = 0
        summary = pull_integrator_datasets(
            planck_keys=planck_keys,
            force_refresh=force,
            batch_limit=batch_limit,
            force_rescan=str(options.get("force_rescan", "no")).lower() in {"yes", "y", "true", "1"},
        )
        print(f"[TAV ENGINE] Fetched: {summary.fetched}")
        if summary.failed:
            for key, msg in summary.failed.items():
                print(f"  FAIL {key}: {msg}")
        print(f"[TAV ENGINE] Datasets in {DATASETS_DIR}/ | archive -> {FINISHED2_DIR}/")
        return

    config, planck_keys = _build_config(options)
    integrator = TavFrameworkIntegrator(config)

    if config_entry.get("calibration_only"):
        calibration = integrator.run_final_calibration()
        report_path = integrator.generate_summary_report()
        print(f"[TAV ENGINE] Void-inflation peaks: {calibration['void_inflation_simulation']['predicted_peaks'][:5]}...")
        print(f"[TAV ENGINE] Calibration report: {report_path}")
        return

    if config_entry.get("report_only"):
        if not integrator.results.planck and not integrator.results.analytic:
            print("[TAV ENGINE] No prior pipeline results — run Full Correlation Pipeline first.")
            return
        integrator.generate_summary_report()
        return

    if config_entry.get("pipeline"):
        integrator.full_correlation_pipeline(run_planck_keys=planck_keys)
        if config_entry.get("report"):
            integrator.generate_summary_report()

        show_popup = parse_show_graphics(options, default="artifacts") if show_plots else False
        if show_popup:
            _show_saved_plots(Path(config["output"]["artifacts_dir"]))
        else:
            print(
                "[TAV ENGINE] Graphics saved under artifacts/. "
                "Set show_graphics=popup to open plot windows."
            )