"""
FRB cosmic-web Tav module ↔ research_tool.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from menus.astronomical.frb.cosmic_web_tav import ARTIFACTS_DIR, run_pipeline
from menus.astronomical.frb.dispersion_test import run_frb_dispersion_test
from menus.astronomical.frb.fetcher import DATASETS_DIR, ensure_frb_datasets, pull_open_archives
from tav_shared.llm_analysis import analyze_run_output as query_llm_analysis
from tav_shared.run_output import parse_show_graphics

MODULE_TAG = "FRB_COSMIC_WEB_TAV"
SUBMENU_TITLE = "FRB Cosmic-Web Tav-Scan"

MENU_ACTIONS = [
    "Pull Datasets from Open Archives",
    "Classify FRB Paths",
    "Analyze DM Residuals by Path",
    "Scan 1/7 Tav Harmonics",
    "FRB Dispersion Tav Test (KS)",
    "Full FRB Tav Analysis",
]

_N_POINTS_ACTIONS = frozenset(
    {
        "Classify FRB Paths",
        "Analyze DM Residuals by Path",
        "Scan 1/7 Tav Harmonics",
        "Full FRB Tav Analysis",
    }
)


def _n_points_field() -> dict[str, Any]:
    return {
        "key": "n_points",
        "label": "Sample size n (optional)",
        "default": "25",
        "required": False,
        "hint": "Leave blank or use interactive selector in menu",
    }


def _maybe_append_n_points(fields: list[dict], action: str) -> list[dict]:
    from tav_shared.n_selector_registry import action_uses_n_selector, n_points_field

    if action_uses_n_selector("frb_web_extension", action):
        fields.append(n_points_field("frb_web_extension", action))
    return fields


def frb_entry_instructions(action: str) -> list[str]:
    from menus.astronomical.frb.fetcher import DATASETS_DIR
    from tav_shared.tav_project_paths import FINISHED_A_DIR

    instructions: dict[str, list[str]] = {
        "Pull Datasets from Open Archives": [
            "Fetches CHIME FRB catalog (CSV mirrors) into datasets/frb/.",
            "Falls back to CDS VizieR J/ApJS/257/59 (536-burst Catalog 1) when mirrors fail.",
            "Uses cfod catalog API when installed; else 400-burst curated template.",
            "Fetches SDSS DR7 void catalog (Douglass+ 2023, VizieR J/ApJS/265/7).",
            "Skips re-download when cached or marked processed (datasets/processed.txt).",
            f"Processed CSVs move to {FINISHED_A_DIR}/ after analysis runs.",
        ],
        "FRB Dispersion Tav Test (KS)": [
            "KS test: DM distribution near vs far analytic Tav ladder nodes.",
            "Tav peaks from TavFrameworkIntegrator.analytic_tav_ladder() (α=41.341).",
            f"Auto-pull / restore from {FINISHED_A_DIR}/ when CSV missing in {DATASETS_DIR}/.",
            "healpy great-circle geometry; nodes from 7-fold ladder (k, residue) mapping.",
            "node_threshold_deg default 25° | restore_archived default yes.",
            "Report: artifacts/frb_dispersion_tav_test_*.json",
        ],
        "Classify FRB Paths": [
            "Classifies each FRB as void / sheet / filament / node using void proximity.",
            f"Blank catalog fields auto-detect / pull / restore from {DATASETS_DIR}/.",
            "Optional n_points subsamples the catalog for quick scans.",
            "Plot: artifacts/frb_cosmic_web_tav_*_paths.png",
        ],
        "Analyze DM Residuals by Path": [
            "Groups DM residuals by path class from cosmic-web classifier.",
            "Uses full CHIME + void pairing when n_points is blank.",
            "Plots: paths + periodogram under artifacts/.",
        ],
        "Scan 1/7 Tav Harmonics": [
            "Periodogram-only pass on DM residuals (analyze_only mode).",
            "Tests 1/7 Tav harmonic structure without re-running full classifier.",
            "Optional n_points limits FRBs used in the harmonic scan.",
        ],
        "Full FRB Tav Analysis": [
            "End-to-end: classify paths, DM residuals, 1/7 harmonic periodogram.",
            f"Auto-fetch when missing; archives to {FINISHED_A_DIR}/ on success.",
            "Plots: artifacts/frb_cosmic_web_tav_*_paths.png and _periodogram.png",
        ],
    }
    return instructions.get(
        action,
        [
            "FRB CSV: ra, dec, DM columns (CHIME-style names auto-detected).",
            "Void CSV: ra, dec, reff_mpc (SDSS void catalog format).",
            f"Drop files in {DATASETS_DIR}/ — blank fields auto-detect, pull, or restore.",
            "Auto-fetch when missing; skips processed targets unless force_refresh=yes.",
            f"After success, datasets/frb/ files move to {FINISHED_A_DIR}/.",
            "Plots: artifacts/frb_cosmic_web_tav_*_paths.png and _periodogram.png",
        ],
    )


def frb_entry_fields(action: str) -> list[dict]:
    from menus.astronomical.frb.fetcher import DATASETS_DIR
    from tav_shared.tav_project_paths import FINISHED_A_DIR

    fetch_fields = [
        {
            "key": "force_refresh",
            "label": "Force re-download",
            "default": "no",
            "required": False,
            "hint": "Override processed ledger and cached CSVs",
            "choices": ["no", "yes"],
        },
        {
            "key": "restore_archived",
            "label": "Restore archived",
            "default": "yes",
            "required": False,
            "hint": f"Try {FINISHED_A_DIR}/ before remote pull",
            "choices": ["no", "yes"],
        },
    ]
    graphics_field = {
        "key": "show_graphics",
        "label": "Graphics mode",
        "default": "popup",
        "required": False,
        "hint": "popup = Tk windows | artifacts = save PNG only",
        "choices": ["artifacts", "popup"],
    }

    if action == "Pull Datasets from Open Archives":
        return fetch_fields

    if action == "FRB Dispersion Tav Test (KS)":
        return [
            {
                "key": "frb_catalog",
                "label": "FRB catalog CSV (optional)",
                "default": "",
                "required": False,
                "hint": f"Blank = auto-detect / pull / restore in {DATASETS_DIR}/",
            },
            {
                "key": "node_threshold_deg",
                "label": "Node proximity threshold (degrees)",
                "default": "25",
                "required": False,
                "hint": "Great-circle separation to nearest Tav node",
            },
            *fetch_fields,
        ]

    pipeline_fields = [
        {
            "key": "frb_catalog",
            "label": "FRB catalog CSV (optional)",
            "default": "",
            "required": False,
            "hint": f"Blank = auto-detect / pull / restore in {DATASETS_DIR}/",
        },
        {
            "key": "void_catalog",
            "label": "Void catalog CSV (optional)",
            "default": "",
            "required": False,
            "hint": "Blank = auto-detect void/web CSV in datasets/frb/",
        },
        {
            "key": "output_prefix",
            "label": "Plot filename prefix",
            "default": "frb_cosmic_web_tav",
            "required": False,
            "hint": "Base name; dataset tag appended (e.g. frb_cosmic_web_tav_chime)",
        },
        *fetch_fields,
        graphics_field,
    ]
    return _maybe_append_n_points(list(pipeline_fields), action)


_ACTION_CONFIG = {
    "Pull Datasets from Open Archives": {"pull_only": True},
    "Classify FRB Paths": {"plot": True, "analyze_only": False},
    "Analyze DM Residuals by Path": {"plot": True, "analyze_only": False},
    "Scan 1/7 Tav Harmonics": {"plot": True, "analyze_only": True},
    "FRB Dispersion Tav Test (KS)": {"dispersion_test": True},
    "Full FRB Tav Analysis": {"plot": True, "analyze_only": False},
}


def is_module_selection(repo: str | None) -> bool:
    return repo == MODULE_TAG


def _yes(value: str) -> bool:
    return str(value or "").strip().lower() in {"yes", "y", "true", "1"}


def _fetch_options(options: dict) -> tuple[bool, bool]:
    force_refresh = _yes(options.get("force_refresh", "no")) or _yes(
        options.get("force_rescan", "no")
    )
    restore_raw = options.get("restore_archived", "yes")
    restore_archived = _yes(restore_raw) if str(restore_raw).strip() else True
    return force_refresh, restore_archived


def _max_frbs_from_options(options: dict) -> int | None:
    from tav_shared.n_selector_registry import parse_n_points

    return parse_n_points(options.get("n_points")) or parse_n_points(
        options.get("batch_limit")
    )


def _show_saved_plots(output_prefix: str) -> None:
    plot_paths = [
        ARTIFACTS_DIR / f"{output_prefix}_paths.png",
        ARTIFACTS_DIR / f"{output_prefix}_periodogram.png",
    ]
    shown = False
    for plot_path in plot_paths:
        if not plot_path.is_file():
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
# =============================================================================
# BLOCK: Pre-form hook (batch n selector)
# =============================================================================
def handle_pre_form(stdscr, selection: str, params: dict[str, str]) -> str | None:
    """Interactive n picker sets batch_limit / force_rescan before the entry form."""
    from tav_research.curses_shell import select_n_interactive
    from tav_research.n_selector import apply_frb_or_tav_n_selection, should_offer_n_selector

    if not should_offer_n_selector(MODULE_TAG, selection):
        return None
    n = select_n_interactive(
        stdscr,
        prompt=f"Choose n for {selection} (controls batch size)",
    )
    if n is not None:
        apply_frb_or_tav_n_selection(params, n, MODULE_TAG)
    return None


entry_fields = frb_entry_fields


# =============================================================================
# BLOCK: Entry form — instructions
# =============================================================================
entry_instructions = frb_entry_instructions


def run_action(selection: str, show_plots: bool = True, options: dict | None = None) -> None:
    options = options or {}
    config = _ACTION_CONFIG.get(selection)
    if config is None:
        print(f"[TAV ENGINE] Unknown FRB cosmic-web action: {selection}")
        return

    if config.get("pull_only"):
        force = str(options.get("force_refresh", "no")).strip().lower() in {
            "yes",
            "y",
            "true",
            "1",
        }
        restore = _yes(options.get("restore_archived", "no"))
        summary = pull_open_archives(force_refresh=force, restore_archived=restore)
        print(f"[TAV ENGINE] FRB source: {summary.frb_source}")
        print(f"[TAV ENGINE] Void source: {summary.void_source}")
        print(f"[TAV ENGINE] Datasets in {DATASETS_DIR}/")
        return

    if config.get("dispersion_test"):
        frb_path = (options.get("frb_catalog") or "").strip() or None
        try:
            threshold = float(options.get("node_threshold_deg") or 25.0)
        except ValueError:
            threshold = 25.0
        force_refresh, restore_archived = _fetch_options(options)
        if not frb_path:
            print(f"[TAV ENGINE] Auto-pulling FRB catalog into {DATASETS_DIR}/")
            ensure_frb_datasets(
                force_refresh=force_refresh,
                restore_archived=restore_archived,
            )
        run_frb_dispersion_test(
            frb_catalog_path=frb_path,
            node_threshold_deg=threshold,
            force_refresh=force_refresh,
            restore_archived=restore_archived,
        )
        return

    frb_path = (options.get("frb_catalog") or "").strip()
    void_path = (options.get("void_catalog") or "").strip()
    output_prefix = (options.get("output_prefix") or "frb_cosmic_web_tav").strip()
    show_popup = parse_show_graphics(options, default="popup") if show_plots else False
    force_refresh, restore_archived = _fetch_options(options)

    if not frb_path and not void_path:
        print(f"[TAV ENGINE] Auto-detecting / auto-pulling datasets in {DATASETS_DIR}/")
        ensure_frb_datasets(
            force_refresh=force_refresh,
            restore_archived=restore_archived,
        )

    result = run_pipeline(
        frb_path=Path(frb_path) if frb_path else None,
        void_path=Path(void_path) if void_path else None,
        plot=config["plot"],
        show_plot=False,
        output_prefix=output_prefix,
        archive=True,
        force_refresh=force_refresh,
        restore_archived=restore_archived,
        max_frbs=_max_frbs_from_options(options),
    )

    artifact_prefix = result.output_prefix
    if config["plot"] and show_popup:
        _show_saved_plots(artifact_prefix)
    elif config["plot"]:
        print(
            "[TAV ENGINE] Graphics mode is artifacts — PNGs saved under artifacts/. "
            "Set show_graphics=popup to open plot windows."
        )
        print(f"[TAV ENGINE] Paths: {ARTIFACTS_DIR / f'{artifact_prefix}_paths.png'}")
        print(
            f"[TAV ENGINE] Periodogram: "
            f"{ARTIFACTS_DIR / f'{artifact_prefix}_periodogram.png'}"
        )