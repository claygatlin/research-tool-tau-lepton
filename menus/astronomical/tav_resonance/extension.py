"""
tav-resonance package ↔ research_tool.py

Bridges the standalone ``tav-resonance`` PyPI package (or sibling source tree)
into the Tau-Superblock interactive research tool.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from tav_shared.llm_analysis import analyze_run_output as query_llm_analysis
from tav_shared.run_output import parse_show_graphics

MODULE_TAG = "TAV_RESONANCE"
SUBMENU_TITLE = "Tav-Resonance Package"

MENU_ACTIONS = [
    "Verify Geometric Anchors",
    "Analytic Ladder vs Reference",
    "Export Pre-registration Config",
    "FRB Resonance Scan (tav-resonance)",
    "Full Tav-Resonance Demo",
]

_N_POINTS_ACTIONS = frozenset(
    {
        "FRB Resonance Scan (tav-resonance)",
        "Full Tav-Resonance Demo",
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

    if action_uses_n_selector("tav_resonance_extension", action):
        fields.append(n_points_field("tav_resonance_extension", action))
    return fields


def tav_resonance_entry_instructions(action: str) -> list[str]:
    from menus.astronomical.frb.fetcher import DATASETS_DIR
    from tav_shared.tav_project_paths import FINISHED_A_DIR

    instructions: dict[str, list[str]] = {
        "Verify Geometric Anchors": [
            "Checks fixed M₀=313.1 MeV, 1/7 harmonic, and ladder constants.",
            "No datasets required; JSON anchor report printed to stdout.",
        ],
        "Analytic Ladder vs Reference": [
            "Compares analytic Tav ladder peaks to reference Planck multipoles.",
            "α=41.341 fixed; reports mean |residual| over reference peaks.",
        ],
        "Export Pre-registration Config": [
            "Writes frozen TavResonanceConfig JSON under artifacts/.",
            "harmonic_phase_tolerance tunable before freeze=yes export.",
        ],
        "FRB Resonance Scan (tav-resonance)": [
            "Runs tav-resonance path classifier on CHIME FRB + SDSS void catalogs.",
            f"Blank paths auto-detect / pull in {DATASETS_DIR}/; restore from {FINISHED_A_DIR}/.",
            "Optional n_points subsamples FRBs for quick resonance checks.",
            "Plot: artifacts/tav_resonance_frb_paths.png",
        ],
        "Full Tav-Resonance Demo": [
            "Anchors → ladder → pre-reg export → FRB resonance scan in one run.",
            "Uses same dataset auto-pull as the standalone FRB scan action.",
            "Text output saved to artifacts/run_*.txt automatically.",
        ],
    }
    return instructions.get(
        action,
        [
            "Uses the standalone tav-resonance package (pip install -e ../tav-resonance).",
            "Fixed anchors: M₀=313.1 MeV, 1/7 harmonic, α=41.341 — never fitted.",
            "FRB scan reuses datasets/frb/ via frb_fetcher (CHIME + SDSS voids).",
            "Pre-registration export writes JSON under artifacts/.",
            "Text output saved to artifacts/run_*.txt automatically.",
        ],
    )


def tav_resonance_entry_fields(action: str) -> list[dict]:
    from menus.astronomical.frb.fetcher import DATASETS_DIR
    from tav_shared.tav_project_paths import FINISHED_A_DIR

    fetch_fields = [
        {
            "key": "force_refresh",
            "label": "Force re-download",
            "default": "no",
            "required": False,
            "hint": "Override processed ledger",
            "choices": ["no", "yes"],
        },
        {
            "key": "restore_archived",
            "label": "Restore archived",
            "default": "yes",
            "required": False,
            "hint": f"Copy from {FINISHED_A_DIR}/ when missing",
            "choices": ["no", "yes"],
        },
    ]
    graphics_field = {
        "key": "show_graphics",
        "label": "Graphics mode",
        "default": "artifacts",
        "required": False,
        "hint": "popup = Tk window | artifacts = PNG only",
        "choices": ["artifacts", "popup"],
    }

    if action == "Export Pre-registration Config":
        return [
            {
                "key": "harmonic_phase_tolerance",
                "label": "Harmonic phase tolerance",
                "default": "0.04",
                "required": False,
                "hint": "Tunable threshold before freeze",
            },
            {
                "key": "freeze",
                "label": "Freeze before export",
                "default": "yes",
                "required": False,
                "hint": "Pin config for pre-registration",
                "choices": ["yes", "no"],
            },
            {
                "key": "output_name",
                "label": "JSON filename",
                "default": "tav_resonance_prereg.json",
                "required": False,
                "hint": "Saved under artifacts/",
            },
        ]

    if action in {"Verify Geometric Anchors", "Analytic Ladder vs Reference"}:
        return []

    scan_fields = [
        {
            "key": "frb_catalog",
            "label": "FRB catalog CSV (optional)",
            "default": "",
            "required": False,
            "hint": f"Blank = auto-detect / pull in {DATASETS_DIR}/",
        },
        {
            "key": "void_catalog",
            "label": "Void catalog CSV (optional)",
            "default": "",
            "required": False,
            "hint": "Blank = auto-detect void CSV in datasets/frb/",
        },
        {
            "key": "freeze",
            "label": "Freeze config before scan",
            "default": "yes",
            "required": False,
            "hint": "Pre-registration workflow",
            "choices": ["yes", "no"],
        },
        *fetch_fields,
        {
            "key": "output_prefix",
            "label": "Plot filename prefix",
            "default": "tav_resonance_frb",
            "required": False,
            "hint": "Path-mix bar chart under artifacts/",
        },
        graphics_field,
    ]
    return _maybe_append_n_points(scan_fields, action)


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
_SIBLING_SRC = PROJECT_ROOT.parent / "tav-resonance" / "src"


def _ensure_tav_resonance():
    try:
        import tav_resonance  # noqa: F401
    except ImportError:
        if _SIBLING_SRC.is_dir():
            src = str(_SIBLING_SRC)
            if src not in sys.path:
                sys.path.insert(0, src)
        import tav_resonance  # noqa: F401


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _yes(value: str) -> bool:
    return str(value or "").strip().lower() in {"yes", "y", "true", "1"}


def is_module_selection(repo: str | None) -> bool:
    return repo == MODULE_TAG


def _run_anchor_check() -> None:
    _ensure_tav_resonance()
    from tav_resonance import geometric_anchor_check

    report = geometric_anchor_check()
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["ok"]:
        print("[TAV RESONANCE] All geometric anchors pass.")
    else:
        print("[TAV RESONANCE] Anchor mismatch detected — check values above.")


def _run_ladder_evaluation() -> None:
    _ensure_tav_resonance()
    from tav_resonance import TavResonanceConfig, evaluate_ladder_vs_observed
    from tav_resonance.anchors import REFERENCE_LADDER_PEAKS

    cfg = TavResonanceConfig.from_defaults()
    report = evaluate_ladder_vs_observed(REFERENCE_LADDER_PEAKS, config=cfg)
    print(f"α (fixed): {report['alpha_used']}")
    print(f"Peaks compared: {report['n_peaks_compared']}")
    print(f"Mean |residual|: {report['mean_absolute_residual']:.4f}")
    for row in report["comparison_table"][:8]:
        print(
            f"  ℓ_obs={row['observed']:4d}  ℓ_pred={row['predicted']:4d}  "
            f"residual={row['residual']:+7.1f}"
        )


def _run_prereg_export(options: dict) -> None:
    _ensure_tav_resonance()
    from tav_resonance import TavResonanceConfig

    cfg = TavResonanceConfig.from_defaults()
    try:
        cfg.harmonic_phase_tolerance = float(options.get("harmonic_phase_tolerance") or 0.04)
    except ValueError:
        pass
    if _yes(options.get("freeze", "yes")):
        cfg.freeze()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_name = (options.get("output_name") or f"tav_resonance_prereg_{_utc_stamp()}.json").strip()
    out_path = ARTIFACTS_DIR / out_name
    cfg.to_prereg_json(out_path)
    print(f"[TAV RESONANCE] Pre-registration config written to {out_path}")


def _resolve_frb_void_paths(options: dict) -> tuple[Path, Path]:
    from menus.astronomical.frb.fetcher import DATASETS_DIR, resolve_dataset_paths

    force_refresh = _yes(options.get("force_refresh", "no")) or _yes(
        options.get("force_rescan", "no")
    )
    restore_archived = _yes(options.get("restore_archived", "yes"))
    frb_raw = (options.get("frb_catalog") or "").strip() or None
    void_raw = (options.get("void_catalog") or "").strip() or None
    if not frb_raw and not void_raw:
        print(f"[TAV ENGINE] Auto-detecting / auto-pulling datasets in {DATASETS_DIR}/")
    return resolve_dataset_paths(
        frb_raw,
        void_raw,
        force_refresh=force_refresh,
        restore_archived=restore_archived,
    )


def _run_frb_scan(options: dict, *, plot: bool) -> None:
    _ensure_tav_resonance()
    from tav_resonance import TavResonanceConfig, run_resonance_scan
    from tav_resonance.core import load_frb_catalog, load_void_catalog

    frb_path, void_path = _resolve_frb_void_paths(options)
    frb = load_frb_catalog(frb_path)
    voids = load_void_catalog(void_path)
    from tav_shared.n_selector_registry import parse_n_points, subsample_rows

    n_menu = parse_n_points(options.get("n_points")) or parse_n_points(
        options.get("batch_limit")
    )
    if n_menu is not None and len(frb) > n_menu:
        print(f"[TAV ENGINE] Subsampling FRB catalog for tav-resonance: {len(frb)} → {n_menu}")
        frb = subsample_rows(frb, n_menu)
    cfg = TavResonanceConfig.from_defaults()
    if _yes(options.get("freeze", "yes")):
        cfg.freeze()

    result = run_resonance_scan(frb, voids, config=cfg)
    print("\n".join(result.summary_lines()))
    print(result.group_stats.to_string())

    if plot:
        output_prefix = (options.get("output_prefix") or "tav_resonance_frb").strip()
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        plot_path = ARTIFACTS_DIR / f"{output_prefix}_paths.png"
        counts = result.frame["path_type"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        counts.plot(kind="bar", ax=ax, title="FRB path mix (tav-resonance)")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"[TAV ENGINE] Path plot: {plot_path}")
        if parse_show_graphics(options, default="artifacts") == "popup":
            image = plt.imread(plot_path)
            plt.figure(figsize=(10, 6))
            plt.imshow(image)
            plt.axis("off")
            plt.title(plot_path.name)
            plt.tight_layout()
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


entry_fields = tav_resonance_entry_fields


# =============================================================================
# BLOCK: Entry form — instructions
# =============================================================================
entry_instructions = tav_resonance_entry_instructions


def run_action(selection: str, show_plots: bool = True, options: dict | None = None) -> None:
    options = options or {}
    try:
        _ensure_tav_resonance()
    except ImportError as exc:
        print("[TAV ENGINE] tav-resonance package not available.")
        print(f"[ERROR] {exc}")
        print("[TAV ENGINE] Install with:")
        print("  pip install -e ../tav-resonance")
        print("  # or: pip install tav-resonance")
        return

    if selection == "Verify Geometric Anchors":
        _run_anchor_check()
    elif selection == "Analytic Ladder vs Reference":
        _run_ladder_evaluation()
    elif selection == "Export Pre-registration Config":
        _run_prereg_export(options)
    elif selection == "FRB Resonance Scan (tav-resonance)":
        _run_frb_scan(options, plot=show_plots)
    elif selection == "Full Tav-Resonance Demo":
        _run_anchor_check()
        print()
        _run_ladder_evaluation()
        print()
        _run_prereg_export({**options, "freeze": "yes"})
        print()
        _run_frb_scan(options, plot=show_plots)
    else:
        print(f"[TAV ENGINE] Unknown tav-resonance action: {selection}")