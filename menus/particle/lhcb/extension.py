"""
LHCb Tav-Echo module ↔ research_tool.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from menus.particle.lhcb.echo import DEFAULT_DATA_PATH, run_pipeline
from tav_shared.llm_analysis import analyze_run_output as query_llm_analysis
from tav_shared.run_output import parse_show_graphics

MODULE_TAG = "LHCB_TAV_ECHO"
SUBMENU_TITLE = "LHCb Tav-Echo"

MENU_ACTIONS = [
    "Run Tav-Echo Correlation",
    "Plot Residuals vs Kernel",
    "Full Analysis (Stats + Plot)",
]

_ACTION_CONFIG = {
    "Run Tav-Echo Correlation": {"plot": False},
    "Plot Residuals vs Kernel": {"plot": True},
    "Full Analysis (Stats + Plot)": {"plot": True},
}


def is_module_selection(repo: str | None) -> bool:
    return repo == MODULE_TAG


# =============================================================================
# BLOCK: Entry form — fields
# =============================================================================
def entry_fields(action: str) -> list[dict]:
    """Curses entry-form field specs for LHCb Tav-Echo actions."""
    return [
        {
            "key": "data_file",
            "label": "LHCb data file (.npy)",
            "default": str(DEFAULT_DATA_PATH),
            "required": False,
            "hint": "Structured array: q2_bin, C9_exp, C9_exp_err, C9_SM",
        },
        {
            "key": "mass_gap",
            "label": "Mass-gap anchor m₀ (MeV)",
            "default": "313.1",
            "required": False,
            "hint": "Resonance kernel uses 1/(m₀² + q²)",
        },
        {
            "key": "output_name",
            "label": "Plot filename (optional)",
            "default": "lhcb_tav_echo_correlation.png",
            "required": False,
            "hint": "Saved under artifacts/ for plot actions",
        },
        {
            "key": "show_graphics",
            "label": "Graphics mode",
            "default": "artifacts",
            "required": False,
            "hint": "popup = Tk window | artifacts = PNG only",
            "choices": ["artifacts", "popup"],
        },
    ]


# =============================================================================
# BLOCK: Entry form — instructions
# =============================================================================
def entry_instructions(action: str) -> list[str]:
    """Short help bullets shown above the LHCb Tav-Echo entry form."""
    return [
        "Loads structured NumPy file: q2_bin, C9_exp, C9_exp_err, C9_SM.",
        f"Default path: {DEFAULT_DATA_PATH}",
        "Correlates (C9_exp - C9_SM) with 1/(m₀² + q²); m₀ defaults to 313.1 MeV.",
        "Plot actions save PNG under artifacts/; popup opens Tk window after curses exits.",
        "Text output saved to artifacts/run_*.txt automatically.",
    ]


def run_action(selection: str, show_plots: bool = True, options: dict | None = None) -> None:
    options = options or {}
    config = _ACTION_CONFIG.get(selection)
    if config is None:
        print(f"[TAV ENGINE] Unknown LHCb Tav-Echo action: {selection}")
        return

    data_path = (options.get("data_file") or str(DEFAULT_DATA_PATH)).strip()
    try:
        mass_gap = float(options.get("mass_gap") or "313.1")
    except ValueError:
        mass_gap = 313.1

    output_name = (options.get("output_name") or "lhcb_tav_echo_correlation.png").strip()
    if not output_name.endswith(".png"):
        output_name += ".png"

    show_popup = parse_show_graphics(options, default="artifacts") if show_plots else False

    run_pipeline(
        data_path=Path(data_path),
        mass_gap_mev=mass_gap,
        plot=config["plot"],
        show_plot=show_popup,
        output_name=output_name,
    )

    if show_popup and config["plot"]:
        plt.show()