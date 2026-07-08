"""
Planck CMB Tav-resonance module ↔ research_tool.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt

from menus.planck_cmb.tav import ARTIFACTS_DIR, FITS_DIR, run_pipeline
from tav_shared.llm_analysis import analyze_run_output as query_llm_analysis
from tav_shared.run_output import parse_show_graphics

MODULE_TAG = "PLANCK_CMB_TAV"
SUBMENU_TITLE = "Planck CMB Tav-Scan"

MENU_ACTIONS = [
    "Compute Power Spectrum",
    "Scan 1/7 Tav Harmonics",
    "Full CMB Tav Analysis",
]

_ACTION_CONFIG = {
    "Compute Power Spectrum": {"plot": True, "analyze_only": False},
    "Scan 1/7 Tav Harmonics": {"plot": True, "analyze_only": True},
    "Full CMB Tav Analysis": {"plot": True, "analyze_only": False},
}


def is_module_selection(repo: str | None) -> bool:
    return repo == MODULE_TAG


def _show_saved_plots(output_prefix: str) -> None:
    """Open the PNG artifacts in Tk windows (same pattern as other modules)."""
    plot_paths = [
        ARTIFACTS_DIR / f"{output_prefix}_spectrum.png",
        ARTIFACTS_DIR / f"{output_prefix}_fft.png",
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
def entry_fields(action: str) -> list[dict]:
    """Curses entry-form field specs for Planck CMB Tav-Scan actions."""
    return [
        {
            "key": "cmb_map",
            "label": "CMB map FITS (optional)",
            "default": "",
            "required": False,
            "hint": f"Blank = auto-detect in {FITS_DIR}/ (any .fits map)",
        },
        {
            "key": "cmb_mask",
            "label": "CMB mask FITS (optional)",
            "default": "",
            "required": False,
            "hint": "Blank = auto-detect (filename with 'mask' preferred)",
        },
        {
            "key": "lmax",
            "label": "lmax (optional)",
            "default": "2000",
            "required": False,
            "hint": "Maximum multipole for anafast",
        },
        {
            "key": "output_prefix",
            "label": "Plot filename prefix",
            "default": "planck_cmb_tav",
            "required": False,
            "hint": "Base name; FITS batch tag appended (e.g. planck_cmb_tav_hm2)",
        },
        {
            "key": "show_graphics",
            "label": "Graphics mode",
            "default": "popup",
            "required": False,
            "hint": "popup = Tk windows | artifacts = save PNG only",
            "choices": ["artifacts", "popup"],
        },
    ]


# =============================================================================
# BLOCK: Entry form — instructions
# =============================================================================
def entry_instructions(action: str) -> list[str]:
    """Short help bullets shown above the Planck CMB entry form."""
    from menus.planck_cmb.tav import EXAMPLE_CMB_MAP, EXAMPLE_CMB_MASK
    from tav_shared.tav_project_paths import FINISHED_DIR

    return [
        "Any compatible CMB map + mask .fits files (names not fixed).",
        f"Drop files in {FITS_DIR}/ — blank form fields auto-detect them.",
        "Mask auto-pick: filename contains 'mask', or wget Planck common mask for the map.",
        f"Example map: {EXAMPLE_CMB_MAP} | example mask: {EXAMPLE_CMB_MASK}",
        "Or set explicit paths in the form for non-standard locations.",
        f"After a successful run, files from ./fits/ move to {FINISHED_DIR}/.",
        "Plots: artifacts/planck_cmb_tav_spectrum.png and _fft.png",
    ]


def run_action(selection: str, show_plots: bool = True, options: dict | None = None) -> None:
    options = options or {}
    config = _ACTION_CONFIG.get(selection)
    if config is None:
        print(f"[TAV ENGINE] Unknown Planck CMB action: {selection}")
        return

    cmb_path = (options.get("cmb_map") or "").strip()
    mask_path = (options.get("cmb_mask") or "").strip()

    try:
        lmax = int(options.get("lmax") or "2000")
    except ValueError:
        lmax = 2000

    output_prefix = (options.get("output_prefix") or "planck_cmb_tav").strip()
    show_popup = parse_show_graphics(options, default="popup") if show_plots else False

    if not cmb_path and not mask_path:
        print(f"[TAV ENGINE] Auto-detecting .fits files in {FITS_DIR}/")

    result = run_pipeline(
        cmb_path=Path(cmb_path) if cmb_path else None,
        mask_path=Path(mask_path) if mask_path else None,
        lmax=lmax,
        plot=config["plot"],
        show_plot=False,
        output_prefix=output_prefix,
    )

    artifact_prefix = result.output_prefix
    if config["plot"] and show_popup:
        _show_saved_plots(artifact_prefix)
    elif config["plot"]:
        print(
            "[TAV ENGINE] Graphics mode is artifacts — PNGs saved under artifacts/. "
            "Set show_graphics=popup to open plot windows."
        )
        print(f"[TAV ENGINE] Spectrum: {ARTIFACTS_DIR / f'{artifact_prefix}_spectrum.png'}")
        print(f"[TAV ENGINE] FFT:      {ARTIFACTS_DIR / f'{artifact_prefix}_fft.png'}")