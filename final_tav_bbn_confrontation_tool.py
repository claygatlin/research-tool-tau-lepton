#!/usr/bin/env python3
"""
final_tav_bbn_confrontation_tool.py
Full Tav BBN scanner with improved reaction network, best-config JSON,
standard vs best comparison plots, and research_tool hooks.

Project root: Public/TauSuperblock/
  final_tav_bbn_confrontation_tool.py  — this script
  menus/prime_past/bbn_enhanced.py      — core engine
  artifacts/prime_past/                — best_tav_bbn_config.json, plots, scans

Usage:
  ./venv/bin/python final_tav_bbn_confrontation_tool.py

Also: research_tool.py → Prime Past Harmonic → Final BBN Confrontation Tool
"""

from __future__ import annotations

from menus.prime_past.bbn_enhanced import (
    BEST_CONFIG_PATH,
    COMPARISON_PLOT_PATH,
    confront,
    fetch_empirical_data,
    load_and_highlight_best,
    plot_comparison,
    run_final_confrontation_tool,
    run_improved_bbn,
    save_best_config,
)

__all__ = [
    "fetch_empirical_data",
    "run_improved_bbn",
    "confront",
    "save_best_config",
    "plot_comparison",
    "load_and_highlight_best",
    "run_final_confrontation_tool",
    "BEST_CONFIG_PATH",
    "COMPARISON_PLOT_PATH",
]

# ============================================================
# Scan configuration — edit here
# ============================================================
A_POINTS = 8
PHI_POINTS = 5
DELTA_K_WIND = 0.469

# Set to analyze a prior scan JSON instead of running a new scan.
HIGHLIGHT_JSON = None
# HIGHLIGHT_JSON = "artifacts/prime_past/final_tav_bbn_scan_20260701_205157.json"


if __name__ == "__main__":
    if HIGHLIGHT_JSON:
        load_and_highlight_best(HIGHLIGHT_JSON)
    else:
        run_final_confrontation_tool(
            a_points=A_POINTS,
            phi_points=PHI_POINTS,
            delta_k_wind=DELTA_K_WIND,
        )