#!/usr/bin/env python3
"""
enhanced_tav_bbn_confrontation.py
Improved minimal BBN engine + plots + JSON analysis for Tav framework.

Project root: Public/TauSuperblock/
  enhanced_tav_bbn_confrontation.py       — this script
  menus/prime_past/bbn_enhanced.py        — core module
  artifacts/prime_past/                   — heatmap PNG + JSON

Usage:
  ./venv/bin/python enhanced_tav_bbn_confrontation.py

Also: research_tool.py → Prime Past Harmonic → Enhanced BBN Confrontation
"""

from __future__ import annotations

from menus.prime_past.bbn_enhanced import (
    confront,
    fetch_empirical_data,
    load_and_analyze_scan,
    plot_lithium_tension_heatmap,
    run_enhanced_scan,
    run_improved_bbn,
    save_enhanced_scan,
)

__all__ = [
    "fetch_empirical_data",
    "run_improved_bbn",
    "confront",
    "plot_lithium_tension_heatmap",
    "load_and_analyze_scan",
    "run_enhanced_scan",
    "save_enhanced_scan",
]

# ============================================================
# Scan configuration — edit here
# ============================================================
A_POINTS = 8
PHI_POINTS = 5
DELTA_K_WIND = 0.469
T_SPAN = (0.0, 400.0)

# Set to a path under artifacts/prime_past/ to re-analyze a prior JSON, or None.
ANALYZE_JSON = None
# ANALYZE_JSON = "artifacts/prime_past/tav_bbn_scan_20260701_204517.json"


if __name__ == "__main__":
    if ANALYZE_JSON:
        load_and_analyze_scan(ANALYZE_JSON)
    else:
        empirical = fetch_empirical_data()
        scan_results = run_enhanced_scan(
            a_points=A_POINTS,
            phi_points=PHI_POINTS,
            delta_k_wind=DELTA_K_WIND,
            t_span=T_SPAN,
        )
        heatmap = plot_lithium_tension_heatmap(scan_results)
        save_enhanced_scan(scan_results, heatmap_path=heatmap)
        print("\nScan complete. Heatmap generated.")