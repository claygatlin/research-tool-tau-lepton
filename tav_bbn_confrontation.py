#!/usr/bin/env python3
"""
tav_bbn_confrontation.py — Complete Tav Framework BBN Scanner + Empirical Confrontation

Framework: Tav Topology / Tau Universe merged with Superblock.
Date: 2026-07-01

Related files (Public/TauSuperblock/):
  tav_bbn_confrontation.py              — this script
  menus/prime_past/bbn_confrontation.py — core module
  artifacts/prime_past/                 — JSON + PNG output

Usage:
  ./venv/bin/python tav_bbn_confrontation.py

Also: research_tool.py → Prime Past Harmonic → BBN Confrontation Scan
"""

from __future__ import annotations

import numpy as np

from menus.prime_past.bbn_confrontation import (
    confront_with_data,
    default_scan_ranges,
    fetch_empirical_data,
    plot_scan_summary,
    print_summary_table,
    run_minimal_bbn,
    run_tav_scan,
    save_scan_report,
)

__all__ = [
    "fetch_empirical_data",
    "run_minimal_bbn",
    "confront_with_data",
    "default_scan_ranges",
    "run_tav_scan",
    "save_scan_report",
    "print_summary_table",
    "plot_scan_summary",
]

# ============================================================
# Scan configuration — edit here
# ============================================================
N_POINTS = 4
T_SPAN = (0.0, 300.0)
SAVE_PLOT = True
SHOW_PLOT = False

# Set SCAN_RANGES = None to use default grid from N_POINTS.
# Example custom grid:
# SCAN_RANGES = {
#     "A": np.linspace(0.0, 0.05, 6),
#     "delta_phi_cyl": [0.0, np.pi / 4, np.pi / 2],
#     "delta_k_wind": [0.0, 0.469],
# }
SCAN_RANGES = None


if __name__ == "__main__":
    print("Running Tav BBN Confrontation Scanner...")
    ranges = SCAN_RANGES if SCAN_RANGES is not None else default_scan_ranges(N_POINTS)
    results_list, best = run_tav_scan(
        scan_ranges=ranges,
        n_points=N_POINTS,
        t_span=T_SPAN,
        plot=SAVE_PLOT,
        show_plot=SHOW_PLOT,
    )
    save_scan_report(results_list, best)