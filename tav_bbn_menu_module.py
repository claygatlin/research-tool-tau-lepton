#!/usr/bin/env python3
"""
tav_bbn_menu_module.py
Tav BBN module for menu integration in research_tool.py

Menu:
  Tav BBN
  ├── Run Improved Scan
  ├── Analyze Best Config
  ├── Compare Standard vs Tav
  ├── High-Resolution Scan around Best Config
  └── Settings (Neutron Lifetime, etc.)

Project root: Public/TauSuperblock/
"""

from menus.prime_past.bbn_menu import (
    analyze_best_config,
    compare_standard_vs_tav,
    get_empirical_data,
    run_high_resolution_around_best,
    run_improved_bbn,
    run_improved_scan,
    settings_menu,
)

__all__ = [
    "get_empirical_data",
    "run_improved_bbn",
    "run_improved_scan",
    "analyze_best_config",
    "compare_standard_vs_tav",
    "run_high_resolution_around_best",
    "settings_menu",
]


if __name__ == "__main__":
    print("Tav BBN Menu Module - Test Run")
    run_improved_scan(a_points=4, phi_points=3)