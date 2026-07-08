"""
Modular bridge: Tav BBN ↔ research_tool.py

Menu:
  Tav BBN
  ├── Run Improved Scan
  ├── Analyze Best Config
  ├── Compare Standard vs Tav
  ├── High-Resolution Scan around Best Config
  └── Settings (Neutron Lifetime, etc.)
"""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt

from menus.prime_past.bbn_menu import (
    BEST_CONFIG_PATH,
    analyze_best_config,
    compare_standard_vs_tav,
    run_high_resolution_around_best,
    run_improved_scan,
    settings_menu,
)
from tav_shared.llm_analysis import analyze_run_output as query_llm_analysis

SUBMENU_TITLE = "Tav BBN"
MODULE_TAG = "TAV_BBN"

MENU_ACTIONS = [
    "Run Improved Scan",
    "Analyze Best Config",
    "Compare Standard vs Tav",
    "High-Resolution Scan around Best Config",
    "Settings (Neutron Lifetime, etc.)",
]

_ACTION_KEYS: Dict[str, str] = {
    "Run Improved Scan": "run_scan",
    "Analyze Best Config": "analyze_best",
    "Compare Standard vs Tav": "compare",
    "High-Resolution Scan around Best Config": "hires_scan",
    "Settings (Neutron Lifetime, etc.)": "settings",
}


def is_module_selection(repo: str | None) -> bool:
    return repo == SUBMENU_TITLE or repo == MODULE_TAG


# =============================================================================
# BLOCK: Entry form — fields
# =============================================================================
def entry_fields(action: str) -> list[dict]:
    tau_field = {
        "key": "neutron_lifetime",
        "label": "Neutron lifetime (s)",
        "default": "879.4",
        "required": False,
        "hint": "Weak-rate timescale; PDG central ~879.4 s",
    }
    if action == "Run Improved Scan":
        return [
            tau_field,
            {"key": "a_points", "label": "A grid points", "default": "9", "required": False},
            {"key": "phi_points", "label": "Phase grid points", "default": "6", "required": False},
            {"key": "delta_k_wind", "label": "Winding contribution", "default": "0.469", "required": False},
        ]
    if action == "Analyze Best Config":
        return [
            {
                "key": "config_json",
                "label": "Best config JSON path (optional)",
                "default": str(BEST_CONFIG_PATH),
                "required": False,
            },
        ]
    if action == "Compare Standard vs Tav":
        return [tau_field]
    if action == "High-Resolution Scan around Best Config":
        return [
            tau_field,
            {"key": "config_json", "label": "Seed config JSON", "default": str(BEST_CONFIG_PATH), "required": False},
            {"key": "grid", "label": "Grid points per axis", "default": "7", "required": False},
        ]
    if action == "Settings (Neutron Lifetime, etc.)":
        return [tau_field]
    return []


def entry_instructions(action: str) -> list[str]:
    base = [
        "Tav BBN menu — improved reaction network with adjustable τ_n.",
        "Artifacts: Public/TauSuperblock/artifacts/prime_past/",
        "Script: Public/TauSuperblock/tav_bbn_menu_module.py",
    ]
    if action == "Run Improved Scan":
        base.append("Saves best_tav_bbn_config.json + full scan JSON.")
    if action == "High-Resolution Scan around Best Config":
        base.append("Requires prior best config; refines A and phase locally.")
    return base


def run_action(
    selection: str,
    show_plots: bool = True,
    options: dict | None = None,
) -> str | None:
    options = options or {}
    action = _ACTION_KEYS.get(selection)
    if action is None:
        print(f"[TAV ENGINE] Unknown Tav BBN action: {selection}")
        return None

    print(f"\n[TAV ENGINE] Tav BBN — {selection}")
    tau = float(options.get("neutron_lifetime") or 879.4)

    if action == "run_scan":
        out = run_improved_scan(
            a_points=int(options.get("a_points") or 9),
            phi_points=int(options.get("phi_points") or 6),
            neutron_lifetime=tau,
            delta_k_wind=float(options.get("delta_k_wind") or 0.469),
        )
        return out.get("scan_path")

    if action == "analyze_best":
        path = (options.get("config_json") or "").strip() or None
        analyze_best_config(path)
        return path or str(BEST_CONFIG_PATH)

    if action == "compare":
        compare_standard_vs_tav(neutron_lifetime=tau, show_plot=show_plots)
        return str(BEST_CONFIG_PATH)

    if action == "hires_scan":
        path = (options.get("config_json") or "").strip() or None
        out = run_high_resolution_around_best(
            json_path=path,
            grid=int(options.get("grid") or 7),
            neutron_lifetime=tau,
        )
        return out.get("scan_path")

    if action == "settings":
        settings_menu(neutron_lifetime=tau)
        return None

    return None