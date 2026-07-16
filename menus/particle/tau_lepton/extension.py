"""
Tau Lepton 1/7-Mode module ↔ research_tool.py
Dedicated fork for searching the 0.142857 harmonic lattice in Tau lepton observables.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from menus.particle.tau_lepton.tau_lepton_1_7_analyzer import TauLepton17Analyzer
from menus.particle.tau_lepton.preregistered_tau_study import PreregisteredTauStudy

MODULE_TAG = "TAU_LEPTON_1_7"
SUBMENU_TITLE = "Tau Lepton 1/7 Mode"

MENU_ACTIONS = [
    "Run 1/7-Mode Smoke Test (Synthetic)",
    "Run Preregistered Analysis (Synthetic)",
    "Show Preregistration Plan",
    "Show Residue Summary",
]

def entry_fields(action: str) -> list[dict[str, Any]]:
    """Curses entry-form field specs."""
    if action in ("Run 1/7-Mode Smoke Test (Synthetic)", "Run Preregistered Analysis (Synthetic)"):
        return [
            {
                "key": "n_events",
                "label": "Number of synthetic events",
                "default": "50000",
                "required": False,
                "hint": "Size of the synthetic sample",
            },
            {
                "key": "inject_excess",
                "label": "Inject controlled 1/7 excess?",
                "default": "yes",
                "required": False,
                "hint": "yes = inject ~18% excess into residue 2 | no = pure Gaussian",
                "choices": ["yes", "no"],
            },
            {
                "key": "preferred_residue",
                "label": "Preferred residue (0-6)",
                "default": "2",
                "required": False,
                "hint": "Which residue bin to boost when inject_excess=yes",
            },
            {
                "key": "notes",
                "label": "Run notes (optional)",
                "default": "",
                "required": False,
                "hint": "Annotation for this run",
            },
        ]
    return []


def entry_instructions(action: str) -> list[str]:
    """Help bullets shown above the entry form."""
    return [
        f"Tau Lepton 1/7-Mode — {action}",
        "Searches for the 0.142857 harmonic lattice (mod-7 residues) in Tau observables.",
        "Currently uses synthetic data. Real LEP / Belle / LHCb / CMS τ loaders coming next.",
        "All engine scores are internal only and carry the mandatory label:",
        "  internal_engine_score_not_hep_significance",
        "Preregistration is frozen in PREREGISTRATION_TAU_LEPTON_1_7.md",
    ]


def run_action(
    selection: str,
    show_plots: bool = True,
    options: dict | None = None,
) -> None:
    """Execute the selected menu action."""
    options = options or {}
    print(f"[TAV ENGINE] {SUBMENU_TITLE} — {selection}")

    if selection == "Show Preregistration Plan":
        study = PreregisteredTauStudy()
        print(study.to_markdown())
        return

    # Common synthetic generator
    n_events = int(options.get("n_events") or 50_000)
    inject = (options.get("inject_excess") or "yes").lower().startswith("y")
    try:
        preferred = int(options.get("preferred_residue") or 2) % 7
    except ValueError:
        preferred = 2

    rng = np.random.default_rng(42)
    base = rng.normal(loc=1.777, scale=0.12, size=n_events)

    if inject:
        n_signal = int(0.18 * n_events)
        signal = preferred + 7 * rng.uniform(0.1, 0.9, size=n_signal)
        base[:n_signal] = signal

    analyzer = TauLepton17Analyzer()
    results = analyzer.analyze(base, label=f"synthetic_{selection.replace(' ', '_')}")

    print(analyzer.summary())
    print("\n--- Residue fractions ---")
    for i, f in enumerate(results["residue_mod7"]["fractions"]):
        print(f"  residue {i}: {f:.4f}")

    if selection == "Show Residue Summary":
        print("\n[TAV ENGINE] Residue summary complete.")
