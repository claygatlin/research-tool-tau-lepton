"""
RGC Resonance Mock Catalog ↔ research_tool Particle menu.

Generates unbinned 1/7 + Δn = 1/14 ladder mocks for peak/graph/ML validation.
Cosmological DESI mocks remain under Astronomical → Tau-SB DESI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from menus.particle.rgc.mock_catalog import (
    DEFAULT_NUM_EVENTS,
    DEFAULT_NUM_DELTA_STEPS,
    DEFAULT_SEED,
    DEFAULT_SIG_FRAC,
    DEFAULT_SIGMA,
    STEP_AMPLITUDE_DECAY,
    MockCatalogConfig,
    run_mock_catalog_pipeline,
)
from tav_shared.run_output import parse_show_graphics

MODULE_TAG = "RGC_MOCK_CATALOG"
SUBMENU_TITLE = "RGC Resonance Mock Catalog"

MENU_ACTIONS = [
    "Generate Mock Catalog (1/7 + Δn)",
    "Quick Mock (10k events)",
    "Stress Test (broad σ, overlapping peaks)",
]

_ACTION_DEFAULTS: dict[str, dict[str, Any]] = {
    "Generate Mock Catalog (1/7 + Δn)": {
        "num_events": DEFAULT_NUM_EVENTS,
        "sigma": DEFAULT_SIGMA,
        "quick": False,
    },
    "Quick Mock (10k events)": {
        "num_events": 10_000,
        "sigma": DEFAULT_SIGMA,
        "quick": True,
    },
    "Stress Test (broad σ, overlapping peaks)": {
        "num_events": DEFAULT_NUM_EVENTS,
        "sigma": 0.04,
        "quick": False,
    },
}


def is_module_selection(repo: str | None) -> bool:
    return repo == MODULE_TAG


def entry_fields(action: str) -> list[dict[str, Any]]:
    defaults = _ACTION_DEFAULTS.get(action, _ACTION_DEFAULTS["Generate Mock Catalog (1/7 + Δn)"])
    return [
        {
            "key": "seed",
            "label": "Random seed",
            "default": str(DEFAULT_SEED),
            "required": False,
            "hint": "Reproducible catalog generation",
        },
        {
            "key": "num_events",
            "label": "Total events",
            "default": str(defaults["num_events"]),
            "required": False,
            "hint": "Background + signal; signal fraction set below",
        },
        {
            "key": "sig_frac",
            "label": "Signal fraction",
            "default": str(DEFAULT_SIG_FRAC),
            "required": False,
            "hint": "Injected purity f_sig (default 0.10)",
        },
        {
            "key": "sigma",
            "label": "Gaussian σ",
            "default": str(defaults["sigma"]),
            "required": False,
            "hint": "Resolution + intrinsic width; stress action uses 0.04",
        },
        {
            "key": "num_delta_steps",
            "label": "Ladder half-width N_steps",
            "default": str(DEFAULT_NUM_DELTA_STEPS),
            "required": False,
            "hint": "k ∈ [-N, +N]; μ_k = 1/7 + k·(1/14)",
        },
        {
            "key": "step_decay",
            "label": "Satellite amplitude decay r",
            "default": str(STEP_AMPLITUDE_DECAY),
            "required": False,
            "hint": "w_k ∝ r^|k| for |k|≥1; primary k=0 weight normalized",
        },
        {
            "key": "output_prefix",
            "label": "Artifact prefix",
            "default": "mock_resonance_catalog_1over7_deltan",
            "required": False,
            "hint": "Writes under artifacts/rgc/",
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


def entry_instructions(action: str) -> list[str]:
    return [
        f"RGC particle mock — {action}",
        "Mixture: uniform background + Gaussian ladder at μ = 1/7 + k·(1/14).",
        "Truth labels: component_k, type (background/signal), resonance_center.",
        "Outputs: CSV (unbinned), PNG diagnostic, stats TXT, JSON summary → artifacts/rgc/.",
        "For DESI BAO mocks use Astronomical → Tau-SB DESI → Mock Catalog Recovery.",
    ]


def prepare_run_options(action: str, raw: dict[str, str]) -> dict[str, str]:
    merged = dict(raw)
    merged["action"] = action
    defaults = _ACTION_DEFAULTS.get(action, {})
    if defaults.get("quick"):
        merged.setdefault("num_events", "10000")
    if "sigma" in defaults and "sigma" not in merged:
        merged["sigma"] = str(defaults["sigma"])
    return merged


def _parse_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value else default
    except ValueError:
        return default


def _parse_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value else default
    except ValueError:
        return default


def run_action(
    selection: str,
    show_plots: bool = True,
    options: dict | None = None,
) -> str | None:
    if selection not in MENU_ACTIONS:
        print(f"[TAV ENGINE] Unknown RGC action: {selection}")
        return None

    options = options or {}
    defaults = _ACTION_DEFAULTS.get(selection, {})

    cfg = MockCatalogConfig(
        seed=_parse_int(options.get("seed"), DEFAULT_SEED),
        num_events=_parse_int(
            options.get("num_events"), int(defaults.get("num_events", DEFAULT_NUM_EVENTS))
        ),
        sig_frac=_parse_float(options.get("sig_frac"), DEFAULT_SIG_FRAC),
        sigma=_parse_float(options.get("sigma"), float(defaults.get("sigma", DEFAULT_SIGMA))),
        num_delta_steps=_parse_int(options.get("num_delta_steps"), DEFAULT_NUM_DELTA_STEPS),
        step_amplitude_decay=_parse_float(options.get("step_decay"), STEP_AMPLITUDE_DECAY),
        output_prefix=(options.get("output_prefix") or "mock_resonance_catalog_1over7_deltan").strip(),
    )

    show_popup = parse_show_graphics(options, default="artifacts") if show_plots else False
    report = run_mock_catalog_pipeline(cfg, plot=True, show_popup=show_popup)
    return report.get("summary_path")