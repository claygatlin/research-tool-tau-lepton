"""
Tau-Superblock Research Engine ↔ research_tool.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from menus.tsb_research.core import (
    ARTIFACTS_DIR,
    HAS_SYMPY,
    LAMBDA_TSB,
    N_HIER,
    run_calibrate_sound_horizon,
    run_evolve_cylinder_state,
    run_generate_mocks,
    run_residual_diagnostics_demo,
)

MODULE_TAG = "TSB_RESEARCH"
DOMAIN_TITLE = "Tau-Superblock Research Engine"
SUBMENU_TITLE = DOMAIN_TITLE

REPO_ACTIONS = [
    "Generate Mocks",
    "Generate Full DESI Summary Dashboard",
    "Evolve Cylinder State",
    "Run Residual Diagnostics",
    "Calibrate SoundHorizon",
]
MENU_ACTIONS = REPO_ACTIONS


def is_module_selection(repo: str | None) -> bool:
    return repo == MODULE_TAG


def _save_report(name: str, payload: dict[str, Any]) -> str:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = ARTIFACTS_DIR / f"tsb_research_{name}_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return str(path)


def _float_option(options: dict[str, Any], key: str, default: float) -> float:
    raw = options.get(key)
    if raw is None or str(raw).strip() == "":
        return float(default)
    return float(raw)


def _int_option(options: dict[str, Any], key: str, default: int) -> int:
    raw = options.get(key)
    if raw is None or str(raw).strip() == "":
        return int(default)
    return int(float(raw))


def _bool_option(options: dict[str, Any], key: str, default: bool = False) -> bool:
    raw = options.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


# =============================================================================
# BLOCK: Entry form — fields
# =============================================================================
def entry_fields(action: str) -> list[dict]:
    """Curses entry-form field specs for Tau-Superblock Research Engine actions."""
    if action == "Generate Full DESI Summary Dashboard":
        from tav_shared.n_selector_registry import n_points_field

        return [n_points_field("tsb_research_extension", action)]
    if action == "Evolve Cylinder State":
        return [
            {
                "key": "clockwork_steps",
                "label": "Clockwork steps",
                "default": "2",
                "required": False,
                "hint": "Octonionic phase advance steps",
            },
            {
                "key": "s_test",
                "label": "Cylinder s",
                "default": "5.0",
                "required": False,
                "hint": "Axial coordinate on ℝ_s",
            },
            {
                "key": "phi_test",
                "label": "Phase ϕ",
                "default": "0.0",
                "required": False,
                "hint": "Starting phase on S¹_ϕ",
            },
            {
                "key": "gamma",
                "label": "γ for r_d",
                "default": "8.0",
                "required": False,
                "hint": "SoundHorizon calibration γ",
            },
        ]
    if action == "Run Residual Diagnostics":
        return [
            {
                "key": "use_desi_residuals",
                "label": "Use DESI fit residuals",
                "default": "yes",
                "required": False,
                "hint": "yes = DR2 Cobaya data; no = cylinder evolution",
                "choices": ["yes", "no"],
            },
            {
                "key": "auto_calibrate_gamma",
                "label": "Auto-calibrate γ",
                "default": "yes",
                "required": False,
                "hint": "DESI mode only",
                "choices": ["yes", "no"],
            },
        ]
    if action == "Calibrate SoundHorizon":
        return [
            {
                "key": "s_ref",
                "label": "s_ref for twistor pressure",
                "default": "0.0",
                "required": False,
                "hint": "Axial reference for Φ-anchored r_d",
            },
        ]
    return []


# =============================================================================
# BLOCK: Entry form — instructions
# =============================================================================
def entry_instructions(action: str) -> list[str]:
    """Short help bullets shown above the TSB research entry form."""
    instructions = {
        "Generate Mocks": [
            "Cylinder template + SoundHorizon + clockwork + operator algebra.",
            "Interactive n selector (6…50 or Custom) in the curses menu.",
            "Saved under artifacts/mocks/ with residual diagnostics.",
        ],
        "Generate Full DESI Summary Dashboard": [
            "ACF, Residuals vs Fitted, Q-Q, SoundHorizon γ-curve, summary panel.",
            "Uses live DESI DR2 fit residuals when Cobaya tables are cached.",
            "Saved under artifacts/desi_dashboard/ with JSON report.",
        ],
        "Evolve Cylinder State": [
            "Octonionic clockwork evolution with twistor + SoundHorizon + exclusion.",
            "s_test / phi_test / clockwork_steps tune the evolution step.",
        ],
        "Run Residual Diagnostics": [
            "Lag ACF + Breusch–Pagan on fit residuals.",
            "use_desi_residuals=yes (default): genuine DESI DR2 Tau-SB fit residuals.",
        ],
        "Calibrate SoundHorizon": [
            "Φ-anchored r_d calibration across γ = 7.95, 8.0, 8.8511, 12.0.",
            "Reports γ-swing reduction vs legacy compute_tsb_rd.",
        ],
    }
    return instructions.get(
        action,
        [
            "Tau-Superblock Research Engine: Φ, clockwork, operator algebra, SoundHorizon.",
            "Reports: artifacts/tsb_research_*_*.json",
        ],
    )


# =============================================================================
# BLOCK: Pre-form hook (mock generator n selector)
# =============================================================================
def handle_pre_form(stdscr, selection: str, params: dict[str, str]) -> str | None:
    """Interactive n picker + mock run for Generate Mocks (skips entry form)."""
    if selection != "Generate Mocks":
        return None

    from menus.tsb_research.mock_generator import run_mock_generator
    from tav_research.curses_shell import select_n_interactive

    n = select_n_interactive(
        stdscr,
        prompt="Choose n for mock generation",
    )
    if n is not None:
        params["n_points"] = str(n)
        params["batch_limit"] = str(n)
        run_mock_generator(stdscr, n=n)
    return "continue"


def run_action(selection: str, show_plots: bool = True, options: dict | None = None) -> None:
    _ = show_plots
    options = options or {}

    print(f"\n[TAV ENGINE] {DOMAIN_TITLE} — {selection}")
    print(f"[TAV ENGINE] Λ_TSB = {LAMBDA_TSB} MeV | n_hier = {N_HIER} | sympy = {HAS_SYMPY}")

    if selection == "Generate Mocks":
        report = run_generate_mocks(
            n=_int_option(
                options,
                "n_points",
                _int_option(options, "n", _int_option(options, "n_mocks", 25)),
            ),
            gamma=_float_option(options, "gamma", 9.5),
            verbose=True,
        )
        path = _save_report("generate_mocks", report)
        print(f"[TAV ENGINE] Report saved: {path}")
        return

    if selection == "Generate Full DESI Summary Dashboard":
        from menus.astronomical.desi.dashboard import run_full_desi_dashboard

        report = run_full_desi_dashboard(
            n=_int_option(options, "n_points", _int_option(options, "n", 25)),
        )
        n_req = report.get("n_requested")
        n_used = report.get("n_residuals")
        if n_req is not None and int(n_req) != int(n_used):
            print(
                f"[TAV ENGINE] Dashboard: {report['data_class']} | "
                f"n_requested={n_req} → n_residuals={n_used} (resampled for plots) | "
                f"report={report['report_path']}"
            )
        else:
            print(
                f"[TAV ENGINE] Dashboard: {report['data_class']} | "
                f"n={n_used} | report={report['report_path']}"
            )
        return

    if selection == "Evolve Cylinder State":
        report = run_evolve_cylinder_state(
            steps=_int_option(options, "clockwork_steps", 2),
            s=_float_option(options, "s_test", 5.0),
            phi=_float_option(options, "phi_test", 0.0),
            gamma=_float_option(options, "gamma", 8.0),
            apply_twistor=_bool_option(options, "apply_twistor", True),
            update_sound_horizon=_bool_option(options, "update_sound_horizon", True),
            verbose=True,
        )
        path = _save_report("evolve_cylinder_state", report)
        print(f"[TAV ENGINE] Report saved: {path}")
        return

    if selection == "Run Residual Diagnostics":
        use_desi = _bool_option(options, "use_desi_residuals", True)
        report = run_residual_diagnostics_demo(
            source="desi_fit" if use_desi else "cylinder_evolution",
            data_mode=str(options.get("data_mode", "")).strip() or None,
            tracer=str(options.get("tracer", "")).strip() or None,
            auto_calibrate_gamma=_bool_option(options, "auto_calibrate_gamma", True),
            use_whim_model=_bool_option(options, "use_whim_model", False),
            verbose=True,
        )
        path = _save_report("residual_diagnostics", report)
        print(f"[TAV ENGINE] Report saved: {path}")
        return

    if selection == "Calibrate SoundHorizon":
        report = run_calibrate_sound_horizon(
            s_ref=_float_option(options, "s_ref", 0.0),
            verbose=True,
        )
        path = _save_report("calibrate_sound_horizon", report)
        print(f"[TAV ENGINE] Report saved: {path}")
        return

    print(f"[TAV ENGINE] Unknown TSB research action: {selection}")