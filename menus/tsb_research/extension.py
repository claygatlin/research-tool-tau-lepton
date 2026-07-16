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
    "Fractal Tau Circle Likelihood",
    "Run Falsification Suite (Methods 1,4,9)",
]
MENU_ACTIONS = REPO_ACTIONS


def is_module_selection(repo: str | None) -> bool:
    return repo == MODULE_TAG


def _save_report(name: str, payload: dict[str, Any]) -> str:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    from tav_shared.artifact_paths import TestSlug, artifact_path, compose_dataset_slug

    path = artifact_path(TestSlug.TSB_RESEARCH, compose_dataset_slug(name), "report", "json")
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
    if action == "Fractal Tau Circle Likelihood":
        return [
            {
                "key": "winding_density",
                "label": "Winding density ρ_wind",
                "default": "1.0",
                "required": False,
            },
            {
                "key": "fractal_level",
                "label": "Fractal level (n_hier scale)",
                "default": "45.8",
                "required": False,
            },
            {
                "key": "phase_slip_alpha",
                "label": "Phase slip α",
                "default": "0.33",
                "required": False,
            },
            {
                "key": "observed_mass_gap_mev",
                "label": "Observed mass gap (MeV)",
                "default": "313.1",
                "required": False,
                "hint": "313.1 MeV TEP geometric friction floor",
            },
            {
                "key": "observed_spectral_dim",
                "label": "Target spectral dim (CDT UV)",
                "default": "2.0",
                "required": False,
            },
            {
                "key": "likelihood_mode",
                "label": "Likelihood mode",
                "default": "dual",
                "required": False,
                "choices": ["dual", "tep_ansatz"],
                "hint": "tep_ansatz = torsion Dirac + TEP hard constraint",
            },
            {
                "key": "sigma_qcd_mev",
                "label": "σ_qcd (MeV, tep_ansatz)",
                "default": "5.0",
                "required": False,
            },
            {
                "key": "do_grid_scan",
                "label": "Run parameter grid scan",
                "default": "yes",
                "required": False,
                "choices": ["yes", "no"],
            },
            {
                "key": "do_mcmc",
                "label": "Run TEP-hard MCMC (emcee)",
                "default": "no",
                "required": False,
                "choices": ["yes", "no"],
                "hint": "Morris sensitivity → freeze low-impact params; TEP in log_prior",
            },
            {
                "key": "mcmc_walkers",
                "label": "MCMC walkers",
                "default": "32",
                "required": False,
            },
            {
                "key": "mcmc_steps",
                "label": "MCMC steps per walker",
                "default": "1000",
                "required": False,
            },
            {
                "key": "morris_trajectories",
                "label": "Morris trajectories (313.1 MeV screen)",
                "default": "24",
                "required": False,
            },
            {
                "key": "morris_freeze_threshold",
                "label": "Morris μ* freeze threshold",
                "default": "0.05",
                "required": False,
                "hint": "Parameters below μ* threshold are frozen at defaults",
            },
        ]
    if action == "Run Falsification Suite (Methods 1,4,9)":
        return [
            {
                "key": "n_points",
                "label": "Grid points",
                "default": "4096",
                "required": False,
                "hint": "1D grid size (or n^(1/3) for 3D)",
            },
            {
                "key": "box_size_mpc",
                "label": "Box size [h⁻¹ Mpc]",
                "default": "3000",
                "required": False,
                "hint": "Comoving volume scale",
            },
            {
                "key": "noise_level",
                "label": "Noise σ",
                "default": "0.25",
                "required": False,
                "hint": "Gaussian δ noise",
            },
            {
                "key": "dimension",
                "label": "Grid dimension",
                "default": "1",
                "required": False,
                "hint": "1 or 3 (fftn cube)",
                "choices": ["1", "3"],
            },
            {
                "key": "use_jax",
                "label": "JAX FFT",
                "default": "yes",
                "required": False,
                "hint": "Accelerated FFT when JAX installed",
                "choices": ["yes", "no"],
            },
            {
                "key": "output_prefix",
                "label": "Report prefix",
                "default": "falsification_suite",
                "required": False,
                "hint": "artifacts/lss_falsification_*.json",
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
        "Fractal Tau Circle Likelihood": [
            "Pre-run auto-fetch: empirical:glueball_lattice + empirical:neutron_lifetime.",
            "1D Fractal-Conformal Tau Circle: ρ_wind, fractal_level, phase_slip_alpha.",
            "dual mode: rotation dynamics + CDT spectral_dim constraint.",
            "tep_ansatz: m_tors·exp(−1.0607·fractal_level) vs 313.1 MeV floor; TEP closure required.",
            "For tep_ansatz use fractal_level ≈ N_dom = 3 (not full n_hier=45.8).",
            "TEP failure → log L = −∞; optional 3D grid scan.",
            "MCMC: TEP + 142857 hard-coded in log_prior; Morris ranks params for 313.1 MeV floor.",
        ],
        "Run Falsification Suite (Methods 1,4,9)": [
            "Unified LSS falsification: S(n) nodes, P(k) comb, gridded mock recovery.",
            "Returns global PASS / POTENTIAL FALSIFICATION verdict JSON.",
            "Uses menus/astronomical/desi/lss_falsification.py (JAX FFT when available).",
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

    if selection == "Fractal Tau Circle Likelihood":
        from menus.tsb_research.fractal_tau_circle import run_fractal_tau_likelihood

        mode = str(options.get("likelihood_mode") or "dual").strip().lower()
        report = run_fractal_tau_likelihood(
            params={
                "winding_density": _float_option(options, "winding_density", 1.0),
                "fractal_level": _float_option(
                    options,
                    "fractal_level",
                    3.0 if mode == "tep_ansatz" else 45.8,
                ),
                "phase_slip_alpha": _float_option(options, "phase_slip_alpha", 0.33),
            },
            observed_data={
                "mass_gap_mev": _float_option(options, "observed_mass_gap_mev", 313.1),
                "spectral_dim": _float_option(options, "observed_spectral_dim", 2.0),
                "sigma_qcd_mev": _float_option(options, "sigma_qcd_mev", 5.0),
                "likelihood_mode": mode,
            },
            do_grid_scan=_bool_option(options, "do_grid_scan", True),
            do_mcmc=_bool_option(options, "do_mcmc", False),
            mcmc_walkers=_int_option(options, "mcmc_walkers", 32),
            mcmc_steps=_int_option(options, "mcmc_steps", 1000),
            morris_trajectories=_int_option(options, "morris_trajectories", 24),
            morris_freeze_threshold=_float_option(options, "morris_freeze_threshold", 0.05),
            empirical_provenance=options.get("_empirical_provenance"),
            verbose=True,
        )
        path = _save_report("fractal_tau_circle_likelihood", report)
        print(f"[TAV ENGINE] Report saved: {path}")
        return

    if selection == "Run Falsification Suite (Methods 1,4,9)":
        from menus.astronomical.desi.lss_falsification import run_falsification_suite

        report = run_falsification_suite(
            n_points=_int_option(options, "n_points", 4096),
            box_size_mpc=_float_option(options, "box_size_mpc", 3000.0),
            noise_level=_float_option(options, "noise_level", 0.25),
            use_jax=_bool_option(options, "use_jax", True),
            dimension=_int_option(options, "dimension", 1),
            output_prefix=str(options.get("output_prefix") or "falsification_suite").strip(),
            plot=_bool_option(options, "plot", False),
            verbose=True,
        )
        print(f"[TAV ENGINE] Falsification suite: {report.get('verdict')}")
        print(f"[TAV ENGINE] Report: {report.get('report_path')}")
        return

    print(f"[TAV ENGINE] Unknown TSB research action: {selection}")