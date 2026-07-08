"""
Tau-SB DESI BAO scanner ↔ research_tool.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from tav_shared.llm_analysis import analyze_run_output as query_llm_analysis
from tav_shared.run_output import parse_show_graphics
from menus.astronomical.desi.scanner import (
    ARTIFACTS_DIR,
    COMBINED_DR2_TRACER,
    DATA_MODE,
    DATA_MODES,
    DATA_MODE_DEFAULT,
    DESI_TRACERS,
    EXTENDED_BAO_TRACER,
    TRACER,
    DEFAULT_COBAYA_ROOT,
    JOINT_DH_DM,
    MCMC_STEPS_DEFAULT,
    N_HIER_BINDING,
    PRODUCTION_PIPELINE_MIN_N,
    TAU_RESONANCE_PERIOD,
    list_desi_tracers,
    load_desi_from_cobaya_repo,
    normalize_quantity_filter,
    run_desi_scan,
)

MODULE_TAG = "TAU_SB_DESI"
SUBMENU_TITLE = "DESI BAO Tau-SB Scan"

MENU_ACTIONS = [
    "Real DESI DR2 Scan (Cobaya)",
    "Periodicity Scan (1/7 Tracks)",
    "Harmonic Binding Detection Kit",
    "Model Compare (ΛCDM vs aDE vs Tau-SB)",
    "MCMC Posteriors (emcee)",
    "Healpy Dipole Fit",
    "Change-Point Detection",
    "Injection/Recovery Tests",
    "Joint DESI+SN+Planck Fit",
    "Batch Tracer Scan (All DR2)",
    "Full Production Pipeline",
]

# Fixed Tau-SB sequence applied on every live DESI run (not user-tunable via form).
TAU_SEQUENCE: dict[str, Any] = {
    "data_source": "cobaya",
    "tracer": TRACER,
    "quantity": DATA_MODE,
    "auto_calibrate_gamma": "yes",
    "n_hier": str(N_HIER_BINDING),
    "gamma": "10.0",
    "diagonal_only": "no",
    "tau_period": str(TAU_RESONANCE_PERIOD),
    "m0_mev": "313.1",
}

_ACTION_CONFIG: dict[str, dict[str, Any]] = {
    "Real DESI DR2 Scan (Cobaya)": {
        "run_fit": True,
        "compare_models": False,
        "run_dipole": False,
        "plot": True,
        "auto_calibrate_gamma": True,
    },
    "Periodicity Scan (1/7 Tracks)": {
        "run_fit": False,
        "compare_models": False,
        "run_dipole": False,
        "plot": True,
        "auto_calibrate_gamma": True,
    },
    "Harmonic Binding Detection Kit": {
        "run_fit": True,
        "compare_models": False,
        "run_dipole": False,
        "plot": True,
        "run_binding_kit": True,
        "auto_calibrate_gamma": True,
    },
    "Model Compare (ΛCDM vs aDE vs Tau-SB)": {
        "run_fit": False,
        "compare_models": True,
        "run_dipole": False,
        "plot": True,
        "auto_calibrate_gamma": True,
    },
    "MCMC Posteriors (emcee)": {
        "run_fit": False,
        "compare_models": False,
        "run_dipole": False,
        "plot": False,
        "run_mcmc": True,
        "auto_calibrate_gamma": True,
    },
    "Healpy Dipole Fit": {
        "run_fit": False,
        "compare_models": False,
        "run_dipole": True,
        "plot": True,
        "auto_calibrate_gamma": True,
    },
    "Change-Point Detection": {
        "run_fit": False,
        "compare_models": False,
        "run_dipole": False,
        "plot": False,
        "run_changepoints": True,
        "auto_calibrate_gamma": True,
    },
    "Injection/Recovery Tests": {
        "run_fit": False,
        "compare_models": False,
        "run_dipole": False,
        "plot": False,
        "run_injection_recovery": True,
        "auto_calibrate_gamma": True,
    },
    "Joint DESI+SN+Planck Fit": {
        "run_fit": False,
        "compare_models": False,
        "run_dipole": False,
        "plot": False,
        "run_joint_fit": True,
        "auto_calibrate_gamma": True,
    },
    "Batch Tracer Scan (All DR2)": {
        "batch_tracers": True,
    },
    "Full Production Pipeline": {
        "run_fit": True,
        "compare_models": True,
        "run_dipole": True,
        "plot": True,
        "run_mcmc": True,
        "run_changepoints": True,
        "run_binding_kit": True,
        "run_injection_recovery": True,
        "run_joint_fit": True,
        "auto_calibrate_gamma": True,
    },
}

_TRACER_CHOICES = [
    "ALL_GCcomb",
    EXTENDED_BAO_TRACER,
    COMBINED_DR2_TRACER,
    "BGS_BRIGHT-21.35_GCcomb",
    "LRG_GCcomb_z0.4-0.6",
    "LRG_GCcomb_z0.6-0.8",
    "ELG_LOPnotqso_GCcomb_z1.1-1.6",
    "QSO_GCcomb",
    "Lya_GCcomb",
]


def is_module_selection(repo: str | None) -> bool:
    return repo == MODULE_TAG


def _yes(value: str) -> bool:
    return str(value or "").strip().lower() in {"yes", "y", "true", "1"}


def _action_config(action: str) -> dict[str, Any]:
    return _ACTION_CONFIG.get(action, {})


def _non_empty(raw: dict[str, Any]) -> dict[str, str]:
    return {k: str(v).strip() for k, v in raw.items() if v is not None and str(v).strip()}


def _quantity_filter_from_option(quantity: str) -> str | None:
    return normalize_quantity_filter(quantity or TAU_SEQUENCE["quantity"])


def _resolve_quantity_option(action: str, merged: dict[str, str]) -> None:
    """
    Upgrade underpowered quantity filters (e.g. DV_over_rs → DH_over_rs).

    Full Production Pipeline needs n≥PRODUCTION_PIPELINE_MIN_N; never
    auto-upgrades to mixed quantity=all.
    """
    config = _action_config(action)
    if config.get("batch_tracers"):
        return

    quantity = merged.get("quantity") or TAU_SEQUENCE["quantity"]
    quantity_filter = _quantity_filter_from_option(quantity)
    if quantity_filter is None:
        return

    min_n = PRODUCTION_PIPELINE_MIN_N if action == "Full Production Pipeline" else 1
    tracer = merged.get("tracer") or TAU_SEQUENCE["tracer"]
    cobaya_path = (merged.get("cobaya_path") or "").strip() or None
    if not cobaya_path and not DEFAULT_COBAYA_ROOT.is_dir():
        return

    try:
        preview = load_desi_from_cobaya_repo(
            cobaya_path,
            tracer=tracer,
            quantity_filter=quantity_filter,
        )
        n = int(preview.get("n_data", 0))
    except (FileNotFoundError, ValueError):
        return

    if n >= min_n:
        return

    # Single-channel fallbacks only — never upgrade into mixed joint_DH_DM.
    for fallback in ("DH_over_rs", "DM_over_rs"):
        try:
            fb = load_desi_from_cobaya_repo(
                cobaya_path,
                tracer=tracer,
                quantity_filter=fallback,
            )
            n_fb = int(fb.get("n_data", 0))
        except (FileNotFoundError, ValueError):
            continue
        if n_fb > n:
            merged["quantity"] = fallback
            merged["quantity_resolved_from"] = quantity_filter
            return


def prepare_run_options(action: str, raw_options: dict | None = None) -> dict[str, str]:
    """
    Merge user form values with the fixed Tau sequence and action profile.

    Always forces live DR2 Cobaya tables. Strips irrelevant form defaults before
    logging or passing to the scanner.
    """
    raw = _non_empty(raw_options or {})
    config = _action_config(action)
    merged: dict[str, str] = dict(TAU_SEQUENCE)
    merged.update(raw)
    merged["data_source"] = "cobaya"
    merged["action"] = action

    pipelines: list[str] = []
    if config.get("run_fit"):
        pipelines.append("tau_sb_fit")
    if config.get("compare_models"):
        pipelines.append("model_compare")
    if config.get("run_dipole"):
        pipelines.append("healpy_dipole")
    if config.get("run_mcmc"):
        pipelines.append("emcee_mcmc")
        merged.setdefault("mcmc_steps", str(MCMC_STEPS_DEFAULT))
    if config.get("run_changepoints"):
        pipelines.append("changepoints")
    if config.get("run_binding_kit"):
        pipelines.append("harmonic_binding_kit")
    if config.get("run_injection_recovery"):
        pipelines.append("injection_recovery")
    if config.get("run_joint_fit"):
        pipelines.append("joint_desi_sn_planck")
    if config.get("batch_tracers"):
        pipelines.append("batch_tracers")
        merged.setdefault("batch_limit", "0")
        merged.setdefault("force_rescan", "no")
    if not pipelines:
        pipelines.append("periodicity_scan")
    merged["pipelines_enabled"] = ",".join(pipelines)

    if not config.get("run_mcmc") and not config.get("batch_tracers"):
        merged.pop("mcmc_steps", None)
    if not config.get("batch_tracers"):
        # Keep batch_limit / force_rescan when set by the interactive n selector.
        if not raw.get("batch_limit"):
            merged.pop("batch_limit", None)
        if not raw.get("force_rescan"):
            merged.pop("force_rescan", None)

    merged.setdefault("output_prefix", "tau_sb_desi")
    merged.setdefault("show_graphics", "popup")
    _resolve_quantity_option(action, merged)
    return merged


def options_for_run_log(action: str, prepared: dict[str, str]) -> dict[str, str]:
    """Subset of prepared options safe to print in run logs and LLM prompts."""
    config = _action_config(action)
    keys = [
        "action",
        "data_source",
        "tracer",
        "quantity",
        "quantity_resolved_from",
        "auto_calibrate_gamma",
        "n_hier",
        "tau_period",
        "m0_mev",
        "diagonal_only",
        "pipelines_enabled",
        "output_prefix",
        "show_graphics",
        "llm_analysis",
        "llm_providers",
    ]
    if config.get("run_mcmc"):
        keys.append("mcmc_steps")
    if config.get("batch_tracers"):
        keys.extend(["batch_limit", "force_rescan"])
    cobaya = (prepared.get("cobaya_path") or "").strip()
    if cobaya:
        keys.append("cobaya_path")
    if (prepared.get("n_points") or "").strip():
        keys.append("n_points")
    return {k: prepared[k] for k in keys if k in prepared}


_LEGACY_BAO_ACTIONS = frozenset(
    {
        "Real DESI DR2 Scan (Cobaya)",
        "Periodicity Scan (1/7 Tracks)",
        "Model Compare (ΛCDM vs aDE vs Tau-SB)",
        "Joint DESI+SN+Planck Fit",
    }
)


def desi_entry_fields(action: str) -> list[dict]:
    """Action-specific form fields; Tau sequence constants are auto-applied on run."""
    tracer_field = {
        "key": "tracer",
        "label": "DESI DR2 tracer",
        "default": TRACER,
        "required": False,
        "hint": "Live DR2 tables from datasets/desi/bao_data/desi_bao_dr2",
        "choices": _TRACER_CHOICES,
    }
    quantity_choices = list(DATA_MODES) + ["all"]
    if action != "Full Production Pipeline":
        quantity_choices.extend(["DH_over_rs", "DM_over_rs", JOINT_DH_DM, "DV_over_rs"])
    quantity_field = {
        "key": "quantity",
        "label": "DATA_MODE / BAO quantity",
        "default": DATA_MODE,
        "required": False,
        "hint": (
            "DH_only = 6 clean D_H/r_d pts (recommended); DM_only = 6 D_M/r_d; "
            f"Joint_DH_DM = 12 DM+DH; all = 13 mixed channels (not recommended)"
        ),
        "choices": quantity_choices,
    }
    graphics_field = {
        "key": "show_graphics",
        "label": "Graphics mode",
        "default": "popup",
        "required": False,
        "hint": "popup = Tk window | artifacts = PNG only",
        "choices": ["artifacts", "popup"],
    }

    if action == "Batch Tracer Scan (All DR2)":
        fields = [
            quantity_field,
            {
                "key": "batch_limit",
                "label": "Tracer batch limit (0 = all pending)",
                "default": "0",
                "required": False,
                "hint": "Resumes via datasets/desi/batch_done.txt",
            },
            {
                "key": "force_rescan",
                "label": "Force re-scan tracers",
                "default": "no",
                "required": False,
                "hint": "Ignore batch_done.txt ledger",
                "choices": ["no", "yes"],
            },
            {
                "key": "output_prefix",
                "label": "Batch report prefix",
                "default": "tau_sb_desi_batch",
                "required": False,
                "hint": "Artifacts under artifacts/",
            },
            graphics_field,
        ]
        return fields

    fields = [tracer_field, quantity_field]
    if action not in {"Batch Tracer Scan (All DR2)"}:
        fields.extend(
            [
                {
                    "key": "fit_phase",
                    "label": "Fit oscillation phase",
                    "default": "yes",
                    "required": False,
                    "hint": "Set no to fix phase=0 and gain 1 dof on small vectors",
                    "choices": ["yes", "no"],
                },
                {
                    "key": "fit_hier",
                    "label": "Fit hierarchical step (hier_frac)",
                    "default": "auto",
                    "required": False,
                    "hint": "auto = on when n≥8; no = drop hier_frac for more dof",
                    "choices": ["auto", "yes", "no"],
                },
            ]
        )
    if action in {"MCMC Posteriors (emcee)", "Full Production Pipeline"}:
        fields.append(
            {
                "key": "mcmc_steps",
                "label": "MCMC steps (emcee)",
                "default": str(MCMC_STEPS_DEFAULT),
                "required": False,
                "hint": f"Only used when MCMC is enabled ({MCMC_STEPS_DEFAULT}+ recommended)",
            }
        )
    fields.append(
        {
            "key": "output_prefix",
            "label": "Plot/report prefix",
            "default": "tau_sb_desi",
            "required": False,
            "hint": "Artifacts saved under artifacts/",
        }
    )
    if action in _LEGACY_BAO_ACTIONS:
        fields.append(
            {
                "key": "include_legacy_bao",
                "label": "Include Legacy BOSS/eBOSS BAO",
                "default": "no",
                "required": False,
                "hint": "Stacks DESI + BOSS/eBOSS to increase n toward ~15+",
                "choices": ["no", "yes"],
            }
        )
    fields.append(graphics_field)
    return fields


# =============================================================================
# BLOCK: Entry form — fields
# =============================================================================
entry_fields = desi_entry_fields


# =============================================================================
# BLOCK: Entry form — instructions
# =============================================================================
def entry_instructions(action: str) -> list[str]:
    """Short help bullets shown above the DESI BAO entry form."""
    return [
        "Live DR2 only: clone Cobaya bao_data → datasets/desi/bao_data (desi_bao_dr2/).",
        f"Default Cobaya path: {DEFAULT_COBAYA_ROOT}",
        "Tau sequence auto-applied: n_hier≈45.8, τ period 1/7, M₀=313.1 MeV, auto γ calibration.",
        "Form shows tracer/quantity only; MCMC/batch fields appear only for those actions.",
        "n selector appears only on actions where n is wired (periodogram grid, injection trials, batch limit, etc.) — not BAO count.",
        "Tracers: ALL_GCcomb, BGS, LRG, ELG, QSO, Lya (see --list-tracers in CLI).",
        "Cobaya = bao_data repo path, not the Cobaya MCMC sampler.",
        "Model compare fits ΛCDM poly, aDE (axion+Λ), and Tau-SB (1/7 + hierarchical).",
        "Harmonic Binding Kit: maps n_hier/Δn → z, detects binding transitions, extended 1/7 search.",
        "Production: emcee MCMC, healpy dipole, ruptures change-points, binding kit, injection/recovery.",
        "Joint fit: Pantheon+ μ(z) + Planck r_d prior + DESI BAO (auto-downloads SN catalog).",
        "Batch tracer ledger: datasets/desi/batch_done.txt (resume next pending tracer).",
        "Batch: Batch Tracer Scan or CLI --batch-tracers (batch_limit, force_rescan).",
        "MCMC default: 1500 steps / 48 walkers / 500 burn-in (mcmc_steps form field).",
        "CLI: python tau_sb_desi_scanner.py --data cobaya --mcmc --mcmc-steps 1500",
        "Deps: pip install -r requirements-desi-production.txt",
        "Plots/reports: artifacts/tau_sb_desi_*_summary.png and *_report_*.json",
    ]


# =============================================================================
# BLOCK: Pre-form hook (n selector, warnings)
# =============================================================================
def _show_desi_low_n_warning(stdscr, effective_n: int) -> None:
    """Strong warning when effective n is below target for real DESI data."""
    import curses

    from tav_research.curses_shell import _safe_addstr

    stdscr.clear()
    _safe_addstr(
        stdscr,
        2,
        2,
        "═══════════════════════════════════════════════════════════════",
        curses.color_pair(1),
    )
    _safe_addstr(
        stdscr,
        3,
        2,
        "   LOW STATISTICAL POWER WARNING — REAL DATA ONLY",
        curses.A_BOLD | curses.color_pair(1),
    )
    _safe_addstr(
        stdscr,
        4,
        2,
        "═══════════════════════════════════════════════════════════════",
        curses.color_pair(1),
    )
    _safe_addstr(
        stdscr,
        6,
        2,
        f"Current effective n = {effective_n}   (Target: n ≥ 15)",
        curses.A_BOLD,
    )
    _safe_addstr(
        stdscr,
        8,
        2,
        "With n < 15 the following tests are unreliable:",
        curses.A_DIM,
    )
    _safe_addstr(stdscr, 9, 4, "• 1/7 periodicity detection (tav_resonance)", curses.A_DIM)
    _safe_addstr(stdscr, 10, 4, "• γ jackknife stability", curses.A_DIM)
    _safe_addstr(stdscr, 11, 4, "• Model comparison (AIC / evidence)", curses.A_DIM)
    _safe_addstr(stdscr, 13, 2, "Recommended action for real data:", curses.A_BOLD)
    _safe_addstr(
        stdscr,
        14,
        4,
        "→ Enable Legacy BOSS/eBOSS data (strongly recommended)",
        curses.A_DIM,
    )
    _safe_addstr(stdscr, 15, 4, "→ Use batch tracer mode with n ≥ 15", curses.A_DIM)
    _safe_addstr(
        stdscr,
        17,
        2,
        "Press any key to continue (or cancel and increase n)...",
        curses.A_DIM,
    )
    stdscr.refresh()
    stdscr.getch()


def _apply_desi_n_selection(stdscr, params: dict[str, str], n: int) -> None:
    """Apply interactive n choice to DESI batch / legacy-BAO form defaults."""
    import curses

    from tav_research.curses_shell import _safe_addstr

    params["n_points"] = str(n)
    params["force_rescan"] = "yes"
    # Strengthen batch behavior for real data
    params["batch_limit"] = str(max(int(n), 20))
    params["include_legacy_bao"] = "yes" if int(n) >= 12 else "no"

    if int(n) >= 12:
        stdscr.clear()
        _safe_addstr(
            stdscr,
            2,
            2,
            f"n = {n} → Legacy BOSS/eBOSS data will be included",
            curses.A_BOLD,
        )
        _safe_addstr(
            stdscr,
            3,
            2,
            "This is the recommended way to increase real data volume.",
            curses.A_DIM,
        )
        stdscr.refresh()
        stdscr.getch()


def handle_pre_form(stdscr, selection: str, params: dict[str, str]) -> str | None:
    """Low-n warning, interactive n selector, and DESI batch tuning before entry form."""
    from tav_research.curses_shell import select_n_interactive
    from tav_research.n_selector import should_offer_n_selector

    effective_n = int(params.get("n_points", 7))
    if effective_n < 15:
        _show_desi_low_n_warning(stdscr, effective_n)

    if should_offer_n_selector(MODULE_TAG, selection):
        n = select_n_interactive(
            stdscr,
            prompt=f"Choose n for {selection} (real data)",
        )
        if n is not None:
            _apply_desi_n_selection(stdscr, params, n)
    return None


def _show_saved_plot(plot_path: str | None) -> None:
    if not plot_path or not Path(plot_path).is_file():
        return
    image = plt.imread(plot_path)
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title(os.path.basename(plot_path))
    plt.tight_layout()
    plt.show()


def run_action(selection: str, show_plots: bool = True, options: dict | None = None) -> str | None:
    """Run a DESI action. Returns JSON report path when a scan completes."""
    prepared = prepare_run_options(selection, options)
    config = _action_config(selection)
    if not config:
        print(f"[TAV ENGINE] Unknown DESI Tau-SB action: {selection}")
        return None

    if config.get("batch_tracers"):
        from menus.astronomical.desi.production import run_batch_tracer_scan

        quantity = prepared.get("quantity", "DM_over_rs")
        quantity_filter = None if quantity.lower() in {"all", "*", "none"} else quantity
        try:
            max_tracers = int(prepared.get("batch_limit") or 0)
        except ValueError:
            max_tracers = 0
        from tav_shared.n_selector_registry import desi_n_usage_message, parse_n_points

        n_menu = parse_n_points(prepared.get("n_points"))
        if max_tracers <= 0 and n_menu is not None:
            max_tracers = n_menu
            print(f"[TAV ENGINE] {desi_n_usage_message(selection, n_menu)}")
        cobaya_path = (prepared.get("cobaya_path") or "").strip() or None
        if not cobaya_path and not DEFAULT_COBAYA_ROOT.is_dir():
            raise FileNotFoundError(
                f"DESI DR2 tables missing at {DEFAULT_COBAYA_ROOT}. "
                "Auto-fetch should have run before this action."
            )
        batch = run_batch_tracer_scan(
            cobaya_path=cobaya_path,
            quantity_filter=quantity_filter,
            auto_calibrate_gamma=True,
            compare_models=True,
            use_covariance=True,
            output_prefix=(prepared.get("output_prefix") or "tau_sb_desi_batch").strip(),
            max_tracers=max_tracers,
            force_rescan=_yes(prepared.get("force_rescan", "no")),
        )
        print(f"[TAV ENGINE] Batch summary: {batch['summary_path']}")
        for tracer, info in batch["summaries"].items():
            print(f"  {tracer}: {info}")
        return str(batch.get("summary_path") or "")

    try:
        gamma = float(prepared.get("gamma") or 10.0)
    except ValueError:
        gamma = 10.0
    try:
        n_hier = float(prepared.get("n_hier") or N_HIER_BINDING)
    except ValueError:
        n_hier = N_HIER_BINDING

    cobaya_path = (prepared.get("cobaya_path") or "").strip() or None
    tracer = prepared.get("tracer") or TRACER
    quantity = prepared.get("quantity") or TAU_SEQUENCE["quantity"]
    quantity_filter = _quantity_filter_from_option(quantity)
    output_prefix = (prepared.get("output_prefix") or "tau_sb_desi").strip()
    show_popup = parse_show_graphics(prepared, default="popup") if show_plots else False
    use_cov = not _yes(prepared.get("diagonal_only", "no"))
    include_legacy_bao = _yes(prepared.get("include_legacy_bao", "no"))
    combine_tracers = tracer == COMBINED_DR2_TRACER
    if tracer == EXTENDED_BAO_TRACER:
        include_legacy_bao = True
        tracer = TRACER
    fit_phase = _yes(prepared.get("fit_phase", "yes"))
    fit_hier_raw = (prepared.get("fit_hier") or "auto").strip().lower()
    fit_hier: bool | None
    if fit_hier_raw == "auto":
        fit_hier = None
    elif fit_hier_raw in {"no", "false", "0"}:
        fit_hier = False
    else:
        fit_hier = True

    from tav_shared.n_selector_registry import desi_n_knobs, desi_n_usage_message, parse_n_points

    n_menu = parse_n_points(prepared.get("n_points"))

    if not cobaya_path and not DEFAULT_COBAYA_ROOT.is_dir():
        raise FileNotFoundError(
            f"DESI DR2 tables missing at {DEFAULT_COBAYA_ROOT}. "
            "Auto-fetch should have run before this action."
        )

    mcmc_steps = MCMC_STEPS_DEFAULT
    if config.get("run_mcmc"):
        try:
            mcmc_steps = int(prepared.get("mcmc_steps") or MCMC_STEPS_DEFAULT)
        except ValueError:
            mcmc_steps = MCMC_STEPS_DEFAULT

    n_knobs: dict[str, int] = desi_n_knobs(selection, n_menu) if n_menu is not None else {}
    if n_menu is not None:
        print(f"[TAV ENGINE] {desi_n_usage_message(selection, n_menu)}")

    try:
        batch_limit = int(prepared.get("batch_limit") or 0)
    except ValueError:
        batch_limit = 0
    if batch_limit > 0 and _yes(prepared.get("force_rescan", "no")):
        from menus.astronomical.desi.production import run_batch_tracer_scan

        print(
            f"[TAV ENGINE] Batch tracer prefetch: up to {batch_limit} tracers "
            f"(force_rescan=yes) before {selection}"
        )
        run_batch_tracer_scan(
            cobaya_path=cobaya_path,
            quantity_filter=quantity_filter,
            auto_calibrate_gamma=True,
            compare_models=bool(config.get("compare_models")),
            use_covariance=use_cov,
            output_prefix=f"{output_prefix}_batch_n{batch_limit}",
            max_tracers=batch_limit,
            force_rescan=True,
        )

    try:
        result = run_desi_scan(
            cobaya_path=cobaya_path,
            tracer=tracer,
            quantity_filter=quantity_filter,
            gamma=gamma,
            auto_calibrate_gamma=True,
            n_hier=n_hier,
            use_covariance=use_cov,
            run_fit=config.get("run_fit", False),
            compare_models=config.get("compare_models", False),
            run_dipole=config.get("run_dipole", False),
            run_mcmc=config.get("run_mcmc", False),
            run_changepoints=config.get("run_changepoints", False),
            run_binding_kit=config.get("run_binding_kit", False),
            run_injection_recovery=config.get("run_injection_recovery", False),
            run_joint_fit=config.get("run_joint_fit", False),
            mcmc_steps=mcmc_steps,
            plot=config.get("plot", False),
            output_prefix=output_prefix,
            action_name=selection,
            pipelines_enabled=prepared.get("pipelines_enabled", ""),
            include_legacy_bao=include_legacy_bao,
            combine_tracers=combine_tracers,
            fit_phase=fit_phase,
            fit_hier=fit_hier,
            n_freq=n_knobs.get("n_freq"),
            n_injection_trials=n_knobs.get("n_injection_trials"),
            max_sne=n_knobs.get("max_sne"),
        )
    except (FileNotFoundError, ValueError, NotImplementedError) as exc:
        print(f"[TAV ENGINE] DESI load failed: {exc}")
        try:
            available = list_desi_tracers(cobaya_path)
            print(f"[TAV ENGINE] Available tracers: {', '.join(available)}")
        except Exception:
            pass
        raise
    except Exception as exc:
        label = type(exc).__name__
        detail = str(exc).strip()
        print(f"[TAV ENGINE] DESI run failed: {label}" + (f": {detail}" if detail else ""))
        return None

    print("\n".join(result.summary_lines()))
    if config.get("plot") and show_popup:
        _show_saved_plot(result.plot_path)
    elif config.get("plot") and result.plot_path:
        print(f"[TAV ENGINE] Summary plot: {result.plot_path}")
    return result.report_path