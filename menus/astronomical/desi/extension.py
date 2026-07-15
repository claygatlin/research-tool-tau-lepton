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
    DATA_MODE_DM_ONLY,
    DATA_MODE_JOINT,
    DATA_MODE_METHOD10_DEFAULT,
    DATA_MODE_PUBLICATION_DEFAULT,
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
    load_extended_bao_data,
    normalize_data_mode,
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
    "Mock Catalog Recovery (1/7 + Δn)",
    "LSS Comb Falsification (P(k))",
    "S(n) Node Cross-Correlation",
    "Gridded Mock Comb Recovery",
    "DESI Catalog → LSS Grid",
    "Method 10 Joint Likelihood",
    "Test 2 Redshift Law (λ(z))",
    "LSS Publication Figures",
    "Cosmology Falsification (Methods 3, 5)",
    "EKK BAO Re-analysis (DR2 Covariance)",
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
        "run_nested_sampling": True,
        "auto_calibrate_gamma": True,
    },
    "Mock Catalog Recovery (1/7 + Δn)": {
        "mock_catalog": True,
    },
    "LSS Comb Falsification (P(k))": {
        "lss_comb_falsification": True,
        "plot": True,
    },
    "S(n) Node Cross-Correlation": {
        "lss_sn_nodes": True,
        "plot": True,
    },
    "Gridded Mock Comb Recovery": {
        "lss_gridded_mock": True,
        "plot": True,
    },
    "DESI Catalog → LSS Grid": {
        "desi_catalog_grid": True,
        "plot": False,
    },
    "Method 10 Joint Likelihood": {
        "method10_joint": True,
        "plot": True,
    },
    "Test 2 Redshift Law (λ(z))": {
        "redshift_law_test2": True,
        "plot": True,
    },
    "LSS Publication Figures": {
        "lss_publication_figures": True,
        "plot": True,
    },
    "Cosmology Falsification (Methods 3, 5)": {
        "cosmology_falsification": True,
        "plot": True,
    },
    "EKK BAO Re-analysis (DR2 Covariance)": {
        "ekk_bao_reanalysis": True,
        "plot": True,
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


def _action_requires_single_channel(action: str) -> bool:
    """Structured ΛCDM/Tau-SB fits refuse mixed DH+DM vectors."""
    config = _action_config(action)
    return bool(config.get("run_fit") or config.get("compare_models"))


def _is_joint_dh_dm_choice(quantity: str) -> bool:
    key = str(quantity or "").strip()
    if not key:
        return False
    if key in {DATA_MODE_JOINT, JOINT_DH_DM, "Joint_DH_DM", "joint_DH_DM"}:
        return True
    try:
        return normalize_data_mode(key) == DATA_MODE_JOINT
    except ValueError:
        return False


def _is_mixed_quantity_choice(quantity: str) -> bool:
    key = str(quantity or "").strip()
    if not key:
        return False
    if key.lower() in {"all", "*", "none"}:
        return True
    return _is_joint_dh_dm_choice(key)


def _enforce_single_channel_quantity(action: str, merged: dict[str, str]) -> None:
    """Downgrade quantity=all; Joint_DH_DM runs as split DH+DM scans in run_action."""
    if not _action_requires_single_channel(action):
        return
    quantity = merged.get("quantity") or TAU_SEQUENCE["quantity"]
    if _is_joint_dh_dm_choice(quantity):
        return
    if not _is_mixed_quantity_choice(quantity):
        return
    merged.setdefault("quantity_resolved_from", quantity)
    merged["quantity"] = DATA_MODE_DEFAULT


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

    include_legacy = _yes(merged.get("include_legacy_bao", "no"))
    preview_mode = DATA_MODE_DEFAULT if _is_joint_dh_dm_choice(quantity) else quantity
    try:
        if include_legacy:
            preview = load_extended_bao_data(
                normalize_data_mode(preview_mode),
                local_path=cobaya_path,
                tracer=tracer,
            )
        else:
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
    if config.get("run_nested_sampling"):
        pipelines.append("dynesty_nested")
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
    from tav_shared.n_selector_registry import parse_n_points

    n_sel = parse_n_points(merged.get("n_points"))
    if n_sel is not None and n_sel >= 12:
        merged.setdefault("include_legacy_bao", "yes")
        merged.setdefault("high_power_stack", "yes")
        merged.setdefault("augment_covariance", "yes")
    _enforce_single_channel_quantity(action, merged)
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
        "include_legacy_bao",
        "high_power_stack",
        "augment_covariance",
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
    joint_hint = (
        "Joint_DH_DM = run DH_only then DM_only and collate (no mixed vector fits)"
    )
    if action == "Full Production Pipeline":
        quantity_choices = [DATA_MODE_DEFAULT, "DM_only", DATA_MODE_JOINT]
        quantity_hint = (
            "DH_only or DM_only (single channel); "
            f"{joint_hint}; legacy BOSS/eBOSS auto-included when n≥12"
        )
    else:
        quantity_choices = list(DATA_MODES) + ["all"]
        quantity_choices.extend(["DH_over_rs", "DM_over_rs", JOINT_DH_DM, "DV_over_rs"])
        quantity_hint = (
            "DH_only = 6 clean D_H/r_d pts (recommended); DM_only = 6 D_M/r_d; "
            f"{joint_hint}; all = 13 mixed channels (not recommended)"
        )
    qty_default = DATA_MODE
    if action == "Method 10 Joint Likelihood":
        qty_default = DATA_MODE_METHOD10_DEFAULT
    elif action == "LSS Publication Figures":
        qty_default = DATA_MODE_PUBLICATION_DEFAULT
    quantity_field = {
        "key": "quantity",
        "label": "DATA_MODE / BAO quantity",
        "default": qty_default,
        "required": False,
        "hint": quantity_hint,
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

    _lss_grid_fields = [
        {
            "key": "n_points",
            "label": "Grid points (1D) or n^(1/3) (3D)",
            "default": "4096",
            "required": False,
            "hint": "Power of 2 recommended; 3D uses cube root",
        },
        {
            "key": "box_size_mpc",
            "label": "Box size [h⁻¹ Mpc]",
            "default": "3000",
            "required": False,
            "hint": "Comoving box length",
        },
        {
            "key": "noise_level",
            "label": "Noise amplitude σ",
            "default": "0.25",
            "required": False,
            "hint": "Gaussian δ noise level",
        },
        {
            "key": "dimension",
            "label": "Grid dimension",
            "default": "1",
            "required": False,
            "hint": "1 = FFT line; 3 = fftn cube (comb recovery)",
            "choices": ["1", "3"],
        },
        {
            "key": "use_jax",
            "label": "JAX FFT backend",
            "default": "yes",
            "required": False,
            "hint": "GPU/TPU-accelerated FFT when JAX installed",
            "choices": ["yes", "no"],
        },
        {
            "key": "output_prefix",
            "label": "Report prefix",
            "default": "lss_falsification",
            "required": False,
            "hint": "JSON + PNG under artifacts/",
        },
        graphics_field,
    ]

    if action == "LSS Comb Falsification (P(k))":
        return [
            {
                "key": "inject_signal",
                "label": "Inject geometric comb",
                "default": "no",
                "required": False,
                "hint": "no = null field (falsification control); yes = injected comb",
                "choices": ["no", "yes"],
            },
            *_lss_grid_fields,
        ]

    if action == "DESI Catalog → LSS Grid":
        return [
            tracer_field,
            {
                "key": "n_per_axis",
                "label": "Grid cells per axis (3D)",
                "default": "64",
                "required": False,
                "hint": "CIC deposit resolution; 64³ recommended",
            },
            {
                "key": "box_size_mpc",
                "label": "Box size [h⁻¹ Mpc]",
                "default": "2000",
                "required": False,
                "hint": "Periodic comoving cube edge length",
            },
            {
                "key": "galaxies_per_shell",
                "label": "Galaxies per BAO shell",
                "default": "800",
                "required": False,
                "hint": "Poisson draw per DR2 z_eff shell",
            },
            {
                "key": "catalog_path",
                "label": "Optional CSV catalog path",
                "default": "",
                "required": False,
                "hint": "Leave blank to build from BAO shells",
            },
            {
                "key": "output_prefix",
                "label": "Report prefix",
                "default": "desi_catalog_grid",
                "required": False,
                "hint": "NPZ + JSON under artifacts/",
            },
            graphics_field,
        ]

    if action == "Method 10 Joint Likelihood":
        return [
            tracer_field,
            quantity_field,
            {
                "key": "delta_npz",
                "label": "Optional δ NPZ from catalog grid",
                "default": "",
                "required": False,
                "hint": "Blank = auto-build catalog grid",
            },
            *_lss_grid_fields,
            {
                "key": "galaxies_per_shell",
                "label": "Galaxies per BAO shell",
                "default": "600",
                "required": False,
                "hint": "When auto-building catalog grid",
            },
            {
                "key": "output_prefix",
                "label": "Report prefix",
                "default": "method10_joint",
                "required": False,
                "hint": "JSON + PNG under artifacts/",
            },
        ]

    if action == "EKK BAO Re-analysis (DR2 Covariance)":
        ekk_quantity_choices = ["joint_DH_DM", "DM_only", "DH_only", "all"]
        return [
            tracer_field,
            {
                "key": "quantity",
                "label": "BAO channel vector",
                "default": "joint_DH_DM",
                "required": False,
                "hint": "joint_DH_DM = 12 DM+DH pts with full cov (recommended falsification test)",
                "choices": ekk_quantity_choices,
            },
            {
                "key": "gamma",
                "label": "Cylinder stretch γ",
                "default": "8.8511",
                "required": False,
                "hint": "Plasma-interval r_d = Δn_plasma × r_anchor / γ",
            },
            {
                "key": "output_prefix",
                "label": "Report prefix",
                "default": "ekk_bao_reanalysis",
                "required": False,
                "hint": "JSON + PNG under artifacts/ with χ², α_⊥, α_∥, α_AP, residuals",
            },
            graphics_field,
        ]

    if action == "Cosmology Falsification (Methods 3, 5)":
        return [
            {
                "key": "output_prefix",
                "label": "Report prefix",
                "default": "cosmology_suite",
                "required": False,
                "hint": "JSON + PNG under artifacts/",
            },
            graphics_field,
        ]

    if action == "Test 2 Redshift Law (λ(z))":
        return [
            tracer_field,
            {
                "key": "max_sne",
                "label": "Max Pantheon+ SNe",
                "default": "80",
                "required": False,
                "hint": "Subsample for tractable joint fit",
            },
            {
                "key": "output_prefix",
                "label": "Report prefix",
                "default": "redshift_law_test2",
                "required": False,
                "hint": "JSON + PNG under artifacts/",
            },
            graphics_field,
        ]

    if action == "LSS Publication Figures":
        return [tracer_field, quantity_field] + _lss_grid_fields + [
            {
                "key": "output_prefix",
                "label": "Figure prefix",
                "default": "lss_manuscript",
                "required": False,
                "hint": "dpi=300 PNG suite under artifacts/",
            },
        ]

    if action in {"S(n) Node Cross-Correlation", "Gridded Mock Comb Recovery"}:
        extra: list[dict] = []
        if action == "S(n) Node Cross-Correlation":
            extra = [
                {
                    "key": "inject_nodes",
                    "label": "Inject S(n) node bumps",
                    "default": "yes",
                    "required": False,
                    "hint": "yes = recovery test; no = search only on noise",
                    "choices": ["yes", "no"],
                },
                {
                    "key": "sn_n_max",
                    "label": "S(n) sum limit n_max",
                    "default": "5000",
                    "required": False,
                    "hint": "Weight-6 partial sum depth",
                },
            ]
        return extra + _lss_grid_fields

    if action == "Mock Catalog Recovery (1/7 + Δn)":
        return [
            {
                "key": "mock_quick",
                "label": "Quick catalog (6 entries)",
                "default": "yes",
                "required": False,
                "hint": "yes = 3×2 grid; no = full 5×5 amplitude × Δn grid (25 entries)",
                "choices": ["yes", "no"],
            },
            {
                "key": "multi_tracer_stack",
                "label": "Multi-tracer stack (DR2 combined)",
                "default": "yes",
                "required": False,
                "hint": "Stack BGS+LRG+ELG+… for n≫9 Lomb resolution (needs BAO cache)",
                "choices": ["yes", "no"],
            },
            {
                "key": "retain_all_z_bins",
                "label": "Retain all z bins (no dedupe)",
                "default": "yes",
                "required": False,
                "hint": "yes = dedupe_z_tol=0 (max n); no = |Δz|≤0.02 collapse",
                "choices": ["yes", "no"],
            },
            {
                "key": "nested_sampling",
                "label": "Nested sampling per entry (slow)",
                "default": "no",
                "required": False,
                "hint": "Fit-aligned dynesty + bounded aDE priors (not JAX geometric)",
                "choices": ["no", "yes"],
            },
            {
                "key": "output_prefix",
                "label": "Report prefix",
                "default": "tau_sb_mock_catalog",
                "required": False,
                "hint": "JSON under artifacts/",
            },
        ]

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
    if action in {"Full Production Pipeline", "Model Compare (ΛCDM vs aDE vs Tau-SB)"}:
        fields.append(
            {
                "key": "nested_sampling",
                "label": "Nested sampling evidence (dynesty)",
                "default": "yes" if action == "Full Production Pipeline" else "no",
                "required": False,
                "hint": "ln Z with geometric Tau-SB priors vs physical aDE priors",
                "choices": ["no", "yes"],
            }
        )
        fields.append(
            {
                "key": "jax_geometric",
                "label": "JAX geometric likelihood (4D cylinder)",
                "default": "yes",
                "required": False,
                "hint": "yes = JAX tau_sb_mu + geometric priors; no = fit-aligned poly+dynesty",
                "choices": ["yes", "no"],
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
    if action in _LEGACY_BAO_ACTIONS or action == "Full Production Pipeline":
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
    if action in {"Full Production Pipeline", "Real DESI DR2 Scan (Cobaya)", "Model Compare (ΛCDM vs aDE vs Tau-SB)"}:
        fields.extend(
            [
                {
                    "key": "high_power_stack",
                    "label": "High-power stack (retain legacy z bins)",
                    "default": "no",
                    "required": False,
                    "hint": "n≈9 DH (vs 7 deduped); auto-on when n≥12 via selector",
                    "choices": ["no", "yes"],
                },
                {
                    "key": "augment_covariance",
                    "label": "Augment covariance (Δγ + friction)",
                    "default": "no",
                    "required": False,
                    "hint": "Theory-motivated C block from Tav cylinder session 2026-07-09",
                    "choices": ["no", "yes"],
                },
            ]
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
        "Joint_DH_DM: runs DH_only then DM_only sequentially and writes a collated summary JSON.",
        "high_power_stack + augment_covariance: raise n and embed cylinder drift in C (n≥12 auto).",
        "Mock Catalog Recovery: synthetic 1/7 + Δn injection grid + Tau-SB recovery report.",
        "LSS Comb Falsification: P(k) search at k=n/R_τ h Mpc⁻¹ (configuration-space, Methods 4/9).",
        "S(n) Node Cross-Correlation: Weight-6 cumulative nodes vs density peaks (Method 1).",
        "Gridded Mock Comb Recovery: inject + recover geometric comb on δ(x) grid (Method 9).",
        "DESI Catalog → LSS Grid: BAO shells → CIC δ(x,y,z) for real-data falsification.",
        "Method 10 Joint Likelihood: comb P(k) + S(n) nodes + BAO vs ΛCDM (default DM_only).",
        "Test 2 Redshift Law: λ(z) dispersion fit on Pantheon+ + DESI BAO.",
        "LSS Publication Figures: dpi=300 suite; default DM_only BAO reference embedded.",
        "Cosmology Falsification (Methods 3, 5): Λ regularization + H₀ pressure derivation.",
        "EKK BAO Re-analysis: geometry-first D_M/r_d, D_H/r_d, α_⊥/α_∥/α_AP vs DR2 full covariance.",
        "dynesty nested sampling: geometric Tau-SB vs physical aDE evidence (Full Production).",
        "Tracers: ALL_GCcomb, BGS, LRG, ELG, QSO, Lya (see --list-tracers in CLI).",
        "Cobaya = bao_data repo path, not the Cobaya MCMC sampler.",
        "Model compare fits ΛCDM poly, aDE, w0waCDM (CPL DE), and Tau-SB (1/7 + hierarchical).",
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


def _apply_desi_n_selection(
    stdscr,
    params: dict[str, str],
    n: int,
    selection: str,
) -> None:
    """Apply interactive n choice to DESI batch / legacy-BAO form defaults."""
    import curses

    from tav_research.curses_shell import _safe_addstr

    params["n_points"] = str(n)
    params["force_rescan"] = "yes"
    # Strengthen batch behavior for real data
    params["batch_limit"] = str(max(int(n), 20))
    params["include_legacy_bao"] = "yes" if int(n) >= 12 else "no"
    if int(n) >= 12:
        params["high_power_stack"] = "yes"
        params["augment_covariance"] = "yes"
    if _action_requires_single_channel(selection) and not _is_joint_dh_dm_choice(
        params.get("quantity", "")
    ):
        params.setdefault("quantity", DATA_MODE_DEFAULT)

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
            _apply_desi_n_selection(stdscr, params, n, selection)
    return None


def _collect_desi_scan_settings(
    selection: str,
    prepared: dict[str, str],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Shared run_desi_scan / joint-scan parameters from prepared form options."""
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
    output_prefix = (prepared.get("output_prefix") or "tau_sb_desi").strip()
    use_cov = not _yes(prepared.get("diagonal_only", "no"))
    include_legacy_bao = _yes(prepared.get("include_legacy_bao", "no"))
    combine_tracers = tracer == COMBINED_DR2_TRACER
    if tracer == EXTENDED_BAO_TRACER:
        include_legacy_bao = True
        tracer = TRACER

    fit_phase = _yes(prepared.get("fit_phase", "yes"))
    fit_hier_raw = (prepared.get("fit_hier") or "auto").strip().lower()
    if fit_hier_raw == "auto":
        fit_hier: bool | None = None
    elif fit_hier_raw in {"no", "false", "0"}:
        fit_hier = False
    else:
        fit_hier = True

    from tav_shared.n_selector_registry import desi_n_knobs, desi_n_usage_message, parse_n_points

    n_menu = parse_n_points(prepared.get("n_points"))
    n_knobs: dict[str, int] = desi_n_knobs(selection, n_menu) if n_menu is not None else {}
    if n_menu is not None:
        print(f"[TAV ENGINE] {desi_n_usage_message(selection, n_menu)}")

    mcmc_steps = MCMC_STEPS_DEFAULT
    if config.get("run_mcmc"):
        try:
            mcmc_steps = int(prepared.get("mcmc_steps") or MCMC_STEPS_DEFAULT)
        except ValueError:
            mcmc_steps = MCMC_STEPS_DEFAULT

    high_power_stack = _yes(prepared.get("high_power_stack", "no"))
    augment_covariance = _yes(prepared.get("augment_covariance", "no"))
    if include_legacy_bao and selection == "Full Production Pipeline":
        high_power_stack = high_power_stack or _yes(prepared.get("include_legacy_bao", "no"))
        augment_covariance = augment_covariance or _yes(prepared.get("include_legacy_bao", "no"))

    return {
        "cobaya_path": cobaya_path,
        "tracer": tracer,
        "gamma": gamma,
        "n_hier": n_hier,
        "use_covariance": use_cov,
        "include_legacy_bao": include_legacy_bao,
        "combine_tracers": combine_tracers,
        "high_power_stack": high_power_stack,
        "augment_covariance": augment_covariance,
        "fit_phase": fit_phase,
        "fit_hier": fit_hier,
        "output_prefix": output_prefix,
        "mcmc_steps": mcmc_steps,
        "n_knobs": n_knobs,
        "pipelines_enabled": prepared.get("pipelines_enabled", ""),
        "run_fit": config.get("run_fit", False),
        "compare_models": config.get("compare_models", False),
        "run_dipole": config.get("run_dipole", False),
        "run_mcmc": config.get("run_mcmc", False),
        "run_changepoints": config.get("run_changepoints", False),
        "run_binding_kit": config.get("run_binding_kit", False),
        "run_injection_recovery": config.get("run_injection_recovery", False),
        "run_joint_fit": config.get("run_joint_fit", False),
        "run_nested_sampling": config.get("run_nested_sampling", False)
        or _yes(prepared.get("nested_sampling", "no")),
        "use_jax_geometric": _yes(prepared.get("jax_geometric", "yes")),
        "plot": config.get("plot", False),
    }


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


def _lss_option_int(prepared: dict[str, str], key: str, default: int) -> int:
    try:
        return int(float(prepared.get(key) or default))
    except (TypeError, ValueError):
        return default


def _lss_option_float(prepared: dict[str, str], key: str, default: float) -> float:
    try:
        return float(prepared.get(key) or default)
    except (TypeError, ValueError):
        return default


def _run_desi_catalog_grid_action(
    selection: str,
    prepared: dict[str, str],
    *,
    show_plots: bool,
) -> str | None:
    from menus.astronomical.desi.desi_catalog_to_grid import run_desi_catalog_to_grid

    catalog_path = (prepared.get("catalog_path") or "").strip() or None
    tracer = prepared.get("tracer") or TRACER
    report = run_desi_catalog_to_grid(
        catalog_path=catalog_path,
        tracer=tracer,
        galaxies_per_shell=_lss_option_int(prepared, "galaxies_per_shell", 800),
        n_per_axis=_lss_option_int(prepared, "n_per_axis", 64),
        box_size_mpc=_lss_option_float(prepared, "box_size_mpc", 2000.0),
        output_prefix=(prepared.get("output_prefix") or "desi_catalog_grid").strip(),
    )
    print(f"[TAV ENGINE] {selection} → {report.get('report_path')}")
    if report.get("npz_path"):
        print(f"[TAV ENGINE] δ grid NPZ: {report['npz_path']}")
    return str(report.get("report_path") or "")


def _run_method10_action(
    selection: str,
    prepared: dict[str, str],
    *,
    show_plots: bool,
) -> str | None:
    from menus.astronomical.desi.method_10_likelihood import run_method10_joint_likelihood

    delta_npz = (prepared.get("delta_npz") or "").strip() or None
    tracer = prepared.get("tracer") or TRACER
    plot = _yes(prepared.get("plot", "yes"))
    show_popup = (
        parse_show_graphics(prepared, default="artifacts") == "popup" if show_plots else False
    )
    quantity = prepared.get("quantity") or DATA_MODE_METHOD10_DEFAULT
    quantity_filter = normalize_quantity_filter(quantity) or "DM_over_rs"
    report = run_method10_joint_likelihood(
        delta_npz=delta_npz,
        tracer=tracer,
        quantity_filter=quantity_filter,
        box_size_mpc=_lss_option_float(prepared, "box_size_mpc", 2000.0),
        use_jax=_yes(prepared.get("use_jax", "yes")),
        dimension=_lss_option_int(prepared, "dimension", 3),
        n_per_axis=_lss_option_int(prepared, "n_per_axis", 64),
        galaxies_per_shell=_lss_option_int(prepared, "galaxies_per_shell", 600),
        build_catalog_grid=delta_npz is None,
        output_prefix=(prepared.get("output_prefix") or "method10_joint").strip(),
        plot=plot,
    )
    print(f"[TAV ENGINE] {selection} → {report.get('verdict', report.get('method10', {}).get('verdict'))}")
    print(f"[TAV ENGINE] Report: {report.get('report_path')}")
    if plot and report.get("plot_path"):
        if show_popup:
            _show_saved_plot(report["plot_path"])
        else:
            print(f"[TAV ENGINE] Plot: {report['plot_path']}")
    return str(report.get("report_path") or "")


def _run_redshift_law_action(
    selection: str,
    prepared: dict[str, str],
    *,
    show_plots: bool,
) -> str | None:
    from menus.astronomical.desi.redshift_law import run_redshift_law_test

    tracer = prepared.get("tracer") or TRACER
    plot = _yes(prepared.get("plot", "yes"))
    show_popup = (
        parse_show_graphics(prepared, default="artifacts") == "popup" if show_plots else False
    )
    report = run_redshift_law_test(
        tracer=tracer,
        max_sne=_lss_option_int(prepared, "max_sne", 80),
        output_prefix=(prepared.get("output_prefix") or "redshift_law_test2").strip(),
        plot=plot,
    )
    print(f"[TAV ENGINE] {selection} → {report.get('fit', {}).get('verdict')}")
    print(f"[TAV ENGINE] Report: {report.get('report_path')}")
    if plot and report.get("plot_path"):
        if show_popup:
            _show_saved_plot(report["plot_path"])
        else:
            print(f"[TAV ENGINE] Plot: {report['plot_path']}")
    return str(report.get("report_path") or "")


def _run_publication_figures_action(
    selection: str,
    prepared: dict[str, str],
    *,
    show_plots: bool,
) -> str | None:
    from menus.astronomical.desi.lss_visualization import run_publication_figures

    plot = True
    show_popup = (
        parse_show_graphics(prepared, default="artifacts") == "popup" if show_plots else False
    )
    quantity = prepared.get("quantity") or DATA_MODE_PUBLICATION_DEFAULT
    quantity_filter = normalize_quantity_filter(quantity) or "DM_over_rs"
    report = run_publication_figures(
        n_points=_lss_option_int(prepared, "n_points", 4096),
        box_size_mpc=_lss_option_float(prepared, "box_size_mpc", 3000.0),
        noise_level=_lss_option_float(prepared, "noise_level", 0.25),
        use_jax=_yes(prepared.get("use_jax", "yes")),
        dimension=_lss_option_int(prepared, "dimension", 1),
        tracer=prepared.get("tracer") or TRACER,
        quantity_filter=quantity_filter,
        output_prefix=(prepared.get("output_prefix") or "lss_manuscript").strip(),
    )
    print(f"[TAV ENGINE] {selection} → {report.get('report_path')}")
    if report.get("suite_path"):
        if show_popup:
            _show_saved_plot(report["suite_path"])
        else:
            print(f"[TAV ENGINE] Suite: {report['suite_path']}")
    return str(report.get("report_path") or "")


def _run_cosmology_falsification_action(
    selection: str,
    prepared: dict[str, str],
    *,
    show_plots: bool,
) -> str | None:
    from menus.astronomical.desi.cosmology_falsification import (
        run_cosmology_falsification_suite,
    )

    plot = _yes(prepared.get("plot", "yes"))
    show_popup = (
        parse_show_graphics(prepared, default="artifacts") == "popup" if show_plots else False
    )
    report = run_cosmology_falsification_suite(
        output_prefix=(prepared.get("output_prefix") or "cosmology_suite").strip(),
        plot=plot,
    )
    print(f"[TAV ENGINE] {selection} → {report.get('verdict', report.get('report_path'))}")
    print(f"[TAV ENGINE] Report: {report.get('report_path')}")
    if plot and report.get("plot_path"):
        if show_popup:
            _show_saved_plot(report["plot_path"])
        else:
            print(f"[TAV ENGINE] Plot: {report['plot_path']}")
    return str(report.get("report_path") or "")


def _run_ekk_bao_action(
    selection: str,
    prepared: dict[str, str],
    *,
    show_plots: bool,
) -> str | None:
    from menus.astronomical.desi.bao_ekk_geometry import run_ekk_bao_reanalysis

    quantity = prepared.get("quantity") or "joint_DH_DM"
    qf = normalize_quantity_filter(quantity) or quantity
    try:
        gamma = float(prepared.get("gamma") or "8.8511")
    except ValueError:
        gamma = 8.8511
    plot = _yes(prepared.get("plot", "yes"))
    show_popup = (
        parse_show_graphics(prepared, default="artifacts") == "popup" if show_plots else False
    )
    report = run_ekk_bao_reanalysis(
        tracer=prepared.get("tracer") or TRACER,
        quantity_filter=qf or "joint_DH_DM",
        gamma=gamma,
        output_prefix=(prepared.get("output_prefix") or "ekk_bao_reanalysis").strip(),
        plot=plot,
    )
    comp = report.get("comparison", {})
    print(f"[TAV ENGINE] {selection} → {comp.get('verdict', report.get('report_path'))}")
    print(f"[TAV ENGINE] Report: {report.get('report_path')}")
    if plot and report.get("plot_path"):
        if show_popup:
            _show_saved_plot(report["plot_path"])
        else:
            print(f"[TAV ENGINE] Plot: {report['plot_path']}")
    return str(report.get("report_path") or "")


def _run_lss_falsification_action(
    selection: str,
    prepared: dict[str, str],
    config: dict[str, Any],
    *,
    show_plots: bool,
) -> str | None:
    from menus.astronomical.desi import lss_falsification as lss

    n_points = _lss_option_int(prepared, "n_points", 4096)
    box_size = _lss_option_float(prepared, "box_size_mpc", 3000.0)
    noise = _lss_option_float(prepared, "noise_level", 0.25)
    use_jax = _yes(prepared.get("use_jax", "yes"))
    dimension = _lss_option_int(prepared, "dimension", 1)
    prefix = (prepared.get("output_prefix") or "lss_falsification").strip()
    plot = bool(config.get("plot"))
    show_popup = (
        parse_show_graphics(prepared, default="artifacts") == "popup" if show_plots else False
    )

    if config.get("lss_comb_falsification"):
        report = lss.run_lss_comb_falsification(
            n_points=n_points,
            box_size_mpc=box_size,
            noise_level=noise,
            inject_signal=_yes(prepared.get("inject_signal", "no")),
            use_jax=use_jax,
            dimension=dimension,
            output_prefix=prefix,
            plot=plot,
        )
    elif config.get("lss_sn_nodes"):
        report = lss.run_sn_node_cross_correlation(
            n_points=n_points,
            box_size_mpc=box_size,
            noise_level=noise,
            inject_nodes=_yes(prepared.get("inject_nodes", "yes")),
            n_max=_lss_option_int(prepared, "sn_n_max", 5000),
            output_prefix=prefix,
            plot=plot,
        )
    elif config.get("lss_gridded_mock"):
        report = lss.run_gridded_mock_comb_recovery(
            n_points=n_points,
            box_size_mpc=box_size,
            noise_level=noise,
            use_jax=use_jax,
            dimension=dimension,
            output_prefix=prefix,
            plot=plot,
        )
    else:
        return None

    print(f"[TAV ENGINE] {selection} → {report.get('report_path')}")
    if plot and report.get("plot_path"):
        if show_popup:
            _show_saved_plot(report["plot_path"])
        else:
            print(f"[TAV ENGINE] Plot: {report['plot_path']}")
    return str(report.get("report_path") or "")


def run_action(selection: str, show_plots: bool = True, options: dict | None = None) -> str | None:
    """Run a DESI action. Returns JSON report path when a scan completes."""
    prepared = prepare_run_options(selection, options)
    config = _action_config(selection)
    if not config:
        print(f"[TAV ENGINE] Unknown DESI Tau-SB action: {selection}")
        return None

    if config.get("desi_catalog_grid"):
        return _run_desi_catalog_grid_action(selection, prepared, show_plots=show_plots)

    if config.get("method10_joint"):
        return _run_method10_action(selection, prepared, show_plots=show_plots)

    if config.get("redshift_law_test2"):
        return _run_redshift_law_action(selection, prepared, show_plots=show_plots)

    if config.get("lss_publication_figures"):
        return _run_publication_figures_action(selection, prepared, show_plots=show_plots)

    if config.get("cosmology_falsification"):
        return _run_cosmology_falsification_action(
            selection, prepared, show_plots=show_plots
        )

    if config.get("ekk_bao_reanalysis"):
        return _run_ekk_bao_action(selection, prepared, show_plots=show_plots)

    if config.get("lss_comb_falsification") or config.get("lss_sn_nodes") or config.get("lss_gridded_mock"):
        return _run_lss_falsification_action(
            selection, prepared, config, show_plots=show_plots
        )

    if config.get("mock_catalog"):
        from menus.astronomical.desi.mock_catalog import (
            generate_full_mock_catalog,
            run_full_mock_recovery_pipeline,
        )

        run_nested = _yes(prepared.get("nested_sampling", "no"))
        multi_tracer = _yes(prepared.get("multi_tracer_stack", "yes"))
        retain_bins = _yes(prepared.get("retain_all_z_bins", "yes"))
        quick = _yes(prepared.get("mock_quick", "yes"))
        if quick:
            catalog = generate_full_mock_catalog(
                A_fracs=(0.02, 0.05, 0.08),
                delta_n_values=(0.33, 0.25),
                multi_tracer_stack=multi_tracer,
                retain_all_z_bins=retain_bins,
            )
        else:
            catalog = generate_full_mock_catalog(
                multi_tracer_stack=multi_tracer,
                retain_all_z_bins=retain_bins,
            )
        summary = run_full_mock_recovery_pipeline(
            catalog,
            run_nested=run_nested,
            output_prefix=(prepared.get("output_prefix") or "tau_sb_mock_catalog").strip(),
        )
        return summary.get("summary_path")

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

    quantity = prepared.get("quantity") or TAU_SEQUENCE["quantity"]
    quantity_filter = _quantity_filter_from_option(quantity)
    show_popup = parse_show_graphics(prepared, default="popup") if show_plots else False
    settings = _collect_desi_scan_settings(selection, prepared, config)
    cobaya_path = settings["cobaya_path"]
    output_prefix = settings["output_prefix"]
    use_cov = settings["use_covariance"]
    n_knobs = settings["n_knobs"]

    if not cobaya_path and not DEFAULT_COBAYA_ROOT.is_dir():
        raise FileNotFoundError(
            f"DESI DR2 tables missing at {DEFAULT_COBAYA_ROOT}. "
            "Auto-fetch should have run before this action."
        )

    try:
        batch_limit = int(prepared.get("batch_limit") or 0)
    except ValueError:
        batch_limit = 0
    if batch_limit > 0 and _yes(prepared.get("force_rescan", "no")):
        from menus.astronomical.desi.production import run_batch_tracer_scan

        prefetch_q = (
            _quantity_filter_from_option(DATA_MODE_DEFAULT)
            if _is_joint_dh_dm_choice(quantity)
            else quantity_filter
        )
        print(
            f"[TAV ENGINE] Batch tracer prefetch: up to {batch_limit} tracers "
            f"(force_rescan=yes) before {selection}"
        )
        run_batch_tracer_scan(
            cobaya_path=cobaya_path,
            quantity_filter=prefetch_q,
            auto_calibrate_gamma=True,
            compare_models=bool(config.get("compare_models")),
            use_covariance=use_cov,
            output_prefix=f"{output_prefix}_batch_n{batch_limit}",
            max_tracers=batch_limit,
            force_rescan=True,
        )

    if _is_joint_dh_dm_choice(quantity):
        from menus.astronomical.desi.production import (
            format_joint_dh_dm_summary_lines,
            run_joint_dh_dm_scan,
        )

        try:
            batch = run_joint_dh_dm_scan(
                cobaya_path=cobaya_path,
                tracer=settings["tracer"],
                gamma=settings["gamma"],
                n_hier=settings["n_hier"],
                use_covariance=use_cov,
                run_fit=settings["run_fit"],
                compare_models=settings["compare_models"],
                run_dipole=settings["run_dipole"],
                run_mcmc=settings["run_mcmc"],
                run_changepoints=settings["run_changepoints"],
                run_binding_kit=settings["run_binding_kit"],
                run_injection_recovery=settings["run_injection_recovery"],
                run_joint_fit=settings["run_joint_fit"],
                mcmc_steps=settings["mcmc_steps"],
                plot=settings["plot"],
                output_prefix=output_prefix,
                action_name=selection,
                pipelines_enabled=settings["pipelines_enabled"],
                include_legacy_bao=settings["include_legacy_bao"],
                combine_tracers=settings["combine_tracers"],
                fit_phase=settings["fit_phase"],
                fit_hier=settings["fit_hier"],
                n_freq=n_knobs.get("n_freq"),
                n_injection_trials=n_knobs.get("n_injection_trials"),
                max_sne=n_knobs.get("max_sne"),
                high_power_stack=settings["high_power_stack"],
                augment_covariance=settings["augment_covariance"],
                run_nested_sampling=settings["run_nested_sampling"],
                use_jax_geometric=settings["use_jax_geometric"],
            )
        except (FileNotFoundError, ValueError, NotImplementedError) as exc:
            print(f"[TAV ENGINE] DESI Joint_DH_DM split failed: {exc}")
            try:
                available = list_desi_tracers(cobaya_path)
                print(f"[TAV ENGINE] Available tracers: {', '.join(available)}")
            except Exception:
                pass
            raise
        except Exception as exc:
            label = type(exc).__name__
            detail = str(exc).strip()
            print(f"[TAV ENGINE] DESI Joint_DH_DM run failed: {label}" + (f": {detail}" if detail else ""))
            return None

        print("\n".join(format_joint_dh_dm_summary_lines(batch)))
        if settings["plot"] and show_popup:
            for mode in ("DH_only", "DM_only"):
                ch = (batch.get("channels") or {}).get(mode) or {}
                _show_saved_plot(ch.get("plot_path"))
        elif settings["plot"]:
            for mode in ("DH_only", "DM_only"):
                ch = (batch.get("channels") or {}).get(mode) or {}
                if ch.get("plot_path"):
                    print(f"[TAV ENGINE] {mode} plot: {ch['plot_path']}")
        return batch.get("summary_path")

    try:
        result = run_desi_scan(
            cobaya_path=cobaya_path,
            tracer=settings["tracer"],
            quantity_filter=quantity_filter,
            gamma=settings["gamma"],
            auto_calibrate_gamma=True,
            n_hier=settings["n_hier"],
            use_covariance=use_cov,
            run_fit=settings["run_fit"],
            compare_models=settings["compare_models"],
            run_dipole=settings["run_dipole"],
            run_mcmc=settings["run_mcmc"],
            run_changepoints=settings["run_changepoints"],
            run_binding_kit=settings["run_binding_kit"],
            run_injection_recovery=settings["run_injection_recovery"],
            run_joint_fit=settings["run_joint_fit"],
            mcmc_steps=settings["mcmc_steps"],
            plot=settings["plot"],
            output_prefix=output_prefix,
            action_name=selection,
            pipelines_enabled=settings["pipelines_enabled"],
            include_legacy_bao=settings["include_legacy_bao"],
            combine_tracers=settings["combine_tracers"],
            fit_phase=settings["fit_phase"],
            fit_hier=settings["fit_hier"],
            n_freq=n_knobs.get("n_freq"),
            n_injection_trials=n_knobs.get("n_injection_trials"),
            max_sne=n_knobs.get("max_sne"),
            high_power_stack=settings["high_power_stack"],
            augment_covariance=settings["augment_covariance"],
            run_nested_sampling=settings["run_nested_sampling"],
            use_jax_geometric=settings["use_jax_geometric"],
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
    if settings["plot"] and show_popup:
        _show_saved_plot(result.plot_path)
    elif settings["plot"] and result.plot_path:
        print(f"[TAV ENGINE] Summary plot: {result.plot_path}")
    return result.report_path