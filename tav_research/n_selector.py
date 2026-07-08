"""
Sample-size (n) selector policy for batch-capable submenus.

DESI-specific pre-form flow lives in ``tau_sb_desi_extension.handle_pre_form``.
FRB and Tav-Resonance batch limits are applied here from the main loop.
"""

from __future__ import annotations

from tav_research import registry

# =============================================================================
# BLOCK: Action name normalization
# =============================================================================

_MODULE_TAG_TO_EXTENSION = {
    "TAU_SB_DESI": "tau_sb_desi_extension",
    "FRB_COSMIC_WEB_TAV": "frb_web_extension",
    "TAV_RESONANCE": "tav_resonance_extension",
    "TSB_CASIMIR": "tsb_casimir_extension",
    "SPARC": "sparc_extension",
    "TAV_DATA_INTEGRATOR": "integrator_extension",
    "TSB_RESEARCH": "tsb_research_extension",
}

_N_SELECTOR_ACTION_ALIASES = {
    "Injection-Recovery": "Injection/Recovery Tests",
    "Batch Tracer Scan": "Batch Tracer Scan (All DR2)",
    "Model Compare": "Model Compare (ΛCDM vs aDE vs Tau-SB)",
    "Pull Batch from Repository": "Pull Batch from Repository",
    "Scan Extracted Dataset (batch)": "Scan Extracted Dataset (batch)",
    "Full Correlation Pipeline": "Full Correlation Pipeline",
}


def _extension_module_name(module_tag: str) -> str:
    if module_tag.endswith("_extension"):
        return module_tag
    return _MODULE_TAG_TO_EXTENSION.get(module_tag, module_tag)


def _normalize_n_selector_action(action: str) -> str:
    return _N_SELECTOR_ACTION_ALIASES.get(action, action)


# =============================================================================
# BLOCK: When to show the interactive n picker
# =============================================================================


def should_offer_n_selector(module_tag: str, action: str) -> bool:
    """Central switch for when to show the interactive n selector tool-wide."""
    module_tag = _extension_module_name(module_tag)
    action = _normalize_n_selector_action(action)

    desi = registry.tau_sb_desi_extension
    if module_tag == "tau_sb_desi_extension":
        if desi is None:
            return False
        return action in desi.MENU_ACTIONS

    n_sensitive = {"tav_resonance_extension", "frb_web_extension"}
    if module_tag not in n_sensitive:
        return False

    frb_batch_actions = {
        "Classify FRB Paths",
        "Analyze DM Residuals by Path",
        "Scan 1/7 Tav Harmonics",
        "Full FRB Tav Analysis",
    }
    if module_tag == "frb_web_extension" and action in frb_batch_actions:
        return True

    tav_batch_actions = {
        "FRB Resonance Scan (tav-resonance)",
        "Full Tav-Resonance Demo",
    }
    if module_tag == "tav_resonance_extension" and action in tav_batch_actions:
        return True

    return False


# =============================================================================
# BLOCK: Entry-form helpers after n is chosen in the menu loop
# =============================================================================


def strip_auto_filled_fields(fields: list[dict], params: dict[str, str]) -> list[dict]:
    """Drop form fields already set by the automatic n selector."""
    skip: set[str] = set()
    if params.get("n_points"):
        skip.add("n_points")
    if params.get("batch_limit"):
        skip.update({"batch_limit", "force_rescan"})
    if not skip:
        return fields
    return [field for field in fields if field.get("key") not in skip]


def seed_field_defaults_from_params(
    fields: list[dict],
    params: dict[str, str],
) -> list[dict]:
    """Let the entry form reflect values already chosen by the n selector."""
    seeded: list[dict] = []
    for field in fields:
        key = field.get("key")
        if key and key in params:
            seeded.append({**field, "default": params[key]})
        else:
            seeded.append(field)
    return seeded


def apply_frb_or_tav_n_selection(params: dict[str, str], n: int, module_tag: str) -> None:
    """Set batch_limit / force_rescan for FRB and Tav-Resonance actions."""
    frb = registry.frb_web_extension
    tav = registry.tav_resonance_extension
    if frb is not None and module_tag == frb.MODULE_TAG:
        params["batch_limit"] = str(n)
        params["force_rescan"] = "yes"
    elif tav is not None and module_tag == tav.MODULE_TAG:
        params["batch_limit"] = str(n)
        params["force_rescan"] = "yes"
    else:
        params["n_points"] = str(n)