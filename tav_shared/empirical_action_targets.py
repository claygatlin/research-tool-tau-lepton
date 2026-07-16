"""
Per-action ``empirical:`` target lists for auto_dataset pull→cache.

Keys match ``menus.empirical_tests.tests.REPOS`` (resolved via EMPIRICAL_TESTS provider).
"""

from __future__ import annotations

from typing import Any

# Prime Past — BBN confrontation scans
PRIME_PAST_HARMONIC_ACTION_TARGETS: dict[str, list[str]] = {
    "BBN Interference Scan": ["bbn_abundances", "neutron_lifetime"],
    "BBN Confrontation Scan": ["bbn_abundances", "neutron_lifetime"],
    "Enhanced BBN Confrontation": ["bbn_abundances", "neutron_lifetime"],
    "Final BBN Confrontation Tool": ["bbn_abundances", "neutron_lifetime"],
}

# TSB Research — theory likelihoods anchored to lattice/PDG observables
TSB_RESEARCH_ACTION_TARGETS: dict[str, list[str]] = {
    "Fractal Tau Circle Likelihood": [
        "glueball_lattice",
        "neutron_lifetime",
    ],
    "Run Residual Diagnostics": [
        "neutron_lifetime",
    ],
    "Calibrate SoundHorizon": [
        "neutron_lifetime",
    ],
}

_MODULE_ACTION_MAP: dict[str, dict[str, list[str]]] = {
    "EMPIRICAL_TESTS": {},  # filled from extension at runtime
    "PRIME_PAST_HARMONIC": PRIME_PAST_HARMONIC_ACTION_TARGETS,
    "TSB_RESEARCH": TSB_RESEARCH_ACTION_TARGETS,
}


def empirical_keys_for_action(module_tag: str, action: str) -> list[str]:
    """Return REPOS keys to fetch as ``empirical:{key}`` for this menu action."""
    if module_tag == "EMPIRICAL_TESTS":
        try:
            from menus.empirical_tests.extension import _ACTION_CONFIG

            sources = (_ACTION_CONFIG.get(action) or {}).get("sources") or []
            return list(sources)
        except ImportError:
            return []

    mapping = _MODULE_ACTION_MAP.get(module_tag, {})
    return list(mapping.get(action, []))


def empirical_target_ids(module_tag: str, action: str) -> list[str]:
    return [f"empirical:{key}" for key in empirical_keys_for_action(module_tag, action)]


def ensure_empirical_targets(
    module_tag: str,
    action: str,
    options: dict[str, Any] | None = None,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Pull→cache empirical datasets for *module_tag* / *action* via registry.

    Stores a provenance block on ``options['_empirical_provenance']`` when provided.
    """
    from tav_shared.dataset_registry import fetch_selected

    options = options if options is not None else {}
    keys = empirical_keys_for_action(module_tag, action)
    if not keys:
        provenance = {
            "module_tag": module_tag,
            "action": action,
            "empirical_keys": [],
            "note": "no empirical targets configured for this action",
        }
        options["_empirical_provenance"] = provenance
        return provenance

    target_ids = [f"empirical:{k}" for k in keys]
    if verbose:
        print(
            f"[TAV ENGINE] Auto-fetch empirical targets for {module_tag} / {action}: "
            f"{', '.join(keys)}"
        )
    result = fetch_selected("EMPIRICAL_TESTS", target_ids, options)
    if result.failed:
        for key, msg in result.failed.items():
            raise RuntimeError(f"Empirical fetch failed for {key}: {msg}")

    provenance = {
        "module_tag": module_tag,
        "action": action,
        "empirical_keys": keys,
        "target_ids": target_ids,
        "fetched": list(result.fetched),
        "skipped": list(result.skipped),
        "pipeline": "empirical_pull_cache_log",
    }
    options["_empirical_provenance"] = provenance
    if verbose and result.fetched:
        print(f"[TAV ENGINE] Empirical materialized: {', '.join(result.fetched)}")
    return provenance