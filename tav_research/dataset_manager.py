"""
Dataset Manager submenu — fetch/run registered datasets from the Primary Matrix.
"""

from __future__ import annotations

import menus.astronomical.sparc.extension as sparc_extension
from tav_shared.llm_analysis import append_llm_entry_fields

# =============================================================================
# BLOCK: Constants
# =============================================================================

DATASET_MANAGER_TITLE = "Dataset Manager"
DATASET_FETCH_ACTION = "Fetch Selected Datasets"
DATASET_RUN_ACTION = "Run Selected Datasets"
DATASET_MANAGER_TAG = "__DATASET_MANAGER__"


# =============================================================================
# BLOCK: Entry form fields
# =============================================================================


def dataset_manager_fetch_fields() -> list[dict]:
    from tav_shared.tav_project_paths import FINISHED2_DIR, FINISHED_A_DIR

    return [
        {
            "key": "force_refresh",
            "label": "Force re-download",
            "default": "no",
            "required": False,
            "hint": "Re-fetch cached or already-processed targets",
            "choices": ["no", "yes"],
        },
        {
            "key": "restore_archived",
            "label": "Restore archived",
            "default": "yes",
            "required": False,
            "hint": f"Copy from {FINISHED_A_DIR}/ or {FINISHED2_DIR}/ before remote pull",
            "choices": ["no", "yes"],
        },
    ]


def dataset_manager_run_fields(module_tag: str) -> list[dict]:
    """Common run options shown after Dataset Manager target selection."""
    fields = [
        {
            "key": "show_graphics",
            "label": "Graphics mode",
            "default": "artifacts",
            "required": False,
            "hint": "popup = Tk windows | artifacts = PNG only",
            "choices": ["artifacts", "popup"],
        },
    ]
    if module_tag == sparc_extension.MODULE_TAG:
        fields.insert(
            0,
            {
                "key": "batch_limit",
                "label": "Batch limit (0 = all)",
                "default": "10",
                "required": False,
                "hint": "Max datasets to process this run",
            },
        )
    return append_llm_entry_fields(fields)


def _query_fallback_action(module_tag: str) -> str | None:
    from tav_shared.dataset_registry import get_provider

    provider = get_provider(module_tag)
    if provider and provider.run_actions:
        return provider.run_actions[0]
    return None


# =============================================================================
# BLOCK: Run action
# =============================================================================


def run_dataset_manager_action(params: dict) -> None:
    from tav_shared.dataset_registry import fetch_selected, run_selected

    module_tag = params.get("module_tag", "")
    target_ids = params.get("target_ids") or []
    mode = params.get("dataset_manager_mode", "fetch")
    if not module_tag or not target_ids:
        print("[TAV ENGINE] Dataset Manager: no module or targets selected.")
        return

    if mode == "fetch":
        print(f"\n[TAV ENGINE] Fetching {len(target_ids)} dataset(s) for {module_tag}...")
        params.setdefault("force_refresh", "no")
        result = fetch_selected(module_tag, target_ids, params)
        print(
            f"[TAV ENGINE] Fetch complete — downloaded: {len(result.fetched)}, "
            f"skipped: {len(result.skipped)}, failed: {len(result.failed)}"
        )
        if result.fetched:
            print(f"  fetched: {', '.join(result.fetched[:12])}")
        if result.failed:
            for key, msg in result.failed.items():
                print(f"  FAIL {key}: {msg}")
        return

    action = params.get("dataset_manager_action") or _query_fallback_action(module_tag)
    if not action:
        print(f"[TAV ENGINE] No run action for module {module_tag}")
        return
    print(f"\n[TAV ENGINE] Running {action} on {len(target_ids)} dataset(s)...")
    run_selected(module_tag, action, target_ids, params)