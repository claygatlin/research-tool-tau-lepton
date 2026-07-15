"""LISA / IGWN mock-data module (Pelican OSDF)."""

from __future__ import annotations

from typing import Any

from menus.gravitic.lisa.fetcher import (
    DATASETS_DIR,
    LISA_OSDF_TARGETS,
    fetch_and_graph,
    list_cached_files,
)

MODULE_TAG = "LISA_PRE_RUNS"
SUBMENU_TITLE = "LISA Pre-runs"

MENU_ACTIONS = [
    "Pull OSDF Target",
    "List Cached Mock Data",
    "OSDF Connectivity Check",
]


def entry_fields(action: str) -> list[dict]:
    from tav_shared.llm_analysis import append_llm_entry_fields

    if action == "List Cached Mock Data":
        return append_llm_entry_fields([])
    if action == "OSDF Connectivity Check":
        return append_llm_entry_fields([])
    keys = ", ".join(sorted(LISA_OSDF_TARGETS))
    return append_llm_entry_fields(
        [
            {
                "key": "query",
                "label": "Manifest key / URI",
                "default": "gwdata_zenodo_readme",
                "required": True,
                "hint": f"Keys: {keys}",
            },
            {
                "key": "force_refresh",
                "label": "Force refresh",
                "default": "no",
                "required": False,
            },
        ]
    )


def entry_instructions(action: str) -> list[str]:
    if action == "Pull OSDF Target":
        return [
            "Downloads IGWN/LISA mock data via requests-pelican (public) or pelican CLI (protected).",
            "Cached under datasets/lisa/ with resume ledger done.txt.",
        ]
    if action == "OSDF Connectivity Check":
        return [
            "Probes public OSDF namespaces (gwdata/zenodo, lisa-mock-data, igwn/gwdata).",
            "Protected igwn/ligo paths may require SciTokens.",
        ]
    return ["Lists files currently cached under datasets/lisa/."]


def run_action(action: str, options: dict[str, Any] | None = None) -> None:
    options = options or {}
    if action == "Pull OSDF Target":
        query = str(options.get("query", "")).strip()
        if not query:
            print("[LISA] Missing query/target.")
            return
        fetch_and_graph(query, options)
        return

    if action == "List Cached Mock Data":
        files = list_cached_files()
        if not files:
            print("[LISA] No cached files under datasets/lisa/.")
            return
        print("[LISA] Cached files:")
        for path in files:
            print(f"  - {path}")
        return

    if action == "OSDF Connectivity Check":
        from menus.gravitic.common.download import download_uri
        from menus.gravitic.common.pelican_tools import pelican_status

        print("[LISA] Pelican:", pelican_status())
        checks = list(LISA_OSDF_TARGETS.items())
        for label, uri in checks:
            try:
                name = uri.rsplit("/", 1)[-1] or label
                dest = DATASETS_DIR / "checks" / name
                download_uri(uri, dest, force=True)
                print(f"  OK  {label}: {uri}")
            except Exception as exc:
                print(f"  FAIL {label}: {exc}")
        return

    print(f"[LISA] Unknown action: {action}")