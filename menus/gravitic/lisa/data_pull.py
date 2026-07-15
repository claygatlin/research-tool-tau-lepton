"""LISA Pre-runs — Pelican/OSDF fetch for IGWN mock data."""

from __future__ import annotations

from tav_research.data_pull_common import COMMON_BATCH_HINT, standard_entry_fields

from menus.gravitic.lisa.fetcher import LISA_OSDF_TARGETS, fetch_and_graph

MENU_LABEL = "LISA Pre-runs"


def entry_instructions() -> list[str]:
    keys = ", ".join(sorted(LISA_OSDF_TARGETS))
    return [
        "Query: manifest key, osdf:/// URI, local file path, or cached relative path.",
        f"Manifest keys: {keys}",
        "Transport: requests-pelican (Python) with pelican CLI fallback.",
        "Cached under datasets/lisa/ with resume ledger done.txt.",
        "Install stack: scripts/research_tool/install_gravitic_deps.sh",
        COMMON_BATCH_HINT,
    ]


def entry_fields() -> list[dict]:
    return standard_entry_fields(
        MENU_LABEL,
        query_hint="igwn_readme, osdf:///igwn/…, or local .hdf5/.dat",
    )


__all__ = ["MENU_LABEL", "entry_fields", "entry_instructions", "fetch_and_graph"]