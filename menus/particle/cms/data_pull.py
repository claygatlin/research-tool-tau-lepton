"""CMS Open Data — local ROOT placeholder."""

from __future__ import annotations

from tav_research.data_pull_common import COMMON_BATCH_HINT, standard_entry_fields

MENU_LABEL = "CMS Open Data (NanoAOD / Electrons)"


def entry_instructions() -> list[str]:
    return [
        "Query: local .root path (e.g. nanoaod.root) or open-data index stem.",
        "Files: CMS NanoAOD .root with uproot-readable trees.",
        "URLs: https://opendata.cern.ch/ — download ROOT, then pass local path.",
        COMMON_BATCH_HINT,
    ]


def entry_fields() -> list[dict]:
    return standard_entry_fields(
        MENU_LABEL,
        query_hint="Local .root path or open-data record",
    )


def fetch_and_graph(query: str, params: dict | None = None) -> None:
    print(f"[TAV ENGINE] {MENU_LABEL}: wire local uproot handler for {query!r}")