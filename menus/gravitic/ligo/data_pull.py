"""LIGO GWOSC (Strain Data) — local file placeholder."""

from __future__ import annotations

from tav_research.data_pull_common import COMMON_BATCH_HINT, standard_entry_fields

MENU_LABEL = "LIGO GWOSC (Strain Data)"


def entry_instructions() -> list[str]:
    return [
        "Query: GWOSC event ID or local strain filename (e.g. H1_GW150914).",
        "Files: .hdf5 / .gwf / .txt summary tables stored locally.",
        "URLs: https://gwosc.org/ — download, then point to local file.",
        COMMON_BATCH_HINT,
    ]


def entry_fields() -> list[dict]:
    return standard_entry_fields(MENU_LABEL)


def fetch_and_graph(query: str, params: dict | None = None) -> None:
    print(f"[TAV ENGINE] {MENU_LABEL}: local fetch not fully wired for {query!r}")