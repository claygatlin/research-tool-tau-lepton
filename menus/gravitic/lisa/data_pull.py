"""LISA Pre-runs — local simulation placeholder."""

from __future__ import annotations

from tav_research.data_pull_common import COMMON_BATCH_HINT, standard_entry_fields

MENU_LABEL = "LISA Pre-runs"


def entry_instructions() -> list[str]:
    return [
        "Query: LISA mock-data run label or local simulation output name.",
        "Files: .hdf5, .dat, .csv time-series stored locally.",
        COMMON_BATCH_HINT,
    ]


def entry_fields() -> list[dict]:
    return standard_entry_fields(MENU_LABEL)


def fetch_and_graph(query: str, params: dict | None = None) -> None:
    print(f"[TAV ENGINE] {MENU_LABEL}: local fetch not fully wired for {query!r}")