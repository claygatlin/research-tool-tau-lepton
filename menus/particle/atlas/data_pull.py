"""ATLAS Open Data — local ROOT placeholder."""

from __future__ import annotations

from tav_research.data_pull_common import COMMON_BATCH_HINT, standard_entry_fields

MENU_LABEL = "ATLAS Open Data (Lepton/Photon Tracks)"


def entry_instructions() -> list[str]:
    return [
        "Query: local .root path or ATLAS open-data file index entry.",
        "Files: ATLAS DAOD/ESD .root readable via uproot.",
        "URLs: https://opendata.atlas.cern.ch/ — download first.",
        COMMON_BATCH_HINT,
    ]


def entry_fields() -> list[dict]:
    return standard_entry_fields(MENU_LABEL)


def fetch_and_graph(query: str, params: dict | None = None) -> None:
    print(f"[TAV ENGINE] {MENU_LABEL}: wire local uproot handler for {query!r}")