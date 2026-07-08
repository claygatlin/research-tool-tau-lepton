"""HALOGAS (HI Data Cubes) — local file placeholder."""

from __future__ import annotations

from tav_research.data_pull_common import COMMON_BATCH_HINT, standard_entry_fields

MENU_LABEL = "HALOGAS (HI Data Cubes)"


def entry_instructions() -> list[str]:
    return [
        "Query: HALOGAS source ID or prepared local filename stem.",
        "Files: .csv / .fits cubes placed locally (parser not fully wired).",
        "URLs: https://www.halogas.org/ — fetch archives manually for now.",
        COMMON_BATCH_HINT,
    ]


def entry_fields() -> list[dict]:
    return standard_entry_fields(MENU_LABEL)


def fetch_and_graph(query: str, params: dict | None = None) -> None:
    print(f"[TAV ENGINE] {MENU_LABEL}: local fetch not fully wired for {query!r}")