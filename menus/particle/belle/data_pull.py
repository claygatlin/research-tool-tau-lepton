"""Belle II (Meson Oscillation) — local / HEPData placeholder."""

from __future__ import annotations

from tav_research.data_pull_common import COMMON_BATCH_HINT, standard_entry_fields

MENU_LABEL = "Belle II (Meson Oscillation)"


def entry_instructions() -> list[str]:
    return [
        "Query: Belle II open-data filename or HEPData record ID.",
        "Files: local .root / .csv from Belle II releases.",
        "URLs: https://belle2.jp/ — download locally.",
        COMMON_BATCH_HINT,
    ]


def entry_fields() -> list[dict]:
    return standard_entry_fields(MENU_LABEL)


def fetch_and_graph(query: str, params: dict | None = None) -> None:
    print(f"[TAV ENGINE] {MENU_LABEL}: local fetch not fully wired for {query!r}")