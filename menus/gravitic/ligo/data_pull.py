"""LIGO GWOSC (Strain Data) — live fetch via GWOSC API, GWDataFind, and Pelican."""

from __future__ import annotations

from tav_research.data_pull_common import COMMON_BATCH_HINT, standard_entry_fields

from menus.gravitic.ligo.fetcher import (
    EVENT_API,
    GWOSC_DATA,
    GWOSC_HOME,
    fetch_and_graph,
)

MENU_LABEL = "LIGO GWOSC (Strain Data)"


def entry_instructions() -> list[str]:
    return [
        "Query: GW event ID (e.g. GW150914), GWDataFind segment, or Pelican/OSDF URI.",
        "Segment syntax: OBS:CHANNEL:GPS_START:GPS_END (e.g. H:H1_R:1126259447:1126259479).",
        "Pelican/OSDF examples: osdf:///gwdata/zenodo/README.zenodo",
        f"Catalog API: {EVENT_API}",
        f"Data portal: {GWOSC_DATA}",
        "Cached under datasets/gwosc/ with resume ledger done.txt.",
        "Install stack: scripts/research_tool/install_gravitic_deps.sh",
        COMMON_BATCH_HINT,
    ]


def entry_fields() -> list[dict]:
    return standard_entry_fields(
        MENU_LABEL,
        query_hint="GW150914, H:H1_R:start:end, or osdf:///… URI",
    )


__all__ = ["MENU_LABEL", "entry_fields", "entry_instructions", "fetch_and_graph", "GWOSC_HOME"]