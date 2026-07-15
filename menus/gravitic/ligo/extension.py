"""LIGO GWOSC strain fetch + ringdown analysis module."""

from __future__ import annotations

from typing import Any

from menus.gravitic.ligo.fetcher import fetch_and_graph as pull_strain
from menus.gravitic.ligo.ringdown import run_ringdown_scan
from tav_shared.run_output import parse_show_graphics

MODULE_TAG = "LIGO_GWOSC"
SUBMENU_TITLE = "LIGO GWOSC (Strain Data)"

MENU_ACTIONS = [
    "Pull Strain from GWOSC",
    "Ringdown Harmonic Scan (1/7)",
    "Full Ringdown Report",
]


def entry_fields(action: str) -> list[dict]:
    from tav_shared.llm_analysis import append_llm_entry_fields

    if action == "Pull Strain from GWOSC":
        return append_llm_entry_fields(
            [
                {
                    "key": "query",
                    "label": "Event / segment / URI",
                    "default": "GW150914",
                    "required": True,
                    "hint": "GW150914, H:H1_R:start:end, or osdf:///…",
                },
                {
                    "key": "force_refresh",
                    "label": "Force refresh",
                    "default": "no",
                    "required": False,
                    "hint": "yes = re-download even if cached",
                },
            ]
        )
    return append_llm_entry_fields(
        [
            {
                "key": "event",
                "label": "GW event ID",
                "default": "GW150914",
                "required": True,
                "hint": "Uses cached files under datasets/gwosc/events/",
            },
            {
                "key": "strain_path",
                "label": "Strain file (optional)",
                "default": "",
                "required": False,
                "hint": "Specific cached .hdf5/.txt.gz; blank = all detectors",
            },
            {
                "key": "show_graphics",
                "label": "Save plots",
                "default": "artifacts",
                "required": False,
                "hint": "artifacts | popup | no",
            },
        ]
    )


def entry_instructions(action: str) -> list[str]:
    if action == "Pull Strain from GWOSC":
        return [
            "Fetches open strain via GWOSC Event API, GWDataFind, or Pelican/OSDF.",
            "Caches under datasets/gwosc/ with resume ledger done.txt.",
            "Requires: gwdatafind, requests-pelican, pelican CLI (install_gravitic_deps.sh).",
        ]
    return [
        "Operates on cached GWOSC strain (.hdf5 or .txt.gz).",
        "Bandpasses ringdown window and scans 1/7 sub-harmonic excess (Tav falsification hook).",
        "Writes artifacts/gwosc_ringdown_*.json and optional PNG spectrograms.",
        "Run Pull Strain first if datasets/gwosc/events/<EVENT>/ is empty.",
    ]


def run_action(action: str, options: dict[str, Any] | None = None) -> None:
    options = options or {}
    if action == "Pull Strain from GWOSC":
        query = str(options.get("query", "")).strip()
        if not query:
            print("[LIGO GWOSC] Missing query/event ID.")
            return
        pull_strain(query, options)
        return

    event = str(options.get("event", "")).strip()
    if not event:
        print("[LIGO GWOSC] Missing event ID.")
        return
    strain_path = str(options.get("strain_path", "")).strip() or None
    default_graphics = "artifacts" if action == "Full Ringdown Report" else "artifacts"
    show_graphics = parse_show_graphics(options.get("show_graphics", default_graphics))
    payload = run_ringdown_scan(event, strain_path=strain_path, show_graphics=show_graphics)
    if action == "Full Ringdown Report":
        print("[LIGO GWOSC] Ringdown summary:")
        for report in payload.get("reports", []):
            excess = report.get("subharmonic_excess", {}).get("f_over_7")
            print(
                f"  {report.get('detector')}: f0={report.get('fundamental_hz')} Hz, "
                f"1/7 excess={excess}"
            )