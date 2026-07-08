"""
Router for legacy repository data-pull menus (HEPData, ALICE, CMS, …).

Each repository handler lives under ``menus/<domain>/<repo>/data_pull.py``.
This module re-exports shared batch helpers and dispatches by menu label.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from tav_research.data_pull_common import (
    COMMON_BATCH_HINT,
    is_batch_list_file,
    iter_resumable_batch_lines,
    standard_entry_fields,
)

from menus.astronomical.halogas import data_pull as halogas_pull
from menus.gravitic.ligo import data_pull as ligo_pull
from menus.gravitic.lisa import data_pull as lisa_pull
from menus.particle.alice import data_pull as alice_pull
from menus.particle.atlas import data_pull as atlas_pull
from menus.particle.belle import data_pull as belle_pull
from menus.particle.cms import data_pull as cms_pull
from menus.particle.hepdata import data_pull as hepdata_pull

# Backward-compatible alias used by runner.py
_iter_resumable_batch_lines = iter_resumable_batch_lines

_REPO_MODULES: dict[str, Any] = {
    halogas_pull.MENU_LABEL: halogas_pull,
    ligo_pull.MENU_LABEL: ligo_pull,
    lisa_pull.MENU_LABEL: lisa_pull,
    hepdata_pull.MENU_LABEL: hepdata_pull,
    cms_pull.MENU_LABEL: cms_pull,
    atlas_pull.MENU_LABEL: atlas_pull,
    alice_pull.MENU_LABEL: alice_pull,
    belle_pull.MENU_LABEL: belle_pull,
}

_DEFAULT_INSTRUCTIONS = [
    "Query: dataset ID, local filename, or prepared index token.",
    "Files: .csv, .txt, .root as supported by the selected repository handler.",
    "URLs: only HEPData INSPIRE IDs are fetched live; other hosts require local files.",
    COMMON_BATCH_HINT,
]


def _handler_for(source: str) -> Any | None:
    return _REPO_MODULES.get(source)


def data_pull_entry_instructions(source: str) -> list[str]:
    module = _handler_for(source)
    if module is not None and hasattr(module, "entry_instructions"):
        return module.entry_instructions()
    return list(_DEFAULT_INSTRUCTIONS)


def data_pull_entry_fields(source: str) -> list[dict]:
    module = _handler_for(source)
    if module is not None and hasattr(module, "entry_fields"):
        return module.entry_fields()
    return standard_entry_fields(source)


def fetch_and_graph(repo: str, query: str, params: dict[str, Any] | None = None) -> None:
    params = params or {}
    print(f"\n[TAV ENGINE] Initiating fetch for: {query} in {repo}...")
    if params:
        print(f"[TAV ENGINE] Entry parameters: {params}")

    batch_file = (params.get("batch_file") or "").strip()
    if batch_file and os.path.isfile(batch_file) and is_batch_list_file(batch_file):
        print(f"[TAV ENGINE] Batch mode via entry form: {batch_file}")
        for batch_query in iter_resumable_batch_lines(batch_file, params):
            fetch_and_graph(
                repo,
                batch_query,
                {**params, "batch_file": "", "_record_batch_done": batch_query},
            )
        return

    module = _handler_for(repo)
    if module is None:
        print(f"[TAV ENGINE] Domain {repo} not wired.")
        return

    fetch_fn: Callable[[str, dict[str, Any] | None], None] = module.fetch_and_graph
    fetch_fn(query, params)

    record_key = (params.get("_record_batch_done") or "").strip()
    if record_key:
        from tav_shared.batch_ledger import HEPDATA_BATCH_DONE, append_done_entries

        append_done_entries(HEPDATA_BATCH_DONE, [record_key])