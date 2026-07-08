"""
Shared helpers for legacy repository data-pull menus.
"""

from __future__ import annotations

import os

from tav_shared.batch_ledger import HEPDATA_BATCH_DONE, format_batch_banner, select_batch_items

BATCH_TEXT_EXTENSIONS = {".txt", ".csv", ".lst", ".list"}

COMMON_BATCH_HINT = (
    "Batch file (optional): .txt, .csv, .lst, .list — one dataset target per line, UTF-8 text."
)


def is_batch_list_file(path: str) -> bool:
    """Only plain-text list files should be read line-by-line."""
    return os.path.splitext(path)[1].lower() in BATCH_TEXT_EXTENSIONS


def iter_resumable_batch_lines(batch_file: str, params: dict) -> list[str]:
    """Return the next pending slice from a batch list file."""
    with open(batch_file, "r", encoding="utf-8", errors="replace") as handle:
        queries = [
            line.strip()
            for line in handle
            if line.strip() and not line.strip().startswith("#")
        ]
    try:
        limit = int(params.get("batch_limit") or 0)
    except ValueError:
        limit = 0
    force_rescan = str(params.get("force_rescan", "no")).lower() in {"yes", "y", "true", "1"}
    selected, status = select_batch_items(
        queries,
        HEPDATA_BATCH_DONE,
        key_fn=str,
        limit=limit,
        force_rescan=force_rescan,
    )
    print(format_batch_banner("Batch list", status))
    if not selected and status["done_count"] >= status["catalog_total"] and queries:
        print(
            "[TAV ENGINE] All batch-list entries already processed. "
            "Use force_rescan=yes to restart from the first ID."
        )
    return selected


def standard_entry_fields(source: str, *, query_hint: str = "") -> list[dict]:
    """Default query / batch_file / root_entry_limit fields for data-pull repos."""
    from tav_shared.llm_analysis import append_llm_entry_fields

    return append_llm_entry_fields(
        [
            {
                "key": "query",
                "label": "Dataset ID / Filename",
                "default": "",
                "required": True,
                "hint": query_hint or "Primary dataset target for this repository",
            },
            {
                "key": "batch_file",
                "label": "Batch list file (optional)",
                "default": "",
                "required": False,
                "hint": "Plain-text .txt/.csv list; overrides single query when set",
            },
            {
                "key": "root_entry_limit",
                "label": "ROOT entry limit (optional)",
                "default": "50000",
                "required": False,
                "hint": "Max tracks/events to load from .root files",
            },
        ]
    )