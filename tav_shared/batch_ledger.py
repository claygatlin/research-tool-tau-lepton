#!/usr/bin/env python3
"""
Shared done.txt / batch_done.txt ledger for resumable batch pulls and scans.

Each module keeps a ledger under its datasets/ folder. A batch run selects the
next pending slice; successful items are appended so the following run resumes
where the previous one stopped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

T = TypeVar("T")

from tav_shared.tav_project_paths import DATASETS_ROOT

SPARC_PULL_DONE = DATASETS_ROOT / "sparc" / "done.txt"
SPARC_BATCH_DONE = DATASETS_ROOT / "sparc" / "batch_done.txt"
CASIMIR_BATCH_DONE = DATASETS_ROOT / "casimir" / "done.txt"
DESI_BATCH_DONE = DATASETS_ROOT / "desi" / "batch_done.txt"
INTEGRATOR_PULL_DONE = DATASETS_ROOT / "fits" / "done.txt"
FRB_PULL_DONE = DATASETS_ROOT / "frb" / "done.txt"
HEPDATA_BATCH_DONE = DATASETS_ROOT / "hepdata" / "batch_done.txt"

_LEDGER_HEADERS: dict[Path, tuple[str, ...]] = {
    SPARC_PULL_DONE: (
        "# SPARC repository pull ledger — one galaxy ID per line.",
        "# Successful pulls append here; the next batch resumes at the first pending ID.",
    ),
    SPARC_BATCH_DONE: (
        "# SPARC analysis batch ledger — one galaxy ID per line.",
        "# Successful analysis batches append here; the next run resumes at pending IDs.",
    ),
    CASIMIR_BATCH_DONE: (
        "# TSB Casimir batch scan ledger — one relative path per line (under datasets/casimir/).",
        "# Successful batch scans append here; the next run resumes at the first pending file.",
    ),
    DESI_BATCH_DONE: (
        "# DESI DR2 tracer batch ledger — one tracer key per line.",
        "# Successful tracer scans append here; the next batch resumes at pending tracers.",
    ),
    INTEGRATOR_PULL_DONE: (
        "# Integrator dataset pull ledger — catalog key or filename per line.",
        "# Successful fetches append here; the next pull resumes at pending items.",
    ),
    FRB_PULL_DONE: (
        "# FRB catalog pull ledger — one filename per line.",
        "# Cached pulls append here.",
    ),
    HEPDATA_BATCH_DONE: (
        "# HEPData / batch-list ledger — one dataset ID per line.",
        "# Successful fetches append here; batch mode resumes at pending IDs.",
    ),
}


def clear_ledger(done_file: Path | str) -> Path:
    """Truncate a batch/pull ledger back to its header (fresh start)."""
    path = Path(done_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = _LEDGER_HEADERS.get(path.resolve(), ())
    if not header:
        header = (
            "# Batch ledger — one completed item per line.",
            "# The next batch run resumes at the first pending item.",
        )
    path.write_text("\n".join(header) + "\n", encoding="utf-8")
    return path


def ensure_done_file(done_file: Path | str) -> Path:
    """Create an empty ledger with a standard header when missing."""
    path = Path(done_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return path
    header = _LEDGER_HEADERS.get(path.resolve(), ())
    if not header:
        header = (
            "# Batch ledger — one completed item per line.",
            "# The next batch run resumes at the first pending item.",
        )
    path.write_text("\n".join(header) + "\n", encoding="utf-8")
    return path


def read_done_list(done_file: Path | str) -> set[str]:
    """Return all keys recorded in a ledger file (comments and blanks ignored)."""
    path = Path(done_file)
    if not path.is_file():
        return set()
    names: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            name = line.split("#", 1)[0].strip()
            if name:
                names.add(name)
    return names


def append_done_entries(done_file: Path | str, keys: Iterable[str]) -> list[str]:
    """Append newly completed keys; returns the list of keys newly written."""
    ensure_done_file(done_file)
    path = Path(done_file)
    existing = read_done_list(path)
    added: list[str] = []
    with path.open("a", encoding="utf-8") as handle:
        for raw in keys:
            clean = str(raw).strip()
            if clean and clean not in existing:
                handle.write(f"{clean}\n")
                existing.add(clean)
                added.append(clean)
    return added


def batch_status(
    all_items: Iterable[T],
    done_file: Path | str,
    *,
    key_fn: Callable[[T], str],
    is_done: Callable[[T, set[str]], bool] | None = None,
) -> dict[str, Any]:
    """Summarize catalog size, completed count, and pending items."""
    items = list(all_items)
    done = read_done_list(done_file)
    if is_done is None:
        pending = [item for item in items if key_fn(item) not in done]
    else:
        pending = [item for item in items if not is_done(item, done)]
    return {
        "catalog_total": len(items),
        "done_count": len(items) - len(pending),
        "pending": pending,
        "pending_count": len(pending),
    }


def select_batch_items(
    all_items: Iterable[T],
    done_file: Path | str,
    *,
    key_fn: Callable[[T], str],
    limit: int = 0,
    force_rescan: bool = False,
    is_done: Callable[[T, set[str]], bool] | None = None,
) -> tuple[list[T], dict[str, Any]]:
    """
    Return the next batch slice and status metadata.

    ``limit`` of 0 means all pending items. ``force_rescan`` ignores the ledger
    and selects from the full catalog (successful runs still append to the ledger).
    """
    items = list(all_items)
    status = batch_status(items, done_file, key_fn=key_fn, is_done=is_done)
    pending = items if force_rescan else status["pending"]
    cap = int(limit)
    selected = pending if cap <= 0 else pending[:cap]
    status = {
        **status,
        "selected_count": len(selected),
        "force_rescan": force_rescan,
    }
    return selected, status


def format_batch_banner(module: str, status: dict[str, Any]) -> str:
    """Standard one-line batch progress banner."""
    return (
        f"[{module}] Batch: {status['catalog_total']} items | "
        f"done: {status['done_count']} | pending: {status['pending_count']} | "
        f"processing: {status.get('selected_count', 0)}"
    )