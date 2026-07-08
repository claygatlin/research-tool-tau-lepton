#!/usr/bin/env python3
"""
Global ledger for datasets that have already been fetched and processed.

Processed targets are marked in ./datasets/processed.txt and skipped by the
Dataset Manager fetch flow unless force_refresh is set.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Set

from tav_shared.tav_project_paths import DATASETS_ROOT
PROCESSED_FILE = DATASETS_ROOT / "processed.txt"


def ensure_ledger() -> None:
    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)


def _normalize_target_id(target_id: str) -> str:
    clean = str(target_id).strip()
    if not clean or clean.startswith("#"):
        return ""
    return clean


def read_processed_set(processed_file: Path | str = PROCESSED_FILE) -> Set[str]:
    """Return all target IDs recorded as processed (e.g. sparc:NGC3198)."""
    path = Path(processed_file)
    if not path.is_file():
        return set()
    names: Set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            token = line.split("#", 1)[0]
            name = _normalize_target_id(token)
            if name:
                names.add(name)
    return names


def is_processed(target_id: str, processed_file: Path | str = PROCESSED_FILE) -> bool:
    clean = _normalize_target_id(target_id)
    if not clean:
        return False
    return clean in read_processed_set(processed_file)


def mark_processed(
    target_ids: Iterable[str],
    *,
    processed_file: Path | str = PROCESSED_FILE,
    note: str = "",
) -> List[str]:
    """
    Record target IDs as processed so fetch skips them until force_refresh.

    Returns the list of newly added IDs.
    """
    ensure_ledger()
    path = Path(processed_file)
    existing = read_processed_set(path)
    added: List[str] = []
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with path.open("a", encoding="utf-8") as handle:
        for raw in target_ids:
            clean = _normalize_target_id(raw)
            if not clean or clean in existing:
                continue
            if note:
                handle.write(f"{clean}  # processed {stamp} — {note}\n")
            else:
                handle.write(f"{clean}  # processed {stamp}\n")
            existing.add(clean)
            added.append(clean)

    if added:
        print(f"[DATASET LEDGER] Marked processed: {', '.join(added[:8])}")
        if len(added) > 8:
            print(f"[DATASET LEDGER] ... and {len(added) - 8} more")
    return added


def clear_processed_by_prefix(
    prefix: str,
    *,
    processed_file: Path | str = PROCESSED_FILE,
) -> List[str]:
    """Remove all processed ledger rows whose ID starts with ``prefix`` (e.g. ``sparc:``)."""
    path = Path(processed_file)
    if not path.is_file():
        return []

    needle = str(prefix).strip()
    if not needle:
        return []

    kept: List[str] = []
    removed: List[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            token = _normalize_target_id(line.split("#", 1)[0])
            if token and token.startswith(needle):
                removed.append(token)
                continue
            kept.append(line.rstrip("\n"))

    if removed:
        path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        print(
            f"[DATASET LEDGER] Cleared {len(removed)} processed flag(s) with prefix {needle!r}"
        )
    return removed


def unmark_processed(
    target_ids: Iterable[str],
    *,
    processed_file: Path | str = PROCESSED_FILE,
) -> List[str]:
    """Remove target IDs from the processed ledger (e.g. for forced re-fetch)."""
    path = Path(processed_file)
    if not path.is_file():
        return []

    remove = {_normalize_target_id(tid) for tid in target_ids}
    remove.discard("")

    kept: List[str] = []
    removed: List[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            token = _normalize_target_id(line.split("#", 1)[0])
            if token and token in remove:
                removed.append(token)
                continue
            kept.append(line.rstrip("\n"))

    if removed:
        path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        print(f"[DATASET LEDGER] Cleared processed flag: {', '.join(removed[:8])}")
    return removed


def filter_for_fetch(
    target_ids: Iterable[str],
    *,
    force_refresh: bool = False,
    processed_file: Path | str = PROCESSED_FILE,
) -> tuple[List[str], List[str]]:
    """
    Split target IDs into (fetchable, skipped_processed).

    When force_refresh is True, nothing is skipped for being processed.
    """
    ids = [_normalize_target_id(tid) for tid in target_ids]
    ids = [tid for tid in ids if tid]
    if force_refresh:
        return ids, []

    processed = read_processed_set(processed_file)
    fetchable = [tid for tid in ids if tid not in processed]
    skipped = [tid for tid in ids if tid in processed]
    return fetchable, skipped


def sync_archived_as_processed(
    target_ids: Iterable[str],
    *,
    note: str = "archived",
) -> List[str]:
    """Convenience wrapper used after archive/move-to-finished steps."""
    return mark_processed(target_ids, note=note)