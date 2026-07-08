"""
Processed-dataset archive paths for the Tav research tool.

Archives sync under ~/Public/ProtonDrive/tav_project/ (Proton Drive).
Legacy TauSuperblock ./finished* dirs are still searched on restore.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

# TauSuperblock repository root (parent of tav_shared/)
TAU_SUPERBLOCK_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = TAU_SUPERBLOCK_ROOT
DATASETS_ROOT = TAU_SUPERBLOCK_ROOT / "datasets"
ARTIFACTS_ROOT = TAU_SUPERBLOCK_ROOT / "artifacts"

TAV_PROJECT_DIR = (Path.home() / "Public" / "ProtonDrive" / "tav_project").resolve()

FINISHED_A_DIR = TAV_PROJECT_DIR / "finishedA"
FINISHED2_DIR = TAV_PROJECT_DIR / "finished2"
FINISHED_DIR = TAV_PROJECT_DIR / "finished"

LEGACY_FINISHED_A_DIR = PROJECT_ROOT / "finishedA"
LEGACY_FINISHED2_DIR = PROJECT_ROOT / "finished2"
LEGACY_FINISHED_DIR = PROJECT_ROOT / "finished"


def ensure_tav_project_dirs() -> None:
    for path in (TAV_PROJECT_DIR, FINISHED_A_DIR, FINISHED2_DIR, FINISHED_DIR):
        path.mkdir(parents=True, exist_ok=True)


def finished_a_search_dirs() -> tuple[Path, ...]:
    """Archive locations for SPARC / FRB CSV restores (new first, then legacy)."""
    return (FINISHED_A_DIR, LEGACY_FINISHED_A_DIR)


def finished_fits_search_dirs() -> tuple[Path, ...]:
    """Archive locations for Planck / integrator FITS restores (new first, then legacy)."""
    return (FINISHED2_DIR, FINISHED_DIR, LEGACY_FINISHED2_DIR, LEGACY_FINISHED_DIR)


_ARCHIVE_STAMP_RE = re.compile(r"^(.+)_(\d{8}_\d{6})$")


def logical_archive_key(filename: str | Path) -> str:
    """Normalize a filename so timestamp-suffixed archives group with the base dataset."""
    path = Path(filename)
    stem = path.stem
    match = _ARCHIVE_STAMP_RE.match(stem)
    if match:
        stem = match.group(1)
    return f"{stem}{path.suffix.lower()}"


def delete_previous_archived_runs(
    filename: str | Path,
    archive_dir: Path | str,
    *,
    also_search: tuple[Path, ...] = (),
) -> list[Path]:
    """
    Delete prior archived copies of the same logical dataset before saving the current run.

    Removes exact-name matches and ``{stem}_YYYYMMDD_HHMMSS{suffix}`` variants in
    ``archive_dir`` and any optional ``also_search`` directories (e.g. legacy finished/).
    """
    key = logical_archive_key(filename)
    removed: list[Path] = []
    seen: set[Path] = set()

    for directory in (Path(archive_dir), *also_search):
        if not directory.is_dir():
            continue
        for candidate in directory.iterdir():
            if not candidate.is_file() or candidate in seen:
                continue
            if logical_archive_key(candidate) != key:
                continue
            candidate.unlink()
            seen.add(candidate)
            removed.append(candidate)

    if removed:
        names = ", ".join(p.name for p in removed[:4])
        if len(removed) > 4:
            names += f", ... (+{len(removed) - 4})"
        print(f"[TAV ENGINE] Removed {len(removed)} prior archive(s) for {Path(filename).name}: {names}")
    return removed


def archive_destination(
    src: Path | str,
    archive_dir: Path | str,
    *,
    also_search: tuple[Path, ...] = (),
) -> Path:
    """Return the destination path after deleting previous archives of the same dataset."""
    src = Path(src)
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    delete_previous_archived_runs(src.name, archive_dir, also_search=also_search)
    return archive_dir / src.name


def move_to_finished_archive(
    src: Path | str,
    archive_dir: Path | str,
    *,
    also_search: tuple[Path, ...] = (),
) -> Path:
    """Delete prior archives, then move ``src`` into the finished folder."""
    src = Path(src)
    dest = archive_destination(src, archive_dir, also_search=also_search)
    shutil.move(str(src), str(dest))
    return dest