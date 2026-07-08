#!/usr/bin/env python3
"""
Fetch rotation-curve CSVs from the public SPARC repository.

Downloads the Lelli+2016c mass-model table, writes per-galaxy CSV files under
data/sparc/, and tracks completed targets in done.txt so subsequent pulls skip
them automatically.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd
import requests

from tav_shared.batch_ledger import SPARC_PULL_DONE, append_done_entries as _ledger_append
from tav_shared.batch_ledger import read_done_list as _ledger_read_done
from tav_shared.tav_project_paths import (
    FINISHED_A_DIR,
    LEGACY_FINISHED_A_DIR,
    ensure_tav_project_dirs,
    finished_a_search_dirs,
    move_to_finished_archive,
)

from tav_shared.tav_project_paths import TAU_SUPERBLOCK_ROOT

DATASETS_DIR = TAU_SUPERBLOCK_ROOT / "datasets" / "sparc"
LEGACY_DATA_DIR = TAU_SUPERBLOCK_ROOT / "data" / "sparc"
DEFAULT_DATA_DIR = DATASETS_DIR
DEFAULT_DONE_FILE = SPARC_PULL_DONE

SPARC_BASE_URL = "https://astroweb.case.edu/SPARC/"
SPARC_MASS_MODELS_URL = SPARC_BASE_URL + "MassModels_Lelli2016c.mrt"
SPARC_MASS_MODELS_NAME = "MassModels_Lelli2016c.mrt"

# Aggregate tables under datasets/sparc/ — not per-galaxy rotation curves.
NON_GALAXY_CSV_STEMS = frozenset({"SPARC_master_table"})

MASS_MODEL_COLSPECS = [
    (0, 11),
    (12, 18),
    (19, 25),
    (26, 32),
    (33, 38),
    (39, 45),
    (46, 52),
    (53, 59),
]
MASS_MODEL_COLUMNS = ["ID", "D", "R", "Vobs", "e_Vobs", "Vgas", "Vdisk", "Vbul"]

PROJECT_CSV_COLUMNS = ["R", "Vobs", "Vgas", "Vdisk", "Vbulge"]


@dataclass
class PullSummary:
    requested: int = 0
    downloaded: List[str] = field(default_factory=list)
    skipped_done: List[str] = field(default_factory=list)
    skipped_existing: List[str] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)

    @property
    def downloaded_count(self) -> int:
        return len(self.downloaded)


def ensure_dirs() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_tav_project_dirs()


def _migrate_legacy_csvs() -> None:
    """Copy galaxy CSVs from legacy data/sparc/ into datasets/sparc/ once."""
    if not LEGACY_DATA_DIR.is_dir():
        return
    ensure_dirs()
    for legacy in LEGACY_DATA_DIR.glob("*.csv"):
        dest = DATASETS_DIR / legacy.name
        if not dest.is_file():
            shutil.copy2(legacy, dest)
            print(f"[SPARC FETCH] Migrated {legacy.name} -> {dest}")


def read_done_list(done_file: Path | str = DEFAULT_DONE_FILE) -> Set[str]:
    """Galaxy IDs already recorded in done.txt (one name per line)."""
    return set(_ledger_read_done(done_file))


def append_done_entries(done_file: Path | str, galaxy_names: Iterable[str]) -> None:
    """Append newly completed galaxy IDs to done.txt."""
    _ledger_append(done_file, galaxy_names)


def is_galaxy_csv_stem(name: str) -> bool:
    """True if `name` looks like a per-galaxy CSV stem (not an aggregate table)."""
    clean = str(name).strip()
    return bool(clean) and clean not in NON_GALAXY_CSV_STEMS


def sync_done_list(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    done_file: Path | str = DEFAULT_DONE_FILE,
) -> int:
    """
    Record every cached galaxy CSV in done.txt.

    Returns the number of galaxy IDs newly appended.
    """
    data_path = Path(data_dir)
    done_path = Path(done_file)
    existing = read_done_list(done_path)
    cached = [
        path.stem
        for path in iter_local_csv_files(data_path)
        if is_galaxy_csv_stem(path.stem)
    ]
    new_names = sorted(name for name in cached if name not in existing)
    if new_names:
        append_done_entries(done_path, new_names)
    return len(new_names)


def catalog_cache_status(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    done_file: Path | str = DEFAULT_DONE_FILE,
) -> dict:
    """Summarize how many catalog galaxies are cached, pending, or archived."""
    data_path = Path(data_dir)
    ensure_dirs()
    mass_models = fetch_mass_models_table(cache_path=data_path / SPARC_MASS_MODELS_NAME)
    catalog = list_catalog_galaxies(mass_models)
    cached = {
        path.stem
        for path in iter_local_csv_files(data_path)
        if is_galaxy_csv_stem(path.stem)
    }
    done_names = read_done_list(done_file)
    archived_in_finished_a = _archived_galaxy_stems_in_finished_a()
    pending = [
        galaxy
        for galaxy in catalog
        if galaxy not in cached and galaxy not in done_names
    ]
    return {
        "catalog_total": len(catalog),
        "cached_count": len(catalog) - len(pending),
        "cached_local": len(cached & set(catalog)),
        "pending": pending,
        "pending_count": len(pending),
        "done_count": len(done_names),
        "archived_count": len(archived_in_finished_a),
        "extra_local": sorted(cached - set(catalog)),
    }


def _archived_galaxy_stems_in_finished_a() -> Set[str]:
    """Galaxy CSV stems present under tav_project/finishedA/ (archived after analysis)."""
    stems: Set[str] = set()
    for archive_dir in finished_a_search_dirs():
        if not archive_dir.is_dir():
            continue
        for path in archive_dir.glob("*.csv"):
            stem = path.stem
            if "_" in stem:
                # Timestamped archive names: NGC3198_20260625_140255
                base = stem.rsplit("_", 2)[0]
                if base and is_galaxy_csv_stem(base):
                    stems.add(base)
                    continue
            if is_galaxy_csv_stem(stem):
                stems.add(stem)
    return stems


def _find_archived_galaxy_csv(galaxy: str) -> Path | None:
    for archive_dir in finished_a_search_dirs():
        if not archive_dir.is_dir():
            continue
        candidates = sorted(archive_dir.glob(f"{galaxy}*.csv"))
        if candidates:
            return candidates[-1]
    return None


def restore_archived_galaxies(
    galaxy_names: Optional[Iterable[str]] = None,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    done_file: Path | str = DEFAULT_DONE_FILE,
) -> List[str]:
    """
    Copy galaxy CSVs from finishedA/ back into datasets/sparc/.

    If `galaxy_names` is omitted, restores every archived galaxy not already cached.
    """
    ensure_dirs()
    data_path = Path(data_dir)
    archived = _archived_galaxy_stems_in_finished_a()
    if galaxy_names is None:
        targets = sorted(archived - {path.stem for path in iter_local_csv_files(data_path)})
    else:
        targets = sorted(
            str(name).strip()
            for name in galaxy_names
            if str(name).strip() and is_galaxy_csv_stem(str(name).strip())
        )

    restored: List[str] = []
    for galaxy in targets:
        if (data_path / f"{galaxy}.csv").is_file():
            continue
        src = _find_archived_galaxy_csv(galaxy)
        if src is None:
            print(f"[SPARC FETCH] No archived CSV for {galaxy} in {FINISHED_A_DIR}/")
            continue
        dest = data_path / f"{galaxy}.csv"
        shutil.copy2(src, dest)
        append_done_entries(done_file, [galaxy])
        restored.append(galaxy)
        print(f"[SPARC FETCH] Restored {galaxy} <- {src.name}")
    if restored:
        print(f"[SPARC FETCH] Restored {len(restored)} galaxy CSV(s) from {FINISHED_A_DIR}/")
    return restored


def _find_mass_model_data_start(lines: List[str]) -> int:
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("-") or stripped.startswith("Byte"):
            continue
        if stripped.startswith("Note") or stripped.startswith("Title"):
            continue
        # First fixed-width data row in the published table starts with a galaxy ID.
        if len(line) >= 19 and line[12:18].strip().replace(".", "", 1).replace("-", "", 1).isdigit():
            return index
    raise ValueError("Could not locate SPARC mass-model data rows in .mrt file.")


def fetch_mass_models_table(
    cache_path: Optional[Path | str] = None,
    force_refresh: bool = False,
    timeout: int = 60,
) -> pd.DataFrame:
    """
    Download (or load cached) SPARC MassModels_Lelli2016c.mrt and parse all rows.
    """
    ensure_dirs()
    _migrate_legacy_csvs()
    cache = Path(cache_path) if cache_path else DATASETS_DIR / SPARC_MASS_MODELS_NAME

    if cache.is_file() and not force_refresh:
        text = cache.read_text(encoding="utf-8", errors="replace")
        print(f"[SPARC FETCH] Using cached catalog: {cache}")
    else:
        print(f"[SPARC FETCH] Downloading catalog: {SPARC_MASS_MODELS_URL}")
        response = requests.get(SPARC_MASS_MODELS_URL, timeout=timeout)
        response.raise_for_status()
        text = response.text
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text, encoding="utf-8")
        print(f"[SPARC FETCH] Cached catalog at {cache}")

    lines = text.splitlines()
    data_start = _find_mass_model_data_start(lines)
    frame = pd.read_fwf(
        StringIO("\n".join(lines[data_start:])),
        colspecs=MASS_MODEL_COLSPECS,
        names=MASS_MODEL_COLUMNS,
    )
    for column in MASS_MODEL_COLUMNS:
        if column == "ID":
            frame[column] = frame[column].astype(str).str.strip()
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["ID", "R", "Vobs"]).reset_index(drop=True)
    return frame


def list_catalog_galaxies(mass_models: pd.DataFrame) -> List[str]:
    """Stable sorted list of unique galaxy IDs in the SPARC mass-model table."""
    return sorted(mass_models["ID"].astype(str).unique().tolist())


def galaxy_frame_to_project_csv(galaxy_frame: pd.DataFrame) -> pd.DataFrame:
    """Map SPARC mass-model columns to the local research-tool CSV schema."""
    export = pd.DataFrame(
        {
            "R": galaxy_frame["R"],
            "Vobs": galaxy_frame["Vobs"],
            "Vgas": galaxy_frame["Vgas"].fillna(0.0),
            "Vdisk": galaxy_frame["Vdisk"].fillna(0.0),
            "Vbulge": galaxy_frame["Vbul"].fillna(0.0),
        }
    )
    return export.sort_values("R").reset_index(drop=True)


def write_galaxy_csv(
    galaxy_name: str,
    mass_models: pd.DataFrame,
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> Path:
    """Extract one galaxy from the catalog table and write data/sparc/{name}.csv."""
    subset = mass_models[mass_models["ID"] == galaxy_name]
    if subset.empty:
        raise KeyError(f"Galaxy '{galaxy_name}' not found in SPARC catalog.")

    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{galaxy_name}.csv"
    export = galaxy_frame_to_project_csv(subset)
    export.to_csv(out_path, index=False)
    return out_path


def list_fetchable_galaxies(
    data_dir: Path | str = DATASETS_DIR,
    done_file: Path | str = DEFAULT_DONE_FILE,
) -> List[str]:
    """Galaxy IDs in the SPARC catalog that are not yet cached or archived."""
    ensure_dirs()
    done_names = read_done_list(done_file)
    cached = {path.stem for path in iter_local_csv_files(data_dir)}
    mass_models = fetch_mass_models_table(cache_path=Path(data_dir) / SPARC_MASS_MODELS_NAME)
    pending: List[str] = []
    for galaxy in list_catalog_galaxies(mass_models):
        if galaxy in cached:
            continue
        if galaxy in done_names:
            continue
        pending.append(galaxy)
    return pending


def pull_selected_galaxies(
    galaxy_names: Iterable[str],
    data_dir: Path | str = DATASETS_DIR,
    done_file: Path | str = DEFAULT_DONE_FILE,
    force_refresh_catalog: bool = False,
    force_repull: bool = False,
) -> PullSummary:
    """Download explicit galaxy IDs from the SPARC catalog."""
    names = [str(name).strip() for name in galaxy_names if str(name).strip()]
    summary = PullSummary(requested=len(names))
    if not names:
        return summary

    data_path = Path(data_dir)
    ensure_dirs()
    mass_models = fetch_mass_models_table(
        cache_path=data_path / SPARC_MASS_MODELS_NAME,
        force_refresh=force_refresh_catalog,
    )
    done_names = read_done_list(done_file)

    for galaxy in names:
        csv_path = data_path / f"{galaxy}.csv"
        if csv_path.is_file() and not force_repull:
            if galaxy in done_names:
                summary.skipped_done.append(galaxy)
            else:
                summary.skipped_existing.append(galaxy)
                append_done_entries(done_file, [galaxy])
            continue
        try:
            out_path = write_galaxy_csv(galaxy, mass_models, data_dir=data_path)
            append_done_entries(done_file, [galaxy])
            summary.downloaded.append(galaxy)
            print(f"[SPARC FETCH] OK {galaxy} -> {out_path}")
        except Exception as exc:
            summary.failed[galaxy] = str(exc)
            print(f"[SPARC FETCH] FAIL {galaxy}: {exc}")
    return summary


def archive_used_datasets(paths: Iterable[Path | str]) -> List[Path]:
    """Move processed SPARC CSVs from datasets/sparc/ into tav_project/finishedA/ (replaces prior run)."""
    ensure_dirs()
    moved: List[Path] = []
    seen: set[Path] = set()

    for raw in paths:
        src = Path(raw).resolve()
        if src in seen or not src.is_file() or src.suffix.lower() != ".csv":
            continue
        seen.add(src)
        try:
            src.relative_to(DATASETS_DIR.resolve())
        except ValueError:
            print(f"[TAV ENGINE] Leaving in place (not under datasets/sparc/): {src}")
            continue

        dest = move_to_finished_archive(
            src,
            FINISHED_A_DIR,
            also_search=(LEGACY_FINISHED_A_DIR,),
        )
        moved.append(dest)
        print(f"[TAV ENGINE] Archived SPARC CSV: {src.name} -> {dest}")

    if moved:
        print(f"[TAV ENGINE] {len(moved)} SPARC file(s) moved to {FINISHED_A_DIR}/")
    return moved


def pull_sparc_batch(
    limit: int = 100,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    done_file: Path | str = DEFAULT_DONE_FILE,
    skip_existing_csv: bool = True,
    force_refresh_catalog: bool = False,
    force_repull: bool = False,
    restore_archived: bool = False,
) -> PullSummary:
    """
    Connect to the SPARC repository, download up to `limit` new galaxy CSVs, and
    record completed targets in done.txt.

    Parameters
    ----------
    limit:
        Maximum number of *new* galaxies to download this run.
    data_dir:
        Directory for per-galaxy CSV output (default: data/sparc).
    done_file:
        Text file listing completed galaxy IDs (default: data/sparc/done.txt).
    skip_existing_csv:
        If True, treat an on-disk CSV as already done even if missing from done.txt.
    force_refresh_catalog:
        Re-download MassModels_Lelli2016c.mrt instead of using the local cache.
    force_repull:
        Re-extract galaxy CSVs from the catalog even when local files exist.
    restore_archived:
        Copy missing galaxy CSVs from finishedA/ before pulling from the repository.

    Returns
    -------
    PullSummary with downloaded, skipped, and failed galaxy IDs.
    """
    ensure_dirs()
    if restore_archived:
        restore_archived_galaxies(data_dir=data_dir, done_file=done_file)

    status = catalog_cache_status(data_dir=data_dir, done_file=done_file)
    synced = sync_done_list(data_dir=data_dir, done_file=done_file)
    if synced:
        print(f"[SPARC FETCH] Synced {synced} cached galaxy(ies) into {done_file}")

    pending = status["pending"]
    if force_repull:
        data_path = Path(data_dir)
        mass_models = fetch_mass_models_table(
            cache_path=data_path / SPARC_MASS_MODELS_NAME,
            force_refresh=force_refresh_catalog,
        )
        pending = list_catalog_galaxies(mass_models)
        skip_existing_csv = False

    if skip_existing_csv and not force_repull:
        data_path = Path(data_dir)
        pending = [
            galaxy
            for galaxy in pending
            if not (data_path / f"{galaxy}.csv").is_file()
        ]

    cap = max(0, int(limit))
    selected = pending[:cap] if cap else pending

    cached = status["cached_count"]
    total = status["catalog_total"]
    action_label = "repulling" if force_repull else "pulling"
    print(
        f"[SPARC FETCH] Catalog: {total} galaxies | "
        f"cached: {cached}/{total} | pending: {len(pending)} | "
        f"{action_label}: {len(selected)}"
    )
    if not selected:
        if cached >= total:
            print(
                f"[SPARC FETCH] All {total} catalog galaxies already in {data_dir}/. "
                "Nothing to download. Use force_repull=yes to refresh from catalog, "
                "or restore_archived=yes if files were moved to finishedA/."
            )
        elif status["archived_count"]:
            print(
                f"[SPARC FETCH] {status['archived_count']} galaxy CSV(s) in {FINISHED_A_DIR}/ "
                "(not in datasets/sparc/). Try restore_archived=yes."
            )
        return PullSummary()

    return pull_selected_galaxies(
        selected,
        data_dir=data_dir,
        done_file=done_file,
        force_refresh_catalog=force_refresh_catalog,
        force_repull=force_repull,
    )


def resolve_galaxy_csv_path(
    galaxy_name: str,
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> Path:
    """Return expected local CSV path for a SPARC galaxy ID."""
    clean = str(galaxy_name).strip()
    if clean.endswith(".csv"):
        return Path(clean)
    return Path(data_dir) / f"{clean}.csv"


def iter_local_csv_files(data_dir: Path | str = DEFAULT_DATA_DIR) -> List[Path]:
    """Per-galaxy rotation-curve CSV files under datasets/sparc/*.csv (sorted)."""
    ensure_dirs()
    _migrate_legacy_csvs()
    data_path = Path(data_dir)
    files = sorted(
        path
        for path in data_path.glob("*.csv")
        if path.is_file() and is_galaxy_csv_stem(path.stem)
    )
    if files:
        return files
    if LEGACY_DATA_DIR.is_dir():
        return sorted(
            path
            for path in LEGACY_DATA_DIR.glob("*.csv")
            if path.is_file() and is_galaxy_csv_stem(path.stem)
        )
    return []


def list_local_galaxies(data_dir: Path | str = DEFAULT_DATA_DIR) -> List[str]:
    """Galaxy IDs from on-disk CSV files in data/sparc/."""
    return [path.stem for path in iter_local_csv_files(data_dir)]


def ensure_galaxy_csv(
    galaxy_name: str,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    done_file: Path | str = DEFAULT_DONE_FILE,
    force_refresh_catalog: bool = False,
) -> Path:
    """
    Return a local CSV for `galaxy_name`, fetching from the SPARC catalog if needed.
    """
    data_path = Path(data_dir)
    done_path = Path(done_file)
    csv_path = resolve_galaxy_csv_path(galaxy_name, data_dir=data_path)
    if csv_path.is_file():
        return csv_path

    clean = csv_path.stem
    mass_models = fetch_mass_models_table(
        cache_path=data_path / SPARC_MASS_MODELS_NAME,
        force_refresh=force_refresh_catalog,
    )
    out_path = write_galaxy_csv(clean, mass_models, data_dir=data_path)
    append_done_entries(done_path, [clean])
    print(f"[SPARC FETCH] Pulled single target {clean} -> {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pull SPARC rotation-curve CSV batch.")
    parser.add_argument("--limit", type=int, default=100, help="Max new galaxies to fetch")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="CSV output directory")
    parser.add_argument("--done-file", default=str(DEFAULT_DONE_FILE), help="Completed-target ledger")
    parser.add_argument("--refresh-catalog", action="store_true", help="Re-download SPARC .mrt table")
    args = parser.parse_args()

    result = pull_sparc_batch(
        limit=args.limit,
        data_dir=args.data_dir,
        done_file=args.done_file,
        force_refresh_catalog=args.refresh_catalog,
    )
    print(f"Downloaded: {result.downloaded_count}")
    if result.downloaded:
        print("First 10:", ", ".join(result.downloaded[:10]))