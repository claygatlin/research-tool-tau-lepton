#!/usr/bin/env python3
"""
Fetch/cache/store for Tav Framework Integrator datasets.

Planck CMB FITS land in ./datasets/fits/; processed files move to ./finished2/.
SPARC master table is cached under ./datasets/sparc/.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd

from tav_shared.batch_ledger import (
    INTEGRATOR_PULL_DONE,
    append_done_entries,
    format_batch_banner,
    select_batch_items,
)
from tav_shared.tav_project_paths import (
    FINISHED2_DIR,
    LEGACY_FINISHED2_DIR,
    LEGACY_FINISHED_DIR,
    ensure_tav_project_dirs,
    finished_fits_search_dirs,
    move_to_finished_archive,
)

from tav_shared.tav_project_paths import TAU_SUPERBLOCK_ROOT as PROJECT_ROOT

DATASETS_DIR = PROJECT_ROOT / "datasets" / "fits"
SPARC_DATASETS_DIR = PROJECT_ROOT / "datasets" / "sparc"
DONE_FILE = INTEGRATOR_PULL_DONE

IRSA_CMB_BASE = (
    "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/"
    "component-maps/cmb/"
)
IRSA_MASK_BASE = (
    "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/masks/"
)

PLANCK_MAP_CATALOG: Dict[str, Dict[str, str]] = {
    "sevem_hm1": {
        "filename": "COM_CMB_IQU-sevem_2048_R3.00_hm1.fits",
        "url": IRSA_CMB_BASE + "COM_CMB_IQU-sevem_2048_R3.00_hm1.fits",
    },
    "sevem_hm2": {
        "filename": "COM_CMB_IQU-sevem_2048_R3.00_hm2.fits",
        "url": IRSA_CMB_BASE + "COM_CMB_IQU-sevem_2048_R3.00_hm2.fits",
    },
    "nilc_hm1": {
        "filename": "COM_CMB_IQU-nilc_2048_R3.00_hm1.fits",
        "url": IRSA_CMB_BASE + "COM_CMB_IQU-nilc_2048_R3.00_hm1.fits",
    },
    "nilc_hm2": {
        "filename": "COM_CMB_IQU-nilc_2048_R3.00_hm2.fits",
        "url": IRSA_CMB_BASE + "COM_CMB_IQU-nilc_2048_R3.00_hm2.fits",
    },
    "commander_full": {
        "filename": "COM_CMB_IQU-commander_2048_R3.00_full.fits",
        "url": IRSA_CMB_BASE + "COM_CMB_IQU-commander_2048_R3.00_full.fits",
    },
    "smica_full": {
        "filename": "COM_CMB_IQU-smica_2048_R3.00_full.fits",
        "url": IRSA_CMB_BASE + "COM_CMB_IQU-smica_2048_R3.00_full.fits",
    },
}

DEFAULT_MASK_FILENAME = "COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
DEFAULT_MASK_URL = IRSA_MASK_BASE + DEFAULT_MASK_FILENAME
DEFAULT_MASTER_CSV = SPARC_DATASETS_DIR / "SPARC_master_table.csv"


@dataclass
class IntegratorFetchSummary:
    fetched: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)


def ensure_dirs() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    SPARC_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_tav_project_dirs()


def read_done_list(done_file: Path | str = DONE_FILE) -> Set[str]:
    path = Path(done_file)
    if not path.is_file():
        return set()
    names: Set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            name = line.strip()
            if name and not name.startswith("#"):
                names.add(name)
    return names


def append_done_entry(name: str, done_file: Path | str = DONE_FILE) -> None:
    path = Path(done_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_done_list(path)
    clean = str(name).strip()
    if clean and clean not in existing:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{clean}\n")


def _wget(url: str, dest: Path, timeout: int = 300) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    result = subprocess.run(
        ["wget", "-q", "--timeout", str(timeout), "-O", str(partial), url],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and partial.is_file() and partial.stat().st_size > 1000:
        partial.replace(dest)
        return True
    if partial.exists():
        partial.unlink()
    return False


def _restore_from_archives(filename: str, dest: Path) -> bool:
    """Copy a FITS from tav_project archives into datasets/fits/."""
    for archive_dir in finished_fits_search_dirs():
        if not archive_dir.is_dir():
            continue
        exact = archive_dir / filename
        if exact.is_file() and exact.stat().st_size > 1000:
            shutil.copy2(exact, dest)
            print(f"[INTEGRATOR FETCH] Restored {filename} from {archive_dir.name}/")
            return True
        matches = sorted(archive_dir.glob(f"{Path(filename).stem}*{Path(filename).suffix}"))
        for candidate in matches:
            if candidate.is_file() and candidate.stat().st_size > 1000:
                shutil.copy2(candidate, dest)
                print(f"[INTEGRATOR FETCH] Restored {candidate.name} -> {dest.name}")
                return True
    return False


def fetch_planck_map(
    key: str,
    *,
    dest_dir: Path | str = DATASETS_DIR,
    force_refresh: bool = False,
) -> Path:
    ensure_dirs()
    if key not in PLANCK_MAP_CATALOG:
        raise KeyError(
            f"Unknown Planck map key '{key}'. "
            f"Known keys: {', '.join(sorted(PLANCK_MAP_CATALOG))}"
        )
    entry = PLANCK_MAP_CATALOG[key]
    dest = Path(dest_dir) / entry["filename"]
    if dest.is_file() and not force_refresh and dest.stat().st_size > 1000:
        print(f"[INTEGRATOR FETCH] Using cached map: {dest.name}")
        return dest

    if not force_refresh and _restore_from_archives(entry["filename"], dest):
        append_done_entry(dest.name)
        return dest

    print(f"[INTEGRATOR FETCH] Downloading {key}: {entry['url']}")
    if not _wget(entry["url"], dest):
        raise FileNotFoundError(f"wget failed for Planck map {key}: {entry['url']}")
    append_done_entry(dest.name)
    print(f"[INTEGRATOR FETCH] Map ready: {dest}")
    return dest


def fetch_planck_mask(
    *,
    dest_dir: Path | str = DATASETS_DIR,
    force_refresh: bool = False,
) -> Path:
    ensure_dirs()
    dest = Path(dest_dir) / DEFAULT_MASK_FILENAME
    if dest.is_file() and not force_refresh and dest.stat().st_size > 1000:
        return dest

    if not force_refresh and _restore_from_archives(DEFAULT_MASK_FILENAME, dest):
        return dest

    print(f"[INTEGRATOR FETCH] Downloading mask: {DEFAULT_MASK_URL}")
    if not _wget(DEFAULT_MASK_URL, dest):
        raise FileNotFoundError(f"wget failed for Planck mask: {DEFAULT_MASK_URL}")
    append_done_entry(dest.name)
    return dest


def build_sparc_master_table(
    dest: Path | str = DEFAULT_MASTER_CSV,
    *,
    force_refresh: bool = False,
) -> Path:
    """Export SPARC MassModels table to a single master CSV for the integrator."""
    from menus.astronomical.sparc.fetcher import SPARC_MASS_MODELS_NAME, fetch_mass_models_table

    ensure_dirs()
    dest = Path(dest)
    if dest.is_file() and not force_refresh:
        print(f"[INTEGRATOR FETCH] Using cached SPARC master: {dest}")
        return dest

    mass_models = fetch_mass_models_table()
    export = mass_models.rename(columns={"Vbul": "Vbulge"})
    export.to_csv(dest, index=False)
    append_done_entry(dest.name)
    print(f"[INTEGRATOR FETCH] SPARC master table: {dest} ({len(export)} rows)")
    return dest


def list_local_fits(directory: Path | str = DATASETS_DIR) -> List[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    files = list(directory.glob("*.fits")) + list(directory.glob("*.FITS"))
    return sorted({path.resolve() for path in files})


def resolve_planck_paths(
    map_keys: Optional[Iterable[str]] = None,
    *,
    auto_fetch: bool = True,
) -> Dict[str, Path]:
    ensure_dirs()
    keys = list(map_keys) if map_keys else list(PLANCK_MAP_CATALOG)
    resolved: Dict[str, Path] = {}
    for key in keys:
        dest = DATASETS_DIR / PLANCK_MAP_CATALOG[key]["filename"]
        if dest.is_file():
            resolved[key] = dest
        elif auto_fetch:
            resolved[key] = fetch_planck_map(key)
        else:
            raise FileNotFoundError(f"Missing Planck map for key {key}: {dest}")
    return resolved


def _integrator_pull_catalog(
    planck_keys: Optional[Iterable[str]] = None,
    *,
    include_mask: bool = True,
    include_sparc_master: bool = True,
) -> list[str]:
    items = list(planck_keys) if planck_keys else list(PLANCK_MAP_CATALOG)
    if include_mask:
        items.append("mask")
    if include_sparc_master:
        items.append("sparc_master")
    return items


def _integrator_item_done(item: str, done: Set[str]) -> bool:
    if item in done:
        return True
    if item in PLANCK_MAP_CATALOG:
        return PLANCK_MAP_CATALOG[item]["filename"] in done
    if item == "mask":
        return DEFAULT_MASK_FILENAME in done
    if item == "sparc_master":
        return Path(DEFAULT_MASTER_CSV).name in done
    return False


def pull_integrator_datasets(
    *,
    planck_keys: Optional[Iterable[str]] = None,
    include_mask: bool = True,
    include_sparc_master: bool = True,
    force_refresh: bool = False,
    batch_limit: int = 0,
    force_rescan: bool = False,
    done_file: Path | str = DONE_FILE,
) -> IntegratorFetchSummary:
    ensure_dirs()
    summary = IntegratorFetchSummary()
    catalog = _integrator_pull_catalog(
        planck_keys,
        include_mask=include_mask,
        include_sparc_master=include_sparc_master,
    )
    selected, status = select_batch_items(
        catalog,
        done_file,
        key_fn=str,
        limit=batch_limit,
        force_rescan=force_rescan or force_refresh,
        is_done=lambda item, done: _integrator_item_done(item, done),
    )
    print(format_batch_banner("Integrator pull", status))
    completed: list[str] = []

    if not selected:
        if status["done_count"] >= status["catalog_total"] and catalog:
            print(
                "[INTEGRATOR FETCH] All catalog items already fetched. "
                "Use force_refresh=yes to re-pull from the start."
            )
        return summary

    for key in selected:
        if key in PLANCK_MAP_CATALOG:
            try:
                path = fetch_planck_map(key, force_refresh=force_refresh)
                summary.fetched.append(path.name)
                completed.append(key)
            except Exception as exc:
                summary.failed[key] = str(exc)
            continue

        if key == "mask":
            try:
                mask = fetch_planck_mask(force_refresh=force_refresh)
                summary.fetched.append(mask.name)
                completed.append(key)
            except Exception as exc:
                summary.failed["mask"] = str(exc)
            continue

        if key == "sparc_master":
            try:
                master = build_sparc_master_table(force_refresh=force_refresh)
                summary.fetched.append(master.name)
                completed.append(key)
            except Exception as exc:
                summary.failed["sparc_master"] = str(exc)

    if completed:
        append_done_entries(done_file, completed)
        print(f"[INTEGRATOR FETCH] Recorded {len(completed)} item(s) in {done_file}")

    return summary


def archive_used_datasets(paths: Iterable[Path | str]) -> List[Path]:
    """Move processed FITS/CSVs from datasets/fits/ into tav_project/finished2/ (replaces prior run)."""
    ensure_dirs()
    moved: List[Path] = []
    seen: set[Path] = set()

    for raw in paths:
        src = Path(raw).resolve()
        if src in seen or not src.is_file():
            continue
        seen.add(src)

        try:
            src.relative_to(DATASETS_DIR.resolve())
        except ValueError:
            print(f"[TAV ENGINE] Leaving in place (not under datasets/fits/): {src}")
            continue

        dest = move_to_finished_archive(
            src,
            FINISHED2_DIR,
            also_search=(LEGACY_FINISHED2_DIR, LEGACY_FINISHED_DIR),
        )
        moved.append(dest)
        append_done_entry(dest.name)
        print(f"[TAV ENGINE] Archived integrator dataset: {src.name} -> {dest}")

    if moved:
        print(f"[TAV ENGINE] {len(moved)} file(s) moved to {FINISHED2_DIR}/")
    return moved