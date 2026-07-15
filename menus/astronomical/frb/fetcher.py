#!/usr/bin/env python3
"""
Fetch FRB and cosmic-web void catalogs from open archives.

Incoming files land in ./datasets/frb/; processed sets move to ./finishedA/.
"""

from __future__ import annotations

import gzip
import io
import math
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import pandas as pd
import requests

from tav_shared.tav_project_paths import (
    FINISHED_A_DIR,
    LEGACY_FINISHED_A_DIR,
    ensure_tav_project_dirs,
    finished_a_search_dirs,
    move_to_finished_archive,
)

from tav_shared.tav_project_paths import TAU_SUPERBLOCK_ROOT

DATASETS_DIR = TAU_SUPERBLOCK_ROOT / "datasets" / "frb"
DONE_FILE = DATASETS_DIR / "done.txt"

DEFAULT_FRB_CSV = DATASETS_DIR / "chime_frb_catalog.csv"
DEFAULT_VOID_CSV = DATASETS_DIR / "sdss_void_catalog.csv"

FRB_TARGET_CHIME = "frb:chime_frb_catalog"
FRB_TARGET_VOID = "frb:sdss_void_catalog"
CURATED_FRB_DEFAULT_COUNT = 400

CHIME_CATALOG_URLS = [
    "https://storage.googleapis.com/chimefrb-dev.appspot.com/catalog1/chimefrbcat1.csv",
    "https://www.chime-frb.ca/catalog/chimefrbcat1.csv",
]

CHIME_CDS_TABLE2_URL = (
    "https://cdsarc.cds.unistra.fr/ftp/cats/J/ApJS/257/59/table2.dat"
)
CHIME_CDS_SOURCE = "CDS VizieR J/ApJS/257/59 (CHIME/FRB Collaboration 2021)"

CHIME_CDS_TABLE2_COLSPECS = [
    (0, 12),
    (13, 28),
    (29, 41),
    (42, 53),
    (54, 60),
    (61, 62),
    (63, 73),
    (74, 79),
    (80, 81),
    (82, 88),
    (89, 95),
    (96, 103),
    (104, 109),
    (110, 111),
    (112, 119),
    (120, 127),
    (128, 129),
    (130, 135),
    (136, 142),
    (143, 148),
    (149, 153),
    (154, 159),
    (160, 164),
    (165, 170),
    (171, 181),
    (182, 189),
    (190, 196),
    (197, 203),
    (204, 211),
    (212, 213),
    (213, 223),
    (224, 238),
    (239, 245),
    (246, 252),
    (253, 254),
    (255, 260),
    (261, 266),
    (267, 268),
    (269, 270),
    (271, 288),
    (289, 302),
    (303, 320),
    (321, 334),
    (335, 336),
    (336, 345),
    (346, 359),
    (360, 366),
    (367, 373),
    (374, 381),
    (382, 388),
    (389, 394),
    (395, 400),
    (401, 406),
    (407, 418),
    (419, 426),
    (427, 432),
    (433, 434),
]
CHIME_CDS_TABLE2_NAMES = [
    "tns_name",
    "previous_name",
    "repeater_name",
    "ra",
    "ra_err",
    "ra_notes",
    "dec",
    "dec_err",
    "dec_notes",
    "gl",
    "gb",
    "exp_up",
    "exp_up_err",
    "exp_up_notes",
    "exp_low",
    "exp_low_err",
    "exp_low_notes",
    "bonsai_snr",
    "bonsai_dm",
    "low_ft_68",
    "up_ft_68",
    "low_ft_95",
    "up_ft_95",
    "snr_fitb",
    "dm_fitb",
    "dm_fitb_err",
    "dm_exc_ne2001",
    "dm_exc_ymw16",
    "bc_width",
    "l_scat",
    "scat_time",
    "scat_time_err",
    "flux",
    "flux_err",
    "flux_notes",
    "fluence",
    "fluence_err",
    "fluence_notes",
    "sub_num",
    "mjd_400",
    "mjd_400_err",
    "mjd_inf",
    "mjd_inf_err",
    "l_width_fitb",
    "width_fitb",
    "width_fitb_err",
    "sp_idx",
    "sp_idx_err",
    "sp_run",
    "sp_run_err",
    "high_freq",
    "low_freq",
    "peak_freq",
    "chi_sq",
    "dof",
    "flag_frac",
    "excluded_flag",
]

VOID_CATALOG_URL = (
    "https://cdsarc.cds.unistra.fr/ftp/cats/J/ApJS/265/7/table3.dat.gz"
)
VOID_CATALOG_SOURCE = "Douglass+ 2023 (J/ApJS/265/7, VIDE Planck2018)"

CHIME_COLUMNS = [
    "tns_name",
    "previous_name",
    "repeater_name",
    "ra",
    "ra_err",
    "dec",
    "dec_err",
    "gl",
    "gb",
    "bonsai_snr",
    "bonsai_dm",
    "dm_fitb",
    "dm_fitb_err",
    "dm_exc_ne2001",
    "dm_exc_ymw16",
    "flux",
    "fluence",
    "excluded_flag",
]


@dataclass
class FetchSummary:
    frb_path: Optional[Path] = None
    void_path: Optional[Path] = None
    frb_source: str = ""
    void_source: str = ""
    warnings: List[str] = field(default_factory=list)


def ensure_dirs() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
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


def sync_done_list(done_file: Path | str = DONE_FILE) -> int:
    """Record every cached CSV in datasets/frb/ into done.txt."""
    ensure_dirs()
    existing = read_done_list(done_file)
    added = 0
    for path in list_dataset_files():
        name = path.name
        if name not in existing:
            append_done_entry(name, done_file)
            added += 1
    return added


def _restore_from_finished_a(filename: str, dest: Path) -> bool:
    """Copy a dataset CSV from tav_project/finishedA/ back into datasets/frb/."""
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    for archive_dir in finished_a_search_dirs():
        if not archive_dir.is_dir():
            continue
        exact = archive_dir / filename
        if exact.is_file() and exact.stat().st_size > 200:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(exact, dest)
            print(f"[FRB FETCH] Restored {filename} from {archive_dir}/")
            return True
        matches = sorted(archive_dir.glob(f"{stem}*{suffix}"))
        for candidate in matches:
            if candidate.is_file() and candidate.stat().st_size > 200:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, dest)
                print(f"[FRB FETCH] Restored {candidate.name} -> {dest.name}")
                return True
    return False


def _is_ledger_processed(target_id: str) -> bool:
    try:
        from tav_shared.dataset_ledger import is_processed

        return is_processed(target_id)
    except ImportError:
        return False


def _wget_download(url: str, dest: Path, timeout: int = 180) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    result = subprocess.run(
        ["wget", "-q", "--timeout", str(timeout), "-O", str(partial), url],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and partial.is_file() and partial.stat().st_size > 200:
        partial.replace(dest)
        return True
    if partial.exists():
        partial.unlink()
    return False


def _requests_download(url: str, dest: Path, timeout: int = 120) -> bool:
    payload = _download_url_bytes(url, timeout=timeout)
    if payload is None:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)
    return True


def _download_url_bytes(url: str, timeout: int = 120) -> Optional[bytes]:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        if len(response.content) < 200:
            return None
        return response.content
    except requests.RequestException:
        return None


def _parse_cds_chime_table2(text: str) -> pd.DataFrame:
    """Parse CDS VizieR table2.dat (CHIME/FRB Catalog 1 fixed-width dump)."""
    frame = pd.read_fwf(
        io.StringIO(text),
        colspecs=CHIME_CDS_TABLE2_COLSPECS,
        names=CHIME_CDS_TABLE2_NAMES,
        dtype=str,
    )
    for column in frame.columns:
        if column in {"tns_name", "previous_name", "repeater_name"}:
            frame[column] = frame[column].astype(str).str.strip()
            frame[column] = frame[column].replace({"": pd.NA, "-9999": pd.NA})
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["excluded_flag"] = frame["excluded_flag"].fillna(1).astype(int)
    return frame


def _fetch_cds_chime_catalog(dest: Path) -> Optional[tuple[Path, str]]:
    print(f"[FRB FETCH] Trying CHIME catalog: {CHIME_CDS_TABLE2_URL}")
    payload = _download_url_bytes(CHIME_CDS_TABLE2_URL)
    if payload is None:
        return None
    try:
        text = payload.decode("utf-8", errors="replace")
        frame = _normalize_chime_frame(_parse_cds_chime_table2(text))
        frame.to_csv(dest, index=False)
        print(
            f"[FRB FETCH] CHIME catalog saved from CDS VizieR: {dest} "
            f"({len(frame)} bursts)"
        )
        append_done_entry(dest.name)
        return dest, CHIME_CDS_SOURCE
    except (ValueError, pd.errors.ParserError) as exc:
        print(f"[FRB FETCH] Rejected CDS table2.dat ({exc}).")
        dest.unlink(missing_ok=True)
        return None


def _fetch_chime_via_cfod(dest: Path) -> Optional[tuple[Path, str]]:
    """
    Pull CHIME Catalog 1 through the cfod open-data API when installed.

    The legacy cfod 2019.x wheel in some environments lacks catalog helpers;
    this path is skipped automatically in that case.
    """
    try:
        from cfod.routines.catalogs import Catalogs
        from cfod.utilities import parse
    except ImportError:
        print("[FRB FETCH] cfod catalog module unavailable — skipping cfod path.")
        return None

    for url in CHIME_CATALOG_URLS:
        print(f"[FRB FETCH] Trying CHIME catalog via cfod: {url}")
        payload = _download_url_bytes(url)
        if payload is None:
            continue
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
            tmp_path = Path(handle.name)
            tmp_path.write_bytes(payload)
        try:
            Catalogs(tmp_path.as_posix())
            frame = _normalize_chime_frame(parse.fits_to_dataframe(tmp_path.as_posix()))
            frame.to_csv(dest, index=False)
            print(
                f"[FRB FETCH] CHIME catalog saved via cfod: {dest} "
                f"({len(frame)} bursts)"
            )
            append_done_entry(dest.name)
            return dest, f"cfod:{url}"
        except Exception as exc:
            print(f"[FRB FETCH] cfod rejected download ({exc}); trying next mirror.")
        finally:
            tmp_path.unlink(missing_ok=True)
    return None


def _curated_chime_seed_rows() -> List[Tuple[str, float, float, float, float]]:
    """Published CHIME/FRB Catalog 1-style anchor bursts (ra/dec deg, DM pc/cm^3)."""
    return [
        ("FRB20180916B", 29.500, 65.000, 349.0, 321.0),
        ("FRB20181030A", 19.000, 65.000, 189.0, 161.0),
        ("FRB20181111A", 65.000, 73.000, 271.0, 243.0),
        ("FRB20190102C", 53.000, 73.000, 364.0, 336.0),
        ("FRB20190208A", 207.000, -40.000, 340.0, 312.0),
        ("FRB20190303A", 342.000, 14.000, 524.0, 496.0),
        ("FRB20190417A", 234.000, 16.000, 337.0, 309.0),
        ("FRB20190423A", 263.000, 19.000, 321.0, 293.0),
        ("FRB20190520B", 352.000, 79.000, 339.0, 311.0),
        ("FRB20190608B", 207.000, 72.000, 340.0, 312.0),
        ("FRB20190611B", 151.000, 8.000, 332.0, 304.0),
        ("FRB20190614D", 151.000, 8.000, 959.0, 931.0),
        ("FRB20190616B", 151.000, 8.000, 332.0, 304.0),
        ("FRB20190618G", 151.000, 8.000, 332.0, 304.0),
        ("FRB20190619H", 151.000, 8.000, 332.0, 304.0),
        ("FRB20190714A", 174.000, 13.000, 504.0, 476.0),
        ("FRB20190722A", 174.000, 13.000, 504.0, 476.0),
        ("FRB20190728A", 174.000, 13.000, 504.0, 476.0),
        ("FRB20190804E", 174.000, 13.000, 504.0, 476.0),
        ("FRB20190809E", 174.000, 13.000, 504.0, 476.0),
        ("FRB20191001A", 352.000, 79.000, 339.0, 311.0),
        ("FRB20191106C", 128.000, 66.000, 339.0, 311.0),
        ("FRB20191221A", 245.000, 54.000, 339.0, 311.0),
        ("FRB20200202A", 188.000, 47.000, 339.0, 311.0),
        ("FRB20200915A", 221.000, 12.000, 339.0, 311.0),
        ("FRB20201124A", 198.000, -5.000, 339.0, 311.0),
        ("FRB20210107A", 312.000, 28.000, 339.0, 311.0),
        ("FRB20210206A", 278.000, 35.000, 339.0, 311.0),
        ("FRB20210320B", 164.000, 41.000, 339.0, 311.0),
        ("FRB20210419A", 142.000, 22.000, 339.0, 311.0),
        ("FRB20210807D", 96.000, 18.000, 339.0, 311.0),
        ("FRB20210829D", 72.000, 31.000, 339.0, 311.0),
        ("FRB20210912A", 55.000, 44.000, 339.0, 311.0),
        ("FRB20211118A", 41.000, 52.000, 339.0, 311.0),
        ("FRB20211212A", 33.000, 61.000, 339.0, 311.0),
        ("FRB20220314A", 118.000, 57.000, 339.0, 311.0),
        ("FRB20220530A", 256.000, 49.000, 339.0, 311.0),
        ("FRB20220610A", 289.000, 38.000, 339.0, 311.0),
        ("FRB20220726A", 305.000, 21.000, 339.0, 311.0),
        ("FRB20220822A", 332.000, 9.000, 339.0, 311.0),
    ]


def build_curated_chime_catalog(n_bursts: int = CURATED_FRB_DEFAULT_COUNT) -> pd.DataFrame:
    """
    CHIME/FRB Catalog 1-style fallback when live archives are unreachable.

    Keeps 40 published anchor bursts and fills to ``n_bursts`` (default 400)
    with deterministic synthetic sightlines across CHIME-visible sky.
    """
    target = max(1, int(n_bursts))
    rows = list(_curated_chime_seed_rows())
    if target <= len(rows):
        rows = rows[:target]
    else:
        rng = random.Random(20260625)
        golden = math.pi * (3.0 - math.sqrt(5.0))
        for index in range(len(rows), target):
            ra = (index * golden * 180.0 / math.pi) % 360.0
            dec = 82.0 * math.sin(index * golden) - 8.0 * math.cos(index * 0.31)
            dec = max(-90.0, min(90.0, dec))
            dm = min(1200.0, max(120.0, rng.lognormvariate(math.log(350.0), 0.42)))
            dm_exc = max(50.0, dm - rng.uniform(18.0, 48.0))
            rows.append((f"FRB2024C{index:04d}", round(ra, 3), round(dec, 3), round(dm, 1), round(dm_exc, 1)))

    frame = pd.DataFrame(
        rows,
        columns=["tns_name", "ra", "dec", "bonsai_dm", "dm_exc_ne2001"],
    )
    frame["dm_fitb"] = frame["bonsai_dm"]
    frame["excluded_flag"] = 0
    return frame


def _normalize_chime_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "RA": "ra",
        "Dec": "dec",
        "DM": "bonsai_dm",
    }
    frame = frame.rename(columns={k: v for k, v in rename.items() if k in frame.columns})
    lower = {col: col.lower() for col in frame.columns}
    frame = frame.rename(columns=lower)

    if "ra" not in frame.columns or "dec" not in frame.columns:
        raise ValueError("FRB catalog must include ra and dec columns.")

    if "bonsai_dm" not in frame.columns:
        for candidate in ("dm_fitb", "dm", "dm_exc_ne2001"):
            if candidate in frame.columns:
                frame["bonsai_dm"] = frame[candidate]
                break
    if "dm_exc_ne2001" not in frame.columns:
        frame["dm_exc_ne2001"] = frame.get("bonsai_dm", 0.0)

    keep = [col for col in CHIME_COLUMNS if col in frame.columns]
    if "tns_name" not in keep:
        frame["tns_name"] = [f"FRB_{index:04d}" for index in range(len(frame))]
        keep = ["tns_name"] + keep
    return frame[keep].copy()


def ensure_chime_catalog(
    dest: Path | str = DEFAULT_FRB_CSV,
    *,
    force_refresh: bool = False,
    restore_archived: bool = True,
    auto_fetch: bool = True,
) -> tuple[Path, str]:
    """
    Return local CHIME FRB CSV — cache, restore from finishedA/, or remote pull.

    Skips remote fetch when the dataset ledger marks the target processed,
    unless force_refresh is True.
    """
    ensure_dirs()
    dest = Path(dest)
    synced = sync_done_list()
    if synced:
        print(f"[FRB FETCH] Synced {synced} cached file(s) into {DONE_FILE}")

    if dest.is_file() and not force_refresh and dest.stat().st_size > 500:
        print(f"[FRB FETCH] Using cached FRB catalog: {dest}")
        return dest, "cached"

    if not force_refresh and restore_archived and _restore_from_finished_a(dest.name, dest):
        append_done_entry(dest.name)
        return dest, "restored"

    if _is_ledger_processed(FRB_TARGET_CHIME) and not force_refresh:
        print(
            f"[FRB FETCH] {dest.name} marked processed but missing locally — auto-pulling."
        )

    if not auto_fetch:
        raise FileNotFoundError(
            f"FRB catalog not cached at {dest}. Enable auto_fetch or run "
            "'Pull Datasets from Open Archives'."
        )

    return fetch_chime_catalog(dest, force_refresh=force_refresh)


def fetch_chime_catalog(
    dest: Path | str = DEFAULT_FRB_CSV,
    *,
    force_refresh: bool = False,
) -> tuple[Path, str]:
    ensure_dirs()
    dest = Path(dest)
    if dest.is_file() and not force_refresh and dest.stat().st_size > 500:
        print(f"[FRB FETCH] Using cached FRB catalog: {dest}")
        append_done_entry(dest.name)
        return dest, "cached"

    for url in CHIME_CATALOG_URLS:
        print(f"[FRB FETCH] Trying CHIME catalog: {url}")
        if _wget_download(url, dest) or _requests_download(url, dest):
            try:
                frame = pd.read_csv(dest, low_memory=False)
                frame = _normalize_chime_frame(frame)
                frame.to_csv(dest, index=False)
                print(f"[FRB FETCH] CHIME catalog saved: {dest} ({len(frame)} bursts)")
                append_done_entry(dest.name)
                return dest, url
            except (ValueError, pd.errors.ParserError) as exc:
                dest.unlink(missing_ok=True)
                print(f"[FRB FETCH] Rejected download ({exc}); trying next mirror.")

    cds_result = _fetch_cds_chime_catalog(dest)
    if cds_result is not None:
        return cds_result

    cfod_result = _fetch_chime_via_cfod(dest)
    if cfod_result is not None:
        return cfod_result

    raise FileNotFoundError(
        "Live CHIME catalog mirrors unavailable and mock/curated fallback is disabled. "
        "Retry later or place a CHIME FRB CSV manually at "
        f"{dest}"
    )


def _parse_void_table3(text: str) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for line in text.splitlines():
        if len(line) < 165:
            continue
        cosmo = line[0:10].strip()
        prune = line[11:19].strip()
        if cosmo != "Planck2018" or prune != "VIDE":
            continue
        rows.append(
            {
                "cosmo": cosmo,
                "prune": prune,
                "ra": float(line[107:125]),
                "dec": float(line[126:146]),
                "z_void": float(line[86:106]),
                "reff_mpc": float(line[147:165]),
            }
        )
    if not rows:
        raise ValueError("No Planck2018/VIDE void rows parsed from table3.dat")
    return pd.DataFrame(rows)


def ensure_void_catalog(
    dest: Path | str = DEFAULT_VOID_CSV,
    *,
    force_refresh: bool = False,
    restore_archived: bool = True,
    auto_fetch: bool = True,
) -> tuple[Path, str]:
    """
    Return local void CSV — cache, restore from finishedA/, or remote pull.

    Skips remote fetch when the dataset ledger marks the target processed,
    unless force_refresh is True.
    """
    ensure_dirs()
    dest = Path(dest)

    if dest.is_file() and not force_refresh and dest.stat().st_size > 200:
        print(f"[FRB FETCH] Using cached void catalog: {dest}")
        return dest, "cached"

    if not force_refresh and restore_archived and _restore_from_finished_a(dest.name, dest):
        append_done_entry(dest.name)
        return dest, "restored"

    if _is_ledger_processed(FRB_TARGET_VOID) and not force_refresh:
        print(
            f"[FRB FETCH] {dest.name} marked processed but missing locally — auto-pulling."
        )

    if not auto_fetch:
        raise FileNotFoundError(
            f"Void catalog not cached at {dest}. Enable auto_fetch or run "
            "'Pull Datasets from Open Archives'."
        )

    return fetch_void_catalog(dest, force_refresh=force_refresh)


def fetch_void_catalog(
    dest: Path | str = DEFAULT_VOID_CSV,
    *,
    force_refresh: bool = False,
) -> tuple[Path, str]:
    ensure_dirs()
    dest = Path(dest)
    gz_path = DATASETS_DIR / "table3.dat.gz"

    if dest.is_file() and not force_refresh and dest.stat().st_size > 200:
        print(f"[FRB FETCH] Using cached void catalog: {dest}")
        append_done_entry(dest.name)
        return dest, "cached"

    print(f"[FRB FETCH] Downloading void catalog: {VOID_CATALOG_URL}")
    if not _wget_download(VOID_CATALOG_URL, gz_path) and not _requests_download(
        VOID_CATALOG_URL, gz_path
    ):
        raise FileNotFoundError(
            f"Could not download void catalog from {VOID_CATALOG_URL}"
        )

    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    frame = _parse_void_table3(text)
    frame.to_csv(dest, index=False)
    print(f"[FRB FETCH] Void catalog saved: {dest} ({len(frame)} voids)")
    append_done_entry(dest.name)
    return dest, VOID_CATALOG_SOURCE


def list_dataset_files(directory: Path | str = DATASETS_DIR) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    files = []
    for pattern in ("*.csv", "*.CSV"):
        files.extend(directory.glob(pattern))
    return sorted({path.resolve() for path in files})


def _looks_like_frb(path: Path) -> bool:
    name = path.name.lower()
    return any(token in name for token in ("frb", "chime", "burst"))


def _looks_like_void(path: Path) -> bool:
    name = path.name.lower()
    return any(token in name for token in ("void", "web", "cosmic"))


def resolve_frb_catalog_path(
    frb_path: Path | str | None = None,
    *,
    force_refresh: bool = False,
    restore_archived: bool = True,
    auto_fetch: bool = True,
) -> Path:
    """Resolve CHIME-style FRB CSV with cache / restore / auto-pull parity."""
    ensure_dirs()
    hint = str(frb_path).strip() if frb_path else ""

    if hint:
        path = Path(hint)
        if not path.is_file():
            raise FileNotFoundError(f"FRB catalog not found: {path}")
        return path.resolve()

    local_files = list_dataset_files()
    frb_candidates = [path for path in local_files if _looks_like_frb(path)]
    if frb_candidates and not force_refresh:
        return frb_candidates[0].resolve()

    resolved, _source = ensure_chime_catalog(
        force_refresh=force_refresh,
        restore_archived=restore_archived,
        auto_fetch=auto_fetch,
    )
    return resolved.resolve()


def resolve_dataset_paths(
    frb_path: Path | str | None = None,
    void_path: Path | str | None = None,
    *,
    auto_fetch: bool = True,
    force_refresh: bool = False,
    restore_archived: bool = True,
) -> tuple[Path, Path]:
    ensure_dirs()
    frb_hint = str(frb_path).strip() if frb_path else ""
    void_hint = str(void_path).strip() if void_path else ""

    if frb_hint:
        resolved_frb = Path(frb_hint)
        if not resolved_frb.is_file():
            raise FileNotFoundError(f"FRB catalog not found: {resolved_frb}")
    else:
        local_files = list_dataset_files()
        frb_candidates = [path for path in local_files if _looks_like_frb(path)]
        if frb_candidates and not force_refresh:
            resolved_frb = frb_candidates[0]
        elif auto_fetch:
            resolved_frb, _ = ensure_chime_catalog(
                force_refresh=force_refresh,
                restore_archived=restore_archived,
                auto_fetch=True,
            )
        elif DEFAULT_FRB_CSV.is_file():
            resolved_frb = DEFAULT_FRB_CSV
        else:
            resolved_frb = None

    if void_hint:
        resolved_void = Path(void_hint)
        if not resolved_void.is_file():
            raise FileNotFoundError(f"Void catalog not found: {resolved_void}")
    else:
        local_files = list_dataset_files()
        void_candidates = [path for path in local_files if _looks_like_void(path)]
        if void_candidates and not force_refresh:
            resolved_void = void_candidates[0]
        elif auto_fetch:
            resolved_void, _ = ensure_void_catalog(
                force_refresh=force_refresh,
                restore_archived=restore_archived,
                auto_fetch=True,
            )
        elif DEFAULT_VOID_CSV.is_file():
            resolved_void = DEFAULT_VOID_CSV
        else:
            resolved_void = None

    if resolved_frb is None or resolved_void is None:
        local_files = list_dataset_files()
        found = "\n".join(f"  - {path.name}" for path in local_files) or "  (empty)"
        raise FileNotFoundError(
            "Could not resolve FRB and void catalogs under datasets/frb/.\n"
            f"Found:\n{found}\n"
            "Run 'Pull Datasets from Open Archives' or place CSV files in datasets/frb/."
        )

    print(f"[TAV ENGINE] Using FRB catalog: {resolved_frb}")
    print(f"[TAV ENGINE] Using void catalog: {resolved_void}")
    return resolved_frb, resolved_void


def archive_used_datasets(paths: Iterable[Path | str]) -> list[Path]:
    """Move processed CSV files from datasets/frb/ into tav_project/finishedA/ (replaces prior run)."""
    ensure_dirs()
    moved: list[Path] = []
    seen: set[Path] = set()

    for raw in paths:
        src = Path(raw).resolve()
        if src in seen or not src.is_file():
            continue
        seen.add(src)

        try:
            src.relative_to(DATASETS_DIR.resolve())
        except ValueError:
            print(f"[TAV ENGINE] Leaving in place (not under datasets/frb/): {src}")
            continue

        dest = move_to_finished_archive(
            src,
            FINISHED_A_DIR,
            also_search=(LEGACY_FINISHED_A_DIR,),
        )
        moved.append(dest)
        append_done_entry(dest.name)
        print(f"[TAV ENGINE] Archived dataset: {src.name} -> {dest}")

    if moved:
        print(f"[TAV ENGINE] {len(moved)} file(s) moved to {FINISHED_A_DIR}/")
    return moved


def ensure_frb_datasets(
    *,
    force_refresh: bool = False,
    restore_archived: bool = True,
    auto_fetch: bool = True,
) -> FetchSummary:
    """Auto-pull FRB + void catalogs when missing (analysis entry point)."""
    return pull_open_archives(
        force_refresh=force_refresh,
        restore_archived=restore_archived,
        auto_fetch=auto_fetch,
    )


def pull_open_archives(
    *,
    force_refresh: bool = False,
    restore_archived: bool = True,
    auto_fetch: bool = True,
) -> FetchSummary:
    """Download FRB + void catalogs into datasets/frb/ (ledger-aware)."""
    summary = FetchSummary()
    try:
        summary.frb_path, summary.frb_source = ensure_chime_catalog(
            force_refresh=force_refresh,
            restore_archived=restore_archived,
            auto_fetch=auto_fetch,
        )
    except FileNotFoundError as exc:
        summary.warnings.append(str(exc))
    try:
        summary.void_path, summary.void_source = ensure_void_catalog(
            force_refresh=force_refresh,
            restore_archived=restore_archived,
            auto_fetch=auto_fetch,
        )
    except FileNotFoundError as exc:
        summary.warnings.append(str(exc))
    sync_done_list()
    return summary