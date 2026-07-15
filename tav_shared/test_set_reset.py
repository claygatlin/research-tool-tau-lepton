"""
Reset a Tav research-tool test set to its initial run state.

Clears completion ledgers, processed flags, finished archive copies, and
captured run logs (*.out) for the selected module so batch tests can rerun
from the beginning after software bugs or bad evaluations.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from tav_shared.batch_ledger import (
    CASIMIR_BATCH_DONE,
    DESI_BATCH_DONE,
    FRB_PULL_DONE,
    INTEGRATOR_PULL_DONE,
    SPARC_BATCH_DONE,
    SPARC_PULL_DONE,
    clear_ledger,
)
from tav_shared.dataset_ledger import clear_processed_by_prefix
from tav_shared.run_output import ARTIFACTS_DIR, _slugify
from tav_shared.tav_project_paths import (
    LEGACY_FINISHED_A_DIR,
    finished_a_search_dirs,
    finished_fits_search_dirs,
)

from tav_shared.tav_project_paths import TAU_SUPERBLOCK_ROOT as PROJECT_ROOT

PLANCK_FITS_DIR = PROJECT_ROOT / "fits"
INTEGRATOR_FITS_DIR = PROJECT_ROOT / "datasets" / "fits"
FRB_DATA_DIR = PROJECT_ROOT / "datasets" / "frb"
CERN_DATA_DIR = PROJECT_ROOT / "datasets" / "cern"
CERN_PULL_DONE = CERN_DATA_DIR / "done.txt"
SPARC_DATA_DIR = PROJECT_ROOT / "datasets" / "sparc"
INTEGRATOR_ARTIFACT_SUBDIR = ARTIFACTS_DIR / "tav_integrator"

RESET_MENU_LABEL = "Reset to Initial Run"


@dataclass
class TestSetResetSpec:
    module_tag: str
    menu_titles: tuple[str, ...]
    ledgers: tuple[Path, ...] = ()
    processed_prefixes: tuple[str, ...] = ()
    log_slug: str = ""
    restore_fn: Callable[[], list[str]] | None = None
    archive_delete: tuple[tuple[tuple[Path, ...], str], ...] = field(default_factory=tuple)
    archive_delete_fn: Callable[[], list[Path]] | None = None


def _delete_archive_globs(
    search_dirs: tuple[Path, ...],
    pattern: str,
) -> list[Path]:
    removed: list[Path] = []
    seen: set[Path] = set()
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for candidate in directory.glob(pattern):
            if not candidate.is_file() or candidate in seen:
                continue
            candidate.unlink()
            seen.add(candidate)
            removed.append(candidate)
    return removed


def _restore_fits_from_archives(dest_dir: Path, patterns: tuple[str, ...]) -> list[str]:
    restored: list[str] = []
    dest_dir.mkdir(parents=True, exist_ok=True)
    for archive_dir in finished_fits_search_dirs():
        if not archive_dir.is_dir():
            continue
        for pattern in patterns:
            for src in sorted(archive_dir.glob(pattern)):
                if not src.is_file() or src.stat().st_size < 1000:
                    continue
                dest = dest_dir / src.name
                if dest.is_file() and dest.stat().st_size >= src.stat().st_size:
                    continue
                shutil.copy2(src, dest)
                restored.append(str(dest))
                print(f"[RESET] Restored FITS {src.name} -> {dest_dir}/")
    return restored


def _restore_sparc_archives() -> list[str]:
    from menus.astronomical.sparc.fetcher import restore_archived_galaxies

    return restore_archived_galaxies(data_dir=SPARC_DATA_DIR)


def _delete_sparc_galaxy_archives() -> list[Path]:
    from menus.astronomical.sparc.fetcher import is_galaxy_csv_stem

    removed: list[Path] = []
    seen: set[Path] = set()
    for directory in finished_a_search_dirs():
        if not directory.is_dir():
            continue
        for candidate in directory.glob("*.csv"):
            if not candidate.is_file() or candidate in seen:
                continue
            stem = candidate.stem
            if "_" in stem:
                base = stem.rsplit("_", 2)[0]
                if is_galaxy_csv_stem(base):
                    stem = base
            if not is_galaxy_csv_stem(stem):
                continue
            candidate.unlink()
            seen.add(candidate)
            removed.append(candidate)
    return removed


def _restore_frb_archives() -> list[str]:
    from menus.astronomical.frb.fetcher import _restore_from_finished_a

    restored: list[str] = []
    for name in ("chime_frb_catalog.csv", "sdss_void_catalog.csv"):
        dest = FRB_DATA_DIR / name
        if _restore_from_finished_a(name, dest):
            restored.append(str(dest))
    return restored


def _build_specs() -> dict[str, TestSetResetSpec]:
    finished_a = finished_a_search_dirs()
    finished_fits = finished_fits_search_dirs()
    all_finished_a = finished_a + (LEGACY_FINISHED_A_DIR,)
    all_finished_fits = finished_fits

    return {
        "SPARC": TestSetResetSpec(
            module_tag="SPARC",
            menu_titles=("SPARC",),
            ledgers=(SPARC_PULL_DONE, SPARC_BATCH_DONE),
            processed_prefixes=("sparc:",),
            log_slug="sparc",
            restore_fn=_restore_sparc_archives,
            archive_delete_fn=_delete_sparc_galaxy_archives,
        ),
        "FRB_COSMIC_WEB_TAV": TestSetResetSpec(
            module_tag="FRB_COSMIC_WEB_TAV",
            menu_titles=("FRB Cosmic-Web Tav-Scan",),
            ledgers=(FRB_PULL_DONE,),
            processed_prefixes=("frb:",),
            log_slug="frb_cosmic_web_tav",
            restore_fn=_restore_frb_archives,
            archive_delete=(
                (all_finished_a, "chime_frb_catalog*.csv"),
                (all_finished_a, "sdss_void_catalog*.csv"),
            ),
        ),
        "CERN_OPENDATA": TestSetResetSpec(
            module_tag="CERN_OPENDATA",
            menu_titles=("CERN Open Data",),
            ledgers=(CERN_PULL_DONE,),
            processed_prefixes=("cern:",),
            log_slug="cern_opendata",
            archive_delete=((all_finished_a, "cern_*"),),
        ),
        "TAU_SB_DESI": TestSetResetSpec(
            module_tag="TAU_SB_DESI",
            menu_titles=("DESI BAO Tau-SB Scan",),
            ledgers=(DESI_BATCH_DONE,),
            processed_prefixes=("desi:", "tau_sb_desi:"),
            log_slug="tau_sb_desi",
        ),
        "PLANCK_CMB_TAV": TestSetResetSpec(
            module_tag="PLANCK_CMB_TAV",
            menu_titles=("Planck CMB Tav-Scan",),
            ledgers=(INTEGRATOR_PULL_DONE,),
            processed_prefixes=("planck:", "integrator:", "fits:"),
            log_slug="planck_cmb_tav",
            restore_fn=lambda: _restore_fits_from_archives(
                PLANCK_FITS_DIR,
                ("COM_CMB*.fits", "COM_Mask*.fits"),
            ),
            archive_delete=(
                (all_finished_fits, "COM_CMB*.fits"),
                (all_finished_fits, "COM_Mask*.fits"),
            ),
        ),
        "TAV_DATA_INTEGRATOR": TestSetResetSpec(
            module_tag="TAV_DATA_INTEGRATOR",
            menu_titles=("Tav Framework Integrator",),
            ledgers=(INTEGRATOR_PULL_DONE,),
            processed_prefixes=("integrator:", "fits:", "planck:", "sparc:"),
            log_slug="tav_data_integrator",
            restore_fn=lambda: _restore_fits_from_archives(
                INTEGRATOR_FITS_DIR,
                ("COM_CMB*.fits", "COM_Mask*.fits"),
            ),
            archive_delete=(
                (all_finished_fits, "COM_CMB*.fits"),
                (all_finished_fits, "COM_Mask*.fits"),
            ),
        ),
        "TSB_CASIMIR": TestSetResetSpec(
            module_tag="TSB_CASIMIR",
            menu_titles=("Casimir Tau-SB Scan",),
            ledgers=(CASIMIR_BATCH_DONE,),
            processed_prefixes=("casimir:", "tsb_casimir:"),
            log_slug="tsb_casimir",
        ),
        "TAV_RESONANCE": TestSetResetSpec(
            module_tag="TAV_RESONANCE",
            menu_titles=("Tav-Resonance Package",),
            processed_prefixes=("tav_resonance:",),
            log_slug="tav_resonance",
        ),
        "LHCB_TAV_ECHO": TestSetResetSpec(
            module_tag="LHCB_TAV_ECHO",
            menu_titles=("LHCb Tav-Echo",),
            processed_prefixes=("lhcb:",),
            log_slug="lhcb_tav_echo",
        ),
        "PRIME_PAST_HARMONIC": TestSetResetSpec(
            module_tag="PRIME_PAST_HARMONIC",
            menu_titles=("Prime Past Harmonic (Down Quark / Isospin)",),
            processed_prefixes=("prime_past:",),
            log_slug="prime_past_harmonic",
        ),
        "EMPIRICAL_TESTS": TestSetResetSpec(
            module_tag="EMPIRICAL_TESTS",
            menu_titles=("Superblock Empirical Tests",),
            processed_prefixes=("empirical:",),
            log_slug="empirical_tests",
        ),
    }


_SPECS = _build_specs()
_TITLE_TO_TAG: dict[str, str] = {}
for tag, spec in _SPECS.items():
    for title in spec.menu_titles:
        _TITLE_TO_TAG[title] = tag


def resolve_module_tag(menu_title: str) -> str | None:
    """Map the current submenu/domain title to a module tag."""
    return _TITLE_TO_TAG.get(menu_title.strip())


def append_reset_option(actions: list[str]) -> list[str]:
    """Append the reset menu row before navigation items are added."""
    if RESET_MENU_LABEL in actions:
        return list(actions)
    return list(actions) + [RESET_MENU_LABEL]


def clear_run_logs_for_module(log_slug: str) -> list[Path]:
    """Delete captured *.out logs (and legacy run_*.txt) for one module."""
    slug = _slugify(log_slug or "run")
    legacy = f"run_{slug}"
    llm = f"llm_analysis_{slug}"
    removed: list[Path] = []
    if not ARTIFACTS_DIR.is_dir():
        return removed

    for path in sorted(ARTIFACTS_DIR.iterdir()):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.startswith(slug) or name.startswith(legacy) or name.startswith(llm):
            path.unlink()
            removed.append(path)
            continue
        if ".out." in name and slug in name:
            path.unlink()
            removed.append(path)

    if slug == "tav_data_integrator" and INTEGRATOR_ARTIFACT_SUBDIR.is_dir():
        shutil.rmtree(INTEGRATOR_ARTIFACT_SUBDIR, ignore_errors=True)
        print(f"[RESET] Removed integrator artifact dir: {INTEGRATOR_ARTIFACT_SUBDIR}")

    return removed


def reset_test_set(module_tag: str) -> dict[str, Any]:
    """Reset ledgers, processed flags, archives, and run logs for one test set."""
    spec = _SPECS.get(module_tag)
    if spec is None:
        raise ValueError(f"No reset profile for module tag {module_tag!r}")

    summary: dict[str, Any] = {
        "module_tag": module_tag,
        "menu_titles": spec.menu_titles,
        "ledgers_cleared": [],
        "processed_cleared": [],
        "archives_restored": [],
        "archives_deleted": [],
        "logs_deleted": [],
    }

    print(f"[RESET] Test set: {spec.menu_titles[0]} ({module_tag})")

    if spec.restore_fn is not None:
        try:
            summary["archives_restored"] = spec.restore_fn()
        except Exception as exc:
            print(f"[RESET] Archive restore warning: {exc}")

    if spec.archive_delete_fn is not None:
        deleted = spec.archive_delete_fn()
        summary["archives_deleted"].extend(str(p) for p in deleted)
        if deleted:
            print(f"[RESET] Deleted {len(deleted)} module-specific finished archive file(s)")

    for search_dirs, pattern in spec.archive_delete:
        deleted = _delete_archive_globs(search_dirs, pattern)
        summary["archives_deleted"].extend(str(p) for p in deleted)
        if deleted:
            print(f"[RESET] Deleted {len(deleted)} finished archive file(s) matching {pattern!r}")

    for ledger in spec.ledgers:
        clear_ledger(ledger)
        summary["ledgers_cleared"].append(str(ledger))
        print(f"[RESET] Cleared ledger: {ledger}")

    for prefix in spec.processed_prefixes:
        cleared = clear_processed_by_prefix(prefix)
        summary["processed_cleared"].extend(cleared)

    deleted_logs = clear_run_logs_for_module(spec.log_slug or _slugify(module_tag))
    summary["logs_deleted"] = [str(p) for p in deleted_logs]
    if deleted_logs:
        print(f"[RESET] Deleted {len(deleted_logs)} run log file(s) under artifacts/")
    else:
        print("[RESET] No matching run logs found in artifacts/")

    print(
        f"[RESET] Done — ledgers={len(summary['ledgers_cleared'])}, "
        f"logs={len(summary['logs_deleted'])}, "
        f"archives_removed={len(summary['archives_deleted'])}"
    )
    return summary


def confirm_reset(stdscr, menu_title: str) -> bool:
    """Curses yes/no prompt before destructive reset."""
    import curses

    tag = resolve_module_tag(menu_title)
    if tag is None:
        return False

    lines = [
        f"Reset: {menu_title}",
        "",
        "This will:",
        "  - Clear completion ledgers (batch_done / done.txt)",
        "  - Clear processed flags for this test set",
        "  - Restore archived datasets where applicable",
        "  - Delete finished archive copies",
        "  - Delete captured run logs (*.out) in artifacts/",
        "",
        "Press Y to confirm, any other key to cancel.",
    ]
    curses.curs_set(0)
    stdscr.clear()
    max_y, max_x = stdscr.getmaxyx()
    for row, line in enumerate(lines):
        if row >= max_y - 2:
            break
        stdscr.addstr(row, 2, line[: max_x - 4])
    stdscr.refresh()
    key = stdscr.getch()
    return chr(key).lower() == "y"