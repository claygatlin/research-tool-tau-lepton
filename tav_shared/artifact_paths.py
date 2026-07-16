"""
Canonical artifact layout for the Tav research tool.

Each run writes under::

    artifacts/{test_slug}/{DD-MM-YYYY}_{HH-MM}_{dataset}/
        {test}__{dataset}__{kind}__{MM-DD-YYYY}_{HHMMSS}.{ext}

Example::

    artifacts/cern_analysis/11-07-2026_16-15_tav_full_61m_events_output_run2012bc_doublemuparked_muons_full/
        cern_analysis__tav_full_61m_events_output_...__report__07-11-2026_161506.json
        cern_analysis__tav_full_61m_events_output_...__plot__07-11-2026_161506.png
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from tav_shared.tav_project_paths import ARTIFACTS_ROOT

# Re-export for modules that still import ARTIFACTS_DIR from local scanner modules.
ARTIFACTS_DIR = ARTIFACTS_ROOT

_LEGACY_STAMP_RE = re.compile(r"_(\d{8})_(\d{6})(?=\.[^.]+$)")
_FILENAME_STAMP_RE = re.compile(
    r"__(\d{2}-\d{2}-\d{4})_(\d{6})(?=\.[^.]+$)"
)
_ARTIFACT_RUN_PARSE_RE = re.compile(
    r"^(?P<head>.+)__(?P<mon>\d{2})-(?P<day>\d{2})-(?P<year>\d{4})_(?P<hms>\d{6})(?P<tail>.*)\.(?P<ext>[^.]+)$"
)


class TestSlug:
    """Stable subdirectory names — one folder per menu test / pipeline."""

    EKK_BAO = "ekk_bao_reanalysis"
    COSMOLOGY_FALSIFICATION = "cosmology_falsification"
    LSS_FALSIFICATION = "lss_falsification"
    LSS_PUBLICATION = "lss_publication_figures"
    REDSHIFT_LAW = "redshift_law"
    DESI_CATALOG_GRID = "desi_catalog_grid"
    METHOD10 = "method10_joint_likelihood"
    TAU_SB_DESI = "tau_sb_desi_scan"
    MOCK_CATALOG = "mock_catalog_recovery"
    MCMC = "mcmc_posteriors"
    NESTED_SAMPLING = "nested_sampling"
    RESIDUAL_DIAGNOSTICS = "residual_diagnostics"
    DESI_DASHBOARD = "desi_dashboard"
    CERN = "cern_analysis"
    LIGO = "ligo_ringdown"
    CASIMIR = "casimir"
    TSB_RESEARCH = "tsb_research"
    TSB_TEST_RESULTS = "tsb_test_results"
    PLANCK_CMB = "planck_cmb"
    FRB = "frb_analysis"
    SPARC = "sparc_analysis"
    TAV_RESONANCE = "tav_resonance"
    BERARD = "berard_framework"
    LHCb = "lhcb_echo"
    RGC = "rgc_mock_catalog"
    PRIME_PAST = "prime_past_bbn"
    REMOTE_AI = "remote_ai"
    EMPIRICAL = "empirical_tests"
    INTEGRATOR = "tav_integrator"
    LLM_ANALYSIS = "llm_analysis"
    RUN_LOG = "run_logs"
    LEGACY = "legacy_unsorted"


def slugify(text: str, *, max_len: int = 72) -> str:
    """Filesystem-safe slug; empty → ``default``."""
    clean = "".join(ch if ch.isalnum() else "_" for ch in str(text or "").strip().lower())
    while "__" in clean:
        clean = clean.replace("__", "_")
    clean = clean.strip("_")
    if not clean:
        return "default"
    return clean[:max_len]


def compose_dataset_slug(*parts: str | None, default: str = "default") -> str:
    """Join tracer / quantity / mode fields into one dataset slug."""
    tokens = [slugify(p) for p in parts if p is not None and str(p).strip()]
    tokens = [t for t in tokens if t != "default"]
    if not tokens:
        return default
    return slugify("_".join(tokens))


def _normalize_when(when: datetime | None = None, *, utc: bool = True) -> datetime:
    if when is None:
        when = datetime.now(timezone.utc) if utc else datetime.now()
    elif utc and when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    if utc:
        return when.astimezone(timezone.utc)
    return when


def artifact_timestamp(when: datetime | None = None, *, utc: bool = True) -> str:
    """Return ``MM-DD-YYYY_HHMMSS`` for artifact filenames."""
    return _normalize_when(when, utc=utc).strftime("%m-%d-%Y_%H%M%S")


def artifact_run_folder_label(when: datetime, dataset_slug: str) -> str:
    """
    Per-run subdirectory name: ``DD-MM-YYYY_HH-MM_{dataset}`` (d/m/y + hr:min).
    """
    stamp = _normalize_when(when, utc=True)
    date_part = stamp.strftime("%d-%m-%Y")
    time_part = stamp.strftime("%H-%M")
    dataset = slugify(dataset_slug) or "default"
    return f"{date_part}_{time_part}_{dataset}"


def legacy_stamp_to_artifact_timestamp(stamp: str) -> str | None:
    """Convert ``YYYYMMDD_HHMMSS`` → ``MM-DD-YYYY_HHMMSS``."""
    match = re.fullmatch(r"(\d{4})(\d{2})(\d{2})_(\d{6})", stamp)
    if not match:
        return None
    y, m, d, t = match.groups()
    return f"{m}-{d}-{y}_{t}"


def build_artifact_filename(
    test_slug: str,
    dataset_slug: str,
    artifact_kind: str,
    ext: str,
    *,
    when: datetime | None = None,
) -> str:
    """``{test}__{dataset}__{kind}__{MM-DD-YYYY}_{HHMMSS}.{ext}``"""
    test = slugify(test_slug)
    dataset = slugify(dataset_slug) or "default"
    kind = slugify(artifact_kind) or "output"
    extension = str(ext).lstrip(".").lower() or "bin"
    stamp = artifact_timestamp(when)
    return f"{test}__{dataset}__{kind}__{stamp}.{extension}"


def test_artifacts_dir(test_slug: str, *, create: bool = True) -> Path:
    """Per-test subdirectory under ``artifacts/``."""
    path = ARTIFACTS_ROOT / slugify(test_slug)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_run_dir(
    test_slug: str,
    dataset_slug: str,
    *,
    when: datetime | None = None,
    create: bool = True,
) -> Path:
    """Directory for one run's artifacts (timestamped d/m/y + hr:min prefix)."""
    run_when = _normalize_when(when, utc=True)
    path = test_artifacts_dir(test_slug, create=create) / artifact_run_folder_label(
        run_when,
        dataset_slug,
    )
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def parse_artifact_run_folder(filename: str) -> str | None:
    """
    Infer per-run folder name from a canonical artifact filename.

    Groups report, plot, manifest, and chunk parts from the same run.
    """
    match = _ARTIFACT_RUN_PARSE_RE.match(str(filename))
    if not match:
        return None
    head = match.group("head")
    parts = head.split("__")
    if len(parts) < 2:
        return None
    dataset = slugify(parts[1])
    if not dataset:
        return None
    folder_date = f"{match.group('day')}-{match.group('mon')}-{match.group('year')}"
    folder_time = f"{match.group('hms')[:2]}-{match.group('hms')[2:4]}"
    return f"{folder_date}_{folder_time}_{dataset}"


def artifact_path(
    test_slug: str,
    dataset_slug: str,
    artifact_kind: str,
    ext: str,
    *,
    when: datetime | None = None,
    run_dir: Path | str | None = None,
    use_run_subdir: bool = True,
    create_dir: bool = True,
) -> Path:
    """Full path for a new artifact file (optionally inside a per-run subdirectory)."""
    run_when = _normalize_when(when, utc=True)
    base = test_artifacts_dir(test_slug, create=create_dir)
    if use_run_subdir:
        directory = Path(run_dir) if run_dir is not None else artifact_run_dir(
            test_slug,
            dataset_slug,
            when=run_when,
            create=create_dir,
        )
    else:
        directory = base
    name = build_artifact_filename(test_slug, dataset_slug, artifact_kind, ext, when=run_when)
    return directory / name


def infer_test_slug_from_filename(name: str) -> str:
    """Best-effort mapping of legacy flat filenames → test subdirectory."""
    lower = name.lower()
    rules: list[tuple[str, str]] = [
        (r"^ekk_bao", TestSlug.EKK_BAO),
        (r"^cosmology", TestSlug.COSMOLOGY_FALSIFICATION),
        (r"^lss_falsification", TestSlug.LSS_FALSIFICATION),
        (r"^lss_manuscript", TestSlug.LSS_PUBLICATION),
        (r"^redshift_law", TestSlug.REDSHIFT_LAW),
        (r"^desi_catalog_grid", TestSlug.DESI_CATALOG_GRID),
        (r"^method10|method_10", TestSlug.METHOD10),
        (r"^tau_sb_desi", TestSlug.TAU_SB_DESI),
        (r"^tau_sb_mock_catalog|^mock_\d+_evidence", TestSlug.MOCK_CATALOG),
        (r"^tau_sb_mcmc", TestSlug.MCMC),
        (r"^tau_sb_nested|^tau_sb_jax", TestSlug.NESTED_SAMPLING),
        (r"^cern_", TestSlug.CERN),
        (r"^gwosc_|ringdown", TestSlug.LIGO),
        (r"^tsb_casimir|^casimir", TestSlug.CASIMIR),
        (r"^tsb_research|^tsb_falsification", TestSlug.TSB_RESEARCH),
        (r"^planck|^cmb_", TestSlug.PLANCK_CMB),
        (r"^frb_", TestSlug.FRB),
        (r"^sparc_", TestSlug.SPARC),
        (r"^tav_resonance|^tav_bbn", TestSlug.PRIME_PAST),
        (r"^llm_analysis", TestSlug.LLM_ANALYSIS),
        (r"^connection_test|^custom_prompt", TestSlug.REMOTE_AI),
        (r"^superblock_|^empirical", TestSlug.EMPIRICAL),
        (r"\.out\.", TestSlug.RUN_LOG),
    ]
    for pattern, slug in rules:
        if re.search(pattern, lower):
            return slug
    return TestSlug.LEGACY


def _parse_legacy_filename(name: str) -> tuple[str, str, str] | None:
    """
    Parse legacy ``{prefix}_{kind}_{YYYYMMDD}_{HHMMSS}.ext`` or
    ``{prefix}_{YYYYMMDD}_{HHMMSS}.ext`` into (dataset, kind, stamp).
    """
    path = Path(name)
    stem = path.stem
    match = _LEGACY_STAMP_RE.search(stem)
    if not match:
        return None
    ymd, hms = match.group(1), match.group(2)
    legacy_stamp = f"{ymd}_{hms}"
    base = stem[: match.start()].rstrip("_")
    if not base:
        return ("legacy", "output", legacy_stamp)

    parts = base.split("_")
    # Heuristic: last token(s) before date may be kind (report, plot, summary, …)
    kind_hints = {
        "report",
        "summary",
        "plot",
        "png",
        "recovery",
        "evidence",
        "chain",
        "delta",
        "pk",
        "nodes",
        "suite",
        "scan",
        "json",
    }
    kind = "output"
    dataset_parts = parts
    if len(parts) >= 2 and parts[-1] in kind_hints:
        kind = parts[-1]
        dataset_parts = parts[:-1]
    elif len(parts) >= 3 and "_".join(parts[-2:]) in {"joint_dh_dm"}:
        kind = "summary"
        dataset_parts = parts[:-3] if len(parts) > 3 else parts

    dataset = slugify("_".join(dataset_parts)) if dataset_parts else "legacy"
    return (dataset, kind, legacy_stamp)


def rename_legacy_to_canonical(name: str, test_slug: str) -> str:
    """Build canonical filename from a legacy flat artifact name."""
    parsed = _parse_legacy_filename(name)
    path = Path(name)
    if parsed is None:
        dataset, kind, stamp = "legacy", "output", artifact_timestamp()
    else:
        dataset, kind, stamp = parsed
        converted = legacy_stamp_to_artifact_timestamp(stamp)
        if converted:
            stamp = converted
    test = slugify(test_slug)
    dataset = slugify(dataset)
    kind = slugify(kind)
    ext = path.suffix.lstrip(".") or "bin"
    return f"{test}__{dataset}__{kind}__{stamp}.{ext}"


def migrate_artifacts_root(
    *,
    artifacts_root: Path | str = ARTIFACTS_ROOT,
    dry_run: bool = False,
    rename: bool = True,
) -> dict[str, int]:
    """
    Move flat files in ``artifacts/`` into per-test subdirectories.

    Optionally rename to the canonical ``test__dataset__kind__date.ext`` form.
    Existing subdirectories are consolidated by name (``residual_diagnostics`` →
    ``tau_sb_desi_scan/residual_diagnostics/`` only when flat; whole subdirs move
    to matching test slug when names align).
    """
    root = Path(artifacts_root)
    stats = {"moved": 0, "skipped": 0, "renamed": 0, "errors": 0}
    if not root.is_dir():
        return stats

    # First pass: files directly in root
    for item in sorted(root.iterdir()):
        if item.is_dir():
            continue
        test_slug = infer_test_slug_from_filename(item.name)
        dest_dir = test_artifacts_dir(test_slug, create=not dry_run)
        dest_name = (
            rename_legacy_to_canonical(item.name, test_slug) if rename else item.name
        )
        dest = dest_dir / dest_name
        if dest.exists():
            dest = dest_dir / item.name
        if dest.exists():
            stats["skipped"] += 1
            continue
        if dry_run:
            print(f"[dry-run] {item.name} → {dest}")
            stats["moved"] += 1
            if rename and dest_name != item.name:
                stats["renamed"] += 1
            continue
        try:
            item.rename(dest)
            stats["moved"] += 1
            if rename and dest_name != item.name:
                stats["renamed"] += 1
        except OSError:
            stats["errors"] += 1

    # Second pass: known legacy subdirs → test slug root (merge contents)
    legacy_subdir_map = {
        "residual_diagnostics": TestSlug.TAU_SB_DESI,
        "tsb_test_results": TestSlug.TSB_TEST_RESULTS,
        "desi_dashboard": TestSlug.DESI_DASHBOARD,
        "rgc": TestSlug.RGC,
        "prime_past": TestSlug.PRIME_PAST,
        "remote_ai": TestSlug.REMOTE_AI,
        "tav_integrator": TestSlug.INTEGRATOR,
    }
    for sub_name, test_slug in legacy_subdir_map.items():
        sub = root / sub_name
        if not sub.is_dir():
            continue
        dest_root = test_artifacts_dir(test_slug, create=not dry_run)
        for item in sorted(sub.rglob("*")):
            if not item.is_file():
                continue
            rel = item.relative_to(sub)
            dest_name = (
                rename_legacy_to_canonical(item.name, test_slug) if rename else item.name
            )
            # Preserve one level of legacy structure when multiple files share a theme
            if len(rel.parts) > 1:
                dest = dest_root / slugify(rel.parent.as_posix()) / dest_name
            else:
                dest = dest_root / dest_name
            if dest.exists():
                dest = dest_root / item.name
            if dest.exists():
                stats["skipped"] += 1
                continue
            if dry_run:
                print(f"[dry-run] {item} → {dest}")
                stats["moved"] += 1
                continue
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                item.rename(dest)
                stats["moved"] += 1
            except OSError:
                stats["errors"] += 1
        if not dry_run:
            try:
                sub.rmdir() if not any(sub.iterdir()) else None
            except OSError:
                pass

    return stats


def migrate_run_subdirectories(
    *,
    artifacts_root: Path | str = ARTIFACTS_ROOT,
    test_slug: str | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """
    Move flat files in each test folder into per-run subdirectories.

    Run folder names use ``DD-MM-YYYY_HH-MM_{dataset}`` parsed from each
    artifact filename timestamp.
    """
    root = Path(artifacts_root)
    stats = {"moved": 0, "skipped": 0, "errors": 0, "folders": 0}
    if not root.is_dir():
        return stats

    test_dirs: Iterable[Path]
    if test_slug:
        candidate = root / slugify(test_slug)
        test_dirs = [candidate] if candidate.is_dir() else []
    else:
        test_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]

    seen_folders: set[str] = set()
    for test_dir in test_dirs:
        for item in sorted(test_dir.iterdir()):
            if not item.is_file():
                continue
            folder_name = parse_artifact_run_folder(item.name)
            if folder_name is None:
                stats["skipped"] += 1
                continue
            dest_dir = test_dir / folder_name
            dest = dest_dir / item.name
            if dest.exists():
                stats["skipped"] += 1
                continue
            if dry_run:
                print(f"[dry-run] {item} → {dest}")
            else:
                try:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    item.rename(dest)
                except OSError:
                    stats["errors"] += 1
                    continue
            stats["moved"] += 1
            if folder_name not in seen_folders:
                seen_folders.add(folder_name)
                stats["folders"] += 1

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Migrate artifacts to canonical layout")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-rename", action="store_true")
    parser.add_argument(
        "--run-subdirs",
        action="store_true",
        help="Move flat per-test files into DD-MM-YYYY_HH-MM_{dataset} run folders",
    )
    parser.add_argument("--test-slug", default=None, help="Limit run-subdir migration to one test")
    args = parser.parse_args()
    if args.run_subdirs:
        result = migrate_run_subdirectories(
            dry_run=args.dry_run,
            test_slug=args.test_slug,
        )
    else:
        result = migrate_artifacts_root(dry_run=args.dry_run, rename=not args.no_rename)
    print(result)