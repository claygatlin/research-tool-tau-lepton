"""Fetch and cache LISA / IGWN mock data via Pelican OSDF."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from tav_shared.batch_ledger import LISA_PULL_DONE, append_done_entries, read_done_list
from tav_shared.tav_project_paths import ensure_tav_project_dirs, move_to_finished_archive

from menus.gravitic.common.download import copy_local_file, download_uri, is_pelican_uri, is_remote_uri

DATASETS_DIR = Path(__file__).resolve().parents[3] / "datasets" / "lisa"
DEFAULT_DONE_FILE = LISA_PULL_DONE

# Seed targets reachable through the OSDF Pelican federation.
# Extend this list as LISA mock releases are published on IGWN namespaces.
LISA_OSDF_TARGETS: dict[str, str] = {
    "gwdata_zenodo_readme": "osdf:///gwdata/zenodo/README.zenodo",
    "igwn_gwdata_readme": "osdf:///igwn/gwdata/README",
    "lisa_mock_index": "osdf:///lisa-mock-data/index.html",
    "igwn_ligo_readme": "osdf:///igwn/ligo/README",
}


@dataclass
class PullSummary:
    requested: int = 0
    downloaded: list[str] = field(default_factory=list)
    skipped_done: list[str] = field(default_factory=list)
    skipped_existing: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


def ensure_dirs() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_tav_project_dirs()


def list_cached_files() -> list[Path]:
    ensure_dirs()
    return sorted(path for path in DATASETS_DIR.rglob("*") if path.is_file())


def list_fetchable_targets() -> list[tuple[str, str]]:
    ensure_dirs()
    targets = list(LISA_OSDF_TARGETS.items())
    for path in list_cached_files():
        rel = path.relative_to(DATASETS_DIR)
        targets.append((str(rel), f"cached:{rel}"))
    return targets


def _target_uri(target: str) -> str:
    clean = target.strip()
    if clean in LISA_OSDF_TARGETS:
        return LISA_OSDF_TARGETS[clean]
    if is_remote_uri(clean):
        return clean
    cached = DATASETS_DIR / clean
    if cached.is_file():
        return f"cached:{cached}"
    raise ValueError(
        f"Unknown LISA target {target!r}. Use a manifest key ({', '.join(LISA_OSDF_TARGETS)}), "
        "an osdf:/// URI, or a cached relative path."
    )


def pull_target(target: str, *, force_refresh: bool = False) -> PullSummary:
    ensure_dirs()
    summary = PullSummary(requested=1)
    ledger_key = f"lisa:{target.strip()}"
    done = read_done_list(DEFAULT_DONE_FILE)
    if ledger_key in done and not force_refresh:
        summary.skipped_done.append(ledger_key)
        return summary

    try:
        uri = _target_uri(target)
    except Exception as exc:
        summary.failed[target] = str(exc)
        return summary

    if uri.startswith("cached:"):
        summary.skipped_existing.append(uri.split(":", 1)[-1])
        return summary

    name = uri.rsplit("/", 1)[-1] or target.replace("/", "_")
    dest = DATASETS_DIR / "osdf" / name
    if dest.is_file() and not force_refresh:
        summary.skipped_existing.append(str(dest.relative_to(DATASETS_DIR)))
        return summary

    try:
        download_uri(uri, dest, force=force_refresh, prefer_cli=is_pelican_uri(uri))
        summary.downloaded.append(str(dest.relative_to(DATASETS_DIR)))
        append_done_entries(DEFAULT_DONE_FILE, [ledger_key])
    except Exception as exc:
        summary.failed[target] = str(exc)
    return summary


def pull_selected_targets(
    targets: Iterable[str],
    *,
    force_refresh: bool = False,
) -> PullSummary:
    merged = PullSummary()
    for target in targets:
        merged.requested += 1
        item = str(target).strip()
        if not item:
            continue
        if Path(item).is_file():
            name = Path(item).name
            dest = DATASETS_DIR / "local" / name
            try:
                copy_local_file(Path(item), dest, force=force_refresh)
                merged.downloaded.append(str(dest.relative_to(DATASETS_DIR)))
            except Exception as exc:
                merged.failed[item] = str(exc)
            continue
        result = pull_target(item, force_refresh=force_refresh)
        merged.downloaded.extend(result.downloaded)
        merged.skipped_done.extend(result.skipped_done)
        merged.skipped_existing.extend(result.skipped_existing)
        merged.failed.update(result.failed)
    return merged


def archive_used_datasets(paths: Iterable[Path]) -> list[Path]:
    moved: list[Path] = []
    for path in paths:
        if path.is_file():
            moved.append(move_to_finished_archive(path))
    return moved


def fetch_and_graph(query: str, params: dict | None = None) -> None:
    params = params or {}
    force = str(params.get("force_refresh", "no")).lower() in {"yes", "y", "true", "1"}
    print(f"[LISA FETCH] Target: {query!r}")
    print("[LISA FETCH] Transport: requests-pelican + pelican CLI (OSDF)")
    summary = pull_selected_targets([query], force_refresh=force)
    if summary.downloaded:
        print("[LISA FETCH] Downloaded:")
        for item in summary.downloaded:
            print(f"  - {item}")
    if summary.skipped_done:
        print(f"[LISA FETCH] Skipped (ledger): {', '.join(summary.skipped_done)}")
    if summary.skipped_existing:
        print(f"[LISA FETCH] Skipped (cached): {len(summary.skipped_existing)} file(s)")
    if summary.failed:
        print("[LISA FETCH] Failed:")
        for key, err in summary.failed.items():
            print(f"  - {key}: {err}")