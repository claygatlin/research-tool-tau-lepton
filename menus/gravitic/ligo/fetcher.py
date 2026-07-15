"""Fetch and cache LIGO/GWOSC strain data via GWOSC API, GWDataFind, and Pelican."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from tav_shared.batch_ledger import GWOSC_PULL_DONE, append_done_entries, read_done_list
from tav_shared.tav_project_paths import ensure_tav_project_dirs, move_to_finished_archive

from menus.gravitic.common.download import copy_local_file, download_uri, is_pelican_uri, is_remote_uri
from menus.gravitic.common.gwdatafind_tools import find_frame_urls, parse_segment_query
from menus.gravitic.common.gwosc_api import (
    EVENT_API,
    GWOSC_DATA,
    GWOSC_HOME,
    cache_event_catalog,
    is_event_name,
    list_strain_files,
)

DATASETS_DIR = Path(__file__).resolve().parents[3] / "datasets" / "gwosc"
EVENTS_DIR = DATASETS_DIR / "events"
FRAMES_DIR = DATASETS_DIR / "frames"
CATALOG_CACHE = DATASETS_DIR / "event_catalog.json"
DEFAULT_DONE_FILE = GWOSC_PULL_DONE

DEFAULT_STRAIN_FMT = "hdf5"
DEFAULT_STRAIN_DURATION = 32
DEFAULT_STRAIN_RATE = 4096


@dataclass
class PullSummary:
    requested: int = 0
    downloaded: list[str] = field(default_factory=list)
    skipped_done: list[str] = field(default_factory=list)
    skipped_existing: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


def ensure_dirs() -> None:
    for path in (DATASETS_DIR, EVENTS_DIR, FRAMES_DIR):
        path.mkdir(parents=True, exist_ok=True)
    ensure_tav_project_dirs()


def list_cached_files() -> list[Path]:
    ensure_dirs()
    files: list[Path] = []
    for root in (EVENTS_DIR, FRAMES_DIR):
        files.extend(sorted(root.rglob("*")))
    return [path for path in files if path.is_file()]


def list_catalog_events(*, force_refresh: bool = False) -> list[str]:
    catalog = cache_event_catalog(CATALOG_CACHE, force_refresh=force_refresh)
    return sorted({str(meta.get("commonName", "")).upper() for meta in catalog.values() if meta.get("commonName")})


def _strain_options(params: dict | None) -> dict:
    params = params or {}
    fmt = str(params.get("strain_format") or DEFAULT_STRAIN_FMT).strip().lower()
    try:
        duration = int(params.get("strain_duration") or DEFAULT_STRAIN_DURATION)
    except ValueError:
        duration = DEFAULT_STRAIN_DURATION
    try:
        sampling_rate = int(params.get("strain_rate") or DEFAULT_STRAIN_RATE)
    except ValueError:
        sampling_rate = DEFAULT_STRAIN_RATE
    detectors = params.get("detectors")
    if isinstance(detectors, str) and detectors.strip():
        detector_set = {part.strip().upper() for part in detectors.split(",") if part.strip()}
    else:
        detector_set = None
    return {
        "fmt": fmt,
        "duration": duration,
        "sampling_rate": sampling_rate,
        "detectors": detector_set,
    }


def pull_event_strain(
    event_name: str,
    *,
    params: dict | None = None,
    force_refresh: bool = False,
) -> PullSummary:
    ensure_dirs()
    summary = PullSummary(requested=1)
    clean = event_name.strip().upper()
    ledger_key = f"event:{clean}"
    done = read_done_list(DEFAULT_DONE_FILE)
    if ledger_key in done and not force_refresh:
        summary.skipped_done.append(ledger_key)
        return summary

    opts = _strain_options(params)
    catalog = cache_event_catalog(CATALOG_CACHE)
    try:
        strain_files = list_strain_files(clean, catalog=catalog, **opts)
    except Exception as exc:
        summary.failed[clean] = str(exc)
        return summary

    event_dir = EVENTS_DIR / clean
    event_dir.mkdir(parents=True, exist_ok=True)
    downloaded_any = False
    for strain in strain_files:
        dest = event_dir / strain.local_name
        if dest.is_file() and not force_refresh:
            summary.skipped_existing.append(str(dest.relative_to(DATASETS_DIR)))
            downloaded_any = True
            continue
        try:
            download_uri(strain.url, dest, force=force_refresh)
            summary.downloaded.append(str(dest.relative_to(DATASETS_DIR)))
            downloaded_any = True
        except Exception as exc:
            summary.failed[strain.local_name] = str(exc)

    if downloaded_any and not summary.failed:
        append_done_entries(DEFAULT_DONE_FILE, [ledger_key])
    return summary


def pull_datafind_segment(
    query: str,
    *,
    force_refresh: bool = False,
) -> PullSummary:
    ensure_dirs()
    summary = PullSummary(requested=1)
    obs, channel, gps_start, gps_end = parse_segment_query(query)
    ledger_key = f"segment:{obs}:{channel}:{gps_start}:{gps_end}"
    done = read_done_list(DEFAULT_DONE_FILE)
    if ledger_key in done and not force_refresh:
        summary.skipped_done.append(ledger_key)
        return summary

    try:
        urls = find_frame_urls(obs, channel, gps_start, gps_end)
    except Exception as exc:
        summary.failed[query] = str(exc)
        return summary

    if not urls:
        summary.failed[query] = "GWDataFind returned no frame URLs"
        return summary

    segment_dir = FRAMES_DIR / obs / channel / str(gps_start)
    segment_dir.mkdir(parents=True, exist_ok=True)
    for index, url in enumerate(urls):
        name = url.rsplit("/", 1)[-1] or f"frame_{index}.gwf"
        dest = segment_dir / name
        if dest.is_file() and not force_refresh:
            summary.skipped_existing.append(str(dest.relative_to(DATASETS_DIR)))
            continue
        try:
            download_uri(url, dest, force=force_refresh, prefer_cli=is_pelican_uri(url))
            summary.downloaded.append(str(dest.relative_to(DATASETS_DIR)))
        except Exception as exc:
            summary.failed[name] = str(exc)

    if summary.downloaded or summary.skipped_existing:
        if not summary.failed:
            append_done_entries(DEFAULT_DONE_FILE, [ledger_key])
    return summary


def pull_remote_uri(uri: str, *, force_refresh: bool = False) -> PullSummary:
    ensure_dirs()
    summary = PullSummary(requested=1)
    ledger_key = f"uri:{uri}"
    done = read_done_list(DEFAULT_DONE_FILE)
    if ledger_key in done and not force_refresh:
        summary.skipped_done.append(ledger_key)
        return summary

    name = uri.rsplit("/", 1)[-1] or "download.bin"
    dest = FRAMES_DIR / "uris" / name
    try:
        download_uri(uri, dest, force=force_refresh, prefer_cli=is_pelican_uri(uri))
        summary.downloaded.append(str(dest.relative_to(DATASETS_DIR)))
        append_done_entries(DEFAULT_DONE_FILE, [ledger_key])
    except Exception as exc:
        summary.failed[uri] = str(exc)
    return summary


def pull_selected_targets(
    targets: Iterable[str],
    *,
    params: dict | None = None,
    force_refresh: bool = False,
) -> PullSummary:
    merged = PullSummary()
    for target in targets:
        merged.requested += 1
        item = str(target).strip()
        if not item:
            continue
        if is_event_name(item):
            result = pull_event_strain(item, params=params, force_refresh=force_refresh)
        elif is_remote_uri(item):
            result = pull_remote_uri(item, force_refresh=force_refresh)
        elif ":" in item and not Path(item).exists():
            result = pull_datafind_segment(item, force_refresh=force_refresh)
        elif Path(item).is_file():
            name = Path(item).name
            dest = FRAMES_DIR / "local" / name
            try:
                copy_local_file(Path(item), dest, force=force_refresh)
                merged.downloaded.append(str(dest.relative_to(DATASETS_DIR)))
            except Exception as exc:
                merged.failed[item] = str(exc)
            continue
        else:
            merged.failed[item] = (
                "Unrecognized target. Use GW event ID, OBS:CHANNEL:GPS_START:GPS_END, "
                "Pelican/OSDF URI, or a local file path."
            )
            continue

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


def _print_summary(label: str, summary: PullSummary) -> None:
    print(f"[GWOSC FETCH] {label}")
    if summary.downloaded:
        print(f"  downloaded ({len(summary.downloaded)}):")
        for item in summary.downloaded[:12]:
            print(f"    - {item}")
        if len(summary.downloaded) > 12:
            print(f"    ... +{len(summary.downloaded) - 12} more")
    if summary.skipped_done:
        print(f"  skipped (ledger): {', '.join(summary.skipped_done[:6])}")
    if summary.skipped_existing:
        print(f"  skipped (cached): {len(summary.skipped_existing)} file(s)")
    if summary.failed:
        print("  failed:")
        for key, err in summary.failed.items():
            print(f"    - {key}: {err}")


def fetch_and_graph(query: str, params: dict | None = None) -> None:
    params = params or {}
    force = str(params.get("force_refresh", "no")).lower() in {"yes", "y", "true", "1"}
    print(f"[GWOSC FETCH] Target: {query!r}")
    print(f"[GWOSC FETCH] Repositories: {GWOSC_HOME} | {GWOSC_DATA}")
    print(f"[GWOSC FETCH] Event API: {EVENT_API}")
    summary = pull_selected_targets([query], params=params, force_refresh=force)
    _print_summary("complete", summary)
    if summary.downloaded:
        print("[GWOSC FETCH] Cached under datasets/gwosc/")