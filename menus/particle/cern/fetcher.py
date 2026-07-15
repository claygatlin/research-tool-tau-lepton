"""
CERN Open Data fetcher — programmatic pull→cache via opendata.cern.ch API.

Uses the public REST API and HTTP downloads (no browser). Optional
``cernopendata-client`` CLI is not required at runtime.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from menus.particle.cern.manifest import CERN_MANIFEST, resolve_target_key
from tav_shared.dataset_ledger import mark_processed
from tav_shared.tav_project_paths import (
    FINISHED_A_DIR,
    LEGACY_FINISHED_A_DIR,
    ensure_tav_project_dirs,
    finished_a_search_dirs,
    move_to_finished_archive,
)

from tav_shared.tav_project_paths import TAU_SUPERBLOCK_ROOT

DATASETS_DIR = TAU_SUPERBLOCK_ROOT / "datasets" / "cern"
DONE_FILE = DATASETS_DIR / "done.txt"
CERN_SERVER = "https://opendata.cern.ch"
CERN_API = f"{CERN_SERVER}/api/records"

TARGET_PREFIX = "cern:"


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


def _ledger_key(target_key: str) -> str:
    return f"{TARGET_PREFIX}{target_key}"


def read_done_list() -> set[str]:
    if not DONE_FILE.is_file():
        return set()
    return {line.strip() for line in DONE_FILE.read_text(encoding="utf-8").splitlines() if line.strip()}


def append_done_entry(key: str) -> None:
    ensure_dirs()
    done = read_done_list()
    if key in done:
        return
    with DONE_FILE.open("a", encoding="utf-8") as fh:
        fh.write(key + "\n")


def sync_done_list() -> int:
    """Append ledger entries for cached targets missing from done.txt."""
    synced = 0
    for key in CERN_MANIFEST:
        dest = cache_dir_for_target(key)
        if dest.is_dir() and any(dest.rglob("*")):
            ledger = _ledger_key(key)
            if ledger not in read_done_list():
                append_done_entry(ledger)
                synced += 1
    return synced


def cache_dir_for_target(target_key: str) -> Path:
    spec = CERN_MANIFEST[target_key]
    return DATASETS_DIR / target_key / f"recid_{spec['recid']}"


def fetch_record_json(recid: int, *, server: str = CERN_SERVER) -> dict[str, Any]:
    url = f"{server}/api/records/{int(recid)}"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _uri_to_http(uri: str, *, server: str = CERN_SERVER) -> str:
    """Map EOS root:// URIs from the API to opendata.cern.ch HTTP paths."""
    uri = str(uri)
    root_uri = "root://eospublic.cern.ch/"
    if not uri.startswith(root_uri):
        return uri
    path = uri[len(root_uri) :].lstrip("/")
    if path.startswith("eos/"):
        path = path[4:]
    return f"{server}/eos/{path}"


def expand_file_urls(
    record_json: dict[str, Any],
    *,
    server: str = CERN_SERVER,
    expand: bool = True,
) -> list[tuple[str, int, str]]:
    """Return (http_url, size, checksum) for each file in a record."""
    metadata = record_json.get("metadata", record_json)
    files: list[tuple[str, int, str]] = []

    for file_info in metadata.get("files", []):
        uri = _uri_to_http(str(file_info["uri"]), server=server)
        files.append((uri, int(file_info.get("size", 0)), str(file_info.get("checksum", ""))))

    for index in metadata.get("_file_indices", []):
        if expand:
            for inner in index.get("files", []):
                uri = _uri_to_http(str(inner["uri"]), server=server)
                files.append((uri, int(inner.get("size", 0)), str(inner.get("checksum", ""))))
        else:
            recid = metadata.get("recid", "")
            files.append(
                (
                    f"{server}/record/{recid}/file_index/{index.get('key', '')}",
                    int(index.get("size", 0)),
                    "",
                )
            )
    return files


def _apply_filters(
    files: list[tuple[str, int, str]],
    *,
    filter_regexp: str | None = None,
    filter_range: str | None = None,
    filter_name: str | None = None,
) -> list[tuple[str, int, str]]:
    urls = [f[0] for f in files]
    selected = urls

    if filter_name:
        names = {n.strip() for n in filter_name.split(",") if n.strip()}
        selected = [u for u in selected if u.split("/")[-1] in names]

    if filter_regexp:
        pat = re.compile(filter_regexp)
        pool = selected if filter_name else urls
        selected = [u for u in pool if pat.search(u.split("/")[-1])]

    if filter_range:
        pool = selected if (filter_name or filter_regexp) else urls
        out: list[str] = []
        for part in filter_range.split(","):
            part = part.strip()
            if "-" not in part:
                continue
            lo, hi = part.split("-", 1)
            out.extend(pool[int(lo) - 1 : int(hi)])
        selected = out

    if not selected:
        return []
    lookup = {f[0]: f for f in files}
    return [lookup[u] for u in selected if u in lookup]


def _restore_from_finished_a(target_key: str, dest: Path) -> bool:
    spec = CERN_MANIFEST[target_key]
    recid = spec["recid"]
    for root in finished_a_search_dirs():
        candidate = root / f"cern_{target_key}_recid_{recid}"
        if candidate.is_dir():
            shutil.copytree(candidate, dest, dirs_exist_ok=True)
            print(f"[CERN FETCH] Restored {target_key} from {candidate}")
            return True
    return False


def download_record_files(
    target_key: str,
    *,
    force_refresh: bool = False,
    restore_archived: bool = True,
    auto_fetch: bool = True,
) -> tuple[Path, str]:
    """
    Pull a manifest target into ``datasets/cern/<key>/recid_<N>/``.

    Returns (cache_dir, source_tag).
    """
    if target_key not in CERN_MANIFEST:
        raise KeyError(f"Unknown CERN manifest key: {target_key}")

    ensure_dirs()
    spec = CERN_MANIFEST[target_key]
    recid = int(spec["recid"])
    dest = cache_dir_for_target(target_key)
    ledger = _ledger_key(target_key)

    if dest.is_dir() and any(dest.rglob("*")) and not force_refresh:
        print(f"[CERN FETCH] Using cached {target_key}: {dest}")
        append_done_entry(ledger)
        return dest, "cached"

    if not force_refresh and restore_archived and _restore_from_finished_a(target_key, dest):
        append_done_entry(ledger)
        mark_processed([ledger], note=f"cern restore {target_key}")
        return dest, "restored"

    if not auto_fetch:
        raise FileNotFoundError(
            f"CERN dataset {target_key} not cached at {dest}. "
            "Enable auto_fetch or run Pull Datasets from Open Archives."
        )

    print(f"[CERN FETCH] Pulling record {recid} → {dest}")
    record = fetch_record_json(recid)
    files = expand_file_urls(record, expand=True)
    filtered = _apply_filters(
        files,
        filter_regexp=spec.get("filter_regexp"),
        filter_range=spec.get("filter_range"),
        filter_name=spec.get("filter_name"),
    )
    if not filtered:
        raise FileNotFoundError(f"No files matched filters for {target_key} (recid {recid})")

    dest.mkdir(parents=True, exist_ok=True)
    for url, _size, _checksum in filtered:
        name = url.split("/")[-1]
        out = dest / name
        if out.is_file() and not force_refresh:
            print(f"[CERN FETCH]   skip existing {name}")
            continue
        print(f"[CERN FETCH]   downloading {name}")
        with requests.get(url, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with out.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        fh.write(chunk)

    append_done_entry(ledger)
    mark_processed([ledger], note=f"cern fetch {target_key}")
    meta_path = dest / "record_metadata.json"
    meta_path.write_text(json.dumps(record.get("metadata", record), indent=2), encoding="utf-8")
    print(f"[CERN FETCH] Ready: {dest} ({len(filtered)} file(s))")
    return dest, "fetched"


def ensure_cern_target(
    target_key: str,
    *,
    force_refresh: bool = False,
    restore_archived: bool = True,
    auto_fetch: bool = True,
) -> Path:
    path, _ = download_record_files(
        target_key,
        force_refresh=force_refresh,
        restore_archived=restore_archived,
        auto_fetch=auto_fetch,
    )
    return path


def list_cached_targets() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, spec in CERN_MANIFEST.items():
        dest = cache_dir_for_target(key)
        root_files = sorted(dest.rglob("*.root")) if dest.is_dir() else []
        rows.append(
            {
                "key": key,
                "recid": spec["recid"],
                "label": spec["label"],
                "cache_dir": str(dest),
                "cached": dest.is_dir() and any(dest.rglob("*")),
                "n_root_files": len(root_files),
                "sample_root": str(root_files[0]) if root_files else None,
            }
        )
    return rows


def list_dataset_files() -> list[Path]:
    ensure_dirs()
    return sorted(p for p in DATASETS_DIR.rglob("*") if p.is_file())


def pull_selected_targets(
    names: list[str],
    *,
    params: dict | None = None,
    force_refresh: bool = False,
) -> PullSummary:
    params = params or {}
    restore = str(params.get("restore_archived", "yes")).lower() in {"yes", "y", "true", "1"}
    summary = PullSummary(requested=len(names))
    done = read_done_list()

    for raw in names:
        key = resolve_target_key(raw) or raw
        if key not in CERN_MANIFEST:
            summary.failed[raw] = f"unknown target {raw!r}"
            continue
        ledger = _ledger_key(key)
        if ledger in done and not force_refresh:
            dest = cache_dir_for_target(key)
            if dest.is_dir() and any(dest.rglob("*")):
                summary.skipped_done.append(key)
                continue
        try:
            _, source = download_record_files(
                key,
                force_refresh=force_refresh,
                restore_archived=restore,
                auto_fetch=True,
            )
            if source == "cached":
                summary.skipped_existing.append(key)
            else:
                summary.downloaded.append(key)
        except Exception as exc:
            summary.failed[key] = str(exc)
    return summary


def pull_open_archives(
    *,
    force_refresh: bool = False,
    restore_archived: bool = True,
) -> PullSummary:
    sync_done_list()
    return pull_selected_targets(
        list(CERN_MANIFEST.keys()),
        force_refresh=force_refresh,
        params={"restore_archived": "yes" if restore_archived else "no"},
    )


def archive_used_datasets(paths: list[Path]) -> list[Path]:
    ensure_tav_project_dirs()
    moved: list[Path] = []
    for src in paths:
        if not src.exists():
            continue
        dest = move_to_finished_archive(
            src,
            FINISHED_A_DIR,
            also_search=finished_a_search_dirs(),
        )
        moved.append(dest)
    return moved


def resolve_root_path(query: str) -> Path | None:
    """Resolve manifest key, recid, or local path to a .root file."""
    q = (query or "").strip()
    if not q:
        return None
    path = Path(q).expanduser()
    if path.is_file() and path.suffix.lower() == ".root":
        return path
    key = resolve_target_key(q)
    if key:
        dest = cache_dir_for_target(key)
        roots = sorted(dest.rglob("*.root"))
        return roots[0] if roots else None
    return None