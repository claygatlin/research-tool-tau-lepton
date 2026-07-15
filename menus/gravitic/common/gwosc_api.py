"""GWOSC Event Portal API helpers (https://gwosc.org/)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

GWOSC_HOME = "https://gwosc.org/"
GWOSC_DATA = "https://gwosc.org/data/"
EVENT_API = "https://gwosc.org/eventapi/json"
ALLEVENTS_URL = f"{EVENT_API}/allevents/"

_EVENT_RE = re.compile(r"^GW\d{6}(?:_\d{6})?$", re.IGNORECASE)


@dataclass(frozen=True)
class StrainFile:
    event: str
    detector: str
    gps_start: int
    duration: int
    sampling_rate: int
    fmt: str
    url: str

    @property
    def local_name(self) -> str:
        return self.url.rsplit("/", 1)[-1]


def is_event_name(query: str) -> bool:
    clean = str(query).strip().upper()
    return bool(_EVENT_RE.match(clean))


def fetch_all_events(*, timeout: int = 120) -> dict[str, dict[str, Any]]:
    response = requests.get(ALLEVENTS_URL, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    out: dict[str, dict[str, Any]] = {}
    for key, meta in payload.get("events", {}).items():
        common = str(meta.get("commonName") or key.split("-", 1)[0]).upper()
        out[common] = meta
    return out


def cache_event_catalog(cache_path: Path, *, force_refresh: bool = False) -> dict[str, dict[str, Any]]:
    if cache_path.is_file() and not force_refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    catalog = fetch_all_events()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(catalog, indent=2, sort_keys=True), encoding="utf-8")
    return catalog


def resolve_event_json_url(event_name: str, catalog: dict[str, dict[str, Any]] | None = None) -> str:
    clean = event_name.strip().upper()
    catalog = catalog or fetch_all_events()
    for meta in catalog.values():
        if str(meta.get("commonName", "")).upper() == clean:
            url = meta.get("jsonurl")
            if url:
                return str(url)
    raise ValueError(f"GWOSC event {clean!r} not found in catalog")


def fetch_event_detail(event_name: str, *, catalog: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    url = resolve_event_json_url(event_name, catalog=catalog)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    payload = response.json()
    events = payload.get("events") or {}
    if not events:
        raise ValueError(f"No event payload returned for {event_name!r}")
    return next(iter(events.values()))


def _strain_rank(entry: dict[str, Any]) -> tuple[int, int, int]:
    """Lower is better: prefer short segments, hdf5/txt, 4 kHz."""
    fmt = str(entry.get("format", "")).lower()
    fmt_score = {"hdf5": 0, "txt": 1, "gwf": 2}.get(fmt, 3)
    try:
        duration = int(entry.get("duration", 10**9))
    except (TypeError, ValueError):
        duration = 10**9
    try:
        rate = int(entry.get("sampling_rate", 10**9))
    except (TypeError, ValueError):
        rate = 10**9
    rate_score = abs(rate - 4096)
    return (duration, fmt_score, rate_score)


def list_strain_files(
    event_name: str,
    *,
    catalog: dict[str, dict[str, Any]] | None = None,
    fmt: str | None = "hdf5",
    duration: int | None = 32,
    sampling_rate: int | None = 4096,
    detectors: set[str] | None = None,
    auto_select: bool = True,
) -> list[StrainFile]:
    detail = fetch_event_detail(event_name, catalog=catalog)
    common = str(detail.get("commonName") or event_name).upper()
    raw_entries = list(detail.get("strain") or [])
    if not raw_entries:
        raise ValueError(f"No GWOSC strain files listed for event={common!r}")

    filtered: list[dict[str, Any]] = []
    for entry in raw_entries:
        entry_fmt = str(entry.get("format", "")).lower()
        if fmt and entry_fmt != fmt.lower():
            continue
        if duration is not None and int(entry.get("duration", -1)) != duration:
            continue
        if sampling_rate is not None and int(entry.get("sampling_rate", -1)) != sampling_rate:
            continue
        detector = str(entry.get("detector", "")).upper()
        if detectors and detector not in detectors:
            continue
        if str(entry.get("url", "")).strip():
            filtered.append(entry)

    if not filtered and auto_select:
        filtered = [
            entry
            for entry in raw_entries
            if str(entry.get("url", "")).strip()
            and (not detectors or str(entry.get("detector", "")).upper() in detectors)
        ]
        chosen: dict[str, dict[str, Any]] = {}
        for entry in sorted(filtered, key=_strain_rank):
            detector = str(entry.get("detector", "")).upper()
            chosen.setdefault(detector, entry)
        filtered = list(chosen.values())

    files: list[StrainFile] = []
    for entry in filtered:
        files.append(
            StrainFile(
                event=common,
                detector=str(entry.get("detector", "")).upper(),
                gps_start=int(entry.get("GPSstart", 0)),
                duration=int(entry.get("duration", 0)),
                sampling_rate=int(entry.get("sampling_rate", 0)),
                fmt=str(entry.get("format", "")).lower(),
                url=str(entry.get("url", "")).strip(),
            )
        )
    if not files:
        raise ValueError(
            f"No GWOSC strain files matched event={common!r} "
            f"fmt={fmt!r} duration={duration!r} rate={sampling_rate!r}"
        )
    return files