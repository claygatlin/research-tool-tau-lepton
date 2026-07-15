"""GWDataFind discovery helpers (https://git.ligo.org/computing/gwdatafind/client)."""

from __future__ import annotations

import os
from typing import Any

DEFAULT_GWDATAFIND_SERVER = os.environ.get(
    "GWDATAFIND_SERVER",
    os.environ.get("LIGO_DATAFIND_SERVER", "https://datafind.gwosc.org"),
)


def ensure_gwdatafind_env() -> str:
    host = DEFAULT_GWDATAFIND_SERVER
    os.environ.setdefault("GWDATAFIND_SERVER", host)
    return host


def find_frame_urls(
    observatory: str,
    channel: str,
    gps_start: int,
    gps_end: int,
    *,
    host: str | None = None,
    **kwargs: Any,
) -> list[str]:
    """Return Pelican/OSDF/HTTP URLs for GWF frames in a GPS interval."""
    ensure_gwdatafind_env()
    from gwdatafind import find_urls

    return list(
        find_urls(
            observatory,
            channel,
            int(gps_start),
            int(gps_end),
            host=host or DEFAULT_GWDATAFIND_SERVER,
            **kwargs,
        )
    )


def parse_segment_query(query: str) -> tuple[str, str, int, int]:
    """
    Parse ``OBS:CHANNEL:GPS_START:GPS_END`` (e.g. ``H:H1_R:1126259447:1126259479``).
    """
    parts = [part.strip() for part in str(query).split(":")]
    if len(parts) != 4:
        raise ValueError(
            "Segment query must be OBS:CHANNEL:GPS_START:GPS_END "
            "(example: H:H1_R:1126259447:1126259479)"
        )
    obs, channel, start, end = parts
    return obs, channel, int(start), int(end)