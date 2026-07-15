"""Unified remote download for HTTP, OSDF, and Pelican URIs."""

from __future__ import annotations

import shutil
from pathlib import Path
from urllib.parse import urlparse

import requests

from menus.gravitic.common.pelican_tools import pelican_object_get, resolve_pelican_binary

PELICAN_SCHEMES = {"osdf", "pelican"}


def is_remote_uri(uri: str) -> bool:
    parsed = urlparse(uri)
    return parsed.scheme.lower() in PELICAN_SCHEMES or parsed.scheme in {"http", "https"}


def is_pelican_uri(uri: str) -> bool:
    parsed = urlparse(uri)
    return parsed.scheme.lower() in PELICAN_SCHEMES


def download_http(url: str, dest: Path, *, timeout: int = 600) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return dest


def download_pelican_python(uri: str, dest: Path, *, timeout: int = 600) -> Path:
    import requests_pelican

    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests_pelican.get(uri, stream=True, timeout=timeout)
    response.raise_for_status()
    with dest.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    return dest


def download_uri(
    uri: str,
    dest: Path,
    *,
    force: bool = False,
    prefer_cli: bool = False,
) -> Path:
    """Download ``uri`` to ``dest``, using cache when the file already exists."""
    if dest.is_file() and not force:
        return dest

    if not is_remote_uri(uri):
        raise ValueError(f"Unsupported remote URI: {uri!r}")

    if is_pelican_uri(uri):
        if not prefer_cli:
            try:
                return download_pelican_python(uri, dest)
            except Exception:
                pass
        if resolve_pelican_binary() is not None:
            return pelican_object_get(uri, dest)
        return download_pelican_python(uri, dest)

    return download_http(uri, dest)


def copy_local_file(source: Path, dest: Path, *, force: bool = False) -> Path:
    if dest.is_file() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return dest