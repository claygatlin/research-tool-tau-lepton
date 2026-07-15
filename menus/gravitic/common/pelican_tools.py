"""Resolve and invoke the Pelican OSDF client binary."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen

from tav_shared.tav_project_paths import TAU_SUPERBLOCK_ROOT

PELICAN_VERSION = os.environ.get("PELICAN_VERSION", "7.25.0")
RESEARCH_ROOT = TAU_SUPERBLOCK_ROOT.parent.parent
BUNDLED_PELICAN = RESEARCH_ROOT / "third_party" / "pelican" / "bin" / "pelican"
PELICAN_DOWNLOAD_BASE = "https://dl.pelicanplatform.org"


def pelican_download_url(version: str | None = None) -> str:
    ver = version or PELICAN_VERSION
    machine = platform.machine().lower()
    arch = "x86_64" if machine in {"x86_64", "amd64"} else "arm64"
    return f"{PELICAN_DOWNLOAD_BASE}/{ver}/pelican_Linux_{arch}.tar.gz"


def resolve_pelican_binary() -> Path | None:
    """Return the first usable pelican executable on this host."""
    override = os.environ.get("PELICAN_BIN", "").strip()
    if override:
        path = Path(override)
        if path.is_file() and os.access(path, os.X_OK):
            return path
    if BUNDLED_PELICAN.is_file() and os.access(BUNDLED_PELICAN, os.X_OK):
        return BUNDLED_PELICAN
    found = shutil.which("pelican")
    if found:
        return Path(found)
    local = Path.home() / ".local" / "bin" / "pelican"
    if local.is_file() and os.access(local, os.X_OK):
        return local
    return None


def pelican_status() -> dict[str, str | bool | None]:
    binary = resolve_pelican_binary()
    version = None
    if binary is not None:
        try:
            proc = subprocess.run(
                [str(binary), "--version"],
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
            version = (proc.stdout or proc.stderr or "").strip() or None
        except Exception:
            version = None
    return {
        "installed": binary is not None,
        "path": str(binary) if binary else None,
        "version": version,
        "bundled_path": str(BUNDLED_PELICAN),
    }


def install_pelican_binary(*, version: str | None = None, force: bool = False) -> Path:
    """Download and unpack the Pelican standalone client under Research/third_party/pelican/."""
    if BUNDLED_PELICAN.is_file() and not force:
        return BUNDLED_PELICAN

    url = pelican_download_url(version)
    dest_root = BUNDLED_PELICAN.parent.parent
    dest_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "pelican.tar.gz"
        with urlopen(url, timeout=120) as response, archive.open("wb") as handle:
            handle.write(response.read())
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=dest_root)

    candidates = [
        dest_root / "pelican",
        dest_root / "bin" / "pelican",
    ]
    candidates.extend(dest_root.glob("pelican-*/pelican"))
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise RuntimeError(f"Pelican install from {url} did not produce an executable")

    BUNDLED_PELICAN.parent.mkdir(parents=True, exist_ok=True)
    if BUNDLED_PELICAN.exists() or BUNDLED_PELICAN.is_symlink():
        BUNDLED_PELICAN.unlink()
    try:
        BUNDLED_PELICAN.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, BUNDLED_PELICAN)
    BUNDLED_PELICAN.chmod(0o755)
    return BUNDLED_PELICAN


def pelican_object_get(remote_uri: str, dest: Path, *, timeout: int = 3600) -> Path:
    """Download a Pelican/OSDF object using the CLI client."""
    binary = resolve_pelican_binary()
    if binary is None:
        raise RuntimeError(
            "Pelican client not found. Run scripts/research_tool/install_gravitic_deps.sh "
            "or set PELICAN_BIN to the pelican executable."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [str(binary), "object", "get", remote_uri, str(dest)],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"pelican object get failed ({proc.returncode}): {detail}")
    return dest