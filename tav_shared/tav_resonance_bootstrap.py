"""
Make ``tav_resonance`` importable inside research_tool.

The library lives at ``Research/tav-resonance/`` (sibling of ``scripts/``).
Prefer a venv install (``pip install -e …/tav-resonance``); otherwise add
``src/`` to ``sys.path``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

# research_tool/tav_shared/ → Research/
_RESEARCH_ROOT = Path(__file__).resolve().parents[3]
TAV_RESONANCE_ROOT = _RESEARCH_ROOT / "tav-resonance"
TAV_RESONANCE_SRC = TAV_RESONANCE_ROOT / "src"
TAV_RESONANCE_WHEEL = TAV_RESONANCE_ROOT / "dist" / "tav_resonance-0.1.0-py3-none-any.whl"


def tav_resonance_install_hint() -> str:
    return (
        f"pip install -e {TAV_RESONANCE_ROOT}\n"
        f"  # or: pip install {TAV_RESONANCE_WHEEL}\n"
        f"  # source tree: {TAV_RESONANCE_SRC}"
    )


def _add_src_to_path() -> bool:
    if not TAV_RESONANCE_SRC.is_dir():
        return False
    src = str(TAV_RESONANCE_SRC.resolve())
    if src not in sys.path:
        sys.path.insert(0, src)
    return True


def _try_pip_install_local(*, python: Optional[str] = None) -> bool:
    """Best-effort editable install from Research/tav-resonance."""
    if not (TAV_RESONANCE_ROOT / "pyproject.toml").is_file():
        return False
    py = python or sys.executable
    try:
        proc = subprocess.run(
            [py, "-m", "pip", "install", "-e", str(TAV_RESONANCE_ROOT)],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def ensure_tav_resonance_importable(
    *,
    auto_install: bool = False,
    python: Optional[str] = None,
) -> str:
    """
    Import ``tav_resonance``; return ``__version__``.

    Raises ``ImportError`` with install instructions on failure.
    """
    try:
        import tav_resonance

        return str(getattr(tav_resonance, "__version__", "unknown"))
    except ImportError:
        pass

    _add_src_to_path()
    try:
        import tav_resonance

        return str(getattr(tav_resonance, "__version__", "unknown"))
    except ImportError:
        pass

    if auto_install and _try_pip_install_local(python=python):
        import tav_resonance

        return str(getattr(tav_resonance, "__version__", "unknown"))

    raise ImportError(
        "tav_resonance not installed. From Research/.venv or research_tool venv:\n"
        + tav_resonance_install_hint()
    )


def bootstrap_tav_resonance(*, quiet: bool = True) -> Optional[str]:
    """Non-fatal startup hook; returns version string or None."""
    try:
        ver = ensure_tav_resonance_importable()
        if not quiet:
            print(f"[TAV ENGINE] tav-resonance {ver} ({TAV_RESONANCE_ROOT})")
        return ver
    except ImportError as exc:
        if not quiet:
            print(f"[TAV ENGINE] tav-resonance not loaded: {exc}")
        return None