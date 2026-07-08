"""
Load remote AI API keys from environment and optional config files.

Search order (first wins per variable):
  1. Already set in process environment
  2. ``config/api_keys.env`` under TauSuperblock root
  3. ``.env`` under TauSuperblock root
"""

from __future__ import annotations

import os
from pathlib import Path

from tav_shared.tav_project_paths import TAU_SUPERBLOCK_ROOT

KNOWN_KEY_VARS = (
    "XAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "NVIDIA_API_KEY",
    "NVAPI_KEY",
    "NVIDIA_NIM_MODEL",
    "NVIDIA_NIM_BASE_URL",
    "NVIDIA_NIM_MAX_TOKENS",
    "NVIDIA_NIM_TEMPERATURE",
    "NVIDIA_NIM_TOP_P",
)


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key:
            values[key] = val
    return values


def load_api_keys(*, verbose: bool = False) -> dict[str, str]:
    """
    Populate unset API key environment variables from config files.

    Returns mapping of variable names that were loaded from disk.
    """
    loaded: dict[str, str] = {}
    candidates = [
        TAU_SUPERBLOCK_ROOT / "config" / "api_keys.env",
        TAU_SUPERBLOCK_ROOT / ".env",
    ]
    merged: dict[str, str] = {}
    for path in candidates:
        merged.update(_parse_env_file(path))

    for var in KNOWN_KEY_VARS:
        if os.environ.get(var, "").strip():
            continue
        value = merged.get(var, "").strip()
        if value:
            os.environ[var] = value
            loaded[var] = value

    if verbose and loaded:
        print(f"[TAV ENGINE] Loaded {len(loaded)} API key(s) from config files.")
    elif verbose:
        print("[TAV ENGINE] No API keys loaded from config (using environment only).")
    return loaded