"""
Capture terminal text from research-tool runs into artifacts/*.txt logs.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = _PROJECT_ROOT / "artifacts"


class _TeeStream:
    """Write stdout to terminal and a log file simultaneously."""

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, data: str) -> int:
        if not data:
            return 0
        self.terminal.write(data)
        self.log_file.write(data)
        return len(data)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()


def _slugify(text: str) -> str:
    clean = "".join(ch if ch.isalnum() else "_" for ch in text.lower())
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_")[:48] or "run"


def build_test_output_path(
    test_name: str,
    *,
    output_dir: Path | str = ARTIFACTS_DIR,
    when: datetime | None = None,
) -> Path:
    """
    Build ``{test_name}.out.{date}.out`` for a single test/run.

    ``test_name`` is slugified; ``date`` is ``YYYYMMDD_HHMMSS``.
    """
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    slug = _slugify(test_name)
    return Path(output_dir) / f"{slug}.out.{stamp}.out"


def write_test_output(
    test_name: str,
    content: str,
    *,
    output_dir: Path | str = ARTIFACTS_DIR,
    when: datetime | None = None,
    append: bool = False,
) -> Path:
    """Write text to ``{test_name}.out.{date}.out`` and return the path."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = build_test_output_path(test_name, output_dir=out_dir, when=when)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(content)
        if append and content and not content.endswith("\n"):
            handle.write("\n")
    return path


def build_log_path(
    module_tag: str,
    action: str,
    artifacts_dir: Path | str = ARTIFACTS_DIR,
) -> Path:
    test_name = f"{module_tag}_{action}" if module_tag else action
    return build_test_output_path(test_name, output_dir=artifacts_dir)


def parse_show_graphics(options: Optional[dict], default: str = "artifacts") -> bool:
    """
    Return True for interactive Tk pop-ups, False for artifacts-only PNG saves.

    Accepted popup values: popup, tk, window, show, yes, interactive
    Accepted artifacts values: artifacts, save, file, no, later
    """
    raw = (options or {}).get("show_graphics", default)
    mode = str(raw).strip().lower()
    if mode in {"popup", "tk", "window", "show", "yes", "y", "interactive", "live"}:
        return True
    if mode in {"artifacts", "save", "file", "no", "n", "later", "disk", "offline"}:
        return False
    return default.strip().lower() in {"popup", "tk", "show", "yes", "interactive"}


def parse_batch_limit(options: Optional[dict], default: int = 0) -> int:
    """0 means no limit (process all local CSV files)."""
    raw = (options or {}).get("batch_limit", "")
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = int(str(raw).strip())
    except ValueError:
        return default
    return max(0, value)


@contextmanager
def capture_run_log(
    module_tag: str,
    action: str,
    options: Optional[dict] = None,
    artifacts_dir: Path | str = ARTIFACTS_DIR,
) -> Iterator[Path]:
    """
    Tee stdout to artifacts/{test_name}.out.{date}.out for the run duration.
    """
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = build_log_path(module_tag, action, artifacts_dir=out_dir)

    header_lines = [
        "=== TAV-SUPERBLOCK RUN LOG ===",
        f"Module: {module_tag}",
        f"Action: {action}",
        f"Started: {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC",
        f"Log file: {log_path}",
    ]
    if options:
        header_lines.append("Options:")
        for key, value in sorted(options.items()):
            if str(value).strip():
                header_lines.append(f"  {key}: {value}")
    header_lines.append("=" * 40)
    header_lines.append("")

    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(header_lines) + "\n")
        handle.flush()
        tee = _TeeStream(sys.stdout, handle)
        previous = sys.stdout
        sys.stdout = tee
        try:
            yield log_path
        finally:
            sys.stdout = previous
            footer = (
                f"\n{'=' * 40}\n"
                f"Finished: {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC\n"
            )
            handle.write(footer)
            handle.flush()
            test_label = f"{module_tag}_{action}" if module_tag else action
            print(f"[TAV ENGINE] Test output saved: {log_path}")
            print(f"[TAV ENGINE] Test: {test_label}")