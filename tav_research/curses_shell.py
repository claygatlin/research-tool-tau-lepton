"""
Curses UI shell — menus, entry forms, and terminal-safe text rendering.

Shared by every submenu.  Submenu-specific pre-form logic lives in each
``*_extension.py`` (see ``handle_pre_form``).
"""

from __future__ import annotations

import curses
import subprocess
import sys
import time
from pathlib import Path
from contextlib import contextmanager
from typing import Any

ROOT_TITLE = "Primary Matrix"
NAV_BACK = "<< Back"
NAV_HOME = "<< Home"
NAV_ITEMS = {NAV_BACK, NAV_HOME}


def restore_terminal() -> None:
    """Return the terminal to normal cooked mode after curses exits."""
    try:
        curses.endwin()
    except Exception:
        pass
    try:
        subprocess.run(
            ["stty", "sane"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        pass
    try:
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
    except Exception:
        pass

def append_nav(options):
    return list(options) + [NAV_BACK, NAV_HOME]


def _term_size(stdscr):
    max_y, max_x = stdscr.getmaxyx()
    return max_y, max_x


def _safe_addstr(stdscr, row, col, text, attr=0):
    """Write text without raising curses.error on small terminals or wide glyphs."""
    max_y, max_x = _term_size(stdscr)
    if row < 0 or row >= max_y - 1 or col >= max_x - 1:
        return False
    clip = str(text)[: max(0, max_x - col - 1)]
    if not clip:
        return False
    try:
        if attr:
            stdscr.attron(attr)
        stdscr.addstr(row, col, clip)
        if attr:
            stdscr.attroff(attr)
        return True
    except curses.error:
        return False


def select_n_interactive(
    stdscr,
    prompt: str = "Select sample size n",
    default: int = 25,
) -> int | None:
    """Cursor-driven multi-choice selector for n (arrow keys + Enter)."""
    choices: list[int | str] = [6, 10, 15, 20, 25, 30, 40, 50, "Custom..."]
    current = choices.index(default) if default in choices else 4

    while True:
        stdscr.clear()
        _safe_addstr(stdscr, 2, 2, prompt, curses.A_BOLD)
        _safe_addstr(
            stdscr,
            3,
            2,
            "UP/DOWN or LEFT/RIGHT  |  ENTER = confirm  |  q = cancel",
            curses.A_DIM,
        )

        for idx, choice in enumerate(choices):
            row = 5 + idx
            prefix = "> " if idx == current else "  "
            attr = curses.color_pair(1) if idx == current else 0
            _safe_addstr(stdscr, row, 4, f"{prefix}{choice}", attr)

        stdscr.refresh()
        key = stdscr.getch()

        if key in (ord("q"), ord("Q"), 27):
            return None
        if key in (10, 13):
            selected = choices[current]
            if selected == "Custom...":
                _safe_addstr(stdscr, 16, 2, "Enter custom n: ")
                stdscr.refresh()
                curses.echo()
                try:
                    val = int(stdscr.getstr(16, 18, 5).decode().strip())
                    curses.noecho()
                    return max(3, val)
                except ValueError:
                    curses.noecho()
                    return None
            return int(selected)

        if key in (curses.KEY_UP, curses.KEY_LEFT):
            current = (current - 1) % len(choices)
        elif key in (curses.KEY_DOWN, curses.KEY_RIGHT):
            current = (current + 1) % len(choices)


def _default_n_for_field(field: dict) -> int:
    raw = field.get("default", 25)
    try:
        return int(str(raw).strip())
    except ValueError:
        return 25


def draw_menu(stdscr, title, options, current_row, breadcrumb=""):
    stdscr.clear()
    _safe_addstr(stdscr, 1, 2, "=== TAV-SUPERBLOCK UNIVERSE THEORY ===", curses.A_BOLD)
    _safe_addstr(stdscr, 2, 2, f"Domain: {title}", curses.A_UNDERLINE)
    if breadcrumb:
        _safe_addstr(stdscr, 3, 2, breadcrumb[:76], curses.A_DIM)
    _safe_addstr(stdscr, 5, 2, "UP/DOWN navigate | ENTER select | Back/Home move in tree")
    start_row = 7
    max_y, max_x = _term_size(stdscr)
    for idx, option in enumerate(options):
        row = start_row + idx
        if row >= max_y - 1:
            break
        prefix = "> " if idx == current_row else "  "
        line = f"{prefix}{option}"
        if idx == current_row:
            _safe_addstr(stdscr, row, 4, line[: max(1, max_x - 6)], curses.color_pair(1))
        else:
            _safe_addstr(stdscr, row, 4, line[: max(1, max_x - 6)])
    stdscr.refresh()


def _read_field(stdscr, row, col, width=56):
    curses.echo()
    curses.curs_set(1)
    raw = stdscr.getstr(row, col, width)
    curses.noecho()
    curses.curs_set(0)
    return raw.decode("utf-8", errors="replace").strip()


def _infer_field_choices(field: dict) -> list[str] | None:
    """Return selectable options for a form field, or None for free-text entry."""
    explicit = field.get("choices") or field.get("options")
    if explicit:
        return [str(choice) for choice in explicit]

    key = str(field.get("key", "")).strip().lower()
    default = str(field.get("default", "")).strip().lower()
    label = str(field.get("label", "")).lower()
    field_type = str(field.get("type", "")).strip().lower()

    if key == "n_points":
        return None

    if field_type in {"choice", "select", "option", "options", "yesno", "bool"}:
        if key == "show_graphics":
            return ["artifacts", "popup"]
        return ["no", "yes"]

    if key == "show_graphics":
        return ["artifacts", "popup"]

    if default in {"yes", "no"}:
        return ["no", "yes"]
    if "yes/no" in label or "(yes/no)" in label:
        return ["no", "yes"]
    return None


def _pick_choice_field(stdscr, start_row: int, field: dict) -> str | None:
    """
    Cursor-driven option picker (UP/DOWN or LEFT/RIGHT, ENTER to confirm).
    Returns the selected value, or None if cancelled.
    """
    choices = _infer_field_choices(field) or []
    if not choices:
        return None
    if len(choices) == 1:
        return choices[0]

    default = str(field.get("default", choices[0])).strip().lower()
    current = 0
    for index, choice in enumerate(choices):
        if choice.lower() == default:
            current = index
            break

    while True:
        max_y, max_x = _term_size(stdscr)
        _safe_addstr(
            stdscr,
            start_row,
            2,
            "UP/DOWN or LEFT/RIGHT to change | ENTER confirm | q cancel",
            curses.A_DIM,
        )
        list_row = start_row + 2
        for index, choice in enumerate(choices):
            row = list_row + index
            if row >= max_y - 2:
                break
            prefix = "> " if index == current else "  "
            attr = curses.color_pair(1) if index == current else 0
            _safe_addstr(stdscr, row, 4, f"{prefix}{choice}", attr)
        stdscr.refresh()

        key = stdscr.getch()
        if key in (ord("q"), ord("Q"), 27):
            return None
        if key in (10, 13):
            return choices[current]
        if key in (curses.KEY_LEFT, curses.KEY_UP):
            current = (current - 1) % len(choices)
        elif key in (curses.KEY_RIGHT, curses.KEY_DOWN):
            current = (current + 1) % len(choices)
        elif len(choices) == 2:
            if key in (ord("y"), ord("Y")) and "yes" in (c.lower() for c in choices):
                return "yes"
            if key in (ord("n"), ord("N")) and "no" in (c.lower() for c in choices):
                return "no"


def _wrap_instruction(text, width=70):
    words = text.split()
    lines, current = [], []
    length = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and length + extra > width:
            lines.append(" ".join(current))
            current, length = [word], len(word)
        else:
            current.append(word)
            length += extra
    if current:
        lines.append(" ".join(current))
    return lines or [text[:width]]


def show_entry_form(stdscr, title, subtitle, fields, instructions=None):
    """
    Entry form — one field per screen so the cursor is always visible.
    Empty field list shows a confirm screen (ENTER = run with defaults).
    Returns dict of values, or None if cancelled.
    """
    max_y, max_x = _term_size(stdscr)
    instructions = instructions or []
    fields = fields or []

    def _draw_header(row_start=1):
        row = row_start
        _safe_addstr(stdscr, row, 2, title, curses.A_BOLD)
        row += 1
        _safe_addstr(stdscr, row, 2, subtitle[: max(1, max_x - 4)], curses.A_UNDERLINE)
        return row + 1

    def _draw_instructions(row):
        if not instructions:
            return row
        _safe_addstr(stdscr, row, 2, "ENGINE ACCEPTS:", curses.A_BOLD)
        row += 1
        for block in instructions:
            for line in _wrap_instruction(block, width=max(20, max_x - 8)):
                if row >= max_y - 4:
                    return row
                _safe_addstr(stdscr, row, 4, f"- {line}", curses.A_DIM)
                row += 1
        return row + 1

    if not fields:
        stdscr.clear()
        row = _draw_header()
        row = _draw_instructions(row)
        _safe_addstr(stdscr, row, 2, "ENTER = run with defaults | type q + ENTER = cancel", curses.A_DIM)
        row += 2
        _safe_addstr(stdscr, row, 2, "> ")
        value = _read_field(stdscr, row, 4, width=8)
        if value.strip().lower() in {"q", "quit", "cancel"}:
            return None
        return {}

    values = {}
    total = len(fields)
    for index, field in enumerate(fields, start=1):
        stdscr.clear()
        row = _draw_header()
        _safe_addstr(stdscr, row, 2, f"Field {index} of {total}", curses.A_DIM)
        row += 2

        label = field["label"]
        hint = field.get("hint", "")
        default = field.get("default", "")
        choices = _infer_field_choices(field)
        _safe_addstr(stdscr, row, 2, f"{label}:", curses.A_BOLD)
        row += 1
        if hint:
            _safe_addstr(stdscr, row, 4, hint[: max(1, max_x - 6)], curses.A_DIM)
            row += 2

        if field.get("key") == "n_points" or field.get("type") == "choice":
            picked = select_n_interactive(
                stdscr,
                prompt=str(label),
                default=_default_n_for_field(field),
            )
            if picked is None:
                return None
            value = str(picked)
        elif choices:
            value = _pick_choice_field(stdscr, row, field)
            if value is None:
                return None
        else:
            _safe_addstr(stdscr, row, 2, "Blank = default. Required empty = cancel.", curses.A_DIM)
            row += 1
            _safe_addstr(stdscr, row, 4, f"[{default}] " if default else "> ")
            value = _read_field(stdscr, row, 6 if default else 4)
            if not value:
                value = field.get("default", "")
            if field.get("required") and not value:
                return None
        values[field["key"]] = value

    return values


def _format_duration(seconds: float) -> str:
    if seconds < 0 or not (seconds < 1e12):
        return "—"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _format_count(value: int) -> str:
    return f"{int(value):,}"


def draw_progress_bar(
    stdscr,
    *,
    fraction: float,
    row: int,
    col: int = 4,
    width: int | None = None,
    fill: str = "█",
    empty: str = "░",
) -> int:
    """Draw a horizontal bar; returns the row after the bar."""
    max_y, max_x = _term_size(stdscr)
    if row >= max_y - 1:
        return row
    bar_width = width
    if bar_width is None:
        bar_width = max(10, min(48, max_x - col - 12))
    frac = max(0.0, min(1.0, float(fraction)))
    filled = int(round(bar_width * frac))
    bar = fill * filled + empty * (bar_width - filled)
    pct = f"{frac * 100:5.1f}%"
    _safe_addstr(stdscr, row, col, f"[{bar}] {pct}")
    return row + 1


def draw_chunked_scan_screen(
    stdscr,
    *,
    title: str,
    path: str,
    events_done: int,
    total_events: int,
    muons_binned: int,
    chunk_size: int,
    elapsed_sec: float,
    eta_sec: float,
    status: str = "Scanning",
) -> None:
    """Full-screen progress layout for uproot chunked CMS scans."""
    max_y, max_x = _term_size(stdscr)
    stdscr.clear()
    row = 2
    _safe_addstr(stdscr, row, 2, title, curses.A_BOLD)
    row += 2

    path_label = Path(path).name if path else "—"
    _safe_addstr(stdscr, row, 4, f"File: {path_label[: max(1, max_x - 12)]}", curses.A_DIM)
    row += 2

    total = max(int(total_events), 1)
    done = max(0, int(events_done))
    frac = min(1.0, done / total)
    _safe_addstr(
        stdscr,
        row,
        4,
        f"Events: {_format_count(done)} / {_format_count(total)}",
        curses.A_BOLD,
    )
    row += 1
    row = draw_progress_bar(stdscr, fraction=frac, row=row, col=4, width=max(10, max_x - 20))
    row += 1

    lines = [
        f"Muons binned: {_format_count(muons_binned)}",
        f"Chunk size:   {_format_count(chunk_size)}",
        f"Elapsed:      {_format_duration(elapsed_sec)}",
        f"ETA:          {_format_duration(eta_sec)}",
        f"Status:       {status}",
    ]
    for line in lines:
        if row >= max_y - 2:
            break
        _safe_addstr(stdscr, row, 4, line[: max(1, max_x - 6)])
        row += 1

    if row < max_y - 1:
        _safe_addstr(stdscr, max_y - 2, 2, "Ctrl+C to cancel", curses.A_DIM)
    stdscr.refresh()


class ChunkedScanProgress:
    """Live progress for long chunked uproot scans — ncurses on TTY, else stdout."""

    def __init__(
        self,
        *,
        title: str = "CMS NanoAOD chunked scan",
        use_curses: bool | None = None,
    ) -> None:
        self.title = title
        self.use_curses = use_curses if use_curses is not None else sys.stdout.isatty()
        self._stdscr: Any = None
        self._curses_active = False
        self._start_time = 0.0
        self._total = 0
        self._path = ""
        self._chunk_size = 0
        self._last_print = 0

    @property
    def is_interactive(self) -> bool:
        """True when the UI can redraw in place (ncurses or TTY carriage-return)."""
        return self._curses_active or sys.stdout.isatty()

    def __enter__(self) -> ChunkedScanProgress:
        self._start_time = time.monotonic()
        if self.use_curses:
            try:
                self._stdscr = curses.initscr()
                curses.curs_set(0)
                curses.noecho()
                curses.cbreak()
                if curses.has_colors():
                    curses.start_color()
                    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
                    curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)
                self._curses_active = True
            except Exception:
                self._stdscr = None
                self._curses_active = False
                self.use_curses = False
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._curses_active:
            try:
                curses.endwin()
            except Exception:
                pass
            restore_terminal()
        self._curses_active = False
        self._stdscr = None
        if not self.use_curses and self._last_print:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def begin(self, *, path: str, total_events: int, chunk_size: int) -> None:
        self._path = str(path)
        self._total = int(total_events)
        self._chunk_size = int(chunk_size)
        self.update(0, 0, status="Opening ROOT file…")

    def update(
        self,
        events_done: int,
        muons_binned: int,
        *,
        status: str = "Scanning",
    ) -> None:
        elapsed = time.monotonic() - self._start_time
        total = self._total or 1
        done = max(0, int(events_done))
        rate = done / elapsed if elapsed > 0.05 and done > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else 0.0

        if self._curses_active and self._stdscr is not None:
            draw_chunked_scan_screen(
                self._stdscr,
                title=self.title,
                path=self._path,
                events_done=done,
                total_events=total,
                muons_binned=int(muons_binned),
                chunk_size=self._chunk_size,
                elapsed_sec=elapsed,
                eta_sec=eta,
                status=status,
            )
            return

        if not sys.stdout.isatty():
            return

        frac = min(1.0, done / total) if total > 0 else 0.0
        line = (
            f"[CERN CMS] {frac * 100:5.1f}%  "
            f"{_format_count(done)} / {_format_count(total)} events  "
            f"({_format_count(muons_binned)} muons)  "
            f"ETA {_format_duration(eta)}"
        )
        sys.stdout.write("\r" + line[:120])
        sys.stdout.flush()
        self._last_print = 1

    def finish(self, events_done: int, muons_binned: int) -> None:
        self.update(events_done, muons_binned, status="Complete")
        if self._curses_active and self._stdscr is not None:
            time.sleep(0.35)


@contextmanager
def optional_chunked_scan_progress(
    progress: ChunkedScanProgress | None,
    *,
    title: str = "CMS NanoAOD chunked scan",
    use_curses_progress: bool = True,
):
    """Use an existing progress object or create one for the scan duration."""
    if progress is not None:
        yield progress
        return
    if not use_curses_progress:
        yield None
        return
    with ChunkedScanProgress(title=title) as owned:
        yield owned


