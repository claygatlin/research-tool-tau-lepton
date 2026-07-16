"""
Curses UI for listing modules with cached datasets and clearing one selection.
"""

from __future__ import annotations

import curses
from typing import List, Optional

from tav_shared.clear_cache import (
    CLEAR_CACHE_MENU_LABEL,
    ClearCacheResult,
    ModuleCacheEntry,
    clear_module_cache,
    format_size,
    list_modules_with_cache,
)


def _term_size(stdscr) -> tuple[int, int]:
    max_y, max_x = stdscr.getmaxyx()
    return max_y, max_x


def _safe_addstr(stdscr, row, col, text, attr=0) -> bool:
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


def _pick_module(stdscr, entries: List[ModuleCacheEntry]) -> Optional[ModuleCacheEntry]:
    current = 0
    while True:
        stdscr.clear()
        max_y, max_x = _term_size(stdscr)
        _safe_addstr(stdscr, 1, 2, CLEAR_CACHE_MENU_LABEL, curses.A_BOLD)
        _safe_addstr(
            stdscr,
            2,
            2,
            "UP/DOWN navigate | ENTER clear module cache | q cancel",
            curses.A_DIM,
        )

        start_row = 4
        for idx, entry in enumerate(entries):
            row = start_row + idx
            if row >= max_y - 2:
                break
            prefix = "> " if idx == current else "  "
            size_label = format_size(entry.total_bytes)
            line = (
                f"{prefix}{entry.title} ({entry.module_tag}) — "
                f"{entry.target_count} cached, {size_label}"
            )
            attr = curses.color_pair(1) if idx == current else 0
            _safe_addstr(stdscr, row, 4, line[: max(1, max_x - 8)], attr)

        stdscr.refresh()
        key = stdscr.getch()
        if key in (ord("q"), ord("Q"), 27):
            return None
        if key == curses.KEY_UP and current > 0:
            current -= 1
        elif key == curses.KEY_DOWN and current < len(entries) - 1:
            current += 1
        elif key in (10, 13):
            return entries[current]


def _confirm_clear(stdscr, entry: ModuleCacheEntry) -> bool:
    lines = [
        f"Clear cache: {entry.title}",
        f"Module: {entry.module_tag}",
        "",
        f"This will permanently delete {entry.target_count} cached dataset(s)",
        f"({format_size(entry.total_bytes)}) from disk.",
        "",
        "Pull/batch ledgers and processed flags are kept (work will not re-run).",
        "Artifacts and run logs are not removed.",
        "",
        "Press Y to confirm, any other key to cancel.",
    ]
    stdscr.clear()
    max_y, max_x = _term_size(stdscr)
    for row, line in enumerate(lines):
        if row >= max_y - 2:
            break
        _safe_addstr(stdscr, row, 2, line[: max(1, max_x - 4)])
    stdscr.refresh()
    key = stdscr.getch()
    return chr(key).lower() == "y"


def _show_result(stdscr, result: ClearCacheResult) -> None:
    lines = [
        f"Cleared: {result.title}",
        "",
        f"Paths removed: {len(result.deleted_paths)}",
        f"Space freed: {format_size(result.bytes_freed)}",
    ]
    if result.errors:
        lines.extend(["", "Errors:"])
        lines.extend(f"  - {err}" for err in result.errors[:6])
        if len(result.errors) > 6:
            lines.append(f"  ... and {len(result.errors) - 6} more")
    lines.extend(["", "Press any key to return to the menu."])

    stdscr.clear()
    max_y, max_x = _term_size(stdscr)
    for row, line in enumerate(lines):
        if row >= max_y - 2:
            break
        _safe_addstr(stdscr, row, 2, line[: max(1, max_x - 4)])
    stdscr.refresh()
    stdscr.getch()


def run_clear_cache_menu(stdscr) -> None:
    """Interactive flow: list cached modules, confirm, delete selected cache."""
    entries = list_modules_with_cache()
    stdscr.clear()
    max_y, max_x = _term_size(stdscr)

    if not entries:
        lines = [
            CLEAR_CACHE_MENU_LABEL,
            "",
            "No modules have downloaded datasets on disk.",
            "",
            "Press any key to return to the menu.",
        ]
        for row, line in enumerate(lines):
            if row >= max_y - 2:
                break
            _safe_addstr(stdscr, row, 2, line[: max(1, max_x - 4)])
        stdscr.refresh()
        stdscr.getch()
        return

    picked = _pick_module(stdscr, entries)
    if picked is None:
        return
    if not _confirm_clear(stdscr, picked):
        return

    try:
        result = clear_module_cache(picked.module_tag)
    except Exception as exc:
        result = ClearCacheResult(
            module_tag=picked.module_tag,
            title=picked.title,
            errors=[str(exc).strip() or type(exc).__name__],
        )
    _show_result(stdscr, result)