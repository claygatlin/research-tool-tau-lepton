#!/usr/bin/env python3
"""
Curses UI for picking fetch-capable modules and dataset targets.
"""

from __future__ import annotations

import curses
from typing import Dict, List, Optional, Tuple

from tav_shared.dataset_registry import DatasetProvider, DatasetTarget, TargetState, list_modules


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


def _state_glyph(state: TargetState, selected: bool) -> str:
    if selected:
        return "[*]"
    if state == TargetState.PROCESSED:
        return "[x]"
    if state == TargetState.CACHED:
        return "[+]"
    if state == TargetState.ARCHIVED:
        return "[~]"
    return "[ ]"


def _fetchable_states() -> set[TargetState]:
    return {TargetState.REMOTE, TargetState.CACHED}


def _is_selectable_for_fetch(target: DatasetTarget) -> bool:
    return target.state in _fetchable_states()


def _is_selectable_for_run(target: DatasetTarget) -> bool:
    return target.state in {TargetState.CACHED, TargetState.PROCESSED}


def pick_module(stdscr, *, title: str = "Select Module") -> Optional[DatasetProvider]:
    providers = list_modules()
    if not providers:
        return None

    current = 0
    while True:
        stdscr.clear()
        max_y, max_x = _term_size(stdscr)
        _safe_addstr(stdscr, 1, 2, title, curses.A_BOLD)
        _safe_addstr(stdscr, 2, 2, "UP/DOWN navigate | ENTER select | q cancel", curses.A_DIM)

        start_row = 4
        for idx, provider in enumerate(providers):
            row = start_row + idx
            if row >= max_y - 2:
                break
            prefix = "> " if idx == current else "  "
            line = f"{prefix}{provider.title} ({provider.module_tag})"
            attr = curses.color_pair(1) if idx == current else 0
            _safe_addstr(stdscr, row, 4, line[: max(1, max_x - 8)], attr)

        stdscr.refresh()
        key = stdscr.getch()
        if key in (ord("q"), ord("Q"), 27):
            return None
        if key == curses.KEY_UP and current > 0:
            current -= 1
        elif key == curses.KEY_DOWN and current < len(providers) - 1:
            current += 1
        elif key in (10, 13):
            return providers[current]


def pick_targets(
    stdscr,
    provider: DatasetProvider,
    *,
    mode: str = "fetch",
    title: Optional[str] = None,
) -> Optional[List[str]]:
    """
    Paginated multi-select of dataset targets.

    mode:
      fetch  — show remote + cached targets
      run    — show runnable (cached) targets only
    """
    if mode == "run":
        targets = provider.list_runnable()
    else:
        targets = provider.list_fetchable()

    if not targets:
        return []

    title = title or f"{provider.title} — {'Fetch' if mode == 'fetch' else 'Run'} Targets"
    selected: set[str] = set()
    page = 0
    page_size = 12
    cursor = 0

    while True:
        stdscr.clear()
        max_y, max_x = _term_size(stdscr)
        _safe_addstr(stdscr, 1, 2, title, curses.A_BOLD)
        _safe_addstr(
            stdscr,
            2,
            2,
            "UP/DOWN move | SPACE toggle | a all | c clear | n/p page | ENTER | q",
            curses.A_DIM,
        )
        _safe_addstr(
            stdscr,
            3,
            2,
            "Selected: {} | [ ] remote [+] cached [~] archived [x] processed [*] picked".format(
                len(selected)
            ),
            curses.A_DIM,
        )

        total_pages = max(1, (len(targets) + page_size - 1) // page_size)
        page = min(page, total_pages - 1)
        start = page * page_size
        chunk = targets[start : start + page_size]

        row = 5
        for idx, target in enumerate(chunk):
            if row >= max_y - 3:
                break
            glyph = _state_glyph(target.state, target.id in selected)
            pointer = ">" if idx == cursor else " "
            suffix = ""
            if mode == "fetch" and target.state == TargetState.PROCESSED:
                suffix = " — skip fetch"
            line = f"{pointer} {glyph} {target.label} ({target.state.value}){suffix}"
            attr = curses.color_pair(1) if idx == cursor else 0
            if target.state == TargetState.PROCESSED and mode == "fetch":
                attr |= curses.A_DIM
            _safe_addstr(stdscr, row, 4, line[: max(1, max_x - 8)], attr)
            row += 1

        _safe_addstr(
            stdscr,
            max_y - 2,
            2,
            f"Page {page + 1}/{total_pages}  |  {provider.intake_dir}",
            curses.A_DIM,
        )
        stdscr.refresh()

        key = stdscr.getch()
        if key in (ord("q"), ord("Q"), 27):
            return None
        if key in (10, 13):
            return sorted(selected) if selected else None
        if key == ord("a"):
            if mode == "run":
                selected = {t.id for t in targets if _is_selectable_for_run(t)}
            else:
                selected = {t.id for t in targets if _is_selectable_for_fetch(t)}
        elif key == ord("c"):
            selected.clear()
        elif key == ord("n") and page < total_pages - 1:
            page += 1
            cursor = 0
        elif key == ord("p") and page > 0:
            page -= 1
            cursor = 0
        elif key == curses.KEY_UP and cursor > 0:
            cursor -= 1
        elif key == curses.KEY_DOWN and cursor < len(chunk) - 1:
            cursor += 1
        elif key == ord(" "):
            if chunk and 0 <= cursor < len(chunk):
                target = chunk[cursor]
                tid = target.id
                if mode == "fetch" and target.state == TargetState.PROCESSED:
                    continue
                if mode == "run" and not _is_selectable_for_run(target):
                    continue
                if tid in selected:
                    selected.remove(tid)
                else:
                    selected.add(tid)

        if ord("1") <= key <= ord("9"):
            index = key - ord("1")
            if index < len(chunk):
                target = chunk[index]
                if mode == "fetch" and target.state == TargetState.PROCESSED:
                    continue
                if mode == "run" and not _is_selectable_for_run(target):
                    continue
                tid = target.id
                if tid in selected:
                    selected.remove(tid)
                else:
                    selected.add(tid)
                cursor = index


def pick_action(stdscr, provider: DatasetProvider) -> Optional[str]:
    actions = provider.run_actions or []
    if not actions:
        return None
    if len(actions) == 1:
        return actions[0]

    current = 0
    while True:
        stdscr.clear()
        max_y, max_x = _term_size(stdscr)
        _safe_addstr(stdscr, 1, 2, f"{provider.title} — Select Action", curses.A_BOLD)
        _safe_addstr(stdscr, 2, 2, "UP/DOWN | ENTER | q cancel", curses.A_DIM)
        for idx, action in enumerate(actions):
            row = 4 + idx
            if row >= max_y - 2:
                break
            prefix = "> " if idx == current else "  "
            attr = curses.color_pair(1) if idx == current else 0
            _safe_addstr(stdscr, row, 4, f"{prefix}{action}"[: max(1, max_x - 8)], attr)
        stdscr.refresh()
        key = stdscr.getch()
        if key in (ord("q"), ord("Q"), 27):
            return None
        if key == curses.KEY_UP and current > 0:
            current -= 1
        elif key == curses.KEY_DOWN and current < len(actions) - 1:
            current += 1
        elif key in (10, 13):
            return actions[current]


def run_dataset_manager(stdscr, mode: str) -> Optional[Tuple[str, str, List[str], Dict]]:
    """
    Full picker flow.

    Returns (module_tag, action_or_mode, target_ids, options) or None if cancelled.
    mode: 'fetch' or 'run'
    """
    provider = pick_module(
        stdscr,
        title="Dataset Manager — Select Module",
    )
    if provider is None:
        return None

    targets = pick_targets(
        stdscr,
        provider,
        mode=mode,
        title=f"{provider.title} — {'Fetch' if mode == 'fetch' else 'Run'} Targets (1-9 toggle)",
    )
    if targets is None:
        return None
    if not targets:
        return None

    from research_tool import (
        dataset_manager_fetch_fields,
        dataset_manager_run_fields,
        show_entry_form,
    )

    if mode == "fetch":
        entry = show_entry_form(
            stdscr,
            "Dataset Manager",
            f"{provider.title} — Fetch Options",
            dataset_manager_fetch_fields(),
            instructions=[
                "Force re-download overrides the processed ledger and cached files.",
            ],
        )
        if entry is None:
            return None
        return provider.module_tag, "fetch", targets, entry

    action = pick_action(stdscr, provider)
    if action is None:
        return None

    entry = show_entry_form(
        stdscr,
        "Dataset Manager",
        f"{provider.title} — Run Options",
        dataset_manager_run_fields(provider.module_tag),
        instructions=[
            f"Action: {action}",
            "Text fields accept typed input; yes/no and mode fields use the cursor.",
        ],
    )
    if entry is None:
        return None
    return provider.module_tag, action, targets, entry