"""
Curses UI shell — menus, entry forms, and terminal-safe text rendering.

Shared by every submenu.  Submenu-specific pre-form logic lives in each
``*_extension.py`` (see ``handle_pre_form``).
"""

from __future__ import annotations

import curses

ROOT_TITLE = "Primary Matrix"
NAV_BACK = "<< Back"
NAV_HOME = "<< Home"
NAV_ITEMS = {NAV_BACK, NAV_HOME}

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


