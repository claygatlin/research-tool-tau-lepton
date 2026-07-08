#!/usr/bin/env python3
"""
Tau-Superblock Research Engine — thin entry point.

Each submenu lives in its own ``*_extension.py`` module.  Shared curses UI,
menu tree, and run orchestration live under ``tav_research/``.

Run:  ./venv/bin/python research_tool.py

Remote AI: configure keys in config/api_keys.env (see api_keys.env.example).
Menu domain **Remote AI Processing** — test connections, analyze artifacts, custom prompts.
Post-run review: set **AI verbose review = yes** on any module entry form.
"""

from __future__ import annotations

# =============================================================================
# BLOCK: Backward-compatible re-exports (mock_generator, desi_dashboard, …)
# =============================================================================

from tav_research.curses_shell import (  # noqa: F401
    ROOT_TITLE,
    _safe_addstr,
    append_nav,
    draw_menu,
    select_n_interactive,
    show_entry_form,
)
from tav_research.n_selector import should_offer_n_selector  # noqa: F401
from tav_research.registry import (  # noqa: F401
    MODULE_EXTENSIONS,
    module_entry_fields,
    module_entry_instructions,
)

# =============================================================================
# BLOCK: Main
# =============================================================================

from tav_research.runner import load_api_keys, main

load_api_keys()

if __name__ == "__main__":
    main()