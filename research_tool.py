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
    ChunkedScanProgress,
    _safe_addstr,
    append_nav,
    draw_menu,
    draw_progress_bar,
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
from tav_shared.tav_resonance_bootstrap import bootstrap_tav_resonance

load_api_keys()
bootstrap_tav_resonance(quiet=True)

if __name__ == "__main__":
    main()