"""
MODULE_TEMPLATE — copy this folder to create a new Tav-Superblock submenu.

Steps
-----
1. Copy ``menus/MODULE_TEMPLATE/`` to ``menus/<domain>/<your_submenu>/``
2. Rename ``extension.py`` constants (MODULE_TAG, SUBMENU_TITLE, MENU_ACTIONS)
3. Implement ``run_action``, ``entry_fields``, ``entry_instructions``
4. Register in ``tav_research/registry.py`` (import + MODULE_EXTENSIONS)
5. Add the submenu title to ``tav_research/menu_tree.py`` under the right domain
6. Run ``./venv/bin/python menus/_layout.py`` or ``ensure_menu_folders()`` to verify paths

Global utilities live in ``tav_shared/``.  Curses UI lives in ``tav_research/``.
"""

from __future__ import annotations

from typing import Any

# =============================================================================
# BLOCK: Required constants
# =============================================================================

MODULE_TAG = "EXAMPLE_MODULE"
SUBMENU_TITLE = "Example Submenu"
MENU_ACTIONS = [
    "Example Action",
]

# =============================================================================
# BLOCK: Entry form — fields
# =============================================================================


def entry_fields(action: str) -> list[dict[str, Any]]:
    """Return curses form field specs for ``action``."""
    if action == "Example Action":
        return [
            {
                "key": "notes",
                "label": "Run notes (optional)",
                "default": "",
                "required": False,
                "hint": "Annotation for this run",
            },
        ]
    return []


# =============================================================================
# BLOCK: Entry form — instructions
# =============================================================================


def entry_instructions(action: str) -> list[str]:
    """Help bullets shown above the entry form."""
    return [
        f"Example submenu — {action}",
        "Replace this template with real data paths and engine knobs.",
    ]


# =============================================================================
# BLOCK: Pre-form hook (optional)
# =============================================================================


def handle_pre_form(stdscr, selection: str, params: dict[str, str]) -> str | None:
    """
    Called before the entry form.  Mutate ``params`` in place.

    Return ``\"continue\"`` to skip the form, ``\"cancel\"`` to abort, or None.
    """
    _ = (stdscr, selection, params)
    return None


# =============================================================================
# BLOCK: Run pipeline (optional hooks)
# =============================================================================


def prepare_run_options(action: str, raw: dict[str, str]) -> dict[str, str]:
    """Merge form values with module defaults before ``run_action``."""
    merged = dict(raw)
    merged.setdefault("action", action)
    return merged


def options_for_run_log(action: str, prepared: dict[str, str]) -> dict[str, str]:
    """Subset of options printed in run logs."""
    return {k: prepared[k] for k in ("action", "notes") if k in prepared}


# =============================================================================
# BLOCK: run_action
# =============================================================================


def run_action(
    selection: str,
    show_plots: bool = True,
    options: dict | None = None,
) -> None:
    """Execute the selected menu action."""
    _ = show_plots
    options = options or {}
    print(f"[TAV ENGINE] {SUBMENU_TITLE} — {selection}")
    print(f"[TAV ENGINE] options={options}")