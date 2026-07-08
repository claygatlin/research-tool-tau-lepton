"""
Submenu module contract for Tav-Superblock Research Engine.

Every submenu is a standalone Python module (``*_extension.py``) that implements
the constants and callables documented below.  ``research_tool.py`` stays a thin
entry point; all submenu-specific logic belongs in the extension file.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


# =============================================================================
# BLOCK: Required module-level constants
# =============================================================================

# MODULE_TAG: str
#     Unique tag returned by the curses menu (e.g. "TAU_SB_DESI", "SPARC").
#
# SUBMENU_TITLE: str
#     Label shown under a Primary Matrix domain (e.g. "DESI BAO Tau-SB Scan").
#
# MENU_ACTIONS: list[str]
#     Ordered action labels for that submenu.


# =============================================================================
# BLOCK: Required run hook
# =============================================================================

# def run_action(selection: str, show_plots: bool = True, options: dict | None = None):
#     Execute the chosen action.  Called after the entry form merges into options.


# =============================================================================
# BLOCK: Optional UI hooks (strongly recommended)
# =============================================================================

# def entry_fields(action: str) -> list[dict[str, Any]]:
#     Curses entry-form field specs for ``action``.  Return [] when defaults suffice.
#
# def entry_instructions(action: str) -> list[str]:
#     Short help bullets shown above the entry form.
#
# def handle_pre_form(stdscr, selection: str, params: dict[str, str]) -> str | None:
#     Called after the user picks an action, before the entry form.
#     Mutate ``params`` in place (n selector, batch limits, warnings).
#     Return "continue" to skip the rest of the menu iteration,
#     "cancel" to return to the submenu, or None to proceed normally.


# =============================================================================
# BLOCK: Optional run-pipeline hooks
# =============================================================================

# def prepare_run_options(action: str, raw: dict) -> dict:
# def options_for_run_log(action: str, prepared: dict) -> dict:


@runtime_checkable
class SubmenuModule(Protocol):
    """Structural type checked by the registry at runtime."""

    MODULE_TAG: str
    SUBMENU_TITLE: str
    MENU_ACTIONS: list[str]

    def run_action(
        self,
        selection: str,
        show_plots: bool = True,
        options: dict | None = None,
    ) -> Any: ...

    def entry_fields(self, action: str) -> list[dict[str, Any]]: ...

    def entry_instructions(self, action: str) -> list[str]: ...


def resolve_entry_fields(extension: Any, action: str) -> list[dict[str, Any]]:
    """
    Call the extension's entry-field provider.

    Supports legacy names (``desi_entry_fields``, ``frb_entry_fields``, …)
    so older modules keep working during migration.
    """
    for attr in (
        "entry_fields",
        "desi_entry_fields",
        "frb_entry_fields",
        "tav_resonance_entry_fields",
    ):
        fn = getattr(extension, attr, None)
        if callable(fn):
            return list(fn(action))
    return []


def resolve_entry_instructions(extension: Any, action: str) -> list[str]:
    """Same as ``resolve_entry_fields`` but for instruction bullets."""
    for attr in (
        "entry_instructions",
        "desi_entry_instructions",
        "frb_entry_instructions",
        "tav_resonance_entry_instructions",
    ):
        fn = getattr(extension, attr, None)
        if callable(fn):
            return list(fn(action))
    return ["Follow field hints; leave optional entries blank to use engine defaults."]


def call_pre_form_hook(
    extension: Any,
    stdscr: Any,
    selection: str,
    params: dict[str, str],
) -> str | None:
    """
    Invoke ``handle_pre_form`` when present.

    Returns the hook's string sentinel or None.
    """
    fn = getattr(extension, "handle_pre_form", None)
    if callable(fn):
        return fn(stdscr, selection, params)
    return None