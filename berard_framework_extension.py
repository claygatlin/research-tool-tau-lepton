"""Top-level shim: ``from berard_framework_extension import ...``."""

from menus.astronomical.berard.extension import (  # noqa: F401
    MENU_ACTIONS,
    MODULE_TAG,
    SUBMENU_TITLE,
    entry_fields,
    entry_instructions,
    query_llm_analysis,
    run_action,
)

__all__ = [
    "MENU_ACTIONS",
    "MODULE_TAG",
    "SUBMENU_TITLE",
    "entry_fields",
    "entry_instructions",
    "query_llm_analysis",
    "run_action",
]