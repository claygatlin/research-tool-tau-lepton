"""Backward-compatible re-export of the unified SPARC extension."""

from tav_shared.llm_analysis import analyze_run_output as query_llm_analysis
from menus.astronomical.sparc import extension as sparc_extension import MENU_ACTIONS, MODULE_TAG, SUBMENU_TITLE, run_action

__all__ = ["MODULE_TAG", "SUBMENU_TITLE", "MENU_ACTIONS", "run_action", "query_llm_analysis"]