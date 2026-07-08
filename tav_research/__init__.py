"""
Tav-Superblock Research UI package.

Each submenu lives in its own extension module under ``menus/``.
This package holds shared curses UI, the module registry, and the main loop.

Layout
------
- ``base_module.py``   — documented contract every submenu module should follow
- ``registry.py``      — discover extensions, dispatch entry forms / runs
- ``curses_shell.py``  — terminal UI primitives (menus, forms, n-picker)
- ``n_selector.py``    — sample-size selector policy (FRB, Tav, generic)
- ``menu_tree.py``     — Primary Matrix domain / submenu tree
- ``data_pull.py``     — router for legacy HEPData / ALICE / CMS repository pulls
- ``dataset_manager.py`` — Dataset Manager submenu
- ``main_loop.py``     — curses navigation loop
- ``runner.py``        — post-menu execution + LLM review wrapper
"""

from __future__ import annotations

__all__ = ["main"]


def main() -> None:
    """Lazy entry point so ``import tav_research.curses_shell`` does not load runner."""
    from tav_research.runner import main as _main

    return _main()