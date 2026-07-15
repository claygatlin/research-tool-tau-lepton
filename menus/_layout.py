"""
Menu folder layout — creates domain/submenu directories on demand.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =============================================================================
# BLOCK: Domain → submenu folder map (matches Primary Matrix menus)
# =============================================================================

MENU_LAYOUT: dict[str, list[str]] = {
    "astronomical": ["sparc", "frb", "tav_resonance", "desi", "halogas"],
    "gravitic": ["ligo", "lisa"],
    "particle": ["lhcb", "rgc", "casimir", "hepdata", "cms", "atlas", "alice", "belle"],
    "prime_past": [],
    "empirical_tests": [],
    "tsb_research": [],
    "planck_cmb": [],
    "integrator": [],
}


def ensure_menu_folders(root: Path | None = None) -> None:
    """Create every menu package folder and ``__init__.py`` if missing."""
    root = root or PROJECT_ROOT
    for domain, submenus in MENU_LAYOUT.items():
        domain_path = root / "menus" / domain
        domain_path.mkdir(parents=True, exist_ok=True)
        _touch_init(domain_path)
        for name in submenus:
            sub_path = domain_path / name
            sub_path.mkdir(parents=True, exist_ok=True)
            _touch_init(sub_path)
    for top in ("menus", "tav_shared", "tav_research"):
        path = root / top
        path.mkdir(parents=True, exist_ok=True)
        _touch_init(path)


def _touch_init(path: Path) -> None:
    init = path / "__init__.py"
    if not init.exists():
        rel = path.relative_to(PROJECT_ROOT / "menus") if "menus" in path.parts else path.name
        init.write_text(f'"""Menu package: {rel}"""\n', encoding="utf-8")


def submenu_path(domain: str, name: str) -> Path:
    """Absolute path to a submenu folder."""
    return PROJECT_ROOT / "menus" / domain / name