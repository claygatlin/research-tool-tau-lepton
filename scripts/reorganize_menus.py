#!/usr/bin/env python3
"""
One-shot project layout migration: menu folders + tav_shared + import rewrites.

Run from project root:
  ./venv/bin/python scripts/reorganize_menus.py
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# =============================================================================
# BLOCK: File moves (source relative to ROOT → destination relative to ROOT)
# =============================================================================

MOVES: list[tuple[str, str]] = [
    # --- tav_shared (global infrastructure) ---
    ("llm_analysis.py", "tav_shared/llm_analysis.py"),
    ("run_output.py", "tav_shared/run_output.py"),
    ("batch_ledger.py", "tav_shared/batch_ledger.py"),
    ("dataset_registry.py", "tav_shared/dataset_registry.py"),
    ("dataset_menu.py", "tav_shared/dataset_menu.py"),
    ("dataset_ledger.py", "tav_shared/dataset_ledger.py"),
    ("hepdata_engine.py", "tav_shared/hepdata_engine.py"),
    ("mcmc_engine.py", "tav_shared/mcmc_engine.py"),
    ("require_topology.py", "tav_shared/require_topology.py"),
    ("tav_project_paths.py", "tav_shared/tav_project_paths.py"),
    ("n_selector_registry.py", "tav_shared/n_selector_registry.py"),
    ("test_set_reset.py", "tav_shared/test_set_reset.py"),
    ("data_provenance.py", "tav_shared/data_provenance.py"),
    ("cosmic_web_style.py", "tav_shared/cosmic_web_style.py"),
    # --- Astronomical ---
    ("sparc_extension.py", "menus/astronomical/sparc/extension.py"),
    ("sparc_analyzer.py", "menus/astronomical/sparc/analyzer.py"),
    ("sparc_fetcher.py", "menus/astronomical/sparc/fetcher.py"),
    ("sparc_superblock.py", "menus/astronomical/sparc/superblock.py"),
    ("frb_web_extension.py", "menus/astronomical/frb/extension.py"),
    ("frb_cosmic_web_tav.py", "menus/astronomical/frb/cosmic_web_tav.py"),
    ("frb_fetcher.py", "menus/astronomical/frb/fetcher.py"),
    ("frb_dispersion_test.py", "menus/astronomical/frb/dispersion_test.py"),
    ("tav_resonance_extension.py", "menus/astronomical/tav_resonance/extension.py"),
    ("tau_sb_desi_extension.py", "menus/astronomical/desi/extension.py"),
    ("tau_sb_desi_scanner.py", "menus/astronomical/desi/scanner.py"),
    ("tau_sb_desi_stats.py", "menus/astronomical/desi/stats.py"),
    ("tau_sb_desi_production.py", "menus/astronomical/desi/production.py"),
    ("tau_sb_desi_analysis_v2.py", "menus/astronomical/desi/analysis.py"),
    ("tau_sb_json.py", "menus/astronomical/desi/json_util.py"),
    ("desi_dashboard.py", "menus/astronomical/desi/dashboard.py"),
    # --- Prime past / empirical ---
    ("prime_past_extension.py", "menus/prime_past/extension.py"),
    ("tav_superblock_prime_past_harmonic.py", "menus/prime_past/harmonic.py"),
    ("empirical_tests_extension.py", "menus/empirical_tests/extension.py"),
    ("superblock_empirical_tests.py", "menus/empirical_tests/tests.py"),
    # --- TSB Research Engine ---
    ("tsb_research_extension.py", "menus/tsb_research/extension.py"),
    ("tsb_research_core.py", "menus/tsb_research/core.py"),
    ("mock_generator.py", "menus/tsb_research/mock_generator.py"),
    # --- Planck / Integrator ---
    ("planck_cmb_extension.py", "menus/planck_cmb/extension.py"),
    ("planck_cmb_tav.py", "menus/planck_cmb/tav.py"),
    ("integrator_extension.py", "menus/integrator/extension.py"),
    ("tav_data_integrator.py", "menus/integrator/integrator.py"),
    ("integrator_fetcher.py", "menus/integrator/fetcher.py"),
    # --- Particle ---
    ("lhcb_echo_extension.py", "menus/particle/lhcb/extension.py"),
    ("lhcb_tav_echo.py", "menus/particle/lhcb/echo.py"),
    ("tsb_casimir_extension.py", "menus/particle/casimir/extension.py"),
    ("tsb_casimir_scanner.py", "menus/particle/casimir/scanner.py"),
]

# =============================================================================
# BLOCK: Import rewrite table (old → new)
# =============================================================================

IMPORT_REWRITES: list[tuple[str, str]] = [
    ("from tav_shared.llm_analysis", "from tav_shared.llm_analysis"),
    ("import tav_shared.llm_analysis as llm_analysis", "import tav_shared.llm_analysis as llm_analysis"),
    ("from tav_shared.run_output", "from tav_shared.run_output"),
    ("import tav_shared.run_output as run_output", "import tav_shared.run_output as run_output"),
    ("from tav_shared.batch_ledger", "from tav_shared.batch_ledger"),
    ("from tav_shared.dataset_registry", "from tav_shared.dataset_registry"),
    ("from tav_shared.dataset_menu", "from tav_shared.dataset_menu"),
    ("from tav_shared.dataset_ledger", "from tav_shared.dataset_ledger"),
    ("from tav_shared.hepdata_engine", "from tav_shared.hepdata_engine"),
    ("from tav_shared.mcmc_engine", "from tav_shared.mcmc_engine"),
    ("from tav_shared.require_topology", "from tav_shared.require_topology"),
    ("from tav_shared.tav_project_paths", "from tav_shared.tav_project_paths"),
    ("from tav_shared.n_selector_registry", "from tav_shared.n_selector_registry"),
    ("from tav_shared.test_set_reset", "from tav_shared.test_set_reset"),
    ("from tav_shared.data_provenance", "from tav_shared.data_provenance"),
    ("from tav_shared.cosmic_web_style", "from tav_shared.cosmic_web_style"),
    ("import menus.astronomical.sparc.extension as sparc_extension", "import menus.astronomical.sparc.extension as sparc_extension"),
    ("from menus.astronomical.sparc import extension as sparc_extension", "from menus.astronomical.sparc import extension as sparc_extension"),
    ("from menus.astronomical.sparc.analyzer", "from menus.astronomical.sparc.analyzer"),
    ("from menus.astronomical.sparc.fetcher", "from menus.astronomical.sparc.fetcher"),
    ("from menus.astronomical.sparc.superblock", "from menus.astronomical.sparc.superblock"),
    ("import menus.astronomical.frb.extension as frb_web_extension", "import menus.astronomical.frb.extension as frb_web_extension"),
    ("from menus.astronomical.frb import extension as frb_web_extension", "from menus.astronomical.frb import extension as frb_web_extension"),
    ("from menus.astronomical.frb.cosmic_web_tav", "from menus.astronomical.frb.cosmic_web_tav"),
    ("from menus.astronomical.frb.fetcher", "from menus.astronomical.frb.fetcher"),
    ("from menus.astronomical.frb.dispersion_test", "from menus.astronomical.frb.dispersion_test"),
    ("import menus.astronomical.tav_resonance.extension as tav_resonance_extension", "import menus.astronomical.tav_resonance.extension as tav_resonance_extension"),
    ("from menus.astronomical.tav_resonance import extension as tav_resonance_extension", "from menus.astronomical.tav_resonance import extension as tav_resonance_extension"),
    ("import menus.astronomical.desi.extension as tau_sb_desi_extension", "import menus.astronomical.desi.extension as tau_sb_desi_extension"),
    ("from menus.astronomical.desi import extension as tau_sb_desi_extension", "from menus.astronomical.desi import extension as tau_sb_desi_extension"),
    ("from menus.astronomical.desi.scanner", "from menus.astronomical.desi.scanner"),
    ("from menus.astronomical.desi.stats", "from menus.astronomical.desi.stats"),
    ("from menus.astronomical.desi.production", "from menus.astronomical.desi.production"),
    ("from menus.astronomical.desi.analysis", "from menus.astronomical.desi.analysis"),
    ("from menus.astronomical.desi.json_util", "from menus.astronomical.desi.json_util"),
    ("from menus.astronomical.desi.dashboard", "from menus.astronomical.desi.dashboard"),
    ("import menus.prime_past.extension as prime_past_extension", "import menus.prime_past.extension as prime_past_extension"),
    ("from menus.prime_past import extension as prime_past_extension", "from menus.prime_past import extension as prime_past_extension"),
    ("from menus.prime_past.harmonic", "from menus.prime_past.harmonic"),
    ("import menus.empirical_tests.extension as empirical_tests_extension", "import menus.empirical_tests.extension as empirical_tests_extension"),
    ("from menus.empirical_tests import extension as empirical_tests_extension", "from menus.empirical_tests import extension as empirical_tests_extension"),
    ("from menus.empirical_tests.tests", "from menus.empirical_tests.tests"),
    ("import menus.tsb_research.extension as tsb_research_extension", "import menus.tsb_research.extension as tsb_research_extension"),
    ("from menus.tsb_research import extension as tsb_research_extension", "from menus.tsb_research import extension as tsb_research_extension"),
    ("from menus.tsb_research.core", "from menus.tsb_research.core"),
    ("from menus.tsb_research.mock_generator", "from menus.tsb_research.mock_generator"),
    ("import menus.planck_cmb.extension as planck_cmb_extension", "import menus.planck_cmb.extension as planck_cmb_extension"),
    ("from menus.planck_cmb import extension as planck_cmb_extension", "from menus.planck_cmb import extension as planck_cmb_extension"),
    ("from menus.planck_cmb.tav", "from menus.planck_cmb.tav"),
    ("import menus.integrator.extension as integrator_extension", "import menus.integrator.extension as integrator_extension"),
    ("from menus.integrator import extension as integrator_extension", "from menus.integrator import extension as integrator_extension"),
    ("from menus.integrator.integrator", "from menus.integrator.integrator"),
    ("from menus.integrator.fetcher", "from menus.integrator.fetcher"),
    ("import menus.particle.lhcb.extension as lhcb_echo_extension", "import menus.particle.lhcb.extension as lhcb_echo_extension"),
    ("from menus.particle.lhcb import extension as lhcb_echo_extension", "from menus.particle.lhcb import extension as lhcb_echo_extension"),
    ("from menus.particle.lhcb.echo", "from menus.particle.lhcb.echo"),
    ("import menus.particle.casimir.extension as tsb_casimir_extension", "import menus.particle.casimir.extension as tsb_casimir_extension"),
    ("from menus.particle.casimir import extension as tsb_casimir_extension", "from menus.particle.casimir import extension as tsb_casimir_extension"),
    ("from menus.particle.casimir.scanner", "from menus.particle.casimir.scanner"),
]

SHIM_TEMPLATE = '''"""Compatibility shim — import from new menu package location."""
from {target} import *  # noqa: F403
'''

SHIMS: dict[str, str] = {
    "sparc_extension.py": "menus.astronomical.sparc.extension",
    "frb_web_extension.py": "menus.astronomical.frb.extension",
    "tav_resonance_extension.py": "menus.astronomical.tav_resonance.extension",
    "tau_sb_desi_extension.py": "menus.astronomical.desi.extension",
    "prime_past_extension.py": "menus.prime_past.extension",
    "empirical_tests_extension.py": "menus.empirical_tests.extension",
    "tsb_research_extension.py": "menus.tsb_research.extension",
    "planck_cmb_extension.py": "menus.planck_cmb.extension",
    "integrator_extension.py": "menus.integrator.extension",
    "lhcb_echo_extension.py": "menus.particle.lhcb.extension",
    "tsb_casimir_extension.py": "menus.particle.casimir.extension",
    "mock_generator.py": "menus.tsb_research.mock_generator",
    "desi_dashboard.py": "menus.astronomical.desi.dashboard",
}


def ensure_packages() -> None:
    """Create every menu folder and __init__.py on the fly."""
    packages = set()
    for _, dst in MOVES:
        packages.add(str(Path(dst).parent))
    packages.update(
        [
            "menus",
            "menus/astronomical",
            "menus/gravitic/ligo",
            "menus/gravitic/lisa",
            "menus/particle/hepdata",
            "menus/particle/cms",
            "menus/particle/atlas",
            "menus/particle/alice",
            "menus/particle/belle",
            "menus/astronomical/halogas",
            "tav_shared",
        ]
    )
    for pkg in sorted(packages):
        path = ROOT / pkg
        path.mkdir(parents=True, exist_ok=True)
        init = path / "__init__.py"
        if not init.exists():
            init.write_text(
                f'"""Package: {pkg.replace("/", ".")}"""\n',
                encoding="utf-8",
            )


def move_files() -> None:
    for src_rel, dst_rel in MOVES:
        src = ROOT / src_rel
        dst = ROOT / dst_rel
        if not src.exists():
            print(f"SKIP missing: {src_rel}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            print(f"SKIP exists: {dst_rel}")
            continue
        shutil.move(str(src), str(dst))
        print(f"MOVED {src_rel} -> {dst_rel}")


def write_shims() -> None:
    for shim_name, target in SHIMS.items():
        path = ROOT / shim_name
        path.write_text(SHIM_TEMPLATE.format(target=target), encoding="utf-8")
        print(f"SHIM {shim_name}")


def rewrite_imports() -> None:
    skip_dirs = {"venv", "__pycache__", ".git", "artifacts", "finished", "finished2"}
    for py in ROOT.rglob("*.py"):
        if any(part in skip_dirs for part in py.parts):
            continue
        text = py.read_text(encoding="utf-8")
        original = text
        for old, new in IMPORT_REWRITES:
            text = text.replace(old, new)
        if text != original:
            py.write_text(text, encoding="utf-8")
            print(f"REWRITE {py.relative_to(ROOT)}")


def main() -> None:
    ensure_packages()
    move_files()
    write_shims()
    rewrite_imports()
    print("Done.")


if __name__ == "__main__":
    main()