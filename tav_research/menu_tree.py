"""
Primary Matrix menu tree — domains, repository lists, and submenu actions.

Each submenu title maps to ``MENU_ACTIONS`` on its ``*_extension.py`` module.
"""

from __future__ import annotations

import menus.empirical_tests.extension as empirical_tests_extension
import menus.particle.lhcb.extension as lhcb_echo_extension
import menus.prime_past.extension as prime_past_extension
import menus.prime_past.bbn_menu_extension as tav_bbn_extension
import menus.remote_ai.extension as remote_ai_extension
import menus.astronomical.sparc.extension as sparc_extension
from tav_shared.test_set_reset import append_reset_option

from tav_research.dataset_manager import (
    DATASET_FETCH_ACTION,
    DATASET_MANAGER_TITLE,
    DATASET_RUN_ACTION,
)
from tav_research.gateway_launcher import GATEWAY_GUI_MENU_LABEL
from tav_research.registry import MODULE_EXTENSIONS
from tav_research import registry


def build_menu_tree():
    """Return (domains, repos, submenus, module_action_map) for the curses UI."""
    domains = [
        GATEWAY_GUI_MENU_LABEL,
        "Astronomical (Conformal Shadows)",
        "Gravitic (Planck-Kerr Seeds)",
        "Particle (Dynamic Refresh)",
        "Prime Past Harmonic (Down Quark / Isospin)",
        "Superblock Empirical Tests",
        "Remote AI Processing",
    ]
    if registry.tsb_research_extension is not None:
        domains.append(registry.tsb_research_extension.DOMAIN_TITLE)
    if registry.planck_cmb_extension is not None:
        domains.append(registry.planck_cmb_extension.SUBMENU_TITLE)
    if registry.integrator_extension is not None:
        domains.append(registry.integrator_extension.SUBMENU_TITLE)
    domains.append(DATASET_MANAGER_TITLE)
    domains.append("Exit")

    submenus = {
        tav_bbn_extension.SUBMENU_TITLE: append_reset_option(list(tav_bbn_extension.MENU_ACTIONS)),
        sparc_extension.SUBMENU_TITLE: append_reset_option(list(sparc_extension.MENU_ACTIONS)),
        lhcb_echo_extension.SUBMENU_TITLE: append_reset_option(list(lhcb_echo_extension.MENU_ACTIONS)),
    }
    if registry.frb_web_extension is not None:
        submenus[registry.frb_web_extension.SUBMENU_TITLE] = append_reset_option(
            list(registry.frb_web_extension.MENU_ACTIONS)
        )
    if registry.tau_sb_desi_extension is not None:
        submenus[registry.tau_sb_desi_extension.SUBMENU_TITLE] = append_reset_option(
            list(registry.tau_sb_desi_extension.MENU_ACTIONS)
        )
    if registry.tav_resonance_extension is not None:
        submenus[registry.tav_resonance_extension.SUBMENU_TITLE] = append_reset_option(
            list(registry.tav_resonance_extension.MENU_ACTIONS)
        )
    if registry.tsb_casimir_extension is not None:
        submenus[registry.tsb_casimir_extension.SUBMENU_TITLE] = append_reset_option(
            list(registry.tsb_casimir_extension.MENU_ACTIONS)
        )
    if registry.rgc_mock_extension is not None:
        submenus[registry.rgc_mock_extension.SUBMENU_TITLE] = append_reset_option(
            list(registry.rgc_mock_extension.MENU_ACTIONS)
        )
    if registry.integrator_extension is not None:
        submenus[registry.integrator_extension.SUBMENU_TITLE] = append_reset_option(
            list(registry.integrator_extension.MENU_ACTIONS)
        )
    if registry.ligo_gwosc_extension is not None:
        submenus[registry.ligo_gwosc_extension.SUBMENU_TITLE] = append_reset_option(
            list(registry.ligo_gwosc_extension.MENU_ACTIONS)
        )
    if registry.lisa_pre_runs_extension is not None:
        submenus[registry.lisa_pre_runs_extension.SUBMENU_TITLE] = append_reset_option(
            list(registry.lisa_pre_runs_extension.MENU_ACTIONS)
        )
    if registry.cern_opendata_extension is not None:
        submenus[registry.cern_opendata_extension.SUBMENU_TITLE] = append_reset_option(
            list(registry.cern_opendata_extension.MENU_ACTIONS)
        )

    repos = {
        "Astronomical (Conformal Shadows)": [
            sparc_extension.SUBMENU_TITLE,
            *(
                [registry.frb_web_extension.SUBMENU_TITLE]
                if registry.frb_web_extension is not None
                else []
            ),
            *(
                [registry.tav_resonance_extension.SUBMENU_TITLE]
                if registry.tav_resonance_extension is not None
                else []
            ),
            *(
                [registry.tau_sb_desi_extension.SUBMENU_TITLE]
                if registry.tau_sb_desi_extension is not None
                else []
            ),
            "HALOGAS (HI Data Cubes)",
        ],
        "Gravitic (Planck-Kerr Seeds)": [
            *(
                [registry.ligo_gwosc_extension.SUBMENU_TITLE]
                if registry.ligo_gwosc_extension is not None
                else ["LIGO GWOSC (Strain Data)"]
            ),
            *(
                [registry.lisa_pre_runs_extension.SUBMENU_TITLE]
                if registry.lisa_pre_runs_extension is not None
                else ["LISA Pre-runs"]
            ),
        ],
        "Particle (Dynamic Refresh)": [
            lhcb_echo_extension.SUBMENU_TITLE,
            *(
                [registry.cern_opendata_extension.SUBMENU_TITLE]
                if registry.cern_opendata_extension is not None
                else []
            ),
            *(
                [registry.rgc_mock_extension.SUBMENU_TITLE]
                if registry.rgc_mock_extension is not None
                else []
            ),
            *(
                [registry.tsb_casimir_extension.SUBMENU_TITLE]
                if registry.tsb_casimir_extension is not None
                else []
            ),
            "HEPData (Numerical Correlation)",
            "CMS Open Data (NanoAOD / Electrons)",
            "ATLAS Open Data (Lepton/Photon Tracks)",
            "ALICE Heavy-Ion (Thermal Cooling)",
            "Belle II (Meson Oscillation)",
        ],
        "Prime Past Harmonic (Down Quark / Isospin)": append_reset_option(
            [tav_bbn_extension.SUBMENU_TITLE, *prime_past_extension.MENU_ACTIONS]
        ),
        "Superblock Empirical Tests": append_reset_option(
            list(empirical_tests_extension.MENU_ACTIONS)
        ),
        "Remote AI Processing": append_reset_option(list(remote_ai_extension.MENU_ACTIONS)),
    }
    if registry.tsb_research_extension is not None:
        repos[registry.tsb_research_extension.DOMAIN_TITLE] = append_reset_option(
            list(registry.tsb_research_extension.REPO_ACTIONS)
        )
    if registry.planck_cmb_extension is not None:
        repos[registry.planck_cmb_extension.SUBMENU_TITLE] = append_reset_option(
            list(registry.planck_cmb_extension.MENU_ACTIONS)
        )
    if registry.integrator_extension is not None:
        repos[registry.integrator_extension.SUBMENU_TITLE] = append_reset_option(
            list(registry.integrator_extension.MENU_ACTIONS)
        )
    repos[DATASET_MANAGER_TITLE] = [DATASET_FETCH_ACTION, DATASET_RUN_ACTION]

    module_action_map: dict[str, str] = {}
    for tag, extension in MODULE_EXTENSIONS.items():
        for action in extension.MENU_ACTIONS:
            module_action_map[action] = tag
    return domains, repos, submenus, module_action_map