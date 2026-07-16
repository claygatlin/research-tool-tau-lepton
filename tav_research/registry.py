"""
Module registry — loads every ``*_extension`` submenu and dispatches UI/run hooks.

Each submenu module is imported here once.  Optional imports are guarded so a
missing dependency (healpy, tav-resonance, …) does not break the whole menu.
"""

from __future__ import annotations

from typing import Any

import menus.empirical_tests.extension as empirical_tests_extension
import menus.particle.lhcb.extension as lhcb_echo_extension
import menus.prime_past.extension as prime_past_extension
import menus.prime_past.bbn_menu_extension as tav_bbn_extension
import menus.remote_ai.extension as remote_ai_extension
import menus.astronomical.sparc.extension as sparc_extension

from tav_research.base_module import (
    call_pre_form_hook,
    resolve_entry_fields,
    resolve_entry_instructions,
)

# =============================================================================
# BLOCK: Optional extension imports (graceful degradation)
# =============================================================================

try:
    import menus.planck_cmb.extension as planck_cmb_extension
except ImportError as _planck_import_error:
    planck_cmb_extension = None
    PLANCK_IMPORT_ERROR = _planck_import_error
else:
    PLANCK_IMPORT_ERROR = None

try:
    import menus.astronomical.frb.extension as frb_web_extension
except ImportError as _frb_import_error:
    frb_web_extension = None
    FRB_IMPORT_ERROR = _frb_import_error
else:
    FRB_IMPORT_ERROR = None

try:
    import menus.integrator.extension as integrator_extension
except ImportError as _integrator_import_error:
    integrator_extension = None
    INTEGRATOR_IMPORT_ERROR = _integrator_import_error
else:
    INTEGRATOR_IMPORT_ERROR = None

try:
    import menus.astronomical.desi.extension as tau_sb_desi_extension
except ImportError as _desi_import_error:
    tau_sb_desi_extension = None
    DESI_IMPORT_ERROR = _desi_import_error
else:
    DESI_IMPORT_ERROR = None

try:
    import menus.astronomical.tav_resonance.extension as tav_resonance_extension
except ImportError as _tav_resonance_import_error:
    tav_resonance_extension = None
    TAV_RESONANCE_IMPORT_ERROR = _tav_resonance_import_error
else:
    TAV_RESONANCE_IMPORT_ERROR = None

try:
    import menus.astronomical.berard.extension as berard_framework_extension
except ImportError as _berard_import_error:
    berard_framework_extension = None
    BERARD_IMPORT_ERROR = _berard_import_error
else:
    BERARD_IMPORT_ERROR = None

try:
    import menus.particle.casimir.extension as tsb_casimir_extension
except ImportError as _casimir_import_error:
    tsb_casimir_extension = None
    CASIMIR_IMPORT_ERROR = _casimir_import_error
else:
    CASIMIR_IMPORT_ERROR = None

try:
    import menus.particle.rgc.extension as rgc_mock_extension
except ImportError as _rgc_import_error:
    rgc_mock_extension = None
    RGC_IMPORT_ERROR = _rgc_import_error
else:
    RGC_IMPORT_ERROR = None

try:
    import menus.tsb_research.extension as tsb_research_extension
except ImportError as _tsb_research_import_error:
    tsb_research_extension = None
    TSB_RESEARCH_IMPORT_ERROR = _tsb_research_import_error
else:
    TSB_RESEARCH_IMPORT_ERROR = None

try:
    import menus.gravitic.ligo.extension as ligo_gwosc_extension
except ImportError as _ligo_import_error:
    ligo_gwosc_extension = None
    LIGO_IMPORT_ERROR = _ligo_import_error
else:
    LIGO_IMPORT_ERROR = None

try:
    import menus.gravitic.lisa.extension as lisa_pre_runs_extension
except ImportError as _lisa_import_error:
    lisa_pre_runs_extension = None
    LISA_IMPORT_ERROR = _lisa_import_error
else:
    LISA_IMPORT_ERROR = None

try:
    import menus.particle.cern.extension as cern_opendata_extension
except ImportError as _cern_import_error:
    cern_opendata_extension = None
    CERN_IMPORT_ERROR = _cern_import_error
else:
    CERN_IMPORT_ERROR = None

try:
    import menus.particle.tau_lepton.extension as tau_lepton_extension
except ImportError as _tau_lepton_import_error:
    tau_lepton_extension = None
    TAU_LEPTON_IMPORT_ERROR = _tau_lepton_import_error
else:
    TAU_LEPTON_IMPORT_ERROR = None

# =============================================================================
# BLOCK: MODULE_EXTENSIONS map (tag → extension module)
# =============================================================================

MODULE_EXTENSIONS: dict[str, Any] = {
    prime_past_extension.MODULE_TAG: prime_past_extension,
    empirical_tests_extension.MODULE_TAG: empirical_tests_extension,
    sparc_extension.MODULE_TAG: sparc_extension,
    lhcb_echo_extension.MODULE_TAG: lhcb_echo_extension,
    tav_bbn_extension.MODULE_TAG: tav_bbn_extension,
    remote_ai_extension.MODULE_TAG: remote_ai_extension,
}

if planck_cmb_extension is not None:
    MODULE_EXTENSIONS[planck_cmb_extension.MODULE_TAG] = planck_cmb_extension
if frb_web_extension is not None:
    MODULE_EXTENSIONS[frb_web_extension.MODULE_TAG] = frb_web_extension
if integrator_extension is not None:
    MODULE_EXTENSIONS[integrator_extension.MODULE_TAG] = integrator_extension
if tau_sb_desi_extension is not None:
    MODULE_EXTENSIONS[tau_sb_desi_extension.MODULE_TAG] = tau_sb_desi_extension
if tav_resonance_extension is not None:
    MODULE_EXTENSIONS[tav_resonance_extension.MODULE_TAG] = tav_resonance_extension
if berard_framework_extension is not None:
    MODULE_EXTENSIONS[berard_framework_extension.MODULE_TAG] = berard_framework_extension
if tsb_casimir_extension is not None:
    MODULE_EXTENSIONS[tsb_casimir_extension.MODULE_TAG] = tsb_casimir_extension
if tsb_research_extension is not None:
    MODULE_EXTENSIONS[tsb_research_extension.MODULE_TAG] = tsb_research_extension
if rgc_mock_extension is not None:
    MODULE_EXTENSIONS[rgc_mock_extension.MODULE_TAG] = rgc_mock_extension
if ligo_gwosc_extension is not None:
    MODULE_EXTENSIONS[ligo_gwosc_extension.MODULE_TAG] = ligo_gwosc_extension
if lisa_pre_runs_extension is not None:
    MODULE_EXTENSIONS[lisa_pre_runs_extension.MODULE_TAG] = lisa_pre_runs_extension
if cern_opendata_extension is not None:
    MODULE_EXTENSIONS[cern_opendata_extension.MODULE_TAG] = cern_opendata_extension
if tau_lepton_extension is not None:
    MODULE_EXTENSIONS[tau_lepton_extension.MODULE_TAG] = tau_lepton_extension

# =============================================================================
# BLOCK: Dispatch helpers
# =============================================================================


def get_extension(module_tag: str) -> Any | None:
    """Return the extension module for ``module_tag``, or None."""
    return MODULE_EXTENSIONS.get(module_tag)


def module_entry_fields(module_tag: str, action: str) -> list[dict]:
    """Entry-form fields for a module action (delegates to the extension)."""
    ext = get_extension(module_tag)
    if ext is None:
        return []
    return resolve_entry_fields(ext, action)


def module_entry_instructions(module_tag: str, action: str) -> list[str]:
    """Help bullets for a module action (delegates to the extension)."""
    ext = get_extension(module_tag)
    if ext is None:
        return ["Follow field hints; leave optional entries blank to use engine defaults."]
    return resolve_entry_instructions(ext, action)


def handle_module_pre_form(
    module_tag: str,
    stdscr: Any,
    selection: str,
    params: dict[str, str],
) -> str | None:
    """Run a submenu's optional pre-form hook (n selector, warnings, …)."""
    ext = get_extension(module_tag)
    if ext is None:
        return None
    return call_pre_form_hook(ext, stdscr, selection, params)
