"""Top-level import shim for ``from tav_superblock_cms_muon_analyzer import ...``."""

from menus.particle.cern.tav_superblock_cms_muon_analyzer import (  # noqa: F401
    M0_MEV_ANCHOR,
    TAV_HARMONIC_RATIO,
    TAV_PERIOD,
    _exponential_detrend,
    _fit_7periodic,
    _pt_spectrum_7fold,
    tav_7fold_muon_analysis,
)
from menus.particle.cern.analyzer import (  # noqa: F401
    PT_REDUCTION_CHOICES,
    PT_REDUCTION_FLATTEN,
    PT_REDUCTION_LEADING,
    analyze_cms_nanoaod_full,
    accumulate_weighted_validation_observables,
    extract_dimuon_kinematics_from_nanoaod,
    extract_photon_pt_histogram_from_nanoaod,
    extract_pileup_ntrueint_histogram,
    read_electron_pt_from_tree,
    reduce_jagged_pt_branch,
    scan_cms_nanoaod_chunked,
)
from menus.particle.cern.mod7_phase import (  # noqa: F401
    mod7_fractions_from_histogram,
    mod7_phase_residues,
    mod7_phase_slot_masks,
    weighted_mod7_histogram,
)
from tav_shared.chunked_results import (  # noqa: F401
    load_tav_results_chunked,
    save_tav_results_chunked,
)

__all__ = [
    "M0_MEV_ANCHOR",
    "TAV_HARMONIC_RATIO",
    "TAV_PERIOD",
    "_exponential_detrend",
    "_fit_7periodic",
    "_pt_spectrum_7fold",
    "tav_7fold_muon_analysis",
    "PT_REDUCTION_CHOICES",
    "PT_REDUCTION_FLATTEN",
    "PT_REDUCTION_LEADING",
    "accumulate_weighted_validation_observables",
    "extract_dimuon_kinematics_from_nanoaod",
    "extract_photon_pt_histogram_from_nanoaod",
    "extract_pileup_ntrueint_histogram",
    "read_electron_pt_from_tree",
    "reduce_jagged_pt_branch",
    "scan_cms_nanoaod_chunked",
    "mod7_fractions_from_histogram",
    "mod7_phase_residues",
    "mod7_phase_slot_masks",
    "weighted_mod7_histogram",
    "analyze_cms_nanoaod_full",
    "save_tav_results_chunked",
    "load_tav_results_chunked",
]