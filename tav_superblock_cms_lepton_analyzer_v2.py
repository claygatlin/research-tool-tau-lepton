"""
Top-level import shim — matches external scripts and MC validation suite.

::

    from tav_superblock_cms_lepton_analyzer_v2 import (
        tav_analyze_leptons_from_file,
        tav_compare_data_mc,
        save_tav_results_chunked,
    )
"""

from menus.particle.cern.tav_superblock_cms_lepton_analyzer_v2 import (  # noqa: F401
    discover_available_lepton_branches,
    tav_analyze_leptons_from_file,
    tav_compare_data_mc,
)
from tav_shared.chunked_results import (  # noqa: F401
    DEFAULT_MAX_CHUNK_MB,
    LARGE_DATASET_EVENT_THRESHOLD,
    load_tav_results_chunked,
    save_tav_results_chunked,
    should_chunk_tav_results,
)

__all__ = [
    "DEFAULT_MAX_CHUNK_MB",
    "LARGE_DATASET_EVENT_THRESHOLD",
    "discover_available_lepton_branches",
    "load_tav_results_chunked",
    "save_tav_results_chunked",
    "should_chunk_tav_results",
    "tav_analyze_leptons_from_file",
    "tav_compare_data_mc",
]