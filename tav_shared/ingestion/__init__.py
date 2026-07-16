"""
Automated ingestion validation for Tau-Superblock data pipelines.

Schema enforcement (Pydantic), physical-range filtering, and canonical
normalization before data reaches solvers or log_likelihood functions.
"""

from tav_shared.ingestion.pipeline import (
    IngestionValidationReport,
    validate_and_normalize_bbn_payload,
    validate_and_normalize_cms_dimuon_batch,
    validate_fractal_tau_for_likelihood,
    validate_generic_records,
)
from tav_shared.ingestion.data_manager import (
    DATA_MANAGER_LEGACY_PARAM_KEY,
    DATA_MANAGER_PARAM_KEY,
    DataManager,
    get_runtime_data_manager,
    inject_data_manager,
)

__all__ = [
    "IngestionValidationReport",
    "validate_and_normalize_bbn_payload",
    "validate_and_normalize_cms_dimuon_batch",
    "validate_fractal_tau_for_likelihood",
    "validate_generic_records",
    "DataManager",
    "DATA_MANAGER_PARAM_KEY",
    "DATA_MANAGER_LEGACY_PARAM_KEY",
    "get_runtime_data_manager",
    "inject_data_manager",
]