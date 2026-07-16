"""Compatibility shim for legacy DataManager imports."""

from tav_shared.ingestion.data_manager import (
    DATA_MANAGER_LEGACY_PARAM_KEY,
    DATA_MANAGER_PARAM_KEY,
    DataManager,
    get_runtime_data_manager,
    inject_data_manager,
)

__all__ = [
    "DataManager",
    "DATA_MANAGER_PARAM_KEY",
    "DATA_MANAGER_LEGACY_PARAM_KEY",
    "get_runtime_data_manager",
    "inject_data_manager",
]
