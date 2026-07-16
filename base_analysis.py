"""Shared analysis base classes for Tau-Superblock workflows."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from data_manager import DataManager, get_runtime_data_manager

logger = logging.getLogger("ResearchTool")


class BaseAnalysis:
    """Base class that centralizes data fetch + resonance prechecks."""

    def __init__(self, use_mock: bool = False, data_manager: DataManager | None = None):
        self.data_manager = data_manager or get_runtime_data_manager(
            {"use_mock": use_mock}
        )

    def run_resonance_test(self, tracer_name: str, source_path: str) -> Any:
        data = self.data_manager.fetch(tracer_name, source_path)
        n_rows = self._effective_sample_size(data)

        # Enforce minimum sample size before periodogram-style checks.
        if n_rows < 8:
            logger.error("Test %s failed: n=%s < 8", tracer_name, n_rows)
            raise ValueError("Statistical power insufficient for periodogram.")

        return self._compute_tav_resonance(data)

    @staticmethod
    def _effective_sample_size(data: Any) -> int:
        if isinstance(data, dict):
            for value in data.values():
                try:
                    return int(len(value))
                except TypeError:
                    continue
            return 0
        try:
            arr = np.asarray(data)
            if arr.ndim == 0:
                return int(arr.size)
            return int(arr.shape[0])
        except Exception:
            try:
                return int(len(data))
            except Exception:
                return 0

    def _compute_tav_resonance(self, data: Any) -> Any:
        raise NotImplementedError("Subclasses must implement _compute_tav_resonance")
