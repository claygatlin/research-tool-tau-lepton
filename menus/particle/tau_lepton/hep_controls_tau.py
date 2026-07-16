#!/usr/bin/env python3
"""
HEP-style controls for Tau Lepton 1/7-mode searches.
Mirrors the hardened CMS dimuon controls (trigger-aware nulls,
engine scores, fine binning around the 313.1 MeV floor).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class EngineScore:
    """Internal framework score — NEVER to be reported as HEP significance."""
    engine_score_sigma: float
    engine_score_sigma_raw: float
    interpretation_label: str = "internal_engine_score_not_hep_significance"


class TauHEPControls:
    """
    Strict controls for Tau lepton analyses under Tav-Superblock.
    """

    # Core geometric constants
    TAU = 7.0                          # compact dimension radius factor
    MODE_1_7 = 1.0 / 7.0               # 0.142857...
    M0_FLOOR_MEV = 313.1               # topological friction floor
    WINDOW_313 = (0.25, 0.40)          # GeV window used for fine binning

    def __init__(self):
        self.null_results: Dict[str, Any] = {}

    def residue_mod7(self, values: np.ndarray) -> Dict[str, float]:
        """
        Compute residue distribution mod 7.
        Returns fractions and a simple χ² against uniform + against
        the framework-preferred residue pattern if available.
        """
        residues = np.mod(values, 7)
        counts = np.bincount(residues.astype(int), minlength=7)
        fractions = counts / counts.sum() if counts.sum() > 0 else np.zeros(7)

        # Uniform null
        expected = np.full(7, 1.0 / 7.0)
        chi2_uniform = np.sum((fractions - expected) ** 2 / expected)

        return {
            "fractions": fractions.tolist(),
            "chi2_uniform": float(chi2_uniform),
            "dominant_residue": int(np.argmax(fractions)),
            "dominant_fraction": float(np.max(fractions)),
        }

    def engine_score(self, observed: float, expected: float = 0.0,
                     uncertainty: float = 1.0) -> EngineScore:
        """
        Produce a capped internal engine score.
        Never claim this as HEP discovery significance.
        """
        raw = abs(observed - expected) / max(uncertainty, 1e-12)
        capped = min(raw, 12.0)          # hard cap to avoid overclaiming
        return EngineScore(
            engine_score_sigma=capped,
            engine_score_sigma_raw=raw,
            interpretation_label="internal_engine_score_not_hep_significance"
        )

    def fine_bin_histogram(self, values: np.ndarray,
                           bin_width_mev: float = 10.0,
                           center_mev: float = 313.1) -> Dict[str, Any]:
        """
        Fine histogram around the 313.1 MeV topological floor
        (or any other scale of interest for τ).
        """
        bin_width_gev = bin_width_mev / 1000.0
        half_window = 0.5  # GeV
        bins = np.arange(center_mev/1000 - half_window,
                         center_mev/1000 + half_window + bin_width_gev,
                         bin_width_gev)
        hist, edges = np.histogram(values, bins=bins)
        return {
            "hist": hist.tolist(),
            "bin_edges": edges.tolist(),
            "bin_width_mev": bin_width_mev,
            "center_mev": center_mev,
        }

    def null_tests(self, values: np.ndarray, labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Standard suite of null tests used in the CMS preregistered study.
        """
        results = {}

        # Event-order shuffle
        shuffled = np.random.permutation(values)
        results["shuffle"] = self.residue_mod7(shuffled)

        # Split-sample stability
        mid = len(values) // 2
        results["split_first"] = self.residue_mod7(values[:mid])
        results["split_second"] = self.residue_mod7(values[mid:])

        # Chunk / stride sensitivity
        results["chunk_even"] = self.residue_mod7(values[::2])
        results["chunk_odd"] = self.residue_mod7(values[1::2])

        self.null_results = results
        return results
