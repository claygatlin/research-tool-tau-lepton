#!/usr/bin/env python3
"""
Tau Lepton 1/7-Mode Analyzer
============================
Core analysis class for searching the 0.142857 harmonic lattice
in Tau lepton data under the Tav-Superblock framework.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any
from .hep_controls_tau import TauHEPControls, EngineScore
from .preregistered_tau_study import PreregisteredTauStudy


class TauLepton17Analyzer:
    """
    Main analyzer for 1/7-mode searches in Tau lepton observables.
    """

    def __init__(self):
        self.controls = TauHEPControls()
        self.study = PreregisteredTauStudy()
        self.results: Dict[str, Any] = {}

    def analyze(self,
                values: np.ndarray,
                weights: Optional[np.ndarray] = None,
                label: str = "tau_sample") -> Dict[str, Any]:
        """
        Run the full preregistered analysis pipeline on a set of values
        (mass, qT, energy, etc.).
        """
        if weights is None:
            weights = np.ones_like(values)

        # 1. Residue mod-7
        residue = self.controls.residue_mod7(values)

        # 2. Fine binning around 313.1 MeV floor
        fine_hist = self.controls.fine_bin_histogram(values)

        # 3. Null tests
        nulls = self.controls.null_tests(values)

        # 4. Internal engine score (example using dominant residue excess)
        excess = residue["dominant_fraction"] - (1.0 / 7.0)
        score = self.controls.engine_score(excess, expected=0.0, uncertainty=0.05)

        self.results = {
            "label": label,
            "n_events": len(values),
            "residue_mod7": residue,
            "fine_histogram_313": fine_hist,
            "null_tests": nulls,
            "engine_score": score,
            "preregistration": self.study.to_markdown(),
        }
        return self.results

    def summary(self) -> str:
        """Human-readable summary."""
        if not self.results:
            return "No analysis run yet."

        r = self.results
        lines = [
            f"=== Tau Lepton 1/7-Mode Analysis: {r['label']} ===",
            f"Events: {r['n_events']:,}",
            f"Dominant residue: {r['residue_mod7']['dominant_residue']} "
            f"({r['residue_mod7']['dominant_fraction']:.3f})",
            f"χ² vs uniform: {r['residue_mod7']['chi2_uniform']:.2f}",
            f"Engine score (capped): {r['engine_score'].engine_score_sigma:.2f}σ",
            f"Engine score (raw):    {r['engine_score'].engine_score_sigma_raw:.2f}",
            f"Label: {r['engine_score'].interpretation_label}",
            "",
            "Null tests completed:",
        ]
        for name, res in r["null_tests"].items():
            lines.append(f"  {name}: dominant={res['dominant_residue']} "
                         f"({res['dominant_fraction']:.3f})")
        return "\n".join(lines)


# Quick smoke test
if __name__ == "__main__":
    # Synthetic data for testing
    rng = np.random.default_rng(42)
    synthetic = rng.normal(loc=1.777, scale=0.1, size=50000)  # rough tau mass scale
    analyzer = TauLepton17Analyzer()
    analyzer.analyze(synthetic, label="synthetic_tau_smoke")
    print(analyzer.summary())
