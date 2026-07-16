#!/usr/bin/env python3
"""
Preregistered Tau Lepton 1/7-Mode Study
======================================
Frozen analysis plan — must be committed before any full data run.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
import datetime


@dataclass
class PreregisteredTauStudy:
    """
    Preregistration document for the Tau Lepton 1/7 search.
    """
    title: str = "Search for 1/7-mode harmonic lattice in Tau lepton observables"
    framework: str = "Tau Universe / Tav-Superblock"
    author: str = "Ernest C. Gatlin III (Clay Gatlin)"
    date: str = field(default_factory=lambda: datetime.date.today().isoformat())
    version: str = "0.1.0-pre"

    # === Frozen Analysis Choices (do not change after first full run) ===
    primary_observable: str = "tau_mass_or_qT_or_invariant_mass"   # decide before run
    secondary_observables: List[str] = field(default_factory=lambda: [
        "tau_decay_mode",
        "visible_energy",
        "missing_energy",
        "impact_parameter",
    ])

    binning: Dict[str, Any] = field(default_factory=lambda: {
        "fine_window_mev": 10.0,
        "focus_scale_mev": 313.1,          # topological floor
        "mod7_residue_test": True,
        "142857_periodicity_test": True,
    })

    null_tests: List[str] = field(default_factory=lambda: [
        "event_order_shuffle",
        "split_sample_stability",
        "chunk_stride_sensitivity",
        "trigger_or_selection_bias_check",
    ])

    selection_criteria: Dict[str, Any] = field(default_factory=lambda: {
        "min_pt_gev": 5.0,                # placeholder — freeze before run
        "max_eta": 2.5,
        "isolation": "tight",
        "decay_modes": ["1-prong", "3-prong"],   # or all
    })

    success_criteria: Dict[str, Any] = field(default_factory=lambda: {
        "engine_score_threshold": 5.0,    # internal only
        "residue_2_or_preferred_dominance": True,
        "stable_under_all_nulls": True,
        "cross_check_with_dimuon_and_sparc": True,
    })

    notes: str = """
    This study re-uses the exact HEP control discipline already hardened
    for CMS dimuon analyses (trigger-aware mod-7, engine-score labeling,
    fine 10 MeV binning around 313.1 MeV floor, full null suite).

    Any claim of a 1/7-mode signal must survive:
      1. All null tests
      2. Comparison against the same controls on dimuon and SPARC
      3. Explicit labeling as internal_engine_score_not_hep_significance
    """

    def to_markdown(self) -> str:
        """Generate a preregistration markdown block ready for Zenodo / GitHub."""
        lines = [
            f"# Preregistered Analysis: {self.title}",
            f"**Framework**: {self.framework}",
            f"**Author**: {self.author}",
            f"**Date**: {self.date}",
            f"**Version**: {self.version}",
            "",
            "## Frozen Choices",
            f"- Primary observable: `{self.primary_observable}`",
            f"- Binning: {self.binning}",
            f"- Null tests: {', '.join(self.null_tests)}",
            f"- Selection: {self.selection_criteria}",
            "",
            "## Success Criteria",
            str(self.success_criteria),
            "",
            "## Notes",
            self.notes,
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    study = PreregisteredTauStudy()
    print(study.to_markdown())
