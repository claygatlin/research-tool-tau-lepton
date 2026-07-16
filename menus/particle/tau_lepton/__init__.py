"""
Tau Lepton 1/7-Mode Analysis Module
===================================
Dedicated fork of the Tau Universe / Tav-Superblock research tool
for searching the 1/7 ≈ 0.142857 harmonic lattice in Tau lepton observables.

Author: Ernest C. Gatlin III (Clay Gatlin)
Branch: tau-lepton-1-7-mode
"""

from .tau_lepton_1_7_analyzer import TauLepton17Analyzer
from .hep_controls_tau import TauHEPControls
from .preregistered_tau_study import PreregisteredTauStudy

__all__ = [
    "TauLepton17Analyzer",
    "TauHEPControls",
    "PreregisteredTauStudy",
]
