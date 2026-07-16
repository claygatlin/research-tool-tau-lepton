# Preregistered Analysis Plan  
## Search for the 1/7-Mode Harmonic Lattice in Tau Lepton Observables

**Framework**: Tau Universe / Tav-Superblock  
**Author**: Ernest Clayton Gatlin III (Clay Gatlin)  
**Date**: 2026-07-16  
**Version**: 0.1.0-pre  
**Repository**: https://github.com/claygatlin/research-tool-tau-lepton  
**Branch**: `tau-lepton-1-7-mode`

---

### 1. Scientific Goal

Test whether the same 1/7 ≈ 0.142857 harmonic lattice (induced by the compact fifth dimension of radius \(\tau = 7\, h^{-1}\,\mathrm{Mpc}\)) that appears in:
- CMS dimuon residues
- SPARC galaxy shadow/baryon ratios
- JWST JADES high-z clustering

also appears in Tau lepton observables.

---

### 2. Frozen Analysis Choices (do not change after first full data run)

| Item                        | Decision                                      |
|----------------------------|-----------------------------------------------|
| Primary observable         | Visible energy / reconstructed mass / \(q_T\) (to be frozen before first real run) |
| Secondary observables      | Decay mode, impact parameter, missing energy  |
| Binning around 313.1 MeV   | 10 MeV fine bins                              |
| Residue test               | Modulo 7 + 142857 periodicity check           |
| Engine score               | Internal only — never reported as HEP significance |
| Null tests                 | Event-order shuffle, split-sample, chunk/stride |

---

### 3. Selection Criteria (to be finalized before unblinding)

- Minimum \(p_T\) threshold: *to be frozen*
- \(\lvert\eta\rvert < 2.5\)
- Isolation: tight
- Decay modes: 1-prong + 3-prong (or all)

---

### 4. Success / Failure Criteria

A result will be considered **interesting** only if **all** of the following hold:

1. Dominant residue fraction significantly exceeds \(1/7\) after all null tests
2. The excess is stable under shuffle, split-sample, and chunk tests
3. The same residue pattern is **not** produced by pure background Monte Carlo
4. The internal engine score is reported with the mandatory label  
   `internal_engine_score_not_hep_significance`
5. Cross-check against the already-hardened CMS dimuon and SPARC pipelines shows consistency of the 1/7 lattice

---

### 5. Data Sources (planned)

- LEP (ALEPH, DELPHI, OPAL, L3) τ samples
- Belle / Belle II
- BaBar
- LHCb
- CMS / ATLAS public τ identification or decay samples (where available via CERN Open Data)

---

### 6. Code Location


menus/particle/tau_lepton/
├── init.py
├── tau_lepton_1_7_analyzer.py
├── hep_controls_tau.py
└── preregistered_tau_study.py

---

### 7. Commitment

This document freezes the analysis plan. Any later change to the primary observable, binning, or null-test suite will be recorded as a new version with explicit justification.

**Signed**: Ernest Clayton Gatlin III  
**Date**: 2026-07-16
