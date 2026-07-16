"""
Berard Framework constants and symbolic equation registry.

Source: claygatlin/tau-cosmology-berard-framework (Vernon Arthur Berard, Feb 2026).
Presented as exploratory / not peer-reviewed in the research tool UI.
"""

from __future__ import annotations

from typing import Any

# Phenomenological Berard Constant (dimensionless matter–resonance coupling)
BERARD_CONSTANT: float = 1.054

# Stability Well vacuum oscillation (Hz)
STABILITY_WELL_F0_HZ: float = 0.10

# Present-epoch homogeneous vacuum (Resonance Invariant)
S0_PRESENT_EPOCH: float = 1.0

# Manuscript uses inverse-square inertia coupling; README v1.3 shows S_0^2.
# Default to inverse-square (data-fit convention from session assessment).
DEFAULT_INERTIA_EXPONENT: int = -2

# Reference Hubble for tension demos (km/s/Mpc)
H0_PLANCK_KM_S_MPC: float = 67.4
H0_SHOES_KM_S_MPC: float = 73.0

# Standard MOND scale (m/s²) for comparison overlays
MOND_A0_M_S2: float = 1.2e-10

BERARD_EQUATIONS: dict[str, dict[str, Any]] = {
    "resonance_invariant": {
        "latex": r"S_0(x) = \phi(x) / \phi_0",
        "variables": {
            "S_0": "Dimensionless Resonance Invariant",
            "phi": "Real scalar resonance field",
            "phi_0": "Vacuum expectation value (present epoch: S_0=1)",
        },
    },
    "effective_inertial_mass": {
        "latex": r"m_{\mathrm{eff}} = m \, BC^2 \, S_0^{n}",
        "variables": {
            "m_eff": "Resonance-dependent effective inertial mass",
            "m": "Bare rest mass",
            "BC": f"Berard Constant ({BERARD_CONSTANT})",
            "S_0": "Local Resonance Invariant",
            "n": "Inertia exponent (default -2 manuscript; +2 in README v1.3)",
        },
    },
    "einstein_berard": {
        "latex": r"E = m_{\mathrm{eff}}(\phi)",
        "variables": {"E": "Rest energy", "m_eff": "Effective inertial mass"},
    },
    "stability_well": {
        "latex": r"V(\phi) = \frac{1}{2} m_\phi^2 (\phi - \phi_0)^2,\; m_\phi = 2\pi f_0",
        "variables": {
            "f_0": f"Vacuum resonance frequency ({STABILITY_WELL_F0_HZ} Hz)",
            "m_phi": "Scalar curvature mass",
        },
    },
    "cosmological_rescaling": {
        "latex": r"H_{\mathrm{obs}} = BC \cdot H_B",
        "variables": {
            "H_obs": "Observed Hubble parameter",
            "H_B": "Bare / resonance-corrected expansion rate",
            "BC": "Berard Constant",
        },
    },
    "galactic_acceleration": {
        "latex": r"a_B = a_{\mathrm{Newton}} / BC^2",
        "variables": {
            "a_B": "Observable radial acceleration",
            "a_Newton": "Baryonic Newtonian acceleration GM/r²",
        },
    },
}

FRAMEWORK_METADATA: dict[str, Any] = {
    "name": "Berard Framework",
    "author": "Vernon Arthur Berard",
    "repository": "https://github.com/claygatlin/tau-cosmology-berard-framework",
    "status": "exploratory",
    "peer_reviewed": False,
    "manuscript_date": "2026-02",
    "confidence_label": "phenomenological / empirically tuned BC",
    "local_assumption": "S_0 ≈ 1 in present local vacuum (homogeneous FRW background)",
}