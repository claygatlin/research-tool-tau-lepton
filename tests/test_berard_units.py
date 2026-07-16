"""Unit tests ported from tau-cosmology/test_units.py."""

from __future__ import annotations

import numpy as np

from menus.astronomical.berard.halo_models import G_KPC, v_piso_halo, v_quadrature

RHO0_PC = 0.0160
RC = 8.58
REFERENCE = {
    5.0: 77.6,
    8.58: 116.9,
    10.0: 128.8,
    15.0: 159.3,
    20.0: 178.4,
    30.0: 200.3,
    48.0: 218.6,
}


def vh_no_conversion(r_kpc, rho0_pc3, rc_kpc):
    r = np.maximum(np.asarray(r_kpc, dtype=float), 1e-6)
    term = 1.0 - (float(rc_kpc) / r) * np.arctan(r / float(rc_kpc))
    return np.sqrt(4.0 * np.pi * G_KPC * float(rho0_pc3) * float(rc_kpc) ** 2 * term)


def test_corrected_velocities_match_reference():
    for r, expected in REFERENCE.items():
        got = float(v_piso_halo(np.array([r]), RHO0_PC, RC)[0])
        assert abs(got - expected) < 0.3


def test_missing_conversion_is_caught():
    assert float(vh_no_conversion(10.0, RHO0_PC, RC)) < 0.01
    ratio = float(v_piso_halo([10.0], RHO0_PC, RC)[0]) / float(
        vh_no_conversion(10.0, RHO0_PC, RC)
    )
    assert abs(ratio - np.sqrt(1e9)) / np.sqrt(1e9) < 1e-6


def test_quadrature_adds_in_squares():
    v = float(v_quadrature(90.0, 120.0))
    assert abs(v - 150.0) < 1e-9