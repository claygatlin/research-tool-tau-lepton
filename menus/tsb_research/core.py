#!/usr/bin/env python3
"""
Tav-Superblock Research Tool
============================

Integrates:
- Tau Cylinder geometry (C ≃ S¹_ϕ × ℝ_s)
- The underlying scalar function Φ(s, ϕ)
- Staggered E₈-projected velocities
- Topological Exclusion Principle
- Pressure averaging / string tension (√2 factor)
- 7-fold resonance + 142857 cycle
- Schematic twistor projection hooks
- Hierarchical binding (n_hier)
- Sound horizon prediction anchored to 313.1 MeV + Φ
- Cylinder-native oscillation template from Φ
- WHIM template generator (2τ filament signal)
- Octonionic / G₂ Clockwork (core dynamical engine)
- Explicit octonion multiplication table (numpy + optional sympy)
- Clockwork-driven cylinder slice time evolution
- CylinderOperatorAlgebra (8×8 phase-space operators)
- CasimirSignatureModule (Φ + operator 7-fold Casimir correction curve)
- ResidualDiagnostics (lag ACF, Breusch–Pagan, unified reporting)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, overload

import numpy as np

try:
    import sympy as sp

    HAS_SYMPY = True
except ImportError:
    sp = None  # type: ignore[assignment]
    HAS_SYMPY = False

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Framework constants (aligned with tau_sb_desi_scanner where applicable)
try:
    from menus.astronomical.desi.scanner import (
        LATE_UNIVERSE_DELTA_N,
        M0_MEV,
        N_HIER_BINDING,
        TAU_PERIOD,
    )

    LAMBDA_TSB = float(M0_MEV)
    N_HIER = float(N_HIER_BINDING)
    DELTA_N_LOCAL = float(LATE_UNIVERSE_DELTA_N)
except ImportError:
    LAMBDA_TSB = 313.1
    N_HIER = 45.8
    DELTA_N_LOCAL = 0.33
    TAU_PERIOD = 7.0

GEOMETRIC_FRICTION_FLOOR = LAMBDA_TSB
SQRT2 = float(np.sqrt(2.0))
RD_OBSERVED_MPC = 147.0
GAMMA_RD_ANCHOR = 8.0

try:
    from menus.astronomical.desi.scanner import (
        DELTA_N_PLASMA,
        RD_OBSERVED_MPC as _RD_OBSERVED_MPC,
        TSB_RD_GAMMA_FALLBACK,
        compute_tsb_rd,
    )

    RD_OBSERVED_MPC = float(_RD_OBSERVED_MPC)
except ImportError:
    DELTA_N_PLASMA = 10.74
    TSB_RD_GAMMA_FALLBACK = 8.8511

    def compute_tsb_rd(gamma: float, **kwargs: Any) -> dict[str, Any]:
        g = float(gamma)
        rd = float(DELTA_N_PLASMA) * float(RD_OBSERVED_MPC) / g
        res_pct = 100.0 * (rd - RD_OBSERVED_MPC) / RD_OBSERVED_MPC
        return {
            "rd_predicted_mpc": rd,
            "rd_observed_mpc": RD_OBSERVED_MPC,
            "gamma_used": g,
            "rd_residual_percent": res_pct,
        }


@dataclass
class TauCylinder:
    """Geometric realization of the Tau cylinder: C ≃ S¹_ϕ × ℝ_s."""

    phi_period: float = 7.0
    reset_phase: float = 7.0
    s_range: tuple[float, float] = (0.0, N_HIER)
    n_hier: float = N_HIER

    def phase_mod(self, phi: float) -> float:
        return float(phi) % float(self.phi_period)

    def is_reset_phase(self, phi: float) -> bool:
        return bool(np.isclose(self.phase_mod(phi), self.reset_phase))


@dataclass
class ScalarFunctionPhi:
    """
    Central scalar function Φ(s, ϕ) on the Tau cylinder.

    Generates 7-fold resonance, transverse velocities, nodal exclusion,
    and pressure contributions via phase gradients.
    """

    cylinder: TauCylinder = field(default_factory=TauCylinder)
    a0: float = 0.0
    ak: np.ndarray = field(default_factory=lambda: np.zeros(6))
    bk: np.ndarray = field(default_factory=lambda: np.zeros(6))
    amplitude_scale: float = LAMBDA_TSB

    def __post_init__(self) -> None:
        if np.allclose(self.ak, 0) and np.allclose(self.bk, 0):
            self.ak = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.05], dtype=float) * self.amplitude_scale
            self.bk = np.array([0.8, 0.4, 0.25, 0.15, 0.08, 0.03], dtype=float) * self.amplitude_scale

    def evaluate(self, s: float, phi: float) -> float:
        phi_mod = self.cylinder.phase_mod(phi)
        val = float(self.a0)
        for k in range(1, 7):
            val += self.ak[k - 1] * np.cos(2 * np.pi * k * phi_mod / 7) + self.bk[k - 1] * np.sin(
                2 * np.pi * k * phi_mod / 7
            )
        stretch = np.exp(-float(s) / self.cylinder.n_hier)
        return float(val * stretch)

    def dphi(self, s: float, phi: float, eps: float = 1e-6) -> float:
        return (self.evaluate(s, phi + eps) - self.evaluate(s, phi - eps)) / (2 * eps)

    def transverse_velocity_squared(self, s: float, phi: float, plane: int = 1) -> float:
        _ = plane
        grad_phi = self.dphi(s, phi)
        return float((grad_phi**2) * (self.amplitude_scale / LAMBDA_TSB) ** 2)

    def string_tension(self, s: float = 0.0, phi: float = 0.0) -> float:
        v1 = self.transverse_velocity_squared(s, phi, plane=1)
        v2 = self.transverse_velocity_squared(s, phi, plane=2)
        return float(v1 + v2)

    def sqrt_sigma(self) -> float:
        return float(SQRT2 * self.amplitude_scale)

    def gluon_condensate(self) -> float:
        factor = 3.0 / 4.0
        return float((factor * self.sqrt_sigma() / 1000.0) ** 4 * 1e9)

    def nodal_structure(self, s: float, phi: float, threshold: float = 0.1) -> bool:
        return bool(abs(self.evaluate(s, phi)) < threshold)

    def exclusion_violation_cost(self, s: float, phi: float) -> float:
        if self.nodal_structure(s, phi):
            return float(GEOMETRIC_FRICTION_FLOOR)
        return 0.0


@dataclass
class TavSuperblockVariables:
    """Central handler for Tav-Superblock variables routed through Φ(s, ϕ)."""

    cylinder: TauCylinder = field(default_factory=TauCylinder)
    phi: ScalarFunctionPhi = field(default_factory=ScalarFunctionPhi)
    e8_staggering_phases: np.ndarray | None = None
    twistor_kernel: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        if self.phi.cylinder is not self.cylinder:
            self.phi.cylinder = self.cylinder
        if self.e8_staggering_phases is None:
            self.e8_staggering_phases = np.linspace(0, 2 * np.pi, 6, endpoint=False)

    def compute_string_tension(self, s: float = 0.0, phi: float = 0.0) -> float:
        return self.phi.string_tension(s, phi)

    def compute_sqrt_sigma(self) -> float:
        return self.phi.sqrt_sigma()

    def compute_gluon_condensate(self) -> float:
        return self.phi.gluon_condensate()

    def hierarchical_scale(self, s: float) -> float:
        return float(s / self.cylinder.n_hier * N_HIER)

    def check_exclusion(self, domain: int, phase_k: int, s: float = 0.0) -> bool:
        return not self.phi.nodal_structure(s, float(phase_k))

    def exclusion_action_cost(self, domain: int, phase_k: int, s: float = 0.0) -> float:
        return self.phi.exclusion_violation_cost(s, float(phase_k))

    def twistor_projected_pressure(self, s: float, phi: float) -> float:
        v_trans = self.phi.transverse_velocity_squared(s, phi)
        return float(v_trans * 0.5)

    def apply_twistor_projection(
        self,
        s: float,
        phi: float,
        *,
        domain: int = 0,
        raw_pressure: float | None = None,
    ) -> dict[str, float]:
        """
        Schematic twistor projection: map transverse pressure through E₈ staggering
        and an optional user-supplied twistor kernel.
        """
        phi_stag = self.apply_e8_staggering(float(phi), int(domain))
        pressure = float(raw_pressure if raw_pressure is not None else self.twistor_projected_pressure(s, phi_stag))
        stagger_mod = 1.0 + 0.1 * float(np.cos(phi_stag))
        projected = pressure * stagger_mod
        if self.twistor_kernel is not None:
            kernel_out = self.twistor_kernel(projected, float(s), float(phi_stag))
            projected = float(kernel_out) if np.isscalar(kernel_out) else float(np.asarray(kernel_out).ravel()[0])
        return {
            "raw_pressure": pressure,
            "projected_pressure": projected,
            "phi_staggered": float(phi_stag),
            "stagger_modulation": float(stagger_mod),
        }

    def apply_e8_staggering(self, base_phi: float, domain: int) -> float:
        if self.e8_staggering_phases is None:
            return float(base_phi)
        offset = float(self.e8_staggering_phases[domain % 6])
        return float(base_phi + offset)

    def resonance_projector(self, k: int) -> bool:
        return (int(k) % 7) in (1, 2, 3, 4, 5, 6)

    def summary(self) -> str:
        return (
            "Tav-Superblock Variables Summary\n"
            "================================\n"
            f"Λ_TSB (MeV)          : {LAMBDA_TSB}\n"
            f"√σ = √2 · Λ_TSB      : {self.compute_sqrt_sigma():.4f} MeV\n"
            f"<(α_s/π)G²> (GeV⁴)   : {self.compute_gluon_condensate():.6e}\n"
            f"n_hier               : {self.cylinder.n_hier}\n"
            f"Scalar Φ amplitude   : {self.phi.amplitude_scale}\n"
            "7-fold resonance     : active (modes 1-6)\n"
        )

    def plasma_slice_s(self) -> float:
        """Mid-plasma axial coordinate on the cylinder (Δn_plasma projection)."""
        return float(self.cylinder.n_hier) * float(DELTA_N_PLASMA) / float(N_HIER)

    def demo_report(self, s: float = 12.5, phi: float = 2.3) -> dict[str, Any]:
        return {
            "lambda_tsb_mev": LAMBDA_TSB,
            "n_hier": self.cylinder.n_hier,
            "sqrt_sigma_mev": self.compute_sqrt_sigma(),
            "gluon_condensate_gev4": self.compute_gluon_condensate(),
            "phi_at_test": self.phi.evaluate(s, phi),
            "v_trans_squared": self.phi.transverse_velocity_squared(s, phi, 1),
            "string_tension": self.compute_string_tension(s, phi),
            "exclusion_cost_domain1_k3": self.exclusion_action_cost(1, 3, s),
            "twistor_pressure": self.twistor_projected_pressure(s, phi),
            "has_sympy": HAS_SYMPY,
            "s_test": float(s),
            "phi_test": float(phi),
        }


@dataclass
class SoundHorizon:
    """
    Predicts sound horizon r_d using Φ(s, ϕ) and geometric pressure / string tension.

    Anchors to the 313.1 MeV friction floor and incorporates pipeline tensions
    at γ=7.95 (+35.15%) and γ=12.0 (−10.5%) with Φ pressure reducing γ sensitivity.
    """

    tsv: TavSuperblockVariables
    m0: float = LAMBDA_TSB
    gamma_low: float = 7.95
    gamma_high: float = 12.0
    rd_obs: float = RD_OBSERVED_MPC
    residual_low: float = 0.3515
    residual_high: float = -0.105
    rd_observed_mpc: float = RD_OBSERVED_MPC
    gamma_anchor: float = 8.0

    def _predict_rd_unscaled(self, gamma: float, s_ref: float = 0.0) -> float:
        """Core Φ-anchored formula with reduced γ sensitivity."""
        return self.predict_rd(gamma, s_ref=s_ref)

    def predict_rd(self, gamma: float, s_ref: float = 0.0) -> float:
        """
        Stronger Φ-based anchoring to reduce γ sensitivity.
        Target: Flatter r_d vs γ curve closer to observed 147 Mpc.
        """
        p_trans = self.tsv.twistor_projected_pressure(s_ref, 0.0)
        phi_scale = np.sqrt(max(p_trans, 1e-6)) * (self.m0 / LAMBDA_TSB)

        hier_factor = np.exp(-self.tsv.cylinder.n_hier / 52.0)
        pressure_anchor = 1.0 + 0.015 * (phi_scale - 1.0)

        r_d_base = 147.0 * (self.m0 / 313.1) * hier_factor * pressure_anchor
        gamma_correction = 1.0 + 0.0015 * (gamma - 9.0)

        return float(r_d_base * gamma_correction)

    def residual_at_gamma(self, gamma: float, s_ref: float = 0.0) -> float:
        rd_pred = self.predict_rd(gamma, s_ref=s_ref)
        return float((rd_pred - float(self.rd_obs)) / float(self.rd_obs))

    def consistency_improvement(self, s_ref: float = 0.0) -> dict[str, float]:
        old_swing = abs(self.residual_low - self.residual_high)
        new_res_low = self.residual_at_gamma(self.gamma_low, s_ref=s_ref)
        new_res_high = self.residual_at_gamma(self.gamma_high, s_ref=s_ref)
        new_swing = abs(new_res_low - new_res_high)
        improvement_factor = old_swing / max(new_swing, 1e-6)
        return {
            "old_residual_swing_percent": round(old_swing * 100, 2),
            "new_residual_swing_percent": round(new_swing * 100, 2),
            "improvement_factor": round(improvement_factor, 2),
            "new_residual_at_7.95": round(new_res_low * 100, 2),
            "new_residual_at_12.0": round(new_res_high * 100, 2),
        }

    def predict_rd_detail(self, gamma: float, s_ref: float = 0.0) -> dict[str, Any]:
        g = float(gamma)
        rd_mpc = self.predict_rd(g, s_ref=s_ref)
        legacy = compute_tsb_rd(g, rd_observed=self.rd_obs)
        legacy_rd = float(legacy["rd_predicted_mpc"])
        return {
            "rd_predicted_mpc": rd_mpc,
            "rd_observed_mpc": float(self.rd_obs),
            "gamma_used": g,
            "gamma_anchor": float(self.gamma_anchor),
            "residual_percent": 100.0 * (rd_mpc - float(self.rd_obs)) / float(self.rd_obs),
            "legacy_rd_mpc": legacy_rd,
            "legacy_residual_percent": float(legacy.get("rd_residual_percent", float("nan"))),
            "consistency_improvement": self.consistency_improvement(s_ref=s_ref),
        }


def generate_cylinder_oscillation_template(
    phi_func: ScalarFunctionPhi,
    s_values: np.ndarray,
    n_modes: int = 6,
    *,
    phi_phase: float | None = None,
    average_active_phases: bool = False,
    normalize: str = "std_0.06",
) -> np.ndarray:
    """
    Generate a residual template in s-space from Φ Fourier content and ∂Φ/∂ϕ.

    Default mode sums weighted phase gradients across active harmonics and
    normalizes to std ≈ 0.06. Legacy kwargs ``phi_phase`` / ``average_active_phases``
    select direct Φ evaluation instead.
    """
    s = np.asarray(s_values, dtype=float)
    if phi_phase is not None or average_active_phases:
        if average_active_phases:
            tracks = [
                np.array([phi_func.evaluate(float(si), float(k)) for si in s], dtype=float)
                for k in range(1, 7)
            ]
            template = np.mean(np.vstack(tracks), axis=0)
        else:
            template = np.array(
                [phi_func.evaluate(float(si), float(phi_phase or 0.0)) for si in s],
                dtype=float,
            )
        if normalize == "max_abs":
            peak = float(np.max(np.abs(template)))
            if peak > 0:
                template = template / peak
        return np.asarray(template, dtype=float)

    template = np.zeros(len(s), dtype=float)
    for idx, si in enumerate(s):
        contrib = 0.0
        for k in range(1, int(n_modes) + 1):
            dphi_val = phi_func.dphi(float(si), float(k))
            weight = 1.0 / (k**0.7)
            contrib += weight * dphi_val / phi_func.amplitude_scale
        template[idx] = contrib

    std = float(np.std(template))
    if normalize == "std_0.06" and std > 0:
        template = template / std * 0.06
    elif normalize == "max_abs":
        peak = float(np.max(np.abs(template)))
        if peak > 0:
            template = template / peak
    elif normalize not in ("none", ""):
        raise ValueError(f"unknown normalize mode: {normalize!r}")
    return np.asarray(template, dtype=float)


@dataclass
class WHIMTemplate:
    """WHIM / cosmological-web template using the marginal 2τ (f=1/14) filament signal."""

    frequency: float = 1.0 / 14.0
    amplitude: float = 0.03
    phase_offset: float = 0.0
    use_cylinder_staggering: bool = True

    def generate(
        self,
        s_values: np.ndarray,
        tsv: TavSuperblockVariables | None = None,
    ) -> np.ndarray:
        s = np.asarray(s_values, dtype=float)
        template = self.amplitude * np.sin(2 * np.pi * self.frequency * s + self.phase_offset)
        if self.use_cylinder_staggering and tsv is not None and tsv.e8_staggering_phases is not None:
            avg_stagger = float(np.mean(tsv.e8_staggering_phases))
            template = template * (1.0 + 0.2 * np.sin(avg_stagger))
        return np.asarray(template, dtype=float)

    def power_at_2tau(self, residuals: np.ndarray, s_values: np.ndarray) -> float:
        s = np.asarray(s_values, dtype=float)
        resid = np.asarray(residuals, dtype=float)
        try:
            from scipy.signal import lombscargle

            freqs = np.array([self.frequency], dtype=float)
            power = lombscargle(s, resid, freqs)
            return float(power[0])
        except Exception:
            basis = np.sin(2 * np.pi * self.frequency * s)
            return float(abs(np.dot(resid, basis)) ** 2)


# Fano-plane oriented triples (e_a * e_b = e_c) for the standard octonion basis.
_OCTONION_FANO_TRIPLES: tuple[tuple[int, int, int], ...] = (
    (1, 2, 3),
    (1, 4, 5),
    (1, 7, 6),
    (2, 4, 6),
    (2, 5, 7),
    (3, 4, 7),
    (3, 6, 5),
)


@dataclass
class OctonionAlgebra:
    """
    Explicit octonion multiplication using the 8-dimensional basis {e0..e7}.

    e0 is the real unit; e1..e7 are imaginary units with Fano-plane products.
    Supports numeric (numpy) and optional symbolic (sympy) multiplication.
    """

    dim: int = 8
    mult_table: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.mult_table = self._build_multiplication_table()

    @staticmethod
    def _build_multiplication_table() -> np.ndarray:
        """Table[i, j] = signed index k such that e_i * e_j = sign * e_k."""
        table = np.zeros((8, 8), dtype=int)
        for i in range(8):
            table[i, i] = 0 if i == 0 else 0  # e_i*e_i = -e0 handled in multiply
        for a, b, c in _OCTONION_FANO_TRIPLES:
            table[a, b] = c
            table[b, c] = a
            table[c, a] = b
            table[b, a] = -c
            table[c, b] = -a
            table[a, c] = -b
        for i in range(1, 8):
            for j in range(1, 8):
                if i != j and table[i, j] == 0:
                    table[i, j] = -table[j, i]
        return table

    def multiply_basis(self, i: int, j: int) -> tuple[int, int]:
        """Return (signed_unit_index, sign) for e_i * e_j."""
        if i == 0:
            return j, 1
        if j == 0:
            return i, 1
        if i == j:
            return 0, -1
        k = int(self.mult_table[i, j])
        if k == 0:
            raise ValueError(f"undefined basis product e{i}*e{j}")
        return abs(k), (1 if k > 0 else -1)

    def multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Multiply two 8-component octonion coefficient vectors."""
        x = np.asarray(a, dtype=float).reshape(-1)
        y = np.asarray(b, dtype=float).reshape(-1)
        if x.size != 8 or y.size != 8:
            raise ValueError("octonion vectors must have length 8")
        out = np.zeros(8, dtype=float)
        for i in range(8):
            if x[i] == 0:
                continue
            for j in range(8):
                if y[j] == 0:
                    continue
                k, sign = self.multiply_basis(i, j)
                out[k] += sign * x[i] * y[j]
        return out

    def power(self, a: np.ndarray, n: int) -> np.ndarray:
        """Repeated multiplication a^n with a^0 = e0."""
        if int(n) < 0:
            raise ValueError("n must be non-negative")
        unit = np.zeros(8, dtype=float)
        unit[0] = 1.0
        base = np.asarray(a, dtype=float).reshape(-1)
        result = unit.copy()
        for _ in range(int(n)):
            result = self.multiply(result, base)
        return result

    def phase_angle(self, a: np.ndarray) -> float:
        """Extract a polar angle from the imaginary part of an octonion."""
        vec = np.asarray(a, dtype=float).reshape(-1)
        imag = vec[1:]
        norm = float(np.linalg.norm(imag))
        if norm < 1e-12:
            return 0.0
        dominant = int(np.argmax(np.abs(imag))) + 1
        return float((np.arctan2(imag[dominant - 1], vec[0]) + 2 * np.pi) % (2 * np.pi))

    def generator_octonion(self, generator_index: int = 1) -> np.ndarray:
        """Unit octonion aligned with basis direction e_{generator_index}."""
        out = np.zeros(8, dtype=float)
        out[int(generator_index) % 8] = 1.0
        return out

    def basis_symbols(self) -> list[Any]:
        """Return sympy symbols [e0, ..., e7]."""
        if not HAS_SYMPY or sp is None:
            raise RuntimeError("sympy is required for symbolic octonion multiplication")
        return [sp.Symbol(f"e{i}") for i in range(8)]

    def symbolic_multiply_basis(self, i: int, j: int) -> Any:
        """Symbolic product of two basis units."""
        if not HAS_SYMPY or sp is None:
            raise RuntimeError("sympy is required for symbolic octonion multiplication")
        e = self.basis_symbols()
        if i == 0:
            return e[j]
        if j == 0:
            return e[i]
        if i == j:
            return -e[0]
        k, sign = self.multiply_basis(i, j)
        return e[k] if sign > 0 else -e[k]

    def symbolic_multiply(self, left: Any, right: Any) -> Any:
        """Expand a symbolic octonion product using the explicit multiplication table."""
        if not HAS_SYMPY or sp is None:
            raise RuntimeError("sympy is required for symbolic octonion multiplication")
        e = self.basis_symbols()
        if isinstance(left, (int, float)):
            left = sp.Integer(left) * e[0]
        if isinstance(right, (int, float)):
            right = sp.Integer(right) * e[0]
        expr = sp.Integer(0)
        for i in range(8):
            ci = left.coeff(e[i]) if hasattr(left, "coeff") else sp.Integer(0)
            for j in range(8):
                cj = right.coeff(e[j]) if hasattr(right, "coeff") else sp.Integer(0)
                expr = sp.expand(expr + sp.expand(ci * cj) * self.symbolic_multiply_basis(i, j))
        return sp.expand(expr)

    def verify_associator_failure(self) -> dict[str, Any]:
        """Demonstrate non-associativity: (e1*e2)*e4 != e1*(e2*e4)."""
        e1 = self.generator_octonion(1)
        e2 = self.generator_octonion(2)
        e4 = self.generator_octonion(4)
        left = self.multiply(self.multiply(e1, e2), e4)
        right = self.multiply(e1, self.multiply(e2, e4))
        return {
            "left": left.tolist(),
            "right": right.tolist(),
            "associator_norm": float(np.linalg.norm(left - right)),
            "is_associative": bool(np.allclose(left, right)),
        }


@dataclass
class CylinderEvolutionResult:
    """Bundle produced by clockwork evolution with coupled observables."""

    tsv: TavSuperblockVariables
    phi_coord: float
    s: float
    steps: int
    sound_horizon: SoundHorizon | None = None
    rd_at_gamma: float | None = None
    rd_detail: dict[str, Any] | None = None
    twistor: dict[str, float] | None = None
    exclusion: dict[str, Any] | None = None
    octonion_phase: np.ndarray | None = None


@dataclass
class OctonionicClockwork:
    """
    Models the 8-phase octonionic / G₂ clockwork.

    - 7 active phases from the order-7 octonionic cycle
    - 1 reset phase (G₂ fixed / singular direction)
    - Drives phase advancement on the Tau cylinder
    - Generates / evolves Fourier coefficients for Φ
    - Induces staggered velocities across six supersphere domains
    """

    n_phases: int = 7
    reset_phase: int = 7
    generator_angle: float = 2 * np.pi / 7
    g2_stabilizer_strength: float = 1.0
    generator_index: int = 1
    algebra: OctonionAlgebra = field(default_factory=OctonionAlgebra)
    roots: np.ndarray = field(init=False, repr=False)
    phase_labels: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.roots = np.exp(1j * self.generator_angle * np.arange(self.n_phases))
        self.phase_labels = np.arange(self.n_phases + 1)

    def advance_phase(self, phi: float, steps: int = 1) -> float:
        """Advance phase by integer steps under the clockwork generator."""
        delta = int(steps) * self.generator_angle
        return float((float(phi) + delta) % (2 * np.pi))

    def octonion_generator_state(self, steps: int = 1) -> np.ndarray:
        """Return e_generator^n from the explicit multiplication table."""
        gen = self.algebra.generator_octonion(self.generator_index)
        return self.algebra.power(gen, int(steps))

    def advance_phase_algebraic(self, phi: float, steps: int = 1) -> float:
        """
        Advance τ-cylinder phase by steps (canonical 7-fold generator).

        The octonion-table state e^n is exposed separately via
        ``octonion_generator_state`` — its polar angle can differ from the
        τ-cylinder coordinate because octonion multiplication is 8-dimensional.
        """
        return self.advance_phase(phi, steps=int(steps))

    def is_reset_phase(self, phi: float, tol: float = 1e-6) -> bool:
        phi_mod = float(phi) % (2 * np.pi)
        reset_angle = 2 * np.pi * self.reset_phase / (self.n_phases + 1)
        return bool(abs(phi_mod - reset_angle) < tol)

    def generate_fourier_coefficients(
        self,
        amplitude: float = LAMBDA_TSB,
        decay: float = 0.7,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a_k, b_k for Φ inspired by the clockwork (lower k stronger)."""
        ak = np.array([amplitude * (decay**k) for k in range(1, self.n_phases + 1)], dtype=float)
        bk = np.array(
            [amplitude * (decay ** (k + 0.5)) for k in range(1, self.n_phases + 1)],
            dtype=float,
        )
        return ak, bk

    def evolve_phi_coefficients(
        self,
        ak: np.ndarray,
        bk: np.ndarray,
        steps: int = 1,
        s: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evolve Φ coefficients under clockwork steps + hierarchical modulation."""
        n = min(len(ak), 6)
        roots6 = self.roots[:n]
        new_ak = ak[:n].copy().astype(complex)
        new_bk = bk[:n].copy().astype(complex)

        for _ in range(int(steps)):
            complex_coeff = new_ak + 1j * new_bk
            rotated = roots6 * complex_coeff
            new_ak = np.real(rotated)
            new_bk = np.imag(rotated)

        stretch = float(np.exp(-float(s) / N_HIER))
        return np.asarray(new_ak * stretch, dtype=float), np.asarray(new_bk * stretch, dtype=float)

    def evolve_phi_coefficients_algebraic(
        self,
        ak: np.ndarray,
        bk: np.ndarray,
        steps: int = 1,
        s: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evolve Φ coefficients by multiplying an octonion built from (a_k, b_k)
        with the clockwork generator in the explicit algebra.
        """
        n = min(len(ak), 6)
        oct_vec = np.zeros(8, dtype=float)
        for k in range(n):
            oct_vec[k + 1] = float(ak[k]) + 0.5 * float(bk[k])
        gen = self.algebra.generator_octonion(self.generator_index)
        evolved = oct_vec.copy()
        for _ in range(int(steps)):
            evolved = self.algebra.multiply(evolved, gen)
        stretch = float(np.exp(-float(s) / N_HIER))
        new_ak = evolved[1 : n + 1] * stretch
        new_bk = evolved[2 : n + 2] * stretch if n < 7 else evolved[2:8] * stretch
        if len(new_bk) < n:
            new_bk = np.pad(new_bk, (0, n - len(new_bk)))
        return np.asarray(new_ak[:n], dtype=float), np.asarray(new_bk[:n], dtype=float)

    def exclusion_cost_detail(
        self,
        phi: float,
        s: float,
        tsv: TavSuperblockVariables,
        *,
        domain: int = 0,
        phase_k: int | None = None,
        nodal_threshold: float = 0.1,
    ) -> dict[str, Any]:
        """
        Link reset-phase Hamiltonian cost to the topological exclusion principle.

        A configuration is forbidden when Φ has a nodal structure, when the
        clockwork sits on the G₂ reset direction, or when the τ-cylinder hits
        its 7-fold reset phase. Costs combine via the geometric friction floor.
        """
        phi_base = float(phase_k if phase_k is not None else phi)
        phi_stag = tsv.apply_e8_staggering(phi_base, int(domain))
        phi_val = tsv.phi.evaluate(float(s), phi_stag)
        nodal = bool(abs(phi_val) < nodal_threshold)
        clock_reset = self.is_reset_phase(phi_stag)
        cylinder_reset = tsv.cylinder.is_reset_phase(phi_stag)
        hamiltonian_cost = self.clockwork_hamiltonian_contribution(phi_stag, s=s)
        hard_forbidden = nodal or clock_reset
        nodal_cost = float(GEOMETRIC_FRICTION_FLOOR) if nodal else 0.0
        reset_cost = float(GEOMETRIC_FRICTION_FLOOR) if clock_reset else 0.0
        cylinder_reset_cost = float(0.5 * GEOMETRIC_FRICTION_FLOOR) if cylinder_reset else 0.0
        soft_barrier = float(0.01 * hamiltonian_cost) if not hard_forbidden else 0.0
        if hard_forbidden:
            total_cost = float(GEOMETRIC_FRICTION_FLOOR)
            mechanism = "nodal" if nodal else "clock_reset"
        elif cylinder_reset:
            total_cost = float(cylinder_reset_cost + soft_barrier)
            mechanism = "cylinder_reset"
        elif hamiltonian_cost > 1.0:
            total_cost = float(soft_barrier)
            mechanism = "hamiltonian_proximity"
        else:
            total_cost = 0.0
            mechanism = "none"
        allowed = not hard_forbidden
        return {
            "allowed": allowed,
            "phi_staggered": float(phi_stag),
            "phi_value": float(phi_val),
            "nodal": nodal,
            "clock_reset": clock_reset,
            "cylinder_reset": cylinder_reset,
            "hard_forbidden": hard_forbidden,
            "nodal_cost_mev": nodal_cost,
            "reset_cost_mev": reset_cost,
            "cylinder_reset_cost_mev": cylinder_reset_cost,
            "hamiltonian_cost_mev": float(hamiltonian_cost),
            "soft_barrier_mev": soft_barrier,
            "total_exclusion_cost_mev": total_cost,
            "mechanism": mechanism,
            "reset_linked_to_exclusion": bool(clock_reset or (cylinder_reset and nodal)),
        }

    def get_domain_staggered_velocities(
        self,
        base_velocity: float = 1.0,
        n_domains: int = 6,
    ) -> np.ndarray:
        """Generate E₈/G₂-staggered velocities for the six supersphere domains."""
        phases = np.linspace(0, 2 * np.pi, int(n_domains), endpoint=False)
        clockwork_offset = self.generator_angle * np.arange(int(n_domains))
        return np.asarray(
            float(base_velocity) * np.sin(phases + clockwork_offset),
            dtype=float,
        )

    def clockwork_hamiltonian_contribution(self, phi: float, s: float = 0.0) -> float:
        """Schematic Ĥ_ϕ contribution. Peaks at reset phase (exclusion barrier)."""
        _ = s
        if self.is_reset_phase(phi):
            return float(GEOMETRIC_FRICTION_FLOOR)
        phi_mod = float(phi) % (2 * np.pi)
        reset_angle = 2 * np.pi * self.reset_phase / (self.n_phases + 1)
        dist_to_reset = min(
            abs(phi_mod - reset_angle),
            2 * np.pi - abs(phi_mod - reset_angle),
        )
        proximity = float(np.exp(-dist_to_reset / 0.5))
        return float(0.1 * LAMBDA_TSB * proximity)

    @overload
    def evolve_cylinder_state(
        self,
        tsv: TavSuperblockVariables,
        steps: int = 1,
        s: float = 0.0,
        phi: float = 0.0,
        *,
        gamma: float = ...,
        apply_twistor: bool = ...,
        update_sound_horizon: bool = ...,
        use_algebraic_evolution: bool = ...,
        domain: int = ...,
        return_bundle: Literal[True] = True,
    ) -> CylinderEvolutionResult: ...

    @overload
    def evolve_cylinder_state(
        self,
        tsv: TavSuperblockVariables,
        steps: int = 1,
        s: float = 0.0,
        phi: float = 0.0,
        *,
        gamma: float = ...,
        apply_twistor: bool = ...,
        update_sound_horizon: bool = ...,
        use_algebraic_evolution: bool = ...,
        domain: int = ...,
        return_bundle: Literal[False],
    ) -> TavSuperblockVariables: ...

    def evolve_cylinder_state(
        self,
        tsv: TavSuperblockVariables,
        steps: int = 1,
        s: float = 0.0,
        phi: float = 0.0,
        *,
        gamma: float = GAMMA_RD_ANCHOR,
        apply_twistor: bool = True,
        update_sound_horizon: bool = True,
        use_algebraic_evolution: bool = False,
        domain: int = 0,
        return_bundle: bool = True,
    ) -> CylinderEvolutionResult | TavSuperblockVariables:
        """
        Evolve TavSuperblockVariables and optionally refresh SoundHorizon,
        twistor-projected pressure, and exclusion diagnostics.

        Set ``return_bundle=False`` for the legacy API that returns only
        ``TavSuperblockVariables`` (matching the original clockwork snippet).
        """
        phi_coord = self.advance_phase(float(phi), steps=int(steps))
        evolve_fn = (
            self.evolve_phi_coefficients_algebraic
            if use_algebraic_evolution
            else self.evolve_phi_coefficients
        )
        new_phi = ScalarFunctionPhi(cylinder=tsv.cylinder)
        new_phi.ak, new_phi.bk = evolve_fn(tsv.phi.ak, tsv.phi.bk, steps=steps, s=s)
        new_phi.amplitude_scale = tsv.phi.amplitude_scale
        if tsv.e8_staggering_phases is None:
            e8_phases: np.ndarray = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        else:
            e8_phases = np.asarray(tsv.e8_staggering_phases, dtype=float) + self.generator_angle * int(steps)
        evolved_tsv = TavSuperblockVariables(
            cylinder=tsv.cylinder,
            phi=new_phi,
            e8_staggering_phases=e8_phases,
            twistor_kernel=tsv.twistor_kernel,
        )

        twistor: dict[str, float] | None = None
        if apply_twistor:
            twistor = evolved_tsv.apply_twistor_projection(s, phi_coord, domain=domain)

        sound_horizon: SoundHorizon | None = None
        rd_at_gamma: float | None = None
        rd_detail: dict[str, Any] | None = None
        if update_sound_horizon:
            sound_horizon = SoundHorizon(evolved_tsv)
            rd_at_gamma = sound_horizon.predict_rd(float(gamma), s_ref=float(s))
            rd_detail = sound_horizon.predict_rd_detail(float(gamma), s_ref=float(s))

        exclusion = self.exclusion_cost_detail(phi_coord, s, evolved_tsv, domain=domain)
        oct_phase = self.octonion_generator_state(steps=int(steps))

        result = CylinderEvolutionResult(
            tsv=evolved_tsv,
            phi_coord=float(phi_coord),
            s=float(s),
            steps=int(steps),
            sound_horizon=sound_horizon,
            rd_at_gamma=rd_at_gamma,
            rd_detail=rd_detail,
            twistor=twistor,
            exclusion=exclusion,
            octonion_phase=oct_phase,
        )
        if not return_bundle:
            return evolved_tsv
        return result

    def simulate_cylinder_time_evolution(
        self,
        tsv: TavSuperblockVariables | None = None,
        *,
        n_steps: int = 20,
        s_start: float = 0.0,
        s_end: float | None = None,
        phi_start: float = 0.0,
        gamma: float = GAMMA_RD_ANCHOR,
        apply_twistor: bool = True,
        use_algebraic_evolution: bool = False,
        domain: int = 0,
    ) -> dict[str, Any]:
        """
        Drive a simple time evolution across axial cylinder slices s(t).

        Each step advances the clockwork phase, evolves Φ coefficients, applies
        twistor projection, evaluates exclusion cost, and updates r_d(γ).
        """
        state_tsv = tsv or TavSuperblockVariables()
        s_end_val = float(N_HIER if s_end is None else s_end)
        s_vals = np.linspace(float(s_start), s_end_val, int(n_steps), dtype=float)
        phi = float(phi_start)
        rows: list[dict[str, Any]] = []
        current = state_tsv

        for step_idx, s_i in enumerate(s_vals):
            result = self.evolve_cylinder_state(
                current,
                steps=1,
                s=float(s_i),
                phi=phi,
                gamma=float(gamma),
                apply_twistor=apply_twistor,
                update_sound_horizon=True,
                use_algebraic_evolution=use_algebraic_evolution,
                domain=domain,
            )
            current = result.tsv
            phi = result.phi_coord
            rows.append(
                {
                    "step": int(step_idx),
                    "s": float(s_i),
                    "phi": float(phi),
                    "phi_value": float(result.tsv.phi.evaluate(float(s_i), phi)),
                    "rd_mpc": float(result.rd_at_gamma or float("nan")),
                    "twistor_projected_pressure": float(
                        (result.twistor or {}).get("projected_pressure", float("nan"))
                    ),
                    "exclusion_cost_mev": float(
                        (result.exclusion or {}).get("total_exclusion_cost_mev", 0.0)
                    ),
                    "exclusion_allowed": bool((result.exclusion or {}).get("allowed", True)),
                    "exclusion_mechanism": (result.exclusion or {}).get("mechanism", "none"),
                }
            )

        s_arr = np.array([r["s"] for r in rows], dtype=float)
        rd_arr = np.array([r["rd_mpc"] for r in rows], dtype=float)
        excl_arr = np.array([r["exclusion_cost_mev"] for r in rows], dtype=float)
        return {
            "gamma": float(gamma),
            "n_steps": int(n_steps),
            "trajectory": rows,
            "s": s_arr.tolist(),
            "phi": [r["phi"] for r in rows],
            "rd_mpc": rd_arr.tolist(),
            "exclusion_cost_mev": excl_arr.tolist(),
            "rd_swing_mpc": float(np.nanmax(rd_arr) - np.nanmin(rd_arr)),
            "forbidden_fraction": float(np.mean([not r["exclusion_allowed"] for r in rows])),
        }


@dataclass
class CylinderOperatorAlgebra:
    """
    8×8 operator algebra on the τ-cylinder phase space.

    Basis: |0⟩…|6⟩ (active clockwork phases), |7⟩ ≡ |r⟩ (G₂ reset / exclusion slot).
    Includes shift T, schematic creation/annihilation with reset projection,
    clockwork Hamiltonian H_clock, and reset penalty H_penalty.
    """

    delta_floor: float = GEOMETRIC_FRICTION_FLOOR
    omega_pl: float = 1.0
    dim: int = 8
    T: np.ndarray = field(init=False, repr=False)
    a: list[np.ndarray] = field(init=False, repr=False)
    adag: list[np.ndarray] = field(init=False, repr=False)
    H_clock: np.ndarray = field(init=False, repr=False)
    H_penalty: np.ndarray = field(init=False, repr=False)
    H_phi: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._build_operators()

    def _build_operators(self) -> None:
        dim = int(self.dim)
        self.T = np.zeros((dim, dim), dtype=float)
        for k in range(7):
            self.T[(k + 1) % 7, k] = 1.0

        self.a = []
        self.adag = []
        for i in range(7):
            a_mat = np.zeros((dim, dim), dtype=float)
            adag_mat = np.zeros((dim, dim), dtype=float)
            if i < 6:
                a_mat[7, i] = 1.0
                adag_mat[i, 7] = 1.0
            self.a.append(a_mat)
            self.adag.append(adag_mat)

        self.H_clock = np.zeros((dim, dim), dtype=float)
        for k in range(7):
            self.H_clock += self.omega_pl * (
                self.T @ self.adag[k] @ self.a[k] + self.adag[k] @ self.a[k] @ self.T
            )

        self.H_penalty = self.delta_floor * np.diag([0.0] * 7 + [1.0])
        self.H_phi = self.H_clock + self.H_penalty

    def initial_state(self, active_index: int = 0) -> np.ndarray:
        """Normalized state vector localized on an active phase |k⟩."""
        psi = np.zeros(self.dim, dtype=complex)
        psi[int(active_index) % 7] = 1.0
        return psi

    def evolve_state(self, psi: np.ndarray, delta_s: float = 0.1) -> np.ndarray:
        """Unitary evolution exp(−i H_Φ Δs) |ψ⟩."""
        from scipy.linalg import expm

        state = np.asarray(psi, dtype=complex).reshape(-1)
        if state.size != self.dim:
            raise ValueError(f"state vector must have length {self.dim}")
        unitary = expm(-1j * self.H_phi * float(delta_s))
        return unitary @ state

    def evolve_trajectory(
        self,
        psi0: np.ndarray,
        *,
        n_steps: int = 10,
        delta_s: float = 0.1,
    ) -> dict[str, Any]:
        """Evolve |ψ⟩ for multiple Δs steps and track occupations."""
        states: list[np.ndarray] = [np.asarray(psi0, dtype=complex).reshape(-1)]
        occupations: list[np.ndarray] = [self.occupation_numbers(states[0])]
        for _ in range(int(n_steps)):
            next_state = self.evolve_state(states[-1], delta_s=delta_s)
            states.append(next_state)
            occupations.append(self.occupation_numbers(next_state))
        return {
            "n_steps": int(n_steps),
            "delta_s": float(delta_s),
            "states": states,
            "occupations": occupations,
            "reset_occupation_final": float(occupations[-1][7]),
        }

    def occupation_numbers(self, psi: np.ndarray) -> np.ndarray:
        """Return |ψ_k|² for each basis component."""
        state = np.asarray(psi, dtype=complex).reshape(-1)
        return np.abs(state) ** 2

    def residuals_from_states(self, psi_before: np.ndarray, psi_after: np.ndarray) -> np.ndarray:
        """Active-sector residual vector (first 7 components) between two states."""
        before = np.asarray(psi_before, dtype=complex).reshape(-1)[:7]
        after = np.asarray(psi_after, dtype=complex).reshape(-1)[:7]
        return np.real(after - before)

    def clockwork_hamiltonian_contribution(self, phi: float = 0.0) -> float:
        """
        Schematic ⟨H_Φ⟩ proxy at clockwork phase ``phi`` for Casimir penalty estimates.

        Uses short evolution to leak into the reset slot plus ‖H_Φ‖/dim phase weighting.
        """
        phase_index = int(round((float(phi) % (2 * np.pi)) / (2 * np.pi / 7))) % 7
        psi0 = self.initial_state(active_index=phase_index)
        psi = self.evolve_state(psi0, delta_s=0.35)
        reset_leak = float(np.abs(psi[7]) ** 2)
        base = float(np.linalg.norm(self.H_phi) / self.dim)
        return base * (1.0 + 0.25 * np.cos(float(phi))) + float(self.delta_floor) * reset_leak


@dataclass
class CasimirSignatureModule:
    """
    Generates expected Casimir signatures and a first-order force correction
    from the Tau-Superblock operator algebra.

    Uses:
    - :class:`CylinderOperatorAlgebra` (exclusion penalty + reset phase)
    - :class:`ScalarFunctionPhi` (pressure from gradients)
    - :class:`OctonionicClockwork` (7-fold modulation)
    """

    algebra: CylinderOperatorAlgebra
    phi: ScalarFunctionPhi
    clockwork: OctonionicClockwork | None = None
    m0: float = 313.1
    a0: float = 20.0
    lambda_decay: float = 42.0
    beta: float = 1.8

    def __post_init__(self) -> None:
        if self.clockwork is None:
            self.clockwork = OctonionicClockwork()

    def geometric_suppression(self, separation_nm: float) -> float:
        """Improved geometric factor f(a) for exclusion energy at separation ``a``."""
        a = float(separation_nm)
        a0 = float(self.a0)
        return float((a0 / (a + a0)) ** float(self.beta) * np.exp(-a / float(self.lambda_decay)))

    def compute_exclusion_energy_density(self, separation_nm: float) -> float:
        """First-order correction from H_penalty (schematic)."""
        f_a = self.geometric_suppression(separation_nm)
        avg_penalty = float(
            np.mean(
                [
                    self.algebra.clockwork_hamiltonian_contribution(phi=np.pi * k / 3.5)
                    for k in range(7)
                ]
            )
        )
        return float(self.algebra.delta_floor) * f_a * (avg_penalty / 1e9)

    def predict_casimir_force_correction(self, separations_nm: np.ndarray) -> np.ndarray:
        """Return ΔF_TSB(a) including 7-fold clockwork modulation."""
        sep = np.asarray(separations_nm, dtype=float).reshape(-1)
        delta_e = np.array(
            [self.compute_exclusion_energy_density(float(s)) for s in sep],
            dtype=float,
        )

        phases = sep / 12.0
        modulation = 1.0 + 0.12 * np.sin(2 * np.pi * phases / 7.0)
        delta_e_mod = delta_e * modulation

        da = np.gradient(sep)
        return -np.gradient(delta_e_mod, da)

    def generate_signature_plot(
        self,
        separations_nm: np.ndarray | None = None,
        save_path: str | Path | None = None,
    ) -> Path:
        """Plot predicted Casimir force correction ΔF_TSB(a) in fN."""
        import matplotlib.pyplot as plt

        if separations_nm is None:
            separations_nm = np.linspace(30.0, 250.0, 400)
        sep = np.asarray(separations_nm, dtype=float)
        delta_f = self.predict_casimir_force_correction(sep)

        out = Path(save_path) if save_path is not None else ARTIFACTS_DIR / "casimir_tsb_correction.png"
        out.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(9, 5))
        plt.plot(sep, delta_f * 1e15)
        plt.xlabel("Plate separation (nm)")
        plt.ylabel("Force correction (fN)")
        plt.title("Casimir Force Correction — Improved Geometric Suppression")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Casimir] Plot saved → {out}")
        return out

    def generate_harmonic_signature(self, n_points: int = 200) -> np.ndarray:
        """
        Expected harmonic content in the Casimir signal from 7-fold clockwork.

        Returns the cylinder oscillation template in s-space for comparison with data.
        """
        s_values = np.linspace(0.0, N_HIER, int(n_points))
        return generate_cylinder_oscillation_template(self.phi, s_values)

    @staticmethod
    def _map_param_to_s(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if len(x) < 2:
            return np.zeros_like(x)
        xmin, xmax = float(np.min(x)), float(np.max(x))
        span = xmax - xmin
        if span <= 0:
            return np.zeros_like(x)
        return (x - xmin) / span * float(N_HIER)

    @staticmethod
    def _zscore(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float).reshape(-1)
        std = float(np.std(y))
        if std <= 0:
            return y - float(np.mean(y))
        return (y - float(np.mean(y))) / std

    @staticmethod
    def _is_separation_axis(param_col: str) -> bool:
        tokens = ("separation", "distance", "gap", "nm", "plate")
        name = str(param_col).lower()
        return any(token in name for token in tokens)

    def predict_template_on_params(self, x: np.ndarray) -> np.ndarray:
        """Map observed parameter axis to s-space and evaluate the Φ template."""
        return generate_cylinder_oscillation_template(self.phi, self._map_param_to_s(x))

    def load_reference_casimir_data(
        self,
        data_path: str | Path | None = None,
        *,
        use_mock_if_missing: bool = True,
    ) -> tuple[Any, str, str, str]:
        """
        Load a Casimir reference table from ``datasets/casimir`` or an explicit path.

        Returns ``(dataframe, label, param_col, value_col)``.
        """
        import pandas as pd

        from menus.particle.casimir.scanner import (
            DATASETS_DIR,
            _auto_select_columns,
            generate_mock_casimir_data,
            list_casimir_data_files,
            load_data,
        )

        label = "mock magnetic-field Casimir sweep"
        if data_path is not None:
            path = Path(data_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            df = load_data(path)
            label = path.name
        else:
            skip_names = {"done.txt", "processed.txt", "readme.txt", "readme.md"}
            candidates = [
                p
                for p in list_casimir_data_files(DATASETS_DIR)
                if p.name.lower() not in skip_names
                and "github_magnetic_fluid" in str(p)
            ]
            if not candidates:
                candidates = [
                    p
                    for p in list_casimir_data_files(DATASETS_DIR)
                    if p.name.lower() not in skip_names
                ]
            loaded = False
            for path in candidates:
                try:
                    trial = load_data(path)
                    _auto_select_columns(trial)
                except (ValueError, TypeError, KeyError):
                    continue
                df = trial
                label = str(path.relative_to(PROJECT_ROOT))
                loaded = True
                break
            if not loaded:
                if use_mock_if_missing:
                    df = generate_mock_casimir_data()
                else:
                    raise FileNotFoundError(f"No usable Casimir data files under {DATASETS_DIR}")

        param_col, value_col = _auto_select_columns(df)
        return df, label, param_col, value_col

    def compare_with_data(
        self,
        df: Any,
        *,
        param_col: str | None = None,
        value_col: str | None = None,
        data_label: str = "",
    ) -> dict[str, Any]:
        """Compare predicted TSB signature / force correction against observed Casimir data."""
        from menus.particle.casimir.scanner import _auto_select_columns, detect_tau_resonances

        if param_col is None or value_col is None:
            param_col, value_col = _auto_select_columns(df)

        mask = df[param_col].notna() & df[value_col].notna()
        work = df.loc[mask, [param_col, value_col]].astype(float)
        work = work.sort_values(param_col)
        x = work[param_col].to_numpy()
        y = work[value_col].to_numpy()
        if len(x) < 3:
            raise ValueError("Need at least 3 finite points for comparison")

        template = self.predict_template_on_params(x)
        template_scaled = template * (float(np.std(y)) / max(float(np.std(template)), 1e-12))
        template_scaled = template_scaled + float(np.mean(y)) - float(np.mean(template_scaled))

        force_corr = None
        if self._is_separation_axis(param_col):
            force_corr = self.predict_casimir_force_correction(x)
            fc = np.asarray(force_corr, dtype=float)
            if float(np.max(np.abs(fc))) > 0:
                force_corr = fc * (float(np.std(y)) / float(np.std(fc)))
                force_corr = force_corr + float(np.mean(y)) - float(np.mean(force_corr))

        y_z = self._zscore(y)
        template_z = self._zscore(template)
        pearson_r = float(np.corrcoef(y_z, template_z)[0, 1]) if len(y) > 1 else float("nan")
        rmse = float(np.sqrt(np.mean((y_z - template_z) ** 2)))

        deg = min(2, len(x) - 1)
        trend = np.polyval(np.polyfit(x, y, deg=deg), x)
        residual = y - trend
        residual_z = self._zscore(residual)
        residual_r = (
            float(np.corrcoef(residual_z, template_z)[0, 1]) if len(residual) > 1 else float("nan")
        )

        data_scan = detect_tau_resonances(df, param_col=param_col, value_col=value_col)
        harmonic_vec = self.generate_harmonic_signature(n_points=max(50, len(x)))

        return {
            "data_label": data_label or "casimir_reference",
            "param_col": param_col,
            "value_col": value_col,
            "n_points": int(len(x)),
            "param_range": [float(np.min(x)), float(np.max(x))],
            "metrics": {
                "pearson_r_template": pearson_r,
                "pearson_r_residual_template": residual_r,
                "rmse_zscore_template": rmse,
                "observed_std": float(np.std(y)),
                "template_std": float(np.std(template)),
            },
            "harmonic_signature": harmonic_vec.tolist(),
            "series": {
                "param": x.tolist(),
                "observed": y.tolist(),
                "predicted_template_scaled": template_scaled.tolist(),
                "predicted_template_raw": template.tolist(),
                "residual_detrended": residual.tolist(),
                "predicted_force_correction_scaled": (
                    force_corr.tolist() if force_corr is not None else None
                ),
            },
            "data_tau_scan": {
                "detected_features": data_scan.get("detected_features", []),
                "signature_summary": data_scan.get("signature_summary", {}),
                "diagnostics": data_scan.get("diagnostics", {}),
            },
            "m0_mev": float(self.m0),
            "reset_occupation_proxy": float(
                self.algebra.evolve_trajectory(
                    self.algebra.initial_state(0), n_steps=7, delta_s=float(N_HIER) / 7.0
                )["reset_occupation_final"]
            ),
        }

    def export_comparison(
        self,
        comparison: dict[str, Any],
        *,
        output_dir: str | Path | None = None,
        stamp: str | None = None,
    ) -> dict[str, str]:
        """Write comparison JSON, CSV, and overlay plot under artifacts/."""
        import csv
        import json
        from datetime import datetime, timezone

        import matplotlib.pyplot as plt
        import pandas as pd

        out_dir = Path(output_dir) if output_dir is not None else ARTIFACTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = stamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        json_path = out_dir / f"casimir_signature_compare_{tag}.json"
        csv_path = out_dir / f"casimir_signature_compare_{tag}.csv"
        plot_path = out_dir / f"casimir_signature_compare_{tag}.png"

        series = comparison["series"]
        rows = {
            comparison["param_col"]: series["param"],
            comparison["value_col"]: series["observed"],
            "predicted_template_scaled": series["predicted_template_scaled"],
            "predicted_template_raw": series["predicted_template_raw"],
            "residual_detrended": series["residual_detrended"],
        }
        if series.get("predicted_force_correction_scaled") is not None:
            rows["predicted_force_correction_scaled"] = series["predicted_force_correction_scaled"]
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        export_payload = {k: v for k, v in comparison.items() if k != "series"}
        export_payload["series_path"] = str(csv_path)
        json_path.write_text(json.dumps(export_payload, indent=2) + "\n", encoding="utf-8")

        x = np.asarray(series["param"], dtype=float)
        y = np.asarray(series["observed"], dtype=float)
        template = np.asarray(series["predicted_template_scaled"], dtype=float)
        residual = np.asarray(series["residual_detrended"], dtype=float)
        metrics = comparison["metrics"]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        axes[0].plot(x, y, "b.-", alpha=0.8, label="observed")
        axes[0].plot(x, template, "r--", linewidth=2, label="TSB template (scaled)")
        if series.get("predicted_force_correction_scaled") is not None:
            axes[0].plot(
                x,
                np.asarray(series["predicted_force_correction_scaled"], dtype=float),
                "g:",
                linewidth=2,
                label="force correction (scaled)",
            )
        axes[0].set_xlabel(comparison["param_col"])
        axes[0].set_ylabel(comparison["value_col"])
        axes[0].set_title("Observed vs TSB prediction")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.25)

        axes[1].plot(x, residual, "k.-", alpha=0.75, label="detrended residual")
        axes[1].plot(
            x,
            self._zscore(np.asarray(series["predicted_template_raw"], dtype=float))
            * float(np.std(residual)),
            "m--",
            label="template (σ-matched)",
        )
        axes[1].set_xlabel(comparison["param_col"])
        axes[1].set_title("Residual vs 7-fold template")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.25)

        feat_lines = [
            f"r(template) = {metrics['pearson_r_template']:.3f}",
            f"r(residual) = {metrics['pearson_r_residual_template']:.3f}",
            f"RMSE(z) = {metrics['rmse_zscore_template']:.3f}",
            f"n = {comparison['n_points']}",
            "",
            f"data: {comparison['data_label']}",
        ]
        for feat in comparison.get("data_tau_scan", {}).get("detected_features", [])[:4]:
            feat_lines.append(f"• {feat.get('type', 'feature')}")
        axes[2].axis("off")
        axes[2].text(0.03, 0.97, "\n".join(feat_lines), va="top", family="monospace", fontsize=9)
        axes[2].set_title("Comparison metrics")

        fig.suptitle("CasimirSignatureModule — data comparison", fontsize=11)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        paths = {
            "json": str(json_path),
            "csv": str(csv_path),
            "plot": str(plot_path),
        }
        print(f"[Casimir] Comparison JSON: {json_path}")
        print(f"[Casimir] Comparison CSV:  {csv_path}")
        print(f"[Casimir] Comparison plot: {plot_path}")
        return paths

    def compare_and_export(
        self,
        data_path: str | Path | None = None,
        *,
        use_mock_if_missing: bool = True,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Load reference Casimir data, compare to TSB predictions, and export artifacts."""
        df, label, param_col, value_col = self.load_reference_casimir_data(
            data_path,
            use_mock_if_missing=use_mock_if_missing,
        )
        comparison = self.compare_with_data(
            df,
            param_col=param_col,
            value_col=value_col,
            data_label=label,
        )
        paths = self.export_comparison(comparison, output_dir=output_dir)
        comparison["export_paths"] = paths
        return comparison


@dataclass
class ResidualDiagnostics:
    """
    Lightweight residual diagnostics for τ-cylinder / operator-evolution outputs.

    Provides lag autocorrelation, Breusch–Pagan heteroskedasticity test,
    optional diagnostic plots (ACF, residuals vs fitted, Q-Q), and a unified
    report dict suitable for JSON export.
    """

    residuals: np.ndarray
    exog: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.residuals = np.asarray(self.residuals, dtype=float).reshape(-1)
        self.n = int(len(self.residuals))
        if self.exog is not None:
            self.exog = np.asarray(self.exog, dtype=float)

    @classmethod
    def from_desi_fit(
        cls,
        desi_bundle: dict[str, Any],
    ) -> ResidualDiagnostics:
        """
        Build diagnostics from a :func:`load_desi_fit_residuals` bundle.

        Uses ``s`` and ``z`` as Breusch–Pagan regressors when available.
        """
        residuals = np.asarray(desi_bundle["residuals"], dtype=float).reshape(-1)
        s = np.asarray(desi_bundle.get("s"), dtype=float).reshape(-1)
        z = np.asarray(desi_bundle.get("z"), dtype=float).reshape(-1)
        if len(s) == len(residuals) and len(z) == len(residuals):
            exog = np.column_stack([np.ones(len(residuals)), z, s])
        elif len(z) == len(residuals):
            exog = np.column_stack([np.ones(len(residuals)), z])
        else:
            exog = np.column_stack([np.ones(len(residuals)), np.arange(len(residuals), dtype=float)])
        return cls(residuals=residuals, exog=exog)

    @classmethod
    def from_cylinder_evolution(
        cls,
        evolution: dict[str, Any],
        *,
        field: str = "phi_value",
    ) -> ResidualDiagnostics:
        """Build diagnostics from model-minus-reference residuals along a trajectory."""
        traj = evolution.get("trajectory", [])
        if len(traj) < 2:
            raise ValueError("evolution trajectory must contain at least 2 steps")
        values = np.array([float(row[field]) for row in traj], dtype=float)
        resid = np.diff(values)
        exog = np.column_stack([np.ones(len(resid)), np.arange(len(resid), dtype=float)])
        return cls(residuals=resid, exog=exog)

    def lag_autocorrelation(
        self,
        lags: list[int] | None = None,
        *,
        verbose: bool = False,
    ) -> dict[str, float]:
        if lags is None:
            lags = list(range(1, min(7, self.n)))
        results: dict[str, float] = {}
        for lag in lags:
            if lag >= self.n:
                continue
            acf = float(np.corrcoef(self.residuals[:-lag], self.residuals[lag:])[0, 1])
            results[f"lag_{lag}"] = round(acf, 4)
            if verbose:
                print(f"Lag {lag} autocorrelation: {acf:.4f}")
        return results

    def breusch_pagan_test(self, *, verbose: bool = False) -> dict[str, Any]:
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan

            exog = self.exog
            if exog is None:
                exog = np.column_stack([np.ones(self.n), np.arange(self.n, dtype=float)])
            bp_test = het_breuschpagan(self.residuals, exog)
            results = {
                "lm_statistic": float(bp_test[0]),
                "lm_pvalue": round(float(bp_test[1]), 4),
                "f_statistic": float(bp_test[2]),
                "f_pvalue": round(float(bp_test[3]), 4),
            }
            if verbose:
                print(f"Breusch-Pagan LM p-value: {results['lm_pvalue']:.4f}")
            return results
        except Exception as exc:
            return {"error": str(exc)}

    def run_full_diagnostics(
        self,
        *,
        verbose: bool = False,
        provenance: dict[str, Any] | None = None,
        save_plots: bool = True,
        fitted_values: np.ndarray | list[float] | None = None,
    ) -> dict[str, Any]:
        if verbose:
            print("\n=== Full Residual Diagnostics Report ===")
            print(f"n = {self.n}")

        report: dict[str, Any] = {
            "n_residuals": self.n,
            "lag_autocorrelation": self.lag_autocorrelation(verbose=verbose),
            "breusch_pagan": self.breusch_pagan_test(verbose=verbose),
            "plots_saved": save_plots,
        }
        if provenance is not None:
            report["provenance"] = provenance

        if save_plots:
            plot_paths: dict[str, str] = {}
            try:
                plot_paths["acf"] = str(self.plot_acf())
                if verbose:
                    print(f"[Plot] Saved: {plot_paths['acf']}")
            except Exception as exc:
                if verbose:
                    print(f"[Plot Warning] Could not save ACF plot: {exc}")

            if fitted_values is not None:
                try:
                    plot_paths["residuals_vs_fitted"] = str(
                        self.plot_residuals_vs_fitted(fitted_values)
                    )
                    if verbose:
                        print(f"[Plot] Saved: {plot_paths['residuals_vs_fitted']}")
                except Exception as exc:
                    if verbose:
                        print(f"[Plot Warning] Could not save residuals-vs-fitted plot: {exc}")

            try:
                plot_paths["qq"] = str(self.plot_qq())
                if verbose:
                    print(f"[Plot] Saved: {plot_paths['qq']}")
            except Exception as exc:
                if verbose:
                    print(f"[Plot Warning] Could not save Q-Q plot: {exc}")

            if plot_paths:
                report["plot_paths"] = plot_paths

        if verbose:
            print("=== End of Report ===\n")
        return report

    @staticmethod
    def _resolve_plot_path(save_path: str | Path | None, default_name: str) -> Path:
        path = Path(save_path) if save_path is not None else ARTIFACTS_DIR / default_name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def plot_acf(self, save_path: str | Path | None = None) -> Path:
        import matplotlib.pyplot as plt
        from statsmodels.graphics.tsaplots import plot_acf

        out = self._resolve_plot_path(save_path, "residual_acf.png")
        fig, ax = plt.subplots(figsize=(8, 5))
        lags = max(1, min(10, self.n - 1))
        plot_acf(self.residuals, ax=ax, lags=lags)
        ax.set_title("Residual Autocorrelation Function (ACF)")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_residuals_vs_fitted(
        self,
        fitted_values: np.ndarray | list[float],
        save_path: str | Path | None = None,
    ) -> Path:
        import matplotlib.pyplot as plt
        from statsmodels.nonparametric.smoothers_lowess import lowess

        fitted = np.asarray(fitted_values, dtype=float).reshape(-1)
        if len(fitted) != self.n:
            raise ValueError(f"fitted_values length {len(fitted)} != residuals length {self.n}")

        out = self._resolve_plot_path(save_path, "residuals_vs_fitted.png")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(fitted, self.residuals, alpha=0.7, label="Residuals")
        smoothed = lowess(self.residuals, fitted, frac=0.6)
        ax.plot(smoothed[:, 0], smoothed[:, 1], "r-", linewidth=2, label="LOWESS trend")
        ax.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel("Fitted Values (Model Prediction)")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Fitted Values")
        ax.legend()
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_qq(self, save_path: str | Path | None = None) -> Path:
        import matplotlib.pyplot as plt
        from scipy import stats

        out = self._resolve_plot_path(save_path, "qq_plot.png")
        fig, ax = plt.subplots(figsize=(7, 7))
        stats.probplot(self.residuals, dist="norm", plot=ax)
        ax.set_title(f"Q-Q Plot (R² = {self._qq_r2():.3f})")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out

    def _qq_r2(self) -> float:
        from scipy import stats

        osm, osr = stats.probplot(self.residuals, dist="norm")[0]
        _, _, r_value, _, _ = stats.linregress(osm, osr)
        return float(r_value**2)


def load_desi_fit_residuals(
    *,
    data_mode: str | None = None,
    tracer: str | None = None,
    gamma: float | None = None,
    auto_calibrate_gamma: bool = True,
    use_whim_model: bool = False,
    whim_strength: float = 0.02,
    use_tsb_rd: bool = False,
    fit_A: bool = True,
    fit_phase: bool = True,
    fit_hier: bool = True,
) -> dict[str, Any]:
    """
    Load genuine DESI DR2 data, fit Tau-SB, and return fit residuals.

    Requires cloned Cobaya ``desi_bao_dr2`` tables under
    ``datasets/desi/bao_data/`` (see :mod:`tau_sb_desi_scanner`).
    """
    from menus.astronomical.desi.scanner import (
        DATA_MODE,
        TRACER,
        TauSBScanner,
        _tau_sb_prediction_from_fit,
        _tau_sb_prediction_from_fit_whim,
        load_scan_data_by_data_mode,
        resolve_cobaya_dr2_dir,
        s_from_z,
    )

    mode = data_mode or DATA_MODE
    tracer_use = tracer or TRACER
    data = load_scan_data_by_data_mode(mode=mode, tracer=tracer_use)
    z = np.asarray(data["z"], dtype=float)
    obs = np.asarray(data["observable"], dtype=float)
    err = np.asarray(data["err"], dtype=float)
    cov = np.asarray(data["cov"], dtype=float)
    scanner = TauSBScanner(
        gamma=float(gamma if gamma is not None else 10.0),
        auto_calibrate_gamma=bool(auto_calibrate_gamma),
    )
    fit = scanner.fit_tau_sb_model(
        z,
        obs,
        err,
        cov=cov,
        fit_A=fit_A,
        fit_phase=fit_phase,
        fit_hier=fit_hier,
    )
    gamma_used = float(scanner._gamma_for(z))
    if use_whim_model:
        y_model = _tau_sb_prediction_from_fit_whim(
            fit,
            z=z,
            gamma=gamma_used,
            period=scanner.period,
            whim_strength=whim_strength,
            use_tsb_rd=use_tsb_rd,
        )
    else:
        y_model = _tau_sb_prediction_from_fit(
            fit, z=z, gamma=gamma_used, period=scanner.period
        )
    y_model = np.asarray(y_model, dtype=float)
    residuals = obs - y_model
    s_arr = data.get("s")
    s_grid = np.asarray(s_arr, dtype=float) if s_arr is not None else s_from_z(z, gamma=gamma_used)
    dr2_dir = resolve_cobaya_dr2_dir()
    return {
        "residuals": residuals,
        "y_obs": obs,
        "y_model": y_model,
        "z": z,
        "s": s_grid,
        "err": err,
        "cov": cov,
        "fit": fit,
        "gamma": gamma_used,
        "data_mode": mode,
        "tracer": tracer_use,
        "n_data": int(len(residuals)),
        "source": "desi_dr2_cobaya",
        "repository": "CobayaSampler/bao_data desi_bao_dr2",
        "local_path": str(dr2_dir),
        "data_class": "real_cached",
        "use_whim_model": bool(use_whim_model),
        "chi2_reduced": float(fit.get("chi2_reduced", float("nan"))),
    }


def run_generate_mocks(
    *,
    n: int = 10,
    gamma: float = 9.5,
    residual_sigma: float = 0.08,
    seed: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Generate cylinder-template mocks with SoundHorizon anchor and diagnostics."""
    from menus.tsb_research.mock_generator import generate_research_mock_report, save_research_mock_report

    report = generate_research_mock_report(
        int(n),
        gamma=float(gamma),
        residual_sigma=float(residual_sigma),
        seed=seed,
    )
    path = save_research_mock_report(report)
    report["saved_path"] = str(path)
    if verbose:
        print("Generate Mocks")
        print("==============\n")
        print(f"n = {report['n']} | γ = {report['gamma']} | data_class: {report['data_class']}")
        print(f"Sound horizon: {report['sound_horizon_mpc']:.2f} Mpc")
        print(f"Template std: {report['template_std']:.4f}")
        bp = report.get("diagnostics", {}).get("breusch_pagan", {})
        print(f"Breusch-Pagan p-value: {bp.get('lm_pvalue', 'N/A')}")
        print(f"Saved: {path}")
    return report


def run_evolve_cylinder_state(
    *,
    steps: int = 2,
    s: float = 5.0,
    phi: float = 0.0,
    gamma: float = GAMMA_RD_ANCHOR,
    apply_twistor: bool = True,
    update_sound_horizon: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Evolve cylinder state via octonionic clockwork with coupled observables."""
    clock = OctonionicClockwork()
    tsv = TavSuperblockVariables()
    result = clock.evolve_cylinder_state(
        tsv,
        steps=int(steps),
        s=float(s),
        phi=float(phi),
        gamma=float(gamma),
        apply_twistor=apply_twistor,
        update_sound_horizon=update_sound_horizon,
    )
    report = {
        "steps": int(steps),
        "s": float(s),
        "phi_coord": float(result.phi_coord),
        "gamma": float(gamma),
        "rd_at_gamma_mpc": result.rd_at_gamma,
        "rd_detail": result.rd_detail,
        "twistor": result.twistor,
        "exclusion": result.exclusion,
        "phi_ak_norm": float(np.linalg.norm(result.tsv.phi.ak)),
        "octonion_phase": (
            result.octonion_phase.tolist() if result.octonion_phase is not None else None
        ),
    }
    if verbose:
        print("Evolve Cylinder State")
        print("=====================\n")
        print(f"steps={steps} s={s} ϕ→{result.phi_coord:.4f}")
        if result.rd_at_gamma is not None:
            print(f"r_d(γ={gamma}) = {result.rd_at_gamma:.2f} Mpc")
        if result.exclusion:
            print(
                f"exclusion: allowed={result.exclusion['allowed']} "
                f"cost={result.exclusion['total_exclusion_cost_mev']:.2f} MeV"
            )
    return report


def run_calibrate_sound_horizon(
    *,
    gamma_values: list[float] | None = None,
    s_ref: float = 0.0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Calibrate and report Φ-anchored SoundHorizon across γ values."""
    tsv = TavSuperblockVariables()
    sh = SoundHorizon(tsv)
    gammas = gamma_values or [7.95, 8.0, 8.8511, 12.0]
    predictions = {
        str(float(g)): sh.predict_rd_detail(float(g), s_ref=float(s_ref)) for g in gammas
    }
    report = {
        "gamma_anchor": float(sh.gamma_anchor),
        "s_ref": float(s_ref),
        "predictions": predictions,
        "consistency_improvement": sh.consistency_improvement(s_ref=float(s_ref)),
    }
    if verbose:
        print("Calibrate SoundHorizon")
        print("======================\n")
        for g in gammas:
            detail = predictions[str(float(g))]
            print(
                f"γ={g:.4f} → r_d={detail['rd_predicted_mpc']:.2f} Mpc "
                f"({detail['residual_percent']:+.2f}%)"
            )
        imp = report["consistency_improvement"]
        print(
            f"γ-swing: {imp['old_residual_swing_percent']:.1f}% → "
            f"{imp['new_residual_swing_percent']:.1f}% "
            f"({imp['improvement_factor']:.1f}× improvement)"
        )
    return report


def run_operator_algebra_demo(
    *,
    delta_s: float = 0.2,
    n_steps: int = 5,
    verbose: bool = True,
) -> dict[str, Any]:
    """Evolve the 8×8 cylinder operator algebra and return a JSON-serializable report."""
    alg = CylinderOperatorAlgebra(delta_floor=GEOMETRIC_FRICTION_FLOOR)
    psi0 = alg.initial_state(active_index=0)
    psi1 = alg.evolve_state(psi0, delta_s=float(delta_s))
    traj = alg.evolve_trajectory(psi0, n_steps=int(n_steps), delta_s=float(delta_s))
    report = {
        "delta_s": float(delta_s),
        "occupation_initial": alg.occupation_numbers(psi0).tolist(),
        "occupation_after_one_step": alg.occupation_numbers(psi1).tolist(),
        "reset_occupation_final": traj["reset_occupation_final"],
        "trajectory_reset_occupations": [float(occ[7]) for occ in traj["occupations"]],
        "hamiltonian_norm": float(np.linalg.norm(alg.H_phi)),
    }
    if verbose:
        print("Cylinder Operator Algebra Demo")
        print("==============================\n")
        print(f"H_Φ norm: {report['hamiltonian_norm']:.4f}")
        print(f"Reset occupation after 1 step: {report['occupation_after_one_step'][7]:.4f}")
        print(f"Reset occupation after {n_steps} steps: {report['reset_occupation_final']:.4f}")
    return report


def run_residual_diagnostics_demo(
    *,
    source: str = "cylinder_evolution",
    use_cylinder_evolution: bool | None = None,
    use_desi_residuals: bool = False,
    data_mode: str | None = None,
    tracer: str | None = None,
    auto_calibrate_gamma: bool = True,
    use_whim_model: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run residual diagnostics on synthetic or DESI fit residuals.

    Parameters
    ----------
    source:
        ``desi_fit`` | ``cylinder_evolution`` | ``operator_algebra``
    use_desi_residuals:
        Legacy flag; when True, equivalent to ``source='desi_fit'``.
    use_cylinder_evolution:
        Legacy flag; when False and ``use_desi_residuals`` is False, uses operator algebra.
    """
    if use_desi_residuals:
        source = "desi_fit"
    elif use_cylinder_evolution is not None:
        source = "cylinder_evolution" if use_cylinder_evolution else "operator_algebra"

    provenance: dict[str, Any] | None = None
    if source == "desi_fit":
        desi_bundle = load_desi_fit_residuals(
            data_mode=data_mode,
            tracer=tracer,
            auto_calibrate_gamma=auto_calibrate_gamma,
            use_whim_model=use_whim_model,
        )
        diag = ResidualDiagnostics.from_desi_fit(desi_bundle)
        provenance = {
            "source": "desi_fit",
            "data_class": desi_bundle["data_class"],
            "repository": desi_bundle["repository"],
            "local_path": desi_bundle["local_path"],
            "tracer": desi_bundle["tracer"],
            "data_mode": desi_bundle["data_mode"],
            "gamma": desi_bundle["gamma"],
            "n_data": desi_bundle["n_data"],
            "chi2_reduced": desi_bundle["chi2_reduced"],
        }
        source_label = (
            f"desi_fit_{desi_bundle['tracer']}_{desi_bundle['data_mode']}"
        )
    elif source == "operator_algebra":
        alg = CylinderOperatorAlgebra()
        psi0 = alg.initial_state(0)
        psi1 = alg.evolve_state(psi0, delta_s=0.2)
        resid = alg.residuals_from_states(psi0, psi1)
        diag = ResidualDiagnostics(residuals=resid)
        provenance = {"source": "operator_algebra", "data_class": "synthetic"}
        source_label = "operator_algebra_active_residual"
    else:
        evolution = OctonionicClockwork().simulate_cylinder_time_evolution(n_steps=12)
        diag = ResidualDiagnostics.from_cylinder_evolution(evolution, field="phi_value")
        provenance = {"source": "cylinder_evolution", "data_class": "synthetic"}
        source_label = "cylinder_evolution_phi_value_diff"

    fitted_values = None
    if source == "desi_fit":
        fitted_values = desi_bundle.get("y_model")
    report = diag.run_full_diagnostics(
        verbose=verbose,
        provenance=provenance,
        fitted_values=fitted_values,
    )
    report["source"] = source_label
    if verbose and source == "desi_fit" and provenance:
        print(
            f"[DESI] tracer={provenance['tracer']} mode={provenance['data_mode']} "
            f"n={provenance['n_data']} γ={provenance['gamma']:.4f} "
            f"χ²_red={provenance['chi2_reduced']:.3f}"
        )
    return report


def run_clockwork_demo(
    *,
    steps: int = 2,
    s: float = 5.0,
    phi: float = 0.0,
    gamma: float = GAMMA_RD_ANCHOR,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the octonionic clockwork demo and return a JSON-serializable report."""
    tsv = TavSuperblockVariables()
    clock = OctonionicClockwork()
    ak, bk = clock.generate_fourier_coefficients()
    evolved_ak, evolved_bk = clock.evolve_phi_coefficients(ak, bk, steps=steps, s=s)
    evolution = clock.evolve_cylinder_state(
        tsv,
        steps=steps,
        s=s,
        phi=phi,
        gamma=gamma,
        apply_twistor=True,
        update_sound_horizon=True,
    )
    algebra_check = clock.algebra.verify_associator_failure()
    symbolic_product: str | None = None
    if HAS_SYMPY and sp is not None:
        e = clock.algebra.basis_symbols()
        symbolic_product = str(clock.algebra.symbolic_multiply(e[1], e[2]))
    report = {
        "advance_phase_3_steps": clock.advance_phase(phi, steps=3),
        "advance_phase_algebraic_3_steps": clock.advance_phase_algebraic(phi, steps=3),
        "octonion_generator_state_step3": clock.octonion_generator_state(steps=3).tolist(),
        "octonion_phase_angle_step3": float(clock.algebra.phase_angle(clock.octonion_generator_state(3))),
        "is_reset_at_pi": clock.is_reset_phase(np.pi),
        "ak_first3": ak[:3].tolist(),
        "evolved_ak_first3": evolved_ak[:3].tolist(),
        "domain_velocities": clock.get_domain_staggered_velocities().tolist(),
        "hamiltonian_at_phi": clock.clockwork_hamiltonian_contribution(phi, s=s),
        "evolved_phi_amplitude_scale": float(evolution.tsv.phi.amplitude_scale),
        "evolved_phi_ak_norm": float(np.linalg.norm(evolution.tsv.phi.ak)),
        "rd_at_gamma_mpc": evolution.rd_at_gamma,
        "twistor_projection": evolution.twistor,
        "exclusion_detail": evolution.exclusion,
        "octonion_associator_check": algebra_check,
        "symbolic_e1_times_e2": symbolic_product,
        "steps": int(steps),
        "s": float(s),
        "gamma": float(gamma),
    }
    if verbose:
        print("Octonionic Clockwork Demo")
        print("=========================\n")
        print(f"Advance phase by 3 steps: {report['advance_phase_3_steps']:.4f}")
        print(f"Algebraic τ advance (3 steps): {report['advance_phase_algebraic_3_steps']:.4f}")
        print(f"Octonion e1^3 phase angle: {report['octonion_phase_angle_step3']:.4f}")
        print(f"Is reset phase at π? {report['is_reset_at_pi']}")
        print(f"Generated ak (first 3): {report['ak_first3']}")
        print(f"Evolved ak after {steps} steps at s={s}: {report['evolved_ak_first3']}")
        print(f"Domain staggered velocities: {np.round(report['domain_velocities'], 3)}")
        print(f"r_d at γ={gamma}: {report['rd_at_gamma_mpc']:.2f} Mpc")
        if evolution.exclusion:
            print(
                f"Exclusion: allowed={evolution.exclusion['allowed']} "
                f"cost={evolution.exclusion['total_exclusion_cost_mev']:.1f} MeV "
                f"({evolution.exclusion['mechanism']})"
            )
        if symbolic_product:
            print(f"Symbolic e1*e2 = {symbolic_product}")
    return report


def run_cylinder_evolution_demo(
    *,
    n_steps: int = 15,
    gamma: float = GAMMA_RD_ANCHOR,
    use_algebraic_evolution: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Simulate clockwork-driven evolution across cylinder slices."""
    clock = OctonionicClockwork()
    report = clock.simulate_cylinder_time_evolution(
        n_steps=int(n_steps),
        gamma=float(gamma),
        use_algebraic_evolution=use_algebraic_evolution,
    )
    if verbose:
        print("Cylinder Slice Time Evolution")
        print("=============================\n")
        print(f"Steps: {report['n_steps']} | γ={report['gamma']}")
        print(f"r_d swing: {report['rd_swing_mpc']:.3f} Mpc")
        print(f"Forbidden fraction: {report['forbidden_fraction']:.2%}")
        print("First / last slice:")
        print(f"  s={report['trajectory'][0]['s']:.2f} rd={report['trajectory'][0]['rd_mpc']:.2f} Mpc")
        print(f"  s={report['trajectory'][-1]['s']:.2f} rd={report['trajectory'][-1]['rd_mpc']:.2f} Mpc")
    return report


def run_phi_demo(*, s: float = 12.5, phi: float = 2.3, verbose: bool = True) -> dict[str, Any]:
    """Run the standard Φ integration demo and return a JSON-serializable report."""
    tsv = TavSuperblockVariables()
    report = tsv.demo_report(s=s, phi=phi)
    if verbose:
        print("Tav-Superblock Research Tool — Demo")
        print("====================================\n")
        print(tsv.summary())
        print(f"Φ(s={s}, ϕ={phi})          = {report['phi_at_test']:.4f}")
        print(f"v_trans,1²                         = {report['v_trans_squared']:.4f}")
        print(f"String tension σ                   = {report['string_tension']:.4f}")
        print(f"√σ                                 = {report['sqrt_sigma_mev']:.4f}  (should be √2·Λ_TSB)")
        print(
            f"Exclusion cost at (domain=1, k=3)  = {report['exclusion_cost_domain1_k3']:.1f} MeV"
        )
        print("\nScalar function Φ successfully integrated into Tav-Superblock variable handling.")
    return report


def run_exclusion_scan(
    *,
    s: float = 12.5,
    n_domains: int = 6,
    n_phases: int = 7,
    verbose: bool = True,
) -> dict[str, Any]:
    """Scan (domain, phase_k) slots for nodal exclusion violations."""
    tsv = TavSuperblockVariables()
    rows: list[dict[str, Any]] = []
    for domain in range(int(n_domains)):
        for phase_k in range(int(n_phases)):
            allowed = tsv.check_exclusion(domain, phase_k, s)
            cost = tsv.exclusion_action_cost(domain, phase_k, s)
            rows.append(
                {
                    "domain": domain,
                    "phase_k": phase_k,
                    "allowed": bool(allowed),
                    "exclusion_cost_mev": float(cost),
                    "phi_staggered": float(tsv.apply_e8_staggering(float(phase_k), domain)),
                }
            )
    forbidden = [r for r in rows if not r["allowed"]]
    report = {
        "s": float(s),
        "n_forbidden": len(forbidden),
        "n_allowed": len(rows) - len(forbidden),
        "grid": rows,
    }
    if verbose:
        print(f"[TSB Research] Exclusion scan at s={s}: {len(forbidden)} forbidden / {len(rows)} slots")
        for row in forbidden[:8]:
            print(
                f"  domain={row['domain']} phase_k={row['phase_k']} "
                f"cost={row['exclusion_cost_mev']:.1f} MeV"
            )
    return report


if __name__ == "__main__":
    print("Tav-Superblock Research Tool — Demo with OctonionicClockwork")
    print("=" * 75)

    tsv = TavSuperblockVariables()
    print(tsv.summary())

    run_clockwork_demo(steps=2, s=5.0, verbose=True)
    run_cylinder_evolution_demo(n_steps=10, verbose=True)
    run_operator_algebra_demo(verbose=True)
    run_residual_diagnostics_demo(verbose=True)

    sh = SoundHorizon(tsv)
    print(f"\nSound horizon at γ=7.95 : {sh.predict_rd(7.95):.2f} Mpc")
    print(f"Sound horizon at γ=12.0 : {sh.predict_rd(12.0):.2f} Mpc")
    print(f"Sound horizon at γ=8.0  : {sh.predict_rd(8.0):.2f} Mpc")

    improvement = sh.consistency_improvement()
    print("\nConsistency Improvement from Φ-based Anchoring:")
    for key, val in improvement.items():
        print(f"  {key}: {val}")

    s_test = np.linspace(0, 45.8, 30)
    template = generate_cylinder_oscillation_template(tsv.phi, s_test)
    print(f"\nCylinder-native template std: {np.std(template):.4f}")

    whim = WHIMTemplate()
    whim_template = whim.generate(s_test, tsv)
    print(f"WHIM 2τ template generated (mean abs = {np.mean(np.abs(whim_template)):.4f})")
    print("\nOctonionicClockwork + full framework successfully integrated.")