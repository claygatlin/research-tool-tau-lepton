"""
1D Fractal-Conformal Tau Circle — theoretical model and log-likelihood.

Maps fractal_level (n_hier scale) to spectral-dimension flow, computes emergent
rotation-sector energy from winding density under TEP / 313.1 MeV floor constraints,
and provides a Gaussian log-likelihood for precision testing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from menus.tsb_research.core import LAMBDA_TSB, N_HIER, TAU_PERIOD
from tav_shared.dataset_comparison.pipeline import compare_model_predictions_to_observed
from tav_shared.ingestion import get_runtime_data_manager
from tav_shared.ingestion.pipeline import validate_fractal_tau_for_likelihood

# CDT reference: ds ≈ 2 at deep quantum scales (UV), flows toward 4 (IR).
CDT_TARGET_DS_UV: float = 2.0
CDT_SIGMA_DS_DEFAULT: float = 0.15
MASS_GAP_SIGMA_MEV_DEFAULT: float = 5.0
SIGMA_QCD_DEFAULT: float = 5.0

# Torsion-enriched Dirac operator on the circle (Yang-Mills / E₈ generation factor).
TORSION_GAMMA: float = 1.0607
N_DOM_DEFAULT: float = 3.0
MASS_GAP_FLOOR_MEV: float = LAMBDA_TSB

# Calibrate torsion_factor so m_tors·exp(−γ·N_dom) = 313.1 MeV at N_dom = 3.
TORSION_FACTOR_DEFAULT: float = float(
    MASS_GAP_FLOOR_MEV * np.exp(TORSION_GAMMA * N_DOM_DEFAULT)
)

# 142857 reptend cycle (TEP bosonic closure).
TEP_CYCLE: int = 142857

PARAM_KEYS: tuple[str, ...] = ("winding_density", "fractal_level", "phase_slip_alpha")
DEFAULT_PARAMS: dict[str, float] = {
    "winding_density": 1.0,
    "fractal_level": float(N_HIER),
    "phase_slip_alpha": 0.33,
}


@dataclass
class FractalTauObserved:
    """Observed anchors for likelihood evaluation."""

    mass_gap_mev: float = LAMBDA_TSB
    sigma_mass_gap_mev: float = MASS_GAP_SIGMA_MEV_DEFAULT
    sigma_qcd_mev: float = SIGMA_QCD_DEFAULT
    spectral_dim: float = CDT_TARGET_DS_UV
    sigma_spectral_dim: float = CDT_SIGMA_DS_DEFAULT
    n_hier: float = float(N_HIER)
    torsion_factor: float = TORSION_FACTOR_DEFAULT
    likelihood_mode: str = "dual"  # "dual" | "tep_ansatz"

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def normalize_params(
    params: Mapping[str, float] | Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Accept dict or length-3 vector [winding_density, fractal_level, phase_slip_alpha]."""
    if isinstance(params, Mapping):
        return {
            "winding_density": float(params.get("winding_density", DEFAULT_PARAMS["winding_density"])),
            "fractal_level": float(params.get("fractal_level", DEFAULT_PARAMS["fractal_level"])),
            "phase_slip_alpha": float(
                params.get("phase_slip_alpha", DEFAULT_PARAMS["phase_slip_alpha"])
            ),
        }
    arr = np.asarray(params, dtype=float).ravel()
    if arr.size != 3:
        raise ValueError(
            f"params vector must have length 3 ({PARAM_KEYS}); got {arr.size}"
        )
    return {
        "winding_density": float(arr[0]),
        "fractal_level": float(arr[1]),
        "phase_slip_alpha": float(arr[2]),
    }


def compute_spectral_dimension(
    fractal_level: float,
    *,
    n_hier: float = N_HIER,
) -> float:
    """
    Map fractal scaling (n_hier ladder position) to spectral dimension flow d_s.

    CDT-inspired UV→IR flow: d_s → 2 at deep fractal levels, asymptotes to 4.
    """
    level = max(float(fractal_level), 0.0)
    scale = level / max(float(n_hier), 1e-9)
    return float(2.0 + 2.0 * (1.0 - np.exp(-scale)))


def predicted_mass_gap_torsion_ansatz(
    fractal_level: float,
    *,
    torsion_factor: float = TORSION_FACTOR_DEFAULT,
    gamma: float = TORSION_GAMMA,
) -> float:
    """
    TSB Unification Ansatz: torsion-enriched Dirac operator on the circle.

    m_gap = m_tors · exp(−γ · fractal_level),  γ ≈ 1.0607 (E₈ generation factor).
    With fractal_level = N_dom ≈ 3 and calibrated m_tors, anchors to 313.1 MeV.
    """
    level = float(fractal_level)
    return float(torsion_factor * np.exp(-float(gamma) * level))


def satisfies_tep_142857_cycle_closure(
    params: Mapping[str, float] | Sequence[float] | np.ndarray,
    *,
    n_hier: float = N_HIER,
) -> bool:
    """
    142857 reptend / domain-occupancy closure on the tau circle (TEP bosonic filter).

    Rejects configurations whose composite occupancy collapses to the forbidden
    1/7 reset seam or leaves domain slots outside [1, 6].
    """
    p = normalize_params(params)
    wind_int = int(round(p["winding_density"]))
    if wind_int < 0:
        return False

    fractal = float(p["fractal_level"])
    if fractal <= 10.0:
        domain_slots = int(round(fractal))
    else:
        domain_slots = int(round(fractal / max(float(n_hier), 1e-9) * 6.0))
    if domain_slots < 1 or domain_slots > 6:
        return False

    fractal_micro = int(round(abs(fractal) * 1_000))
    slip_micro = int(round(abs(p["phase_slip_alpha"]) * 1_000_000))
    occupancy = (wind_int * 142_857 + fractal_micro + slip_micro) % TEP_CYCLE
    if occupancy == 0:
        return False

    seam_residue = (occupancy * domain_slots) % 7
    if seam_residue == 0 and wind_int > 0:
        return False
    return True


def tep_closure_diagnostics(
    params: Mapping[str, float] | Sequence[float] | np.ndarray,
    *,
    n_hier: float = N_HIER,
) -> dict[str, Any]:
    """Structured TEP / 142857 / domain diagnostics for MCMC reports."""
    p = normalize_params(params)
    wind_int = int(round(p["winding_density"]))
    fractal = float(p["fractal_level"])
    if fractal <= 10.0:
        domain_slots = int(round(fractal))
    else:
        domain_slots = int(round(fractal / max(float(n_hier), 1e-9) * 6.0))
    fractal_micro = int(round(abs(fractal) * 1_000))
    slip_micro = int(round(abs(p["phase_slip_alpha"]) * 1_000_000))
    occupancy = (wind_int * 142_857 + fractal_micro + slip_micro) % TEP_CYCLE
    return {
        "tep_integer_closure": satisfies_tep_integer_closure(p, n_hier=n_hier),
        "tep_142857_cycle_closure": satisfies_tep_142857_cycle_closure(p, n_hier=n_hier),
        "domain_slots": domain_slots,
        "occupancy_mod_tep_cycle": int(occupancy),
        "seam_residue_mod7": int((occupancy * max(domain_slots, 1)) % 7),
        "winding_integer": wind_int,
        "phase_slip_alpha": float(p["phase_slip_alpha"]),
        "fractal_level": fractal,
    }


def satisfies_tep_integer_closure(
    params: Mapping[str, float] | Sequence[float] | np.ndarray,
    *,
    n_hier: float = N_HIER,
    tau_period: float = TAU_PERIOD,
    winding_tol: float = 0.05,
    slip_max: float | None = None,
) -> bool:
    """
    TEP hard constraint: configuration must close without forbidden reset-phase overlap.

    Fails (→ log L = −∞) when:
    - phase_slip exceeds the 1/7 reset seam budget,
    - winding_density is not integer-closed on S¹_τ,
    - hierarchical domain binding or 142857-cycle proxy does not close.
    """
    p = normalize_params(params)
    wind = p["winding_density"]
    slip = p["phase_slip_alpha"]
    fractal = p["fractal_level"]
    slip_limit = float(tau_period / 7.0) if slip_max is None else float(slip_max)

    if slip > slip_limit:
        return False
    if abs(wind - round(wind)) > winding_tol:
        return False

    active_phase = fractal % tau_period
    if abs(active_phase - round(active_phase)) > 0.25:
        return False

    # fractal_level ≤ 10: interpret as domain depth N_dom (Yang-Mills anchor).
    if fractal <= 10.0:
        n_dom = fractal
        if abs(n_dom - round(n_dom)) > 0.1 or n_dom < 1.0 or n_dom > 6.0:
            return False
    else:
        domain_binding = fractal / max(float(n_hier), 1e-9) * 6.0
        if abs(domain_binding - round(domain_binding)) > 0.15:
            return False

    # Forbidden reset-phase overlap (phase index 7 ≡ r mod active slots).
    wind_int = int(round(wind))
    phase_int = int(round(active_phase)) % 7
    if wind_int > 0 and (wind_int * (phase_int + 1)) % 7 == 0:
        return False

    if not satisfies_tep_142857_cycle_closure(p, n_hier=n_hier):
        return False

    return True


def calculate_rotation_dynamics(
    winding_density: float,
    phase_slip_alpha: float,
    *,
    m0_mev: float = LAMBDA_TSB,
) -> float:
    """
    Emergent rotation-sector energy from winding density under TEP constraints.

    The 313.1 MeV geometric friction floor is the minimal violation cost; coherent
    winding deposition is attenuated by phase-slip α on the 1D tau circle.
    """
    wind = float(winding_density)
    slip = max(float(phase_slip_alpha), 0.0)
    coherent = wind * float(np.exp(-slip / max(float(TAU_PERIOD), 1e-9)))
    return float(m0_mev * (1.0 + 0.15 * coherent))


def theoretical_model(
    params: Mapping[str, float] | Sequence[float] | np.ndarray,
    *,
    n_hier: float = N_HIER,
    torsion_factor: float = TORSION_FACTOR_DEFAULT,
    mode: str = "dual",
) -> dict[str, float]:
    """
    Compute predictions from 1D Fractal-Conformal Tau Circle geometry.

    params: winding_density, fractal_level, phase_slip_alpha
    mode: "dual" (rotation + spectral_dim) | "tep_ansatz" (torsion Dirac ansatz)
    """
    p = normalize_params(params)
    ds = compute_spectral_dimension(p["fractal_level"], n_hier=n_hier)
    rotation_energy = calculate_rotation_dynamics(
        p["winding_density"],
        p["phase_slip_alpha"],
    )
    torsion_gap = predicted_mass_gap_torsion_ansatz(
        p["fractal_level"],
        torsion_factor=torsion_factor,
    )
    tep_ok = satisfies_tep_integer_closure(p, n_hier=n_hier)
    mass_gap = torsion_gap if mode == "tep_ansatz" else rotation_energy
    return {
        "mass_gap": mass_gap,
        "mass_gap_mev": mass_gap,
        "mass_gap_rotation_mev": rotation_energy,
        "mass_gap_torsion_ansatz_mev": torsion_gap,
        "spectral_dim": ds,
        "winding_density": p["winding_density"],
        "fractal_level": p["fractal_level"],
        "phase_slip_alpha": p["phase_slip_alpha"],
        "tep_integer_closure": bool(tep_ok),
        "m0_anchor_mev": float(MASS_GAP_FLOOR_MEV),
        "torsion_factor": float(torsion_factor),
        "torsion_gamma": float(TORSION_GAMMA),
        "n_hier": float(n_hier),
        "mode": str(mode),
    }


def log_likelihood_tep_ansatz(
    params: Mapping[str, float] | Sequence[float] | np.ndarray,
    observed_data: Mapping[str, float] | FractalTauObserved | None = None,
) -> float:
    """
    TEP hard-constraint likelihood (torsion Dirac ansatz).

    predicted_gap = torsion_factor · exp(−1.0607 · fractal_level)
    313.1 MeV treated as asymptotic safety fixed point.
    Returns −∞ if TEP integer closure fails.
    """
    obs_defaults = FractalTauObserved(likelihood_mode="tep_ansatz").as_dict()
    if isinstance(observed_data, FractalTauObserved):
        obs_defaults.update(observed_data.as_dict())
    elif observed_data:
        obs_defaults.update(dict(observed_data))
    v_params, v_obs, _report = validate_fractal_tau_for_likelihood(
        params,
        obs_defaults,
        observed_defaults=obs_defaults,
        strict=True,
        data_manager=get_runtime_data_manager(),
    )
    params = v_params
    obs = FractalTauObserved(**{**obs_defaults, **v_obs})
    if not satisfies_tep_integer_closure(params, n_hier=obs.n_hier):
        return float(-np.inf)

    p = normalize_params(params)
    predicted_gap = predicted_mass_gap_torsion_ansatz(
        p["fractal_level"],
        torsion_factor=obs.torsion_factor,
    )
    sigma_qcd = max(float(obs.sigma_qcd_mev), 1e-9)
    gap_diff = predicted_gap - float(obs.mass_gap_mev)
    return float(-0.5 * (gap_diff / sigma_qcd) ** 2)


def log_likelihood(
    params: Mapping[str, float] | Sequence[float] | np.ndarray,
    observed_data: Mapping[str, float] | FractalTauObserved | None = None,
) -> float:
    """
    Gaussian log-likelihood.

    mode="tep_ansatz": TEP hard constraint + torsion Dirac mass gap only.
    mode="dual" (default): 313.1 MeV floor + CDT spectral-dimension flow.
    """
    obs_defaults = FractalTauObserved().as_dict()
    if isinstance(observed_data, FractalTauObserved):
        obs_defaults.update(observed_data.as_dict())
    elif observed_data:
        obs_defaults.update(dict(observed_data))
    v_params, v_obs, _report = validate_fractal_tau_for_likelihood(
        params,
        obs_defaults,
        observed_defaults=obs_defaults,
        strict=True,
        data_manager=get_runtime_data_manager(),
    )
    params = v_params
    obs = FractalTauObserved(**{**obs_defaults, **v_obs})
    if str(obs.likelihood_mode).lower() == "tep_ansatz":
        return log_likelihood_tep_ansatz(params, obs)

    model_out = theoretical_model(params, n_hier=obs.n_hier, mode="dual")

    sigma_gap = max(float(obs.sigma_mass_gap_mev), 1e-9)
    sigma_ds = max(float(obs.sigma_spectral_dim), 1e-9)

    chi2_gap = ((model_out["mass_gap_mev"] - obs.mass_gap_mev) / sigma_gap) ** 2
    chi2_dim = ((model_out["spectral_dim"] - obs.spectral_dim) / sigma_ds) ** 2

    return float(-0.5 * (chi2_gap + chi2_dim))


def log_likelihood_breakdown(
    params: Mapping[str, float] | Sequence[float] | np.ndarray,
    observed_data: Mapping[str, float] | FractalTauObserved | None = None,
) -> dict[str, Any]:
    """Full likelihood decomposition for reports and MCMC hooks."""
    obs = (
        observed_data
        if isinstance(observed_data, FractalTauObserved)
        else FractalTauObserved(**dict(observed_data or {}))
    )
    tep_ok = satisfies_tep_integer_closure(params, n_hier=obs.n_hier)
    mode = str(obs.likelihood_mode).lower()

    if mode == "tep_ansatz":
        p = normalize_params(params)
        predicted = predicted_mass_gap_torsion_ansatz(
            p["fractal_level"],
            torsion_factor=obs.torsion_factor,
        )
        model_out = theoretical_model(
            params,
            n_hier=obs.n_hier,
            torsion_factor=obs.torsion_factor,
            mode="tep_ansatz",
        )
        sigma_qcd = max(float(obs.sigma_qcd_mev), 1e-9)
        gap_diff = float(predicted - obs.mass_gap_mev)
        chi2_gap = float((gap_diff / sigma_qcd) ** 2)
        ll = float(-np.inf) if not tep_ok else float(-0.5 * chi2_gap)
        return {
            "log_likelihood": ll,
            "likelihood_mode": "tep_ansatz",
            "tep_integer_closure": tep_ok,
            "chi2_mass_gap": chi2_gap,
            "chi2_spectral_dim": None,
            "chi2_total": chi2_gap if tep_ok else None,
            "model": model_out,
            "observed": obs.as_dict(),
            "residual_mass_gap_mev": gap_diff,
            "residual_spectral_dim": None,
            "predicted_gap_mev": predicted,
            "model_vs_observed_tolerance": compare_model_predictions_to_observed(
                model_out, obs.as_dict()
            ),
            "torsion_ansatz": (
                f"m_tors·exp(−{TORSION_GAMMA}·fractal_level), "
                f"m_tors={obs.torsion_factor:.4g}"
            ),
        }

    model_out = theoretical_model(params, n_hier=obs.n_hier, mode="dual")
    sigma_gap = max(float(obs.sigma_mass_gap_mev), 1e-9)
    sigma_ds = max(float(obs.sigma_spectral_dim), 1e-9)
    chi2_gap = float(((model_out["mass_gap_mev"] - obs.mass_gap_mev) / sigma_gap) ** 2)
    chi2_dim = float(((model_out["spectral_dim"] - obs.spectral_dim) / sigma_ds) ** 2)
    ll = float(-0.5 * (chi2_gap + chi2_dim))
    return {
        "log_likelihood": ll,
        "likelihood_mode": "dual",
        "tep_integer_closure": tep_ok,
        "chi2_mass_gap": chi2_gap,
        "chi2_spectral_dim": chi2_dim,
        "chi2_total": chi2_gap + chi2_dim,
        "model": model_out,
        "observed": obs.as_dict(),
        "residual_mass_gap_mev": float(model_out["mass_gap_mev"] - obs.mass_gap_mev),
        "residual_spectral_dim": float(model_out["spectral_dim"] - obs.spectral_dim),
        "mass_gap_torsion_ansatz_mev": model_out.get("mass_gap_torsion_ansatz_mev"),
        "model_vs_observed_tolerance": compare_model_predictions_to_observed(
            model_out, obs.as_dict()
        ),
    }


def scan_parameter_grid(
    *,
    winding_grid: np.ndarray | None = None,
    fractal_grid: np.ndarray | None = None,
    slip_grid: np.ndarray | None = None,
    observed_data: Mapping[str, float] | FractalTauObserved | None = None,
) -> dict[str, Any]:
    """Coarse 3D grid search for best log-likelihood (no MCMC dependency)."""
    obs = (
        observed_data
        if isinstance(observed_data, FractalTauObserved)
        else FractalTauObserved(**dict(observed_data or {}))
    )
    w_grid = winding_grid if winding_grid is not None else np.linspace(0.2, 2.0, 9)
    if str(obs.likelihood_mode).lower() == "tep_ansatz":
        f_grid = fractal_grid if fractal_grid is not None else np.linspace(2.0, 5.0, 7)
    else:
        f_grid = fractal_grid if fractal_grid is not None else np.linspace(30.0, 55.0, 6)
    s_grid = slip_grid if slip_grid is not None else np.linspace(0.0, 1.0, 5)

    best_ll = -np.inf
    best_params: dict[str, float] = {}
    best_breakdown: dict[str, Any] = {}
    n_eval = 0
    for w in w_grid:
        for f in f_grid:
            for s in s_grid:
                params = {
                    "winding_density": float(w),
                    "fractal_level": float(f),
                    "phase_slip_alpha": float(s),
                }
                breakdown = log_likelihood_breakdown(params, obs)
                n_eval += 1
                ll = breakdown["log_likelihood"]
                if ll > best_ll:
                    best_ll = ll
                    best_params = params
                    best_breakdown = breakdown

    return {
        "best_log_likelihood": float(best_ll),
        "best_params": best_params,
        "best_breakdown": best_breakdown,
        "n_evaluations": n_eval,
        "grid_shapes": {
            "winding_density": int(len(w_grid)),
            "fractal_level": int(len(f_grid)),
            "phase_slip_alpha": int(len(s_grid)),
        },
    }


def run_fractal_tau_likelihood(
    *,
    params: Mapping[str, float] | None = None,
    observed_data: Mapping[str, float] | None = None,
    do_grid_scan: bool = True,
    do_mcmc: bool = False,
    mcmc_walkers: int = 32,
    mcmc_steps: int = 1000,
    morris_trajectories: int = 24,
    morris_freeze_threshold: float = 0.05,
    empirical_provenance: dict[str, Any] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Evaluate model + likelihood; optional grid scan for best fit."""
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update({k: float(v) for k, v in params.items() if k in PARAM_KEYS})

    obs_kwargs = dict(observed_data or {})
    breakdown = log_likelihood_breakdown(p, obs_kwargs)
    report: dict[str, Any] = {
        "action": "Fractal-Conformal Tau Circle Likelihood",
        "dataset_provenance": {
            "pipeline": "theory_anchors + empirical_targets",
            "note": (
                "Lattice/PDG empirical caches (glueball, neutron lifetime) pulled pre-run; "
                "likelihood uses preregistered 313.1 MeV / CDT anchors."
            ),
            "empirical_provenance": empirical_provenance,
        },
        "params": p,
        "likelihood": breakdown,
        "equations": {
            "spectral_dim": "d_s = 2 + 2(1 - exp(-fractal_level / n_hier))",
            "mass_gap_rotation": "E = m₀(1 + 0.15·ρ_wind·exp(-α_slip/τ_period))",
            "mass_gap_torsion": f"m_tors·exp(−{TORSION_GAMMA}·fractal_level)",
            "log_likelihood_dual": "-0.5·[(Δm/m_σ)² + (Δd_s/σ_ds)²]",
            "log_likelihood_tep": "TEP closure required; else −∞; −0.5·(Δm/σ_qcd)²",
        },
        "anchors": {
            "m0_mev": LAMBDA_TSB,
            "n_hier": float(N_HIER),
            "tau_period": float(TAU_PERIOD),
            "cdt_target_ds_uv": CDT_TARGET_DS_UV,
        },
    }

    if do_grid_scan:
        report["grid_scan"] = scan_parameter_grid(observed_data=obs_kwargs)

    if do_mcmc:
        from menus.tsb_research.fractal_tau_mcmc import run_fractal_tau_mcmc

        report["mcmc"] = run_fractal_tau_mcmc(
            observed_data=obs_kwargs,
            n_walkers=int(mcmc_walkers),
            n_steps=int(mcmc_steps),
            n_trajectories=int(morris_trajectories),
            freeze_threshold=float(morris_freeze_threshold),
            run_morris=True,
            verbose=verbose,
        )

    if verbose:
        m = breakdown["model"]
        print("\n=== FRACTAL-CONFORMAL TAU CIRCLE LIKELIHOOD ===")
        print(f"  mode            : {breakdown.get('likelihood_mode', 'dual')}")
        print(f"  TEP closure     : {breakdown.get('tep_integer_closure')}")
        ll = breakdown["log_likelihood"]
        print(f"  log L           : {'−∞' if not np.isfinite(ll) else f'{ll:.4f}'}")
        print(f"  mass_gap (pred) : {m['mass_gap_mev']:.3f} MeV")
        if m.get("mass_gap_torsion_ansatz_mev") is not None:
            print(f"  torsion ansatz  : {m['mass_gap_torsion_ansatz_mev']:.3f} MeV")
        if breakdown.get("chi2_spectral_dim") is not None:
            print(f"  spectral_dim    : {m['spectral_dim']:.4f}")
            print(
                f"  χ² gap / dim    : {breakdown['chi2_mass_gap']:.3f} / "
                f"{breakdown['chi2_spectral_dim']:.3f}"
            )
        else:
            print(f"  χ² gap (QCD)    : {breakdown['chi2_mass_gap']:.3f}")
        if do_grid_scan and report.get("grid_scan"):
            gs = report["grid_scan"]
            bp = gs["best_params"]
            print(f"  Grid best log L : {gs['best_log_likelihood']:.4f}")
            print(
                f"  Grid best params: ρ_wind={bp['winding_density']:.3f}, "
                f"fractal={bp['fractal_level']:.2f}, α_slip={bp['phase_slip_alpha']:.3f}"
            )
        print("=" * 46)

    return report