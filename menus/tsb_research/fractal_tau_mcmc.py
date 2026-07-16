"""
Fractal tau circle MCMC with TEP-hard log_prior and Morris sensitivity calibration.

Validation is enforced as a hard constraint in the prior so emcee walkers
never occupy forbidden 142857 / domain-occupancy regions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np

from menus.tsb_research.fractal_tau_circle import (
    DEFAULT_PARAMS,
    FractalTauObserved,
    LAMBDA_TSB,
    MASS_GAP_FLOOR_MEV,
    PARAM_KEYS,
    log_likelihood,
    satisfies_tep_142857_cycle_closure,
    satisfies_tep_integer_closure,
    tep_closure_diagnostics,
    theoretical_model,
)
from menus.tsb_research.sensitivity import morris_screen, sensitivity_driven_param_plan
from tav_shared.artifact_paths import TestSlug, artifact_path, artifact_timestamp
from tav_shared.ingestion.config import load_ingestion_bounds
from tav_shared.ingestion.normalize import normalize_fractal_params
from tav_shared.ingestion.schemas import FractalTauParamsSchema

_NEG_INF = float(-np.inf)


def _mode_default_params(likelihood_mode: str) -> dict[str, float]:
    base = dict(DEFAULT_PARAMS)
    if str(likelihood_mode).lower() == "tep_ansatz":
        base["fractal_level"] = 3.0
        base["phase_slip_alpha"] = 0.0
    return base


def _default_param_bounds(
    *,
    likelihood_mode: str = "dual",
) -> dict[str, tuple[float, float]]:
    b = load_ingestion_bounds().get("fractal_tau", {})
    mode = str(likelihood_mode).lower()
    f_lo, f_hi = tuple(b.get("fractal_level", [0.0, 120.0]))
    if mode == "tep_ansatz":
        f_lo, f_hi = 1.0, 6.0
    return {
        "winding_density": tuple(b.get("winding_density", [0.0, 10.0])),
        "fractal_level": (float(f_lo), float(f_hi)),
        "phase_slip_alpha": tuple(b.get("phase_slip_alpha", [0.0, 7.0])),
    }


def _vector_to_params(
    theta: np.ndarray,
    active_names: Sequence[str],
    frozen: Mapping[str, float],
) -> dict[str, float]:
    out = {k: float(frozen[k]) for k in frozen}
    for i, name in enumerate(active_names):
        out[name] = float(theta[i])
    return out


def log_prior_fractal_tau(
    params: Mapping[str, float] | Sequence[float] | np.ndarray,
    *,
    active_bounds: Mapping[str, tuple[float, float]] | None = None,
    n_hier: float | None = None,
    require_tep: bool = True,
    validate_ingestion: bool = True,
) -> float:
    """
    Hard prior: −∞ outside physical bounds or TEP / 142857 / domain closure.

    TEP violations are rejected here (not softened) so MCMC discards them immediately.
    """
    if isinstance(params, np.ndarray):
        raise TypeError("log_prior_fractal_tau expects a parameter dict, not a raw vector")
    p = dict(params)
    bounds = dict(active_bounds or _default_param_bounds())
    n_hier_val = float(n_hier if n_hier is not None else DEFAULT_PARAMS.get("fractal_level", 45.8))

    for key, (lo, hi) in bounds.items():
        if key not in p:
            return _NEG_INF
        val = float(p[key])
        if not np.isfinite(val) or val < float(lo) or val > float(hi):
            return _NEG_INF

    if require_tep:
        if not satisfies_tep_integer_closure(p, n_hier=n_hier_val):
            return _NEG_INF
        if not satisfies_tep_142857_cycle_closure(p, n_hier=n_hier_val):
            return _NEG_INF

    if validate_ingestion:
        try:
            FractalTauParamsSchema.model_validate(normalize_fractal_params(p))
        except (TypeError, ValueError):
            return _NEG_INF

    return 0.0


def log_likelihood_fractal_tau(
    params: Mapping[str, float],
    observed: FractalTauObserved | Mapping[str, float],
) -> float:
    """Likelihood with ingestion validation; TEP also enforced in prior."""
    obs = (
        observed
        if isinstance(observed, FractalTauObserved)
        else FractalTauObserved(**dict(observed))
    )
    return float(log_likelihood(params, obs))


def log_probability_fractal_tau(
    theta: np.ndarray,
    *,
    active_names: Sequence[str],
    frozen: Mapping[str, float],
    observed: FractalTauObserved,
    active_bounds: Mapping[str, tuple[float, float]],
) -> float:
    params = _vector_to_params(theta, active_names, frozen)
    lp = log_prior_fractal_tau(
        params,
        active_bounds=active_bounds,
        n_hier=observed.n_hier,
        require_tep=True,
    )
    if not np.isfinite(lp):
        return _NEG_INF
    return lp + log_likelihood_fractal_tau(params, observed)


def _mass_gap_floor_objective(
    params: Mapping[str, float],
    *,
    observed: FractalTauObserved,
    target_mev: float = MASS_GAP_FLOOR_MEV,
) -> float:
    """Morris target: deviation from 313.1 MeV floor."""
    mode = str(observed.likelihood_mode).lower()
    if not satisfies_tep_integer_closure(params, n_hier=observed.n_hier):
        return 1.0e6
    if not satisfies_tep_142857_cycle_closure(params, n_hier=observed.n_hier):
        return 1.0e6
    out = theoretical_model(
        params,
        n_hier=observed.n_hier,
        torsion_factor=observed.torsion_factor,
        mode="tep_ansatz" if mode == "tep_ansatz" else "dual",
    )
    gap = float(out.get("mass_gap_mev", out.get("mass_gap", 0.0)))
    return abs(gap - float(target_mev))


def run_morris_mass_gap_screen(
    observed: FractalTauObserved | Mapping[str, float] | None = None,
    *,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    n_trajectories: int = 24,
    freeze_threshold: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Morris screening for parameters that move the 313.1 MeV floor."""
    obs = (
        observed
        if isinstance(observed, FractalTauObserved)
        else FractalTauObserved(**dict(observed or {}))
    )
    b = dict(bounds or _default_param_bounds(likelihood_mode=obs.likelihood_mode))
    morris = morris_screen(
        PARAM_KEYS,
        b,
        lambda p: _mass_gap_floor_objective(p, observed=obs),
        n_trajectories=n_trajectories,
        seed=seed,
    )
    defaults = _mode_default_params(obs.likelihood_mode)
    plan = sensitivity_driven_param_plan(
        morris,
        default_params=defaults,
        bounds=b,
        freeze_threshold=freeze_threshold,
    )
    return {
        "target_mev": float(MASS_GAP_FLOOR_MEV),
        "morris": morris,
        "calibration_plan": plan,
    }


def run_fractal_tau_mcmc(
    *,
    observed_data: Mapping[str, float] | None = None,
    n_walkers: int = 32,
    n_steps: int = 1000,
    burn_in: int | None = None,
    seed: int = 42,
    n_trajectories: int = 24,
    freeze_threshold: float = 0.05,
    run_morris: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    emcee MCMC with TEP-hard prior and optional Morris-driven parameter freezing.
    """
    try:
        import emcee
    except ImportError as exc:
        raise ImportError("Install emcee: pip install emcee") from exc

    obs = FractalTauObserved(**dict(observed_data or {}))
    bounds = _default_param_bounds(likelihood_mode=obs.likelihood_mode)
    defaults = _mode_default_params(obs.likelihood_mode)
    sensitivity: dict[str, Any] | None = None
    frozen: dict[str, float] = {}
    active_names = list(PARAM_KEYS)
    active_bounds = dict(bounds)

    if run_morris:
        sensitivity = run_morris_mass_gap_screen(
            obs,
            bounds=bounds,
            n_trajectories=n_trajectories,
            freeze_threshold=freeze_threshold,
            seed=seed,
        )
        plan = sensitivity["calibration_plan"]
        active_names = list(plan["active_parameters"])
        frozen = dict(plan["frozen_parameters"])
        active_bounds = dict(plan["active_bounds"])

    ndim = len(active_names)
    if ndim == 0:
        raise ValueError("No active MCMC parameters after Morris screening")

    rng = np.random.default_rng(seed)
    p0_center = np.array(
        [
            float(
                defaults.get(
                    name,
                    0.5 * (active_bounds[name][0] + active_bounds[name][1]),
                )
            )
            for name in active_names
        ],
        dtype=float,
    )
    # Ensure center satisfies TEP; jitter only in valid region
    initial = p0_center + 1.0e-3 * rng.standard_normal((int(n_walkers), ndim))
    for i in range(int(n_walkers)):
        for _ in range(20):
            trial = initial[i].copy()
            params = _vector_to_params(trial, active_names, frozen)
            if log_prior_fractal_tau(params, active_bounds=active_bounds, n_hier=obs.n_hier) == 0.0:
                initial[i] = trial
                break
            initial[i] = p0_center + 0.01 * rng.standard_normal(ndim)

    def _log_prob(theta: np.ndarray) -> float:
        return log_probability_fractal_tau(
            theta,
            active_names=active_names,
            frozen=frozen,
            observed=obs,
            active_bounds=active_bounds,
        )

    sampler = emcee.EnsembleSampler(int(n_walkers), ndim, _log_prob)
    sampler.run_mcmc(initial, int(n_steps), progress=False)
    discard = int(burn_in if burn_in is not None else max(int(n_steps) // 3, 1))
    chain = sampler.get_chain(discard=discard, flat=True)

    def _pct(arr: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p16": float(np.percentile(arr, 16)),
            "p50": float(np.percentile(arr, 50)),
            "p84": float(np.percentile(arr, 84)),
        }

    posteriors = {name: _pct(chain[:, j]) for j, name in enumerate(active_names)}
    best_idx = int(np.argmax([_log_prob(chain[i]) for i in range(min(chain.shape[0], 500))]))
    best_theta = chain[best_idx]
    best_params = _vector_to_params(best_theta, active_names, frozen)
    best_ll = log_likelihood_fractal_tau(best_params, obs)

    when = datetime.now(timezone.utc)
    chain_path = artifact_path(
        TestSlug.MCMC,
        "fractal_tau_tep",
        "mcmc_chain",
        "npz",
        when=when,
    )
    np.savez_compressed(
        chain_path,
        chain=chain,
        labels=np.array(active_names),
        frozen_keys=np.array(list(frozen.keys())),
        frozen_vals=np.array(list(frozen.values())),
    )

    report: dict[str, Any] = {
        "action": "Fractal Tau MCMC (TEP-hard prior + Morris calibration)",
        "timestamp": artifact_timestamp(when),
        "likelihood_mode": obs.likelihood_mode,
        "active_parameters": active_names,
        "frozen_parameters": frozen,
        "active_bounds": {k: list(v) for k, v in active_bounds.items()},
        "n_walkers": int(n_walkers),
        "n_steps": int(n_steps),
        "burn_in": discard,
        "acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "posteriors": posteriors,
        "best_params": best_params,
        "best_log_likelihood": float(best_ll),
        "best_tep_diagnostics": tep_closure_diagnostics(best_params, n_hier=obs.n_hier),
        "chain_path": str(chain_path),
        "sensitivity": sensitivity,
        "prior_note": (
            "log_prior = 0 only when TEP integer + 142857 cycle + domain occupancy "
            "and ingestion bounds pass; else −∞."
        ),
    }

    if verbose:
        print("\n=== FRACTAL TAU MCMC (TEP + MORRIS) ===")
        print(f"  mode           : {obs.likelihood_mode}")
        print(f"  active params  : {active_names}")
        if frozen:
            print(f"  frozen params  : {frozen}")
        if sensitivity and sensitivity.get("morris"):
            print(f"  Morris ranking : {sensitivity['morris'].get('ranking')}")
        print(f"  best log L     : {best_ll:.4f}")
        print(f"  acceptance     : {report['acceptance_fraction']:.3f}")
        print(f"  chain saved    : {chain_path}")

    return report