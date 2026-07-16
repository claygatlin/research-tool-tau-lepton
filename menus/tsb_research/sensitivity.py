"""
Global sensitivity screening (Morris one-at-a-time) for parameter prioritization.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

ParamDict = dict[str, float]
BoundsDict = dict[str, tuple[float, float]]


def morris_screen(
    param_names: Sequence[str],
    bounds: BoundsDict,
    evaluate: Callable[[ParamDict], float],
    *,
    n_trajectories: int = 24,
    n_levels: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Morris OAT screening: rank parameters by mean absolute elementary effect (μ*).

    ``evaluate`` receives a parameter dict and returns a scalar objective
    (e.g. |mass_gap_mev − 313.1|). Lower sensitivity → candidate for freezing.
    """
    names = list(param_names)
    k = len(names)
    if k == 0:
        raise ValueError("param_names must be non-empty")
    lo = np.array([float(bounds[n][0]) for n in names], dtype=float)
    hi = np.array([float(bounds[n][1]) for n in names], dtype=float)
    span = np.maximum(hi - lo, 1.0e-30)
    rng = np.random.default_rng(seed)
    delta_grid = 1.0 / (2.0 * (n_levels - 1))
    elementary: dict[str, list[float]] = {n: [] for n in names}

    for _ in range(int(n_trajectories)):
        grid = rng.integers(0, n_levels, size=k) / float(n_levels - 1)
        order = rng.permutation(k)
        x = lo + grid * span
        f_prev = float(evaluate(dict(zip(names, x))))

        for idx in order:
            grid_step = grid.copy()
            step_up = grid[idx] + 2.0 * delta_grid
            if step_up < 1.0:
                grid_step[idx] = step_up
            else:
                grid_step[idx] = max(0.0, grid[idx] - 2.0 * delta_grid)
            x_new = lo + grid_step * span
            f_new = float(evaluate(dict(zip(names, x_new))))
            dx = float(x_new[idx] - x[idx])
            ee = (f_new - f_prev) / dx if abs(dx) > 1.0e-30 else 0.0
            elementary[names[idx]].append(ee)
            grid = grid_step
            x = x_new
            f_prev = f_new

    ranked: list[dict[str, Any]] = []
    for name in names:
        arr = np.asarray(elementary[name], dtype=float)
        ranked.append(
            {
                "parameter": name,
                "mu": float(np.mean(arr)) if arr.size else 0.0,
                "mu_star": float(np.mean(np.abs(arr))) if arr.size else 0.0,
                "sigma": float(np.std(arr)) if arr.size else 0.0,
                "n_effects": int(arr.size),
            }
        )
    ranked.sort(key=lambda row: row["mu_star"], reverse=True)

    return {
        "method": "morris",
        "n_trajectories": int(n_trajectories),
        "n_levels": int(n_levels),
        "parameters": {row["parameter"]: row for row in ranked},
        "ranking": [row["parameter"] for row in ranked],
        "most_sensitive": ranked[0]["parameter"] if ranked else None,
    }


def sensitivity_driven_param_plan(
    morris_report: Mapping[str, Any],
    *,
    default_params: Mapping[str, float],
    bounds: BoundsDict,
    freeze_threshold: float = 0.05,
    narrow_factor: float = 0.5,
) -> dict[str, Any]:
    """
    Build an MCMC parameter plan from Morris μ* rankings.

    - μ* < freeze_threshold → freeze at ``default_params``
    - top-ranked → keep full bounds (optionally narrowed around default)
    """
    params_meta = dict(morris_report.get("parameters", {}))
    active: list[str] = []
    frozen: dict[str, float] = {}
    active_bounds: BoundsDict = {}

    for name, meta in params_meta.items():
        mu_star = float(meta.get("mu_star", 0.0))
        if mu_star < float(freeze_threshold):
            frozen[name] = float(default_params.get(name, 0.0))
        else:
            active.append(name)
            lo, hi = bounds[name]
            center = float(default_params.get(name, 0.5 * (lo + hi)))
            half = 0.5 * (hi - lo) * float(narrow_factor)
            active_bounds[name] = (center - half, center + half)

    # Always keep at least one parameter active
    if not active and params_meta:
        top = str(morris_report.get("most_sensitive") or next(iter(params_meta)))
        active = [top]
        lo, hi = bounds[top]
        frozen.pop(top, None)
        center = float(default_params.get(top, 0.5 * (lo + hi)))
        half = 0.5 * (hi - lo) * float(narrow_factor)
        active_bounds[top] = (center - half, center + half)

    return {
        "active_parameters": active,
        "frozen_parameters": frozen,
        "active_bounds": active_bounds,
        "freeze_threshold": float(freeze_threshold),
        "narrow_factor": float(narrow_factor),
    }