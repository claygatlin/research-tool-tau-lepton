"""
Atemporal Phase-Parity Reset Vector (V_PPR) — cycle-termination boundary matching.

Implements the non-commutative recursion from the Tav-Superblock cylinder engine:

    a_n = a_{n-1} + damping * (-1/2)^{n-1} * prod_{j=1}^{n-1} a_j

``damping_v_ppr`` is frozen at the solid-state / heavy-ion calibration baseline
unless explicitly overridden in MCMC (all other parameters may vary).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

# Frozen baseline (solid-state / heavy-ion calibration)
DAMPING_V_PPR: float = 0.991663


def v_ppr_delta(a_history: Sequence[float] | np.ndarray, n: int) -> float:
    """
    Incremental V_PPR update at recursion step ``n`` (1-indexed).

    Parameters
    ----------
    a_history
        Coefficients [a_1, a_2, …, a_{n-1}] accumulated so far.
    n
        Target step index (≥ 2 for a non-trivial product term).
    """
    if n < 1:
        raise ValueError("recursion index n must be ≥ 1")
    scale = (-0.5) ** (n - 1)
    if n == 1:
        prod_term = 1.0
    else:
        hist = np.asarray(a_history, dtype=float)
        if len(hist) < n - 1:
            raise ValueError(f"need {n - 1} prior coefficients for step n={n}, got {len(hist)}")
        prod_term = float(np.prod(hist[: n - 1]))
    return float(scale * prod_term)


def apply_v_ppr_update(
    a_prev: float,
    n: int,
    a_history: Sequence[float] | np.ndarray,
    *,
    damping: float = DAMPING_V_PPR,
) -> float:
    """Execute V_PPR cycle-termination boundary matching for step ``n``."""
    delta = v_ppr_delta(a_history, n)
    return float(a_prev + damping * delta)


def evolve_v_ppr_trajectory(
    n_max: int = 8,
    *,
    a_init: float = 1.0,
    damping: float = DAMPING_V_PPR,
) -> dict[str, Any]:
    """
    Evolve the full 8-phase recursion trajectory.

    Returns history, terminal boundary values at n ∈ {7, 8}, and topological
    vacuum latency proxy (|a_8 − a_7|).
    """
    n_max = max(1, int(n_max))
    history: list[float] = [float(a_init)]
    for n in range(2, n_max + 1):
        a_new = apply_v_ppr_update(history[-1], n, history, damping=damping)
        history.append(a_new)

    arr = np.asarray(history, dtype=float)
    a7 = float(arr[6]) if len(arr) >= 7 else float("nan")
    a8 = float(arr[7]) if len(arr) >= 8 else float("nan")
    latency = abs(a8 - a7) if len(arr) >= 8 else float("nan")

    return {
        "a_history": arr,
        "n_max": n_max,
        "damping_v_ppr": float(damping),
        "a_terminal_7": a7,
        "a_terminal_8": a8,
        "topological_vacuum_latency": latency,
        "activation_window": [7, 8],
    }


def latency_to_pN_scale(latency: float, *, calibrator: float = 1.731) -> float:
    """Map dimensionless latency to pN scale using hysteresis calibration anchor."""
    ref = evolve_v_ppr_trajectory(n_max=8, damping=DAMPING_V_PPR)["topological_vacuum_latency"]
    if ref <= 0 or not np.isfinite(ref):
        return float("nan")
    return float(calibrator * latency / ref)