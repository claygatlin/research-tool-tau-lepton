"""
Vectorized mod-7 phase slot mapping for CMS Tav validation.

Never use ``int(event_array % 7)`` on NumPy arrays — that raises TypeError.
Use ``mod7_phase_residues`` and boolean slot masks instead.
"""

from __future__ import annotations

import numpy as np

MOD7_SLOT_COUNT = 7


def mod7_phase_residues(
    event_array: np.ndarray,
    *,
    clip_max: int = 64,
) -> np.ndarray:
    """
    Map per-event observables to mod-7 phase residues ``∈ [0, 6]``.

    Correct vector form::

        mod_residues = event_array % 7   # after finite clip/round

    Improper scalar cast (raises on arrays)::

        phase_slot = int(event_array % 7)  # do not use
    """
    arr = np.asarray(event_array, dtype=np.float64).ravel()
    residues = np.zeros(arr.shape, dtype=np.int64)
    finite = np.isfinite(arr)
    if np.any(finite):
        rounded = np.clip(np.rint(arr[finite]), 0, int(clip_max)).astype(np.int64)
        residues[finite] = rounded % MOD7_SLOT_COUNT
    return residues


def mod7_phase_slot_masks(
    event_array: np.ndarray,
    *,
    clip_max: int = 64,
) -> np.ndarray:
    """
    Boolean masks with shape ``(7, n_events)``.

    ``masks[slot, i]`` is True when event ``i`` occupies phase slot ``slot``.
    """
    residues = mod7_phase_residues(event_array, clip_max=clip_max)
    slots = np.arange(MOD7_SLOT_COUNT, dtype=np.int64)[:, np.newaxis]
    return residues[np.newaxis, :] == slots


def weighted_mod7_histogram(
    event_array: np.ndarray,
    event_weights: np.ndarray | None = None,
    *,
    clip_max: int = 64,
) -> np.ndarray:
    """Weighted occupancy histogram over the seven phase slots."""
    arr = np.asarray(event_array, dtype=np.float64).ravel()
    residues = mod7_phase_residues(arr, clip_max=clip_max)
    n = residues.size
    if n == 0:
        return np.zeros(MOD7_SLOT_COUNT, dtype=np.float64)
    finite = np.isfinite(arr)
    if event_weights is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = np.asarray(event_weights, dtype=np.float64).ravel()
        if weights.size != n:
            weights = np.full(n, float(np.mean(weights[np.isfinite(weights)]) or 1.0))
    weights = np.where(finite & np.isfinite(weights), weights, 0.0)
    return np.bincount(residues, weights=weights, minlength=MOD7_SLOT_COUNT).astype(
        np.float64
    )


def mod7_fractions_from_histogram(hist: np.ndarray) -> list[float]:
    """Normalize a 7-bin mod-7 histogram to fractions."""
    h = np.asarray(hist, dtype=np.float64).ravel()
    if h.size < MOD7_SLOT_COUNT:
        h = np.pad(h, (0, MOD7_SLOT_COUNT - h.size))
    total = float(h[:MOD7_SLOT_COUNT].sum())
    if total <= 0:
        return [0.0] * MOD7_SLOT_COUNT
    return (h[:MOD7_SLOT_COUNT] / total).tolist()