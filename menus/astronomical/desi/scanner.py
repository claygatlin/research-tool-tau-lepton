#!/usr/bin/env python3
"""
Tau-SB (Tau-Superblock / Tav Topology) DESI Scanner Module
==========================================================

Scan public DESI BAO / expansion history for Tau cylinder + Superblock tracks.

Data sources
------------
- CobayaSampler/bao_data ``desi_bao_dr2/`` (Gaussian mean + covariance tables)
- https://data.desi.lbl.gov/public/ (clone tables locally; same loader path)

Real-data workflow::

    git clone https://github.com/CobayaSampler/bao_data.git datasets/desi/bao_data
    python tau_sb_desi_scanner.py --data cobaya --tracer ALL_GCcomb --compare-models

When :mod:`tav_resonance` is installed, residuals are cross-checked via
``analyze_tav_harmonics``.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, signal
from scipy.stats import chi2

warnings.filterwarnings("ignore", category=RuntimeWarning)

from tav_shared.artifact_paths import (
    ARTIFACTS_DIR,
    TestSlug,
    artifact_path,
    compose_dataset_slug,
)
from tav_shared.tav_project_paths import TAU_SUPERBLOCK_ROOT
from menus.astronomical.desi.fetcher import DATASETS_DIR, DEFAULT_COBAYA_ROOT

PROJECT_ROOT = TAU_SUPERBLOCK_ROOT

# === TSB framework constants (sound_horizon_tsb_integration.md) ===
M0_MEV: float = 313.1  # conformal mass-gap / geometric friction floor (MeV)
N_HIER_BINDING: float = 45.8
N_HIER: float = N_HIER_BINDING  # alias used in sound-horizon docs
LATE_UNIVERSE_DELTA_N: float = 0.33  # atomic ↔ EM fine-tuning near decoupling
DELTA_N_PLASMA: float = 10.74  # hierarchical steps QCD (313.1 MeV) → recombination
TAU_RESONANCE_PERIOD: float = 7.0  # s-units; expected frequency f = 1/7
TAU_PERIOD: float = TAU_RESONANCE_PERIOD  # alias used in sound-horizon docs
GAMMA_REFERENCE: float = 10.0  # nominal τ-cylinder stretch γ₀
RD_OBSERVED_MPC: float = 147.09  # Planck 2018 / DESI DR2 sound horizon (Mpc)
RD_OBSERVED_ERR_MPC: float = 0.26
KAPPA_LEAK_DEFAULT: float = 0.015  # photon dispersion default (JAX pipeline 2026-07-09)
H0_FIDUCIAL_KM_S_MPC: float = 68.5  # DESI DR2 + BBN reference H₀
H0_H_UNITS: float = 100.0  # h=1 convention for D_C in h⁻¹ Mpc integrals
OMEGA_M_FIDUCIAL: float = 0.2975  # DESI DR2 flat ΛCDM Ω_m
TSB_RD_GAMMA_FALLBACK: float = 8.8511  # typical auto-calibrated γ (ALL_GCcomb DH_only)

# === WHIM / cosmological web (Sonato paper) ===
WHIM_DENSITY_FACTOR: float = 0.05  # fractional contribution to effective density
WEB_1D_CONSTRAINT_STRENGTH: float = 0.02  # 1D spatial constraint modulation strength
BEC_COHERENCE_LENGTH_MPC: float = 5.5  # Mpc; linked to n_hier binding wells
BEC_COHERENCE_LENGTH: float = BEC_COHERENCE_LENGTH_MPC  # alias

# emcee MCMC defaults (publication-grade posteriors)
MCMC_STEPS_DEFAULT: int = 1500
MCMC_WALKERS_DEFAULT: int = 48
MCMC_BURN_IN_DEFAULT: int = 500

# Minimum data points for production sub-pipelines
CHANGPOINTS_MIN_N: int = 2
PRODUCTION_PIPELINE_MIN_N: int = 8
TAV_RESONANCE_MIN_N: int = 8
# Legacy+DESI stack: keep distinct legacy z bins (n≈9 DH) vs default dedupe (n≈7)
HIGH_POWER_DEDUPE_Z_TOL: float = 0.0
STANDARD_DEDUPE_Z_TOL: float = 0.02
R_TAU_MPC: float = 7.0
GAMMA6_HEX: float = 1.4050


def mcmc_burn_in_for_steps(n_steps: int) -> int:
    """Burn-in length scaled for ``n_steps`` (≥500 for the 1500-step default)."""
    return max(int(n_steps) // 3, 500)

DESI_TRACERS: dict[str, str] = {
    "ALL_GCcomb": "Combined DR2 GC (all tracers)",
    "BGS_BRIGHT-21.35_GCcomb": "BGS bright sample",
    "LRG_GCcomb_z0.4-0.6": "LRG z=0.4–0.6",
    "LRG_GCcomb_z0.6-0.8": "LRG z=0.6–0.8",
    "ELG_LOPnotqso_GCcomb_z1.1-1.6": "ELG z=1.1–1.6",
    "LRG+ELG_LOPnotqso_GCcomb": "LRG + ELG combined",
    "QSO_GCcomb": "QSO sample",
    "Lya_GCcomb": "Lyα forest",
}

# Reproducible multi-tracer stack order (BGS → LRG → ELG → QSO → Lyα)
TRACER_STACK_ORDER: tuple[str, ...] = (
    "BGS_BRIGHT-21.35_GCcomb",
    "LRG_GCcomb_z0.4-0.6",
    "LRG_GCcomb_z0.6-0.8",
    "LRG+ELG_LOPnotqso_GCcomb",
    "ELG_LOPnotqso_GCcomb_z1.1-1.6",
    "QSO_GCcomb",
    "Lya_GCcomb",
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_stamp() -> str:
    return _utc_now().strftime("%Y%m%d_%H%M%S")


def _utc_iso() -> str:
    return _utc_now().isoformat(timespec="seconds")


def _baseline_degree(n_data: int, extra_params: int) -> int:
    """Largest poly degree such that (deg+1) + extra_params < n_data."""
    max_deg = 1 if n_data < 8 else 2
    for deg in range(max_deg, -1, -1):
        if (deg + 1) + extra_params < n_data:
            return deg
    return 0


def _oscillation_scale(observable: np.ndarray) -> float:
    """Reference scale for fractional oscillation amplitude on BAO observables."""
    obs = np.asarray(observable, dtype=float)
    return float(max(np.median(np.abs(obs)), 1e-6))


def _fit_parameter_labels(
    deg: int,
    *,
    fit_A: bool = True,
    fit_phase: bool = True,
    use_hier: bool = False,
) -> list[str]:
    labels = [f"poly_c{i}" for i in range(deg + 1)]
    if fit_A:
        labels.append("A_osc_frac")
    if fit_phase:
        labels.append("phase_rad")
    if use_hier:
        labels.append("hier_frac")
    return labels


def _oscillation_param_count(
    *,
    fit_A: bool,
    fit_phase: bool,
    use_hier: bool,
) -> int:
    return int(fit_A) + int(fit_phase) + int(use_hier)


def _format_power_assessment_lines(assessment: dict[str, Any] | None) -> list[str]:
    if not assessment:
        return []
    lines = [
        f"Statistical power: {assessment.get('overall_severity', 'unknown').upper()}",
    ]
    for row in assessment.get("issues", []):
        lines.append(f"  [{row['severity'].upper()}] {row['issue']}: {row['message']}")
    rec = assessment.get("recommendation")
    if rec:
        lines.append(f"  → {rec}")
    return lines


def _ensure_artifacts() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


def _try_tav_harmonic_check(
    s: np.ndarray,
    residuals: np.ndarray,
    *,
    period: float = TAU_RESONANCE_PERIOD,
) -> dict[str, Any]:
    from menus.astronomical.desi.stats import tav_harmonic_check_robust

    return tav_harmonic_check_robust(s, residuals, period=period)


# ============================================================
# 1. DATA LOADING
# ============================================================

# Clean single-channel and joint quantity presets for ALL_GCcomb.
QUANTITY_ALIASES: dict[str, str] = {
    "DM_over_rs_only": "DM_over_rs",
    "DH_over_rs_only": "DH_over_rs",
    "DV_over_rs_only": "DV_over_rs",
}
JOINT_DH_DM: str = "joint_DH_DM"

# ============================================================
# IMPROVED DATA MODE HANDLING + INJECTION RECOVERY
# ============================================================
DATA_MODE_DH_ONLY: str = "DH_only"
DATA_MODE_DM_ONLY: str = "DM_only"
DATA_MODE_JOINT: str = "Joint_DH_DM"
DATA_MODE_DEFAULT: str = DATA_MODE_DH_ONLY
DATA_MODE_METHOD10_DEFAULT: str = DATA_MODE_DM_ONLY  # aligns with DR2 D_M/r_d evaluation
DATA_MODE_PUBLICATION_DEFAULT: str = DATA_MODE_DM_ONLY
DATA_MODE: str = "DH_only"     # default production scan (DH_only); Method 10 uses DM_only
# DATA_MODE = "DM_only"
# DATA_MODE = "Joint_DH_DM"    # Mixed channels — not recommended for single-template fits
TRACER: str = "ALL_GCcomb"

DATA_MODES: tuple[str, ...] = (DATA_MODE_DH_ONLY, DATA_MODE_DM_ONLY, DATA_MODE_JOINT)
DATA_MODE_TO_QUANTITY: dict[str, str] = {
    DATA_MODE_DH_ONLY: "DH_over_rs",
    DATA_MODE_DM_ONLY: "DM_over_rs",
    DATA_MODE_JOINT: JOINT_DH_DM,
}
_DATA_MODE_INPUT_ALIASES: dict[str, str] = {
    "DH_only": DATA_MODE_DH_ONLY,
    "DH_over_rs": DATA_MODE_DH_ONLY,
    "DH_over_rs_only": DATA_MODE_DH_ONLY,
    "DM_only": DATA_MODE_DM_ONLY,
    "DM_over_rs": DATA_MODE_DM_ONLY,
    "DM_over_rs_only": DATA_MODE_DM_ONLY,
    "Joint_DH_DM": DATA_MODE_JOINT,
    "joint_DH_DM": DATA_MODE_JOINT,
    JOINT_DH_DM: DATA_MODE_JOINT,
}

MIXED_QUANTITY_WARNING = (
    "quantity=all mixes DV_over_rs, DM_over_rs (D_M/r_d), and DH_over_rs (D_H/r_d) "
    "in one vector — a single oscillation template is not physically consistent; "
    "use DH_over_rs or DM_over_rs for trustworthy fits."
)
JOINT_DH_DM_WARNING = (
    "Joint_DH_DM mixes DM_over_rs and DH_over_rs in one vector — a single oscillation "
    "template is not physically consistent; use DH_only or DM_only for trustworthy fits."
)


def is_mixed_channel_data(data: dict[str, Any]) -> bool:
    """True when the loaded vector mixes BAO observable types (e.g. DH + DM)."""
    if bool(data.get("mixed_channels")):
        return True
    mode = data.get("data_mode")
    if mode == DATA_MODE_JOINT:
        return True
    quantities = data.get("quantity") or data.get("quantity_types") or []
    labels = set(np.asarray(quantities, dtype=str).tolist())
    return len(labels) > 1


def validate_single_channel_fit_data(
    data: dict[str, Any],
    *,
    action_name: str = "",
    allow_mixed: bool = False,
) -> None:
    """
    Refuse ΛCDM/Tau-SB fits on mixed DH+DM vectors unless explicitly allowed.

    Mixed vectors yield catastrophic reduced χ² even for simple polynomials —
    the failure is in data construction, not model choice.
    """
    if allow_mixed or not is_mixed_channel_data(data):
        return
    mode = data.get("data_mode") or data.get("quantity_filter") or "mixed"
    raise ValueError(
        f"Cannot run structured fits on mixed-channel data ({mode}, n={data.get('n_data', 0)}). "
        f"{JOINT_DH_DM_WARNING} "
        f"Set DATA_MODE='DH_only' (recommended) or 'DM_only'. "
        f"Action: {action_name or 'desi_scan'}"
    )


def mixed_quantity_warning(
    quantities: Sequence[str] | np.ndarray,
    *,
    data_mode: str | None = None,
) -> str | None:
    """Return a channel-mixing warning string when the vector is mixed."""
    labels = set(np.asarray(quantities, dtype=str).tolist())
    if data_mode == DATA_MODE_JOINT or labels == {"DM_over_rs", "DH_over_rs"}:
        return JOINT_DH_DM_WARNING
    if len(labels) > 1:
        return MIXED_QUANTITY_WARNING
    return None


def normalize_data_mode(mode: str | None) -> str:
    """Map UI / legacy quantity labels to a canonical DATA_MODE value."""
    if mode is None:
        return DATA_MODE_DEFAULT
    key = str(mode).strip()
    if not key:
        return DATA_MODE_DEFAULT
    resolved = _DATA_MODE_INPUT_ALIASES.get(key)
    if resolved is None:
        resolved = _DATA_MODE_INPUT_ALIASES.get(QUANTITY_ALIASES.get(key, ""))
    if resolved is None:
        raise ValueError(f"Unknown DATA_MODE: {mode}")
    return resolved


def data_mode_to_quantity_filter(mode: str | None) -> str:
    """Translate DATA_MODE (or alias) to the row filter used in Cobaya tables."""
    return DATA_MODE_TO_QUANTITY[normalize_data_mode(mode)]


def normalize_quantity_filter(quantity_filter: str | None) -> str | None:
    """
    Map UI presets to loader filters.

    Returns ``None`` for the full published vector (all channels), ``joint_DH_DM``
    for DM+DH only, or a single quantity label e.g. ``DH_over_rs``.
    """
    if quantity_filter is None:
        return None
    q = str(quantity_filter).strip()
    if not q or q.lower() in {"all", "*", "none"}:
        return None
    if q in _DATA_MODE_INPUT_ALIASES or q in QUANTITY_ALIASES:
        try:
            return data_mode_to_quantity_filter(q)
        except ValueError:
            pass
    if q == JOINT_DH_DM:
        return JOINT_DH_DM
    return QUANTITY_ALIASES.get(q, q)


def _quantity_selection_mask(quantities: list[str], quantity_filter: str | None) -> np.ndarray:
    """Boolean mask for rows to keep given a normalized quantity filter."""
    if quantity_filter is None:
        return np.ones(len(quantities), dtype=bool)
    if quantity_filter == JOINT_DH_DM:
        return np.array([q in {"DM_over_rs", "DH_over_rs"} for q in quantities], dtype=bool)
    return np.array([q == quantity_filter for q in quantities], dtype=bool)


def print_quantity_composition(
    z: np.ndarray,
    observable: np.ndarray,
    quantities: list[str],
) -> None:
    """Print per-row quantity types and flag mixed-channel vectors."""
    from collections import Counter

    counts = Counter(quantities)
    print("Quantity composition:")
    for name, count in sorted(counts.items()):
        channel = {
            "DM_over_rs": "D_M/r_d",
            "DH_over_rs": "D_H/r_d",
            "DV_over_rs": "D_V/r_d",
        }.get(name, name)
        print(f"  {name} ({channel}): {count} point(s)")
    warn = mixed_quantity_warning(quantities)
    if warn:
        print(f"  ⚠ {warn}")
    print("Row detail (z | quantity | observable):")
    for zi, qi, oi in zip(z, quantities, observable, strict=True):
        print(f"  z={float(zi):.4f}  {str(qi):12s}  obs={float(oi):.4f}")


def _parse_cobaya_mean_table(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows: list[tuple[float, float, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        z_val, obs_val, quantity = float(parts[0]), float(parts[1]), parts[2]
        rows.append((z_val, obs_val, quantity))
    if not rows:
        raise ValueError(f"No data rows parsed from {path}")
    z = np.array([r[0] for r in rows], dtype=float)
    observable = np.array([r[1] for r in rows], dtype=float)
    quantities = [r[2] for r in rows]
    return z, observable, quantities


def _parse_cobaya_cov_table(path: Path) -> np.ndarray:
    values: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values.extend(float(x) for x in line.split())
    if not values:
        raise ValueError(f"No covariance entries in {path}")
    n = int(round(np.sqrt(len(values))))
    if n * n != len(values):
        raise ValueError(f"Covariance file {path} is not square ({len(values)} entries)")
    return np.array(values, dtype=float).reshape(n, n)


def resolve_cobaya_dr2_dir(local_path: str | Path | None = None) -> Path:
    """Resolve path to ``desi_bao_dr2`` inside a cloned bao_data repo."""
    if local_path is None:
        if DEFAULT_COBAYA_ROOT.is_dir():
            return DEFAULT_COBAYA_ROOT
        raise FileNotFoundError(
            "DESI DR2 tables not found. Clone Cobaya bao_data:\n"
            "  git clone https://github.com/CobayaSampler/bao_data.git "
            f"{DATASETS_DIR / 'bao_data'}"
        )
    root = Path(local_path)
    if (root / "desi_bao_dr2").is_dir():
        return root / "desi_bao_dr2"
    if root.name == "desi_bao_dr2":
        return root
    if root.name == "bao_data":
        candidate = root / "desi_bao_dr2"
        if candidate.is_dir():
            return candidate
    if root.is_dir() and any(root.glob("desi_gaussian_bao_*_mean.txt")):
        return root
    raise FileNotFoundError(f"Could not locate desi_bao_dr2 tables under {root}")


def tracer_available_quantities(
    local_path: str | Path | None, tracer: str
) -> list[str]:
    """Return quantity labels present in a tracer mean table."""
    dr2_dir = resolve_cobaya_dr2_dir(local_path)
    mean_path = dr2_dir / f"desi_gaussian_bao_{tracer}_mean.txt"
    if not mean_path.is_file():
        return []
    _z, _obs, quantities = _parse_cobaya_mean_table(mean_path)
    return sorted(set(quantities))


def list_desi_tracers(local_path: str | Path | None = None) -> list[str]:
    """Return tracer keys available in a Cobaya DR2 directory."""
    dr2_dir = resolve_cobaya_dr2_dir(local_path)
    tracers: list[str] = []
    pattern = re.compile(r"desi_gaussian_bao_(.+)_mean\.txt$")
    for mean_file in sorted(dr2_dir.glob("desi_gaussian_bao_*_mean.txt")):
        match = pattern.match(mean_file.name)
        if match:
            tracers.append(match.group(1))
    return tracers


def ordered_tracers_for_stack(
    local_path: str | Path | None = None,
    *,
    stack_order: tuple[str, ...] | None = None,
) -> list[str]:
    """
    Tracers for combined DR2 stack in fixed order, then any extras (sorted).

    Excludes ALL_GCcomb. Unknown cache files append alphabetically for stability.
    """
    available = set(list_desi_tracers(local_path))
    available.discard("ALL_GCcomb")
    order = stack_order or TRACER_STACK_ORDER
    ordered = [t for t in order if t in available]
    extras = sorted(available - set(ordered))
    return ordered + extras


def load_desi_from_cobaya_repo(
    local_path: str | Path | None = None,
    *,
    tracer: str = "ALL_GCcomb",
    quantity_filter: str | None = "DM_over_rs",
) -> dict[str, Any]:
    """
    Load DESI DR2 Gaussian BAO tables from CobayaSampler/bao_data.

    Parameters
    ----------
    local_path:
        Path to cloned ``bao_data`` repo, ``desi_bao_dr2/`` folder, or None
        for ``datasets/desi/bao_data/desi_bao_dr2``.
    tracer:
        Tracer key, e.g. ``ALL_GCcomb``, ``LRG_GCcomb_z0.4-0.6``, ``Lya_GCcomb``.
    quantity_filter:
        Row filter preset:

        - ``None`` / ``all`` — full 13-point ALL_GCcomb vector (mixed channels)
        - ``DH_over_rs`` / ``DH_over_rs_only`` — 6 D_H/r_d points
        - ``DM_over_rs`` / ``DM_over_rs_only`` — 6 D_M/r_d points
        - ``joint_DH_DM`` — 12 DM+DH points (no DV; still mixed unless joint model)
        - ``DV_over_rs`` — 1 point only (underpowered)
    """
    dr2_dir = resolve_cobaya_dr2_dir(local_path)
    mean_path = dr2_dir / f"desi_gaussian_bao_{tracer}_mean.txt"
    cov_path = dr2_dir / f"desi_gaussian_bao_{tracer}_cov.txt"
    if not mean_path.is_file():
        available = ", ".join(list_desi_tracers(dr2_dir))
        raise FileNotFoundError(
            f"Missing {mean_path.name}. Available tracers: {available}"
        )
    if not cov_path.is_file():
        raise FileNotFoundError(f"Missing covariance file: {cov_path}")

    z, observable, quantities = _parse_cobaya_mean_table(mean_path)
    cov = _parse_cobaya_cov_table(cov_path)
    if cov.shape[0] != len(observable):
        raise ValueError(
            f"Mean vector ({len(observable)}) and covariance ({cov.shape[0]}) "
            f"size mismatch for tracer {tracer}"
        )

    norm_filter = normalize_quantity_filter(quantity_filter)
    mask = _quantity_selection_mask(quantities, norm_filter)
    if not np.any(mask):
        raise ValueError(
            f"Tracer {tracer} has no rows for quantity={quantity_filter!r}. "
            f"Available: {sorted(set(quantities))}"
        )
    z = z[mask]
    observable = observable[mask]
    quantities = [q for q, keep in zip(quantities, mask, strict=True) if keep]
    idx = np.where(mask)[0]
    cov = cov[np.ix_(idx, idx)]

    label = f"DESI_DR2_{tracer}"
    if norm_filter:
        label += f"_{norm_filter}"
    mixed_channels = len(set(quantities)) > 1

    from menus.astronomical.desi.analysis import prepare_covariance

    cov, cov_health = prepare_covariance(cov, name=label)
    err = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    return {
        "z": z,
        "observable": observable,
        "err": err,
        "quantity": quantities,
        "quantity_filter": norm_filter,
        "quantity_types": sorted(set(quantities)),
        "mixed_channels": mixed_channels,
        "cov": cov,
        "cov_health": cov_health,
        "label": label,
        "tracer": tracer,
        "source": str(mean_path),
        "n_data": len(z),
        "has_full_covariance": True,
    }


def load_desi_joint_data(
    local_path: str | Path | None = None,
    *,
    tracer: str = "ALL_GCcomb",
    gamma: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Load DESI rows available for joint DH+DM masking on a tracer.

    Uses the published DM+DH subset when present; otherwise falls back to the
    full tracer vector so per-mode masks can still be applied.

    Returns ``(y, cov, s, metadata)`` where ``metadata['quantity']`` labels each row.
    """
    try:
        data = load_desi_from_cobaya_repo(
            local_path,
            tracer=tracer,
            quantity_filter=JOINT_DH_DM,
        )
    except ValueError:
        data = load_desi_from_cobaya_repo(
            local_path,
            tracer=tracer,
            quantity_filter=None,
        )
    z = np.asarray(data["z"], dtype=float)
    y = np.asarray(data["observable"], dtype=float)
    cov = np.asarray(data["cov"], dtype=float)
    s = s_from_z(z, gamma=gamma)
    quantities = np.asarray(data["quantity"], dtype=str)
    metadata: dict[str, Any] = {
        "quantity": quantities,
        "z": z,
        "err": np.asarray(data["err"], dtype=float),
        "tracer": tracer,
        "label": data["label"],
        "source": data.get("source"),
        "gamma": float(gamma),
        "mixed_channels": bool(data.get("mixed_channels", False)),
        "cov_health": data.get("cov_health"),
        "n_data": len(y),
    }
    return y, cov, s, metadata


def get_data_by_mode(
    mode: str | None = None,
    tracer: str = TRACER,
    *,
    local_path: str | Path | None = None,
    gamma: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Clean data loading with safe defaults and clear warnings.

    Modes: ``DH_only``, ``DM_only``, ``Joint_DH_DM`` (alias ``joint_DH_DM``).
    Defaults to module ``DATA_MODE`` when ``mode`` is None.

    Returns ``(y, cov, s, metadata)`` with masked rows in ``metadata``.
    """
    if mode is None:
        mode = DATA_MODE
    mode_key = normalize_data_mode(mode)

    if mode_key == DATA_MODE_JOINT and tracer != TRACER:
        print("⚠️  joint_DH_DM only supported on ALL_GCcomb. Switching to DH_only.")
        mode_key = DATA_MODE_DH_ONLY

    full_y, full_cov, full_s, metadata = load_desi_joint_data(
        local_path,
        tracer=tracer,
        gamma=gamma,
    )
    quantities = np.asarray(metadata["quantity"], dtype=str)

    if mode_key == DATA_MODE_DH_ONLY:
        mask = quantities == "DH_over_rs"
    elif mode_key == DATA_MODE_DM_ONLY:
        mask = quantities == "DM_over_rs"
    elif mode_key == DATA_MODE_JOINT:
        mask = np.ones(len(full_y), dtype=bool)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not np.any(mask):
        raise ValueError(f"No data available for mode='{mode_key}' on tracer='{tracer}'")

    y = full_y[mask]
    cov = full_cov[np.ix_(mask, mask)]
    s = full_s[mask]
    z = np.asarray(metadata["z"], dtype=float)[mask]
    err = np.asarray(metadata["err"], dtype=float)[mask]
    filtered_quantities = quantities[mask]

    label = f"DESI_DR2_{tracer}_{mode_key}"
    from menus.astronomical.desi.analysis import prepare_covariance

    cov, cov_health = prepare_covariance(cov, name=label)
    err = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    filtered_metadata: dict[str, Any] = {
        **metadata,
        "data_mode": mode_key,
        "quantity": filtered_quantities,
        "z": z,
        "err": err,
        "label": label,
        "quantity_filter": DATA_MODE_TO_QUANTITY[mode_key],
        "quantity_types": sorted(set(filtered_quantities.tolist())),
        "mixed_channels": len(set(filtered_quantities.tolist())) > 1,
        "cov_health": cov_health,
        "n_data": len(y),
        "requested_mode": mode,
        "requested_tracer": tracer,
    }

    print(f"\n[DATA MODE] {mode_key} | tracer={tracer} → {len(y)} points selected")
    if mode_key == DATA_MODE_JOINT:
        print("  ⚠ Using mixed DH+DM vector (not recommended for periodicity tests)")

    return y, cov, s, filtered_metadata


def load_scan_data_by_data_mode(
    mode: str = DATA_MODE,
    *,
    local_path: str | Path | None = None,
    tracer: str = TRACER,
    gamma: float = 10.0,
) -> dict[str, Any]:
    """Return a scanner-compatible data dict for the requested DATA_MODE."""
    y, cov, s, metadata = get_data_by_mode(
        mode,
        local_path=local_path,
        tracer=tracer,
        gamma=gamma,
    )
    quantities = metadata["quantity"]
    if isinstance(quantities, np.ndarray):
        quantities = quantities.tolist()
    return {
        "z": metadata["z"],
        "observable": y,
        "err": metadata["err"],
        "quantity": quantities,
        "quantity_filter": metadata["quantity_filter"],
        "data_mode": metadata["data_mode"],
        "quantity_types": metadata["quantity_types"],
        "mixed_channels": metadata["mixed_channels"],
        "cov": cov,
        "cov_health": metadata["cov_health"],
        "label": metadata["label"],
        "tracer": tracer,
        "source": metadata.get("source"),
        "s": s,
        "n_data": metadata["n_data"],
        "has_full_covariance": True,
    }


# Legacy BOSS / eBOSS tables (Cobaya bao_data repo root, not desi_bao_dr2/).
LEGACY_QUANTITY_ALIASES: dict[str, str] = {
    "bao_Hz_rs": "DH_over_rs",
    "Hz_rs": "DH_over_rs",
}
LEGACY_BAO_DATASETS: dict[str, dict[str, str]] = {
    "DR12_LRG": {
        "mean": "sdss_DR12_LRG_BAO_DMDH.dat",
        "cov": "sdss_DR12_LRG_BAO_DMDH_covtot.txt",
        "survey": "BOSS DR12",
    },
    "DR16_LRG": {
        "mean": "sdss_DR16_LRG_BAO_DMDH.dat",
        "cov": "sdss_DR16_LRG_BAO_DMDH_covtot.txt",
        "survey": "eBOSS DR16",
    },
    "DR16_QSO": {
        "mean": "sdss_DR16_QSO_BAO_DMDH.txt",
        "cov": "sdss_DR16_QSO_BAO_DMDH_covtot.txt",
        "survey": "eBOSS DR16",
    },
}
COMBINED_DR2_TRACER: str = "COMBINED_DR2"
EXTENDED_BAO_TRACER: str = "DESI+LEGACY"


def resolve_bao_data_root(local_path: str | Path | None = None) -> Path:
    """Resolve cloned Cobaya ``bao_data`` repo root (parent of ``desi_bao_dr2``)."""
    dr2 = resolve_cobaya_dr2_dir(local_path)
    if dr2.name == "desi_bao_dr2":
        return dr2.parent
    return dr2


def _normalize_legacy_quantity(label: str) -> str:
    return LEGACY_QUANTITY_ALIASES.get(label, label)


def _parse_legacy_mean_table(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows: list[tuple[float, float, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        quantity = _normalize_legacy_quantity(parts[2])
        if quantity in {"f_sigma8", "fsigma8", "f_sigma_8"}:
            continue
        rows.append((float(parts[0]), float(parts[1]), quantity))
    if not rows:
        raise ValueError(f"No data rows parsed from {path}")
    z = np.array([r[0] for r in rows], dtype=float)
    observable = np.array([r[1] for r in rows], dtype=float)
    quantities = [r[2] for r in rows]
    return z, observable, quantities


def load_legacy_bao_dataset(
    dataset_key: str,
    *,
    local_path: str | Path | None = None,
    quantity_filter: str | None = "DH_over_rs",
) -> dict[str, Any]:
    """Load a single BOSS/eBOSS Gaussian BAO table from the Cobaya repo."""
    if dataset_key not in LEGACY_BAO_DATASETS:
        raise ValueError(
            f"Unknown legacy dataset {dataset_key!r}. "
            f"Known: {sorted(LEGACY_BAO_DATASETS)}"
        )
    spec = LEGACY_BAO_DATASETS[dataset_key]
    root = resolve_bao_data_root(local_path)
    mean_path = root / spec["mean"]
    cov_path = root / spec["cov"]
    if not mean_path.is_file():
        raise FileNotFoundError(f"Missing legacy mean table: {mean_path}")
    if not cov_path.is_file():
        raise FileNotFoundError(f"Missing legacy covariance: {cov_path}")

    z, observable, quantities = _parse_legacy_mean_table(mean_path)
    cov = _parse_cobaya_cov_table(cov_path)
    if cov.shape[0] != len(observable):
        raise ValueError(
            f"Legacy {dataset_key}: mean ({len(observable)}) vs cov ({cov.shape[0]}) mismatch"
        )

    norm_filter = normalize_quantity_filter(quantity_filter)
    mask = _quantity_selection_mask(quantities, norm_filter)
    if not np.any(mask):
        raise ValueError(
            f"Legacy {dataset_key} has no rows for quantity={quantity_filter!r}. "
            f"Available: {sorted(set(quantities))}"
        )
    z = z[mask]
    observable = observable[mask]
    quantities = [q for q, keep in zip(quantities, mask, strict=True) if keep]
    idx = np.where(mask)[0]
    cov = cov[np.ix_(idx, idx)]

    label = f"{spec['survey']}_{dataset_key}"
    if norm_filter:
        label += f"_{norm_filter}"
    from menus.astronomical.desi.analysis import prepare_covariance

    cov, cov_health = prepare_covariance(cov, name=label)
    err = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return {
        "z": z,
        "observable": observable,
        "err": err,
        "quantity": quantities,
        "quantity_filter": norm_filter,
        "quantity_types": sorted(set(quantities)),
        "mixed_channels": len(set(quantities)) > 1,
        "cov": cov,
        "cov_health": cov_health,
        "label": label,
        "tracer": dataset_key,
        "source": str(mean_path),
        "survey": spec["survey"],
        "n_data": len(z),
        "has_full_covariance": True,
    }


def _stack_bao_blocks(
    blocks: list[dict[str, Any]],
    *,
    label: str,
    tracer: str,
    data_mode: str,
    dedupe_z_tol: float = 0.02,
    prefer_last: bool = True,
) -> dict[str, Any]:
    """Block-diagonal stack of BAO vectors; dedupe near-duplicate redshifts."""
    if not blocks:
        raise ValueError("No BAO blocks to stack")

    z_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    err_parts: list[np.ndarray] = []
    q_parts: list[list[str]] = []
    cov_blocks: list[np.ndarray] = []
    sources: list[str] = []
    surveys: list[str] = []

    for block in blocks:
        z_parts.append(np.asarray(block["z"], dtype=float))
        y_parts.append(np.asarray(block["observable"], dtype=float))
        err_parts.append(np.asarray(block["err"], dtype=float))
        q_parts.append(list(block["quantity"]))
        cov_blocks.append(np.asarray(block["cov"], dtype=float))
        sources.append(str(block.get("source", block.get("label", "unknown"))))
        surveys.append(str(block.get("survey", block.get("tracer", "unknown"))))

    z_all = np.concatenate(z_parts)
    y_all = np.concatenate(y_parts)
    err_all = np.concatenate(err_parts)
    q_all = [item for sub in q_parts for item in sub]
    n = len(z_all)
    cov_full = np.zeros((n, n), dtype=float)
    offset = 0
    for cov_blk in cov_blocks:
        k = cov_blk.shape[0]
        cov_full[offset : offset + k, offset : offset + k] = cov_blk
        offset += k

    order = np.argsort(z_all)
    z_sorted = z_all[order]
    y_sorted = y_all[order]
    err_sorted = err_all[order]
    q_sorted = [q_all[i] for i in order]
    cov_sorted = cov_full[np.ix_(order, order)]
    src_sorted = []
    for i in order:
        block_idx = 0
        cum = 0
        for j, zp in enumerate(z_parts):
            if i < cum + len(zp):
                block_idx = j
                break
            cum += len(zp)
        src_sorted.append(sources[block_idx])

    keep = np.ones(len(z_sorted), dtype=bool)
    for i in range(1, len(z_sorted)):
        if abs(z_sorted[i] - z_sorted[i - 1]) <= dedupe_z_tol:
            if prefer_last:
                keep[i - 1] = False
            else:
                keep[i] = False

    idx_keep = np.where(keep)[0]
    z_out = z_sorted[idx_keep]
    y_out = y_sorted[idx_keep]
    err_out = err_sorted[idx_keep]
    q_out = [q_sorted[i] for i in idx_keep]
    cov_out = cov_sorted[np.ix_(idx_keep, idx_keep)]
    src_out = [src_sorted[i] for i in idx_keep]

    from menus.astronomical.desi.analysis import prepare_covariance

    cov_out, cov_health = prepare_covariance(cov_out, name=label)
    err_out = np.sqrt(np.clip(np.diag(cov_out), 0.0, None))

    return {
        "z": z_out,
        "observable": y_out,
        "err": err_out,
        "quantity": q_out,
        "quantity_filter": data_mode_to_quantity_filter(data_mode),
        "data_mode": data_mode,
        "quantity_types": sorted(set(q_out)),
        "mixed_channels": len(set(q_out)) > 1,
        "cov": cov_out,
        "cov_health": cov_health,
        "label": label,
        "tracer": tracer,
        "source": "; ".join(dict.fromkeys(sources)),
        "stack_sources": src_out,
        "surveys": sorted(set(surveys)),
        "n_data": len(z_out),
        "has_full_covariance": True,
        "legacy_included": any("DR12" in s or "DR16" in s or "BOSS" in s or "eBOSS" in s for s in surveys),
    }


def load_combined_dr2_data(
    mode: str = DATA_MODE,
    *,
    local_path: str | Path | None = None,
    gamma: float = 10.0,
    dedupe_z_tol: float | None = None,
) -> dict[str, Any]:
    """
    Stack per-tracer DR2 rows (excludes ALL_GCcomb) for the requested DATA_MODE.

    Yields the same redshift coverage as ALL_GCcomb but tags each row with its
    tracer provenance (useful for jackknife / batch diagnostics).

    Set ``dedupe_z_tol=0`` to retain all z bins (maximum n for Lomb resolution).
    """
    mode_key = normalize_data_mode(mode)
    z_tol = STANDARD_DEDUPE_Z_TOL if dedupe_z_tol is None else float(dedupe_z_tol)
    tracers = ordered_tracers_for_stack(local_path)
    blocks: list[dict[str, Any]] = []
    for tracer in tracers:
        try:
            block = load_scan_data_by_data_mode(
                mode_key,
                local_path=local_path,
                tracer=tracer,
                gamma=gamma,
            )
        except (FileNotFoundError, ValueError):
            continue
        if int(block.get("n_data", 0)) > 0:
            block["survey"] = f"DESI DR2 ({tracer})"
            blocks.append(block)
    if not blocks:
        raise ValueError(f"No DR2 tracer rows for mode={mode_key!r}")
    stacked = _stack_bao_blocks(
        blocks,
        label=f"DESI_DR2_{mode_key}_combined",
        tracer=COMBINED_DR2_TRACER,
        data_mode=mode_key,
        dedupe_z_tol=z_tol,
        prefer_last=True,
    )
    stacked["s"] = s_from_z(stacked["z"], gamma=gamma)
    stacked["tracer_stack_order"] = tracers
    stacked["dedupe_z_tol"] = z_tol
    return stacked


def load_extended_bao_data(
    mode: str = DATA_MODE,
    *,
    local_path: str | Path | None = None,
    tracer: str = TRACER,
    gamma: float = 10.0,
    legacy_keys: Sequence[str] | None = None,
    dedupe_z_tol: float = STANDARD_DEDUPE_Z_TOL,
    high_power_stack: bool = False,
) -> dict[str, Any]:
    """
    DESI DR2 + BOSS/eBOSS stack for extended s-space coverage.

    Legacy tables are prepended. Default |Δz|≤0.02 dedupe defers to DESI; set
    ``high_power_stack=True`` (or ``dedupe_z_tol=0``) to retain distinct legacy bins.
    """
    if high_power_stack:
        dedupe_z_tol = HIGH_POWER_DEDUPE_Z_TOL
    mode_key = normalize_data_mode(mode)
    quantity_filter = data_mode_to_quantity_filter(mode_key)
    blocks: list[dict[str, Any]] = []

    keys = list(legacy_keys or LEGACY_BAO_DATASETS)
    for key in keys:
        try:
            legacy = load_legacy_bao_dataset(
                key,
                local_path=local_path,
                quantity_filter=quantity_filter,
            )
            blocks.append(legacy)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  ⚠ Skipping legacy {key}: {exc}")

    desi = load_scan_data_by_data_mode(
        mode_key,
        local_path=local_path,
        tracer=tracer,
        gamma=gamma,
    )
    desi["survey"] = f"DESI DR2 ({tracer})"
    blocks.append(desi)

    label_suffix = "_high_power" if dedupe_z_tol <= 0.0 else ""
    stacked = _stack_bao_blocks(
        blocks,
        label=f"DESI_DR2+LEGACY_{mode_key}_{tracer}{label_suffix}",
        tracer=EXTENDED_BAO_TRACER,
        data_mode=mode_key,
        dedupe_z_tol=dedupe_z_tol,
        prefer_last=True,
    )
    stacked["s"] = s_from_z(stacked["z"], gamma=gamma)
    stacked["high_power_stack"] = dedupe_z_tol <= 0.0
    stacked["dedupe_z_tol"] = dedupe_z_tol
    return stacked


# ============================================================
# 2. TAU-SB THEORETICAL MAPPING
# ============================================================


def calibrate_gamma_from_hierarchy(
    *,
    z_min: float,
    z_max: float,
    n_hier: float = N_HIER_BINDING,
    delta_n: float = LATE_UNIVERSE_DELTA_N,
    n_cycles_in_band: float = 1.0,
    period: float = TAU_RESONANCE_PERIOD,
) -> float:
    """
    Calibrate cylinder stretch γ so that ``n_cycles_in_band`` periods of 1/7
    span the redshift interval [z_min, z_max] in s-space.

    s(z) = -γ ln(1+z).  One Tau resonance cycle spans ``period`` s-units.
    Hierarchical correction scales γ using n_hier ≈ 45.8 and late-universe Δn.
    """
    if z_max <= z_min:
        z_max = z_min + 0.05
    log_stretch = float(np.log((1.0 + z_max) / (1.0 + z_min)))
    gamma_base = n_cycles_in_band * period / log_stretch
    hier_factor = 1.0 + delta_n * (n_hier / N_HIER_BINDING - 1.0) / TAU_RESONANCE_PERIOD
    return float(gamma_base * hier_factor)


def compute_tsb_rd(
    gamma: float,
    n_hier: float = N_HIER,
    *,
    delta_n_plasma: float = DELTA_N_PLASMA,
    tau_period: float = TAU_PERIOD,
    rd_observed: float = RD_OBSERVED_MPC,
) -> dict[str, Any]:
    """
    Compute emergent sound horizon r_d from TSB parameters.

    Uses simplified geometric mapping from Tau cylinder axial advance during the
    plasma epoch (``sound_horizon_tsb_integration.md``).
    """
    g = float(gamma)
    if not np.isfinite(g) or g <= 0:
        raise ValueError(f"gamma must be a positive finite float, got {gamma}")

    # Axial advance on the Tau cylinder during plasma epoch
    delta_s_plasma = float(delta_n_plasma) * float(tau_period) / g

    # Effective phase velocity scaling (placeholder for future v_phase(s) from resonance graph)
    # For now we use a simple scaling that reproduces ~147 Mpc when gamma ≈ 8.85
    v_phase_scale = 1.0

    rd_predicted = delta_s_plasma * v_phase_scale * (float(rd_observed) / float(tau_period))
    rd_residual_percent = 100.0 * (rd_predicted - float(rd_observed)) / float(rd_observed)

    return {
        "rd_predicted_mpc": float(rd_predicted),
        "rd_observed_mpc": float(rd_observed),
        "delta_s_plasma": float(delta_s_plasma),
        "n_plasma": float(delta_n_plasma),
        "gamma_used": g,
        "n_hier": float(n_hier),
        "tau_period": float(tau_period),
        "v_phase_scale": float(v_phase_scale),
        "rd_residual_percent": float(rd_residual_percent),
        "rd_residual_fraction": float(rd_residual_percent / 100.0),
    }


def compute_tsb_sound_horizon(
    gamma: float,
    *,
    rd_observed_mpc: float = RD_OBSERVED_MPC,
    n_hier: float = N_HIER_BINDING,
    **kwargs: Any,
) -> dict[str, Any]:
    """Alias for :func:`compute_tsb_rd` (legacy name used in diagnostics pipeline)."""
    return compute_tsb_rd(
        gamma,
        n_hier=n_hier,
        rd_observed=rd_observed_mpc,
        **kwargs,
    )


def print_tsb_sound_horizon_summary(
    tsb_rd: dict[str, Any],
    *,
    rd_residual: float | None = None,
) -> None:
    """Print the standard TSB sound-horizon comparison block."""
    if rd_residual is not None:
        res_pct = float(rd_residual) * 100.0
    elif tsb_rd.get("rd_residual_percent") is not None:
        res_pct = float(tsb_rd["rd_residual_percent"])
    else:
        res_pct = float(tsb_rd.get("rd_residual_fraction", float("nan"))) * 100.0
    rd_obs = float(tsb_rd.get("rd_observed_mpc", tsb_rd.get("rd_observed", RD_OBSERVED_MPC)))
    print("\n[TSB Sound Horizon]")
    print(f"  Predicted r_d : {float(tsb_rd['rd_predicted_mpc']):.2f} Mpc")
    print(f"  Observed r_d  : {rd_obs:.2f} Mpc")
    print(f"  Residual      : {res_pct:+.2f}%")
    if tsb_rd.get("delta_s_plasma") is not None:
        print(f"  Δs_plasma     : {float(tsb_rd['delta_s_plasma']):.3f} (γ={tsb_rd.get('gamma_used', 'n/a')})")


def whim_web_correction(
    s: np.ndarray,
    gamma: float,
    *,
    strength: float = WEB_1D_CONSTRAINT_STRENGTH,
    period: float = TAU_PERIOD,
    gamma_reference: float = TSB_RD_GAMMA_FALLBACK,
) -> np.ndarray:
    """
    Simple phenomenological correction inspired by WHIM/web 1D constraints.

    Mild scale-dependent modulation along the s-direction; amplitude scales with
    ``gamma / gamma_reference`` (default reference ≈ 8.851).
    """
    s_arr = np.asarray(s, dtype=float)
    g_ref = float(gamma_reference)
    if not np.isfinite(g_ref) or g_ref <= 0:
        g_ref = float(TSB_RD_GAMMA_FALLBACK)
    return (
        float(strength)
        * np.sin(2.0 * np.pi * s_arr / (float(period) * 2.0))
        * (float(gamma) / g_ref)
    )


def whim_web_density_modulation(
    distance_mpc: np.ndarray,
    *,
    whim_density_factor: float = WHIM_DENSITY_FACTOR,
    web_1d_strength: float = WEB_1D_CONSTRAINT_STRENGTH,
    bec_coherence_length_mpc: float = BEC_COHERENCE_LENGTH_MPC,
    n_hier: float = N_HIER_BINDING,
    tau_period: float = TAU_PERIOD,
) -> np.ndarray:
    """
    Fractional effective-density modulation along a 1D cosmological-web coordinate.

    Combines WHIM fractional density, BEC coherence wells (scale ``bec_coherence_length_mpc``,
    linked to ``n_hier``), and mild 1D filament constraint strength.
    """
    d = np.asarray(distance_mpc, dtype=float)
    phase = 2.0 * np.pi * d / float(bec_coherence_length_mpc)
    hier_scale = float(n_hier) / float(tau_period)
    bec_wells = 0.5 * (1.0 + np.cos(phase * hier_scale))
    whim_term = 1.0 + float(whim_density_factor) * bec_wells
    web_term = 1.0 + float(web_1d_strength) * np.sin(phase)
    return whim_term * web_term - 1.0


def compute_whim_web_context(
    distance_mpc: np.ndarray | None = None,
    *,
    s: np.ndarray | None = None,
    gamma: float | None = None,
    n_hier: float = N_HIER_BINDING,
    whim_density_factor: float = WHIM_DENSITY_FACTOR,
    web_1d_strength: float = WEB_1D_CONSTRAINT_STRENGTH,
    bec_coherence_length_mpc: float = BEC_COHERENCE_LENGTH_MPC,
) -> dict[str, Any]:
    """Return Sonato WHIM / web parameters and optional modulation along ``distance_mpc`` / ``s``."""
    ctx: dict[str, Any] = {
        "whim_density_factor": float(whim_density_factor),
        "web_1d_constraint_strength": float(web_1d_strength),
        "bec_coherence_length_mpc": float(bec_coherence_length_mpc),
        "n_hier_binding": float(n_hier),
        "bec_wells_per_coherence_length": float(n_hier) / float(bec_coherence_length_mpc),
        "note": "WHIM / cosmological web parameters (Sonato paper)",
    }
    if distance_mpc is not None:
        mod = whim_web_density_modulation(
            distance_mpc,
            whim_density_factor=whim_density_factor,
            web_1d_strength=web_1d_strength,
            bec_coherence_length_mpc=bec_coherence_length_mpc,
            n_hier=n_hier,
        )
        ctx["modulation_fraction"] = np.asarray(mod, dtype=float).tolist()
        ctx["modulation_mean"] = float(np.mean(mod))
        ctx["modulation_max"] = float(np.max(mod))
    if s is not None and gamma is not None:
        s_corr = whim_web_correction(s, gamma, strength=web_1d_strength)
        ctx["s_correction"] = np.asarray(s_corr, dtype=float).tolist()
        ctx["s_correction_mean"] = float(np.mean(s_corr))
        ctx["s_correction_max"] = float(np.max(np.abs(s_corr)))
        ctx["gamma_used"] = float(gamma)
    return ctx


def print_whim_web_summary(whim_web: dict[str, Any]) -> None:
    """Print the standard WHIM / cosmological-web parameter block."""
    print("\n[WHIM / Cosmological Web]")
    print(f"  WHIM density factor     : {whim_web['whim_density_factor']:.3f}")
    print(f"  1D web constraint       : {whim_web['web_1d_constraint_strength']:.3f}")
    print(f"  BEC coherence length    : {whim_web['bec_coherence_length_mpc']:.1f} Mpc")
    print(
        f"  n_hier / L_BEC          : {whim_web['bec_wells_per_coherence_length']:.2f} wells/Mpc-scale"
    )
    if whim_web.get("modulation_mean") is not None:
        print(f"  Mean ρ_eff excess       : {whim_web['modulation_mean']:+.4f}")
    if whim_web.get("s_correction_mean") is not None:
        print(f"  s-direction correction  : {whim_web['s_correction_mean']:+.5f} (mean)")


def apply_tsb_sound_horizon(
    results: dict[str, Any],
    best_gamma: float,
    *,
    rd_observed_mpc: float = RD_OBSERVED_MPC,
    print_summary: bool = True,
) -> dict[str, Any]:
    """
    Attach TSB sound-horizon prediction to a diagnostics/fit results dict.

    Matches the post-fit workflow::

        tsb_rd = compute_tsb_rd(best_gamma)
        results["tsb_sound_horizon"] = tsb_rd
        results["rd_residual"] = tsb_rd["rd_residual_percent"] / 100
    """
    tsb_rd = compute_tsb_rd(best_gamma, rd_observed=rd_observed_mpc)
    results["tsb_sound_horizon"] = tsb_rd
    results["rd_residual"] = float(tsb_rd["rd_residual_fraction"])
    results["rd_residual_percent"] = float(tsb_rd["rd_residual_percent"])
    if print_summary:
        print_tsb_sound_horizon_summary(tsb_rd, rd_residual=results["rd_residual"])
    return tsb_rd


def plot_rd_vs_gamma(
    gamma_range: tuple[float, float] = (5.0, 15.0),
    n_points: int = 41,
    *,
    best_gamma: float | None = None,
    save_plot: bool = False,
    plot_dir: str | Path = "plots",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot how predicted sound horizon varies with γ, with observed r_d as reference.
    """
    gammas = np.linspace(float(gamma_range[0]), float(gamma_range[1]), int(n_points))
    rd_values = np.array([compute_tsb_rd(float(g))["rd_predicted_mpc"] for g in gammas], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(gammas, rd_values, color="tab:blue", linewidth=2, label="Predicted r_d (TSB)")
    ax.axhline(
        RD_OBSERVED_MPC,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Observed r_d = {RD_OBSERVED_MPC} Mpc",
    )
    cross_gamma = float(DELTA_N_PLASMA)
    if gamma_range[0] <= cross_gamma <= gamma_range[1]:
        ax.axvline(
            cross_gamma,
            color="gray",
            linestyle=":",
            linewidth=1.5,
            label=f"γ = Δn_plasma = {cross_gamma:.2f}",
        )
    if best_gamma is not None and np.isfinite(float(best_gamma)):
        ax.axvline(
            float(best_gamma),
            color="tab:orange",
            linestyle="-.",
            linewidth=1.5,
            label=f"Best γ = {float(best_gamma):.2f}",
        )

    ax.set_xlabel("γ (cylinder stretch)")
    ax.set_ylabel("Predicted Sound Horizon r_d (Mpc)")
    ax.set_title("TSB Emergent Sound Horizon vs γ")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    out_path = Path(save_path) if save_path else None
    if save_plot or out_path is not None:
        if out_path is None:
            out_path = Path(plot_dir) / "rd_vs_gamma.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig


def s_from_z(
    z: np.ndarray,
    gamma: float = 10.0,
    z_pivot: float = 0.0,
    *,
    n_hier: float | None = None,
    delta_n: float = LATE_UNIVERSE_DELTA_N,
    auto_calibrate: bool = False,
) -> np.ndarray:
    """
    Map redshift to Tau cylinder axial coordinate s ≈ -γ ln(1+z).

    When ``auto_calibrate=True``, γ is set from ``n_hier`` and the z-range of
    the input array so the expected frequency is exactly 1/7 in s-units.
    """
    z_arr = np.asarray(z, dtype=float)
    if auto_calibrate:
        gamma = calibrate_gamma_from_hierarchy(
            z_min=float(np.min(z_arr)),
            z_max=float(np.max(z_arr)),
            n_hier=n_hier or N_HIER_BINDING,
            delta_n=delta_n,
        )
    return -gamma * np.log(1.0 + z_arr - z_pivot)


def tau_sb_oscillatory_residual(
    s: np.ndarray,
    A: float = 0.015,
    period: float = 7.0,
    phase: float = 0.0,
    *,
    amplitude_scale: float = 1.0,
) -> np.ndarray:
    """
    Fractional oscillation: ``A`` is dimensionless (typical |A| ≲ 0.05).

    Physical modulation = ``A * amplitude_scale * sin(...)`` where
    ``amplitude_scale`` is usually median(|observable|) for BAO data.
    """
    return A * amplitude_scale * np.sin(2 * np.pi * s / period + phase)


def ade_dark_energy_modulation(
    z: np.ndarray,
    *,
    amplitude: float = 0.01,
    omega: float = 2.0,
    phase: float = 0.0,
    z_star: float = 0.5,
) -> np.ndarray:
    """
    Toy ultralight axion + Λ (aDE) template in redshift space.

    Smooth envelope × cosine — distinct from Tau-SB log-periodic s-space track.
    """
    z_arr = np.asarray(z, dtype=float)
    envelope = 1.0 - np.exp(-z_arr / z_star)
    return amplitude * envelope * np.cos(omega * z_arr + phase)


def comoving_distance_hmpc(
    z: np.ndarray | list[float],
    *,
    om: float = OMEGA_M_FIDUCIAL,
    h0_h: float = H0_H_UNITS,
) -> np.ndarray:
    """Flat ΛCDM comoving distance D_C(z) in h⁻¹ Mpc (H₀=100 h convention)."""
    z_arr = np.asarray(z, dtype=float)
    if z_arr.size == 0:
        return np.array([], dtype=float)
    z_max = float(np.max(z_arr))
    n = max(128, int(z_max * 64) + 1)
    zg = np.linspace(0.0, z_max * 1.05 + 1e-6, n)
    ez = np.sqrt(om * (1.0 + zg) ** 3 + (1.0 - om))
    integrand = 1.0 / ez
    dz = np.diff(zg, prepend=0.0)
    dc = (299792.458 / h0_h) * np.cumsum(0.5 * (integrand + np.roll(integrand, 1)) * dz)
    dc[0] = 0.0
    return np.interp(z_arr, zg, dc)


def hubble_distance_hmpc(
    z: np.ndarray | list[float],
    *,
    om: float = OMEGA_M_FIDUCIAL,
    h0_h: float = H0_H_UNITS,
) -> np.ndarray:
    """Hubble distance D_H(z) = c/H(z) in h⁻¹ Mpc."""
    z_arr = np.asarray(z, dtype=float)
    ez = np.sqrt(om * (1.0 + z_arr) ** 3 + (1.0 - om))
    return 299792.458 / (h0_h * ez)


def w0wa_dark_energy_modulation(
    z: np.ndarray,
    *,
    w0: float = -0.8,
    wa: float = -0.5,
    amplitude_scale: float = 1.0,
) -> np.ndarray:
    """
    CPL dynamical dark energy correction (DESI DR2 preferred quadrant: w₀ > −1, w_a < 0).

    Phenomenological distance-integral proxy on BAO observables:
    w(a) = w₀ + w_a(1 − a),  a = 1/(1+z).
    """
    z_arr = np.asarray(z, dtype=float)
    a = 1.0 / (1.0 + z_arr)
    w_a_cpl = w0 + wa * (1.0 - a)
    integral_proxy = (w0 + 1.0) * np.log1p(z_arr) + wa * z_arr / (1.0 + z_arr)
    return amplitude_scale * 0.01 * integral_proxy * (1.0 + 0.5 * (w_a_cpl + 1.0))


def tau_sb_hierarchical_step(
    z: np.ndarray,
    transition_zs: list[float] | None = None,
    amplitude: float = 0.01,
) -> np.ndarray:
    if transition_zs is None:
        transition_zs = [0.3, 1.0, 2.0]
    step = np.zeros_like(z, dtype=float)
    for zt in transition_zs:
        step += amplitude * np.tanh(10 * (z - zt))
    return step


def tau_sb_fit_prediction(
    z: np.ndarray,
    baseline_coeffs: np.ndarray | Sequence[float],
    *,
    A_osc_frac: float,
    phase_rad: float = 0.0,
    hier_frac: float = 0.0,
    gamma: float = 10.0,
    period: float = TAU_RESONANCE_PERIOD,
    amplitude_scale: float,
    use_hier: bool = True,
) -> np.ndarray:
    """
    Noiseless Tau-SB model vector — identical structure to :meth:`TauSBScanner.fit_tau_sb_model`.
    """
    z_arr = np.asarray(z, dtype=float)
    coeffs = np.asarray(baseline_coeffs, dtype=float)
    s = s_from_z(z_arr, gamma=gamma)
    baseline = np.polyval(coeffs, z_arr)
    osc = tau_sb_oscillatory_residual(
        s,
        A=float(A_osc_frac),
        period=period,
        phase=float(phase_rad),
        amplitude_scale=float(amplitude_scale),
    )
    hier = (
        tau_sb_hierarchical_step(z_arr, amplitude=float(hier_frac) * amplitude_scale * 0.01)
        if use_hier
        else 0.0
    )
    return baseline + osc + hier


def tau_sb_model_scale_dependent(
    s: np.ndarray,
    A_osc_frac: float,
    phase_rad: float,
    gamma: float,
    *,
    hier_frac: float = 0.0,
    baseline_params: list[float] | tuple[float, ...] | None = None,
    scale_dependence: float = 0.0,
    use_tsb_rd: bool = True,
    period: float = TAU_PERIOD,
    rd_observed_mpc: float = RD_OBSERVED_MPC,
) -> np.ndarray:
    """
    Tau-SB model on the s-grid with optional mild scale dependence in amplitude.

    ``scale_dependence > 0`` modulates the oscillation amplitude linearly with
    normalized s (zero mean, unit variance). When ``use_tsb_rd`` is True, the
    oscillation period is stretched by the TSB sound-horizon ratio
    ``r_d(γ) / r_d,obs``.

    Parameters
    ----------
    s:
        Cylinder coordinate grid (from :func:`s_from_z`).
    A_osc_frac, phase_rad:
        Oscillation amplitude (fractional) and phase (radians).
    gamma:
        Cylinder stretch γ; used for TSB ``r_d`` when ``use_tsb_rd=True``.
    hier_frac:
        Weight on the secondary s-space harmonic at period ``2 * period``.
    baseline_params:
        Quadratic baseline ``[c0, c1, c2]`` in s-units.
    scale_dependence:
        Linear amplitude slope vs normalized s; 0 disables scale dependence.
    use_tsb_rd:
        If True, scale oscillation period by TSB predicted/observed sound horizon.

    Examples
    --------
    After building s from redshifts::

        s = s_from_z(z, gamma=8.8511)
        y_model = tau_sb_model_scale_dependent(
            s, A_osc_frac=0.01, phase_rad=0.0, gamma=8.8511,
            baseline_params=[1.0, 0.0, 0.0],
        )
        residuals = y_obs - y_model
    """
    s_arr = np.asarray(s, dtype=float)
    if baseline_params is None:
        baseline_params = [0.0, 0.0, 0.0]
    c0, c1, c2 = (list(baseline_params) + [0.0, 0.0, 0.0])[:3]

    if use_tsb_rd:
        tsb = compute_tsb_rd(gamma, rd_observed=rd_observed_mpc, tau_period=period)
        rd_scale = float(tsb["rd_predicted_mpc"]) / float(rd_observed_mpc)
    else:
        rd_scale = 1.0

    effective_period = float(period) * rd_scale
    baseline = c0 + c1 * s_arr + c2 * s_arr**2

    s_std = float(np.std(s_arr, ddof=0))
    if s_std > 0 and scale_dependence != 0.0:
        s_norm = (s_arr - float(np.mean(s_arr))) / s_std
        amplitude = float(A_osc_frac) * (1.0 + float(scale_dependence) * s_norm)
    else:
        amplitude = np.full_like(s_arr, float(A_osc_frac), dtype=float)

    oscillation = amplitude * np.sin(2.0 * np.pi * s_arr / effective_period + float(phase_rad))
    hier_mod = float(hier_frac) * np.sin(2.0 * np.pi * s_arr / (float(period) * 2.0))
    return baseline + oscillation + hier_mod


def tau_sb_model_with_whim(
    s: np.ndarray,
    A_osc_frac: float,
    phase_rad: float,
    gamma: float,
    *,
    hier_frac: float = 0.0,
    baseline_params: list[float] | tuple[float, ...] | None = None,
    whim_strength: float = WEB_1D_CONSTRAINT_STRENGTH,
    use_tsb_rd: bool = True,
    period: float = TAU_PERIOD,
    rd_observed_mpc: float = RD_OBSERVED_MPC,
) -> np.ndarray:
    """
    Tau-SB model with WHIM / cosmological-web 1D constraint correction.

    Combines TSB sound-horizon period stretch, quadratic s-baseline, primary
    oscillation, :func:`whim_web_correction`, and optional hierarchical harmonic.

    Examples
    --------
    ::

        s = s_from_z(z, gamma=8.8511)
        y_model = tau_sb_model_with_whim(
            s, A_osc_frac=0.01, phase_rad=0.0, gamma=8.8511,
            baseline_params=[1.0, 0.0, 0.0],
        )
        residuals = y_obs - y_model
    """
    s_arr = np.asarray(s, dtype=float)
    if baseline_params is None:
        baseline_params = [0.0, 0.0, 0.0]
    c0, c1, c2 = (list(baseline_params) + [0.0, 0.0, 0.0])[:3]

    if use_tsb_rd:
        tsb = compute_tsb_rd(gamma, rd_observed=rd_observed_mpc, tau_period=period)
        rd_scale = float(tsb["rd_predicted_mpc"]) / float(rd_observed_mpc)
    else:
        rd_scale = 1.0

    effective_period = float(period) * rd_scale
    baseline = c0 + c1 * s_arr + c2 * s_arr**2
    oscillation = float(A_osc_frac) * np.sin(
        2.0 * np.pi * s_arr / effective_period + float(phase_rad)
    )
    web_correction = whim_web_correction(s_arr, gamma, strength=whim_strength, period=period)
    hier_mod = float(hier_frac) * np.sin(2.0 * np.pi * s_arr / (float(period) * 2.0))
    return baseline + oscillation + web_correction + hier_mod


def generate_tau_sb_mock_data(
    z: np.ndarray,
    *,
    baseline: float = 1.0,
    A_osc_frac: float | None = None,
    A_osc: float | None = None,
    gamma: float = 10.0,
    noise_level: float | None = None,
    err: np.ndarray | None = None,
    cov: np.ndarray | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Synthetic Tau-SB signal for injection/recovery tests only (not a scan data source)."""
    rng = np.random.default_rng(seed)
    a_frac = float(A_osc_frac if A_osc_frac is not None else (A_osc if A_osc is not None else 0.02))
    s = s_from_z(z, gamma=gamma)
    amp_scale = _oscillation_scale(np.full(len(z), float(baseline)))
    osc = tau_sb_oscillatory_residual(s, A=a_frac, amplitude_scale=amp_scale)
    hier = tau_sb_hierarchical_step(z, amplitude=0.005 * amp_scale)
    true_model = baseline + osc + hier
    if err is not None:
        err_arr = np.asarray(err, dtype=float)
        noise = rng.normal(0, 1.0, size=len(z)) * err_arr
        cov_use = np.asarray(cov, dtype=float) if cov is not None else np.diag(err_arr**2)
    else:
        sigma = float(noise_level if noise_level is not None else 0.008)
        err_arr = np.full(len(z), sigma)
        noise = rng.normal(0, sigma, size=len(z))
        cov_use = np.diag(err_arr**2)
    observable = true_model + noise
    return {
        "z": z,
        "observable": observable,
        "err": err_arr,
        "cov": cov_use,
        "true_model": true_model,
        "s": s,
        "A_osc_frac": a_frac,
        "amplitude_scale": amp_scale,
        "label": "Tau-SB_injected_mock",
        "tracer": "mock",
        "n_data": len(z),
        "has_full_covariance": cov is not None,
    }


# ============================================================
# 3. LIKELIHOOD / MODEL COMPARISON
# ============================================================


def gaussian_chi2(
    observed: np.ndarray,
    model: np.ndarray,
    err: np.ndarray | None = None,
    cov: np.ndarray | None = None,
) -> float:
    residual = np.asarray(observed, dtype=float) - np.asarray(model, dtype=float)
    if cov is not None:
        cov = np.asarray(cov, dtype=float)
        try:
            return float(residual @ np.linalg.solve(cov, residual))
        except np.linalg.LinAlgError:
            from menus.astronomical.desi.analysis import apply_tikhonov_regularization

            cov, _ = apply_tikhonov_regularization(cov)
            return float(residual @ np.linalg.solve(cov, residual))
    if err is None:
        raise ValueError("Provide err or cov for chi2")
    err = np.asarray(err, dtype=float)
    return float(np.sum((residual / err) ** 2))


def _aic(chi2_val: float, n_params: int) -> float:
    return float(chi2_val + 2 * n_params)


def _bic(chi2_val: float, n_params: int, n_data: int) -> float:
    return float(chi2_val + n_params * np.log(max(n_data, 1)))


def _bayes_factor_from_delta_bic(delta_bic: float) -> float:
    """Rough Occam penalty: BF_12 ≈ exp(-ΔBIC/2) favoring model 1 if ΔBIC<0."""
    return float(np.exp(np.clip(-0.5 * delta_bic, -50.0, 50.0)))


@dataclass
class ModelFitResult:
    name: str
    chi2: float
    n_params: int
    n_data: int
    aic: float
    bic: float
    params: np.ndarray
    model_at_z: Callable[[np.ndarray], np.ndarray]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "chi2": self.chi2,
            "n_params": self.n_params,
            "aic": self.aic,
            "bic": self.bic,
            "params": [float(x) for x in self.params],
        }


# ============================================================
# 4. SCANNERS
# ============================================================


@dataclass
class DesiScanResult:
    data_label: str
    tracer: str = ""
    gamma_used: float = 10.0
    periodicity: dict[str, Any] | None = None
    fit: dict[str, Any] | None = None
    model_comparison: dict[str, Any] | None = None
    dipole: dict[str, Any] | None = None
    mcmc: dict[str, Any] | None = None
    changepoints: dict[str, Any] | None = None
    injection_recovery: dict[str, Any] | None = None
    joint_fit: dict[str, Any] | None = None
    nested_evidence: dict[str, Any] | None = None
    tav_harmonics: dict[str, Any] | None = None
    power_assessment: dict[str, Any] | None = None
    gamma_diagnostics: dict[str, Any] | None = None
    auto_diagnostics: dict[str, Any] | None = None
    residual_diagnostics: dict[str, Any] | None = None
    harmonic_binding_kit: dict[str, Any] | None = None
    recommended_next_steps: list[str] | None = None
    tsb_sound_horizon: dict[str, Any] | None = None
    rd_residual: float | None = None
    plot_path: str | None = None
    report_path: str | None = None

    def summary_lines(self) -> list[str]:
        lines = [
            "Tau-SB DESI BAO scan (live DR2 Cobaya tables)",
            f"Dataset: {self.data_label}",
            f"Tracer: {self.tracer or 'n/a'}",
            f"γ (cylinder stretch): {self.gamma_used:.4f}",
        ]
        lines.extend(_format_power_assessment_lines(self.power_assessment))
        if self.periodicity:
            per = self.periodicity
            max_power = float(per["max_power"])
            power_at_f = float(per["power_at_expected"])
            th = self.tav_harmonics or {}
            lomb_hit = bool(th.get("lomb_detected", False))
            tav_hit = bool(th.get("tav_resonance_detected", False))
            p_boot = th.get("bootstrap_p_value", per.get("pval_bootstrap", per["pval_approx"]))
            lines.append(
                f"1/7 track: Lomb-Scargle peak={lomb_hit} "
                f"(power@f=1/7={power_at_f:.4e}, max={max_power:.4e} @ f={per['max_freq']:.4f}, "
                f"bootstrap p≈{p_boot:.4g})"
            )
        if self.tav_harmonics:
            th = self.tav_harmonics
            if th.get("tav_resonance_skipped"):
                lines.append(f"1/7 track: tav-resonance skipped — {th.get('tav_skip_reason', 'n<8')}")
            else:
                lines.append(
                    f"1/7 track: tav-resonance residual check={tav_hit} "
                    f"(peaks={th.get('harmonic_peaks', [])})"
                )
            if th.get("bootstrap_p_value") is not None:
                lines.append(
                    f"1/7 track: bootstrap p≈{th['bootstrap_p_value']:.4f} "
                    f"(n_freq_bins={th.get('n_freq_bins', 'n/a')})"
                )
            if th.get("underpowered_warning"):
                lines.append(f"⚠ {th['underpowered_warning']}")
            if th.get("s_grid_warning"):
                lines.append(f"⚠ {th.get('s_grid_warning_message', 'narrow s-range')}")
            fixed = self.periodicity.get("fixed_gamma_diagnostic")
            if fixed:
                src = fixed.get("residual_source", "poly_baseline")
                p_nc = fixed.get("bootstrap_p_1_7", fixed.get("bootstrap_p_value"))
                lines.append(
                    f"1/7 track (fixed γ={fixed['gamma']:.1f}, non-circular, {src}): "
                    f"bootstrap p≈{p_nc:.4g}, "
                    f"cycles@τ={fixed['cycles_possible']:.2f}"
                )
            tau_sb_nc = self.periodicity.get("non_circular_tau_sb_fit")
            if tau_sb_nc:
                p_fit = tau_sb_nc.get("bootstrap_p_1_7", tau_sb_nc.get("bootstrap_p_value"))
                lines.append(
                    f"1/7 track (fixed γ={tau_sb_nc['gamma']:.1f}, Tau-SB fit residuals): "
                    f"bootstrap p≈{p_fit:.4g}, cycles@τ={tau_sb_nc['cycles_possible']:.2f}"
                )
        gamma_diag = self.gamma_diagnostics
        if gamma_diag is None and self.periodicity:
            gamma_diag = self.periodicity.get("gamma_diagnostics")
        if self.tsb_sound_horizon:
            tsb = self.tsb_sound_horizon
            if tsb.get("rd_residual_percent") is not None:
                res_pct = float(tsb["rd_residual_percent"])
            elif self.rd_residual is not None:
                res_pct = float(self.rd_residual) * 100.0
            else:
                res_pct = float(tsb.get("rd_residual_fraction", float("nan"))) * 100.0
            lines.append(
                f"TSB sound horizon: r_d^pred={tsb.get('rd_predicted_mpc', float('nan')):.2f} Mpc "
                f"vs obs={tsb.get('rd_observed_mpc', float('nan')):.2f} Mpc "
                f"(residual={res_pct:+.2f}%, Δs_plasma={tsb.get('delta_s_plasma', float('nan')):.3f})"
            )
        if gamma_diag:
            lines.append(
                f"γ sensitivity (post-load): best γ≈{gamma_diag.get('best_gamma', float('nan')):.3f}"
            )
            jk_std = gamma_diag.get("gamma_jackknife_std")
            if jk_std is not None and np.isfinite(jk_std):
                lines.append(f"γ jackknife std (post-load): {jk_std:.4f}")
            if gamma_diag.get("gamma_jackknife_warning"):
                lines.append(f"⚠ {gamma_diag['gamma_jackknife_warning']}")
        if self.fit:
            a_frac = self.fit.get("A_osc_frac", self.fit.get("A_osc"))
            lines.append(
                f"Tau-SB fit: χ²={self.fit['chi2']:.2f} "
                f"(reduced={self.fit.get('reduced_chi2', float('nan')):.2f}, dof={self.fit.get('dof', 'n/a')}), "
                f"A_osc_frac≈{a_frac:.4f}"
            )
            if self.fit.get("null_comparison"):
                nc = self.fit["null_comparison"]
                lines.append(
                    f"  vs null: Δχ²={nc.get('delta_chi2_vs_null', float('nan')):.2f}, "
                    f"ΔAIC={nc.get('delta_aic_vs_null', float('nan')):.2f}"
                )
            if self.fit.get("gamma_jackknife"):
                gj = self.fit["gamma_jackknife"]
                lines.append(
                    f"  γ jackknife: {gj.get('gamma_median', float('nan')):.4f} "
                    f"± {gj.get('gamma_std', 0.0):.4f}"
                )
            mc = self.fit.get("model_complexity")
            if mc:
                labels = ", ".join(mc.get("parameter_labels", []))
                lines.append(
                    f"  Model: {mc.get('n_params')} free params ({labels}), "
                    f"baseline deg={mc.get('baseline_degree')}, dof={mc.get('dof')}"
                )
        if self.model_comparison:
            mc = self.model_comparison
            lines.append("Model comparison (lower AIC/BIC is better):")
            for row in mc.get("ranking", []):
                lines.append(
                    f"  {row['name']}: χ²={row['chi2']:.2f}, AIC={row['aic']:.2f}, BIC={row['bic']:.2f}"
                )
            bf = mc.get("bayes_factors_vs_tau_sb", {})
            if bf:
                lines.append(
                    f"BF(Tau-SB / ΛCDM)≈{bf.get('vs_lcdm', float('nan')):.3f}, "
                    f"BF(Tau-SB / aDE)≈{bf.get('vs_ade', float('nan')):.3f}"
                )
        if self.mcmc:
            a = self.mcmc["posteriors"]["A_osc"]
            lines.append(
                f"MCMC A_osc: {a['median']:.4f} (+{a['high']-a['median']:.4f}/"
                f"-{a['median']-a['low']:.4f})"
            )
        if self.changepoints:
            lines.append(
                f"Change-points: z={self.changepoints.get('transition_zs', [])} "
                f"(ref n_hier z={self.changepoints.get('n_hier_reference_zs', [])})"
            )
        if self.harmonic_binding_kit:
            sm = self.harmonic_binding_kit.get("summary") or {}
            lines.append(
                f"Harmonic binding kit: "
                f"transitions={sm.get('n_detected_transitions', 0)}, "
                f"alignment={sm.get('alignment_fraction', 0.0):.2f}, "
                f"1/7 sig={sm.get('harmonic_ladder_significant', False)}"
            )
        if self.recommended_next_steps:
            lines.append("Recommended next steps:")
            lines.extend(f"  → {step}" for step in self.recommended_next_steps)
        if self.residual_diagnostics:
            rd = self.residual_diagnostics
            shapiro = rd.get("shapiro_wilk") or {}
            dw = rd.get("durbin_watson_stat")
            lines.append(
                "Residual diagnostics: "
                f"Shapiro normal={shapiro.get('is_normal_at_5pct', shapiro.get('normal'))}, "
                f"DW={dw if dw is not None else 'n/a'}"
            )
        if self.injection_recovery:
            ir = self.injection_recovery
            if ir.get("amplitude_sweep"):
                infl = ir.get("median_inflation_factor")
                debiased = ir.get("debiased_A_osc_frac")
                lines.append(
                    f"Injection/recovery sweep: "
                    f"{ir.get('n_trials_per_amplitude', '?')} trials × "
                    f"{len(ir['amplitude_sweep'])} amplitudes, "
                    f"n_z={ir.get('n_z', '?')}"
                )
                if infl is not None:
                    lines.append(
                        f"  median inflation={infl:.2f}× | "
                        f"debiased A_osc_frac≈{debiased:.4f}"
                        if debiased is not None
                        else f"  median inflation={infl:.2f}×"
                    )
                for row in ir["amplitude_sweep"]:
                    a_true = row.get("A_true_frac", row.get("true_amplitude"))
                    a_rec = row.get("A_recovered_mean", row.get("mean_recovered"))
                    a_std = row.get("A_recovered_std", 0.0)
                    det = row.get("detection_fraction", row.get("detection_rate"))
                    lines.append(
                        f"  A_true={a_true:.3f}: "
                        f"rec={a_rec:.3f}±{a_std:.3f}, "
                        f"bias={row['bias']:+.3f}, infl={row['inflation_factor']:.2f}×, "
                        f"detect={det:.0%}, SNR≈{row.get('snr', float('nan')):.1f}"
                    )
            else:
                lines.append(
                    f"Injection/recovery: detect={ir['detection_fraction']:.0%}, "
                    f"recover={ir['recovery_fraction']:.0%}, bias={ir['bias']:.4f}"
                )
        if self.joint_fit:
            jf = self.joint_fit
            lines.append(
                f"Joint DESI+SN+Planck: χ²={jf['chi2_total']:.1f}, "
                f"r_d,eff={jf['rd_mpc_effective']:.2f} Mpc, "
                f"friction shift={jf['friction_shift_fraction']:.2e}"
            )
        if self.nested_evidence:
            ne = self.nested_evidence
            lines.append(
                f"Nested evidence (dynesty): Δln Z(Tau-SB−aDE)="
                f"{ne.get('delta_log_evidence_tau_minus_ade', float('nan')):+.2f} "
                f"→ favors {ne.get('favored_model', 'n/a')}"
            )
        if self.dipole:
            method = self.dipole.get("method", "healpy_l1")
            lines.append(
                f"Dipole ({method}): amplitude≈{self.dipole['fitted_amplitude']:.4f}"
            )
        if self.plot_path:
            lines.append(f"Plot: {self.plot_path}")
        if self.report_path:
            lines.append(f"Report: {self.report_path}")
        lines.append(
            "Framework constants (fixed, not fitted): M₀=313.1 MeV, τ period=1/7, n_hier≈45.8"
        )
        return lines


class TauSBScanner:
    def __init__(
        self,
        gamma: float = 10.0,
        period: float = TAU_RESONANCE_PERIOD,
        *,
        auto_calibrate_gamma: bool = False,
        n_hier: float = N_HIER_BINDING,
        delta_n: float = LATE_UNIVERSE_DELTA_N,
    ):
        self.gamma = gamma
        self.period = period
        self.auto_calibrate_gamma = auto_calibrate_gamma
        self.n_hier = n_hier
        self.delta_n = delta_n
        self.results: dict[str, Any] = {}

    def _gamma_for(self, z: np.ndarray) -> float:
        if self.auto_calibrate_gamma and len(z) >= 2:
            return calibrate_gamma_from_hierarchy(
                z_min=float(np.min(z)),
                z_max=float(np.max(z)),
                n_hier=self.n_hier,
                delta_n=self.delta_n,
                period=self.period,
            )
        if self.auto_calibrate_gamma:
            return calibrate_gamma_from_hierarchy(
                z_min=max(float(np.min(z)) - 0.1, 0.01),
                z_max=float(np.min(z)) + 0.5,
                n_hier=self.n_hier,
                delta_n=self.delta_n,
                period=self.period,
            )
        return self.gamma

    def compute_residuals(
        self, z: np.ndarray, observable: np.ndarray, baseline_model: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if baseline_model is None:
            deg = min(3, max(1, len(z) - 4))
            coeffs = np.polyfit(z, observable, deg=deg)
            baseline_model = np.polyval(coeffs, z)
        return observable - baseline_model, baseline_model

    def scan_periodicity(
        self,
        z: np.ndarray,
        observable: np.ndarray,
        err: np.ndarray,
        baseline_model: np.ndarray | None = None,
        *,
        cov: np.ndarray | None = None,
        model_func: Callable[..., np.ndarray] | None = None,
        freq_range: tuple[float, float] = (0.05, 0.5),
        n_freq: int = 500,
    ) -> dict[str, Any]:
        gamma_used = self._gamma_for(z)
        s = s_from_z(z, gamma=gamma_used)
        residuals, _baseline = self.compute_residuals(z, observable, baseline_model)

        from menus.astronomical.desi.stats import (
            bootstrap_lomb_scargle_pvalue,
            diagnose_frequency_grid,
            search_higher_harmonics,
        )

        grid_diag = diagnose_frequency_grid(s, period=self.period)
        n_freq_use = max(n_freq, grid_diag["recommended_n_bins"])
        freqs = np.linspace(freq_range[0], freq_range[1], n_freq_use)
        power = signal.lombscargle(s, residuals, freqs, normalize=True)

        expected_f = 1.0 / self.period
        idx_expected = int(np.argmin(np.abs(freqs - expected_f)))
        power_at_expected = float(power[idx_expected])
        max_power = float(np.max(power))
        boot = bootstrap_lomb_scargle_pvalue(s, residuals, freq=expected_f)
        pval_approx = float(boot["bootstrap_p_value"])
        tav_check = _try_tav_harmonic_check(s, residuals, period=self.period)
        tav_check["lomb_pval_asymptotic"] = float(np.exp(-max_power))
        higher_harmonics = search_higher_harmonics(
            residuals,
            s,
            fundamental_freq=1.0 / self.period,
            max_harmonic=5,
            verbose=False,
        )

        fixed_gamma_diag = None
        if self.auto_calibrate_gamma:
            from menus.astronomical.desi.analysis import run_non_circular_gamma_diagnostic

            fixed_gamma_diag = run_non_circular_gamma_diagnostic(
                z=z,
                y=observable,
                cov=cov,
                err=err,
                model_func=model_func,
                fixed_gamma=self.gamma,
                period=self.period,
            )

        self.results["periodicity"] = {
            "freqs": freqs,
            "power": power,
            "expected_f": expected_f,
            "power_at_expected": power_at_expected,
            "max_power": max_power,
            "max_freq": float(freqs[int(np.argmax(power))]),
            "pval_approx": pval_approx,
            "pval_bootstrap": pval_approx,
            "s_grid_diagnostics": grid_diag,
            "residuals": residuals,
            "s": s,
            "gamma_used": gamma_used,
            "tav_harmonics": tav_check,
            "higher_harmonics": higher_harmonics,
            "fixed_gamma_diagnostic": fixed_gamma_diag,
            "gamma_diagnostics": self.results.get("gamma_diagnostics"),
        }
        return self.results["periodicity"]

    def compare_models(
        self,
        z: np.ndarray,
        observable: np.ndarray,
        err: np.ndarray | None = None,
        cov: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Fit ΛCDM baseline, aDE template, and Tau-SB; return AIC/BIC ranking."""
        gamma_used = self._gamma_for(z)
        s = s_from_z(z, gamma=gamma_used)
        n_data = len(z)
        observed = np.asarray(observable, dtype=float)

        def _fit(
            model_builder: Callable,
            p0: list[float],
            n_params: int,
            name: str,
            *,
            bounds: list[tuple[float | None, float | None]] | None = None,
            method: str = "Nelder-Mead",
        ) -> ModelFitResult:
            def objective(p: np.ndarray) -> float:
                return gaussian_chi2(observed, model_builder(p, z, s), err=err, cov=cov)

            if bounds is not None:
                res = optimize.minimize(
                    objective, p0, method="L-BFGS-B", bounds=bounds
                )
            else:
                res = optimize.minimize(objective, p0, method=method)
            params = res.x
            pred = model_builder(params, z, s)
            chi2_val = gaussian_chi2(observed, pred, err=err, cov=cov)
            return ModelFitResult(
                name=name,
                chi2=chi2_val,
                n_params=n_params,
                n_data=n_data,
                aic=_aic(chi2_val, n_params),
                bic=_bic(chi2_val, n_params, n_data),
                params=params,
                model_at_z=lambda zz, mb=model_builder, pp=params: mb(pp, zz, s_from_z(zz, gamma_used)),
            )

        from menus.astronomical.desi.prior_bounds import (
            ADE_ZSTAR_MIN,
            ade_amplitude_bounds,
            ade_omega_bounds,
            ade_z_star_bounds,
        )

        amp_scale = _oscillation_scale(observed)
        deg = _baseline_degree(n_data, extra_params=5)

        def lcdm_model(p: np.ndarray, zz: np.ndarray, _ss: np.ndarray) -> np.ndarray:
            return np.polyval(p[: deg + 1], zz)

        def ade_model(p: np.ndarray, zz: np.ndarray, _ss: np.ndarray) -> np.ndarray:
            base = np.polyval(p[: deg + 1], zz)
            ade = ade_dark_energy_modulation(
                zz,
                amplitude=p[deg + 1] * amp_scale * 0.01,
                omega=p[deg + 2],
                phase=p[deg + 3],
                z_star=max(p[deg + 4], ADE_ZSTAR_MIN),
            )
            return base + ade

        def tau_sb_model(p: np.ndarray, zz: np.ndarray, ss: np.ndarray) -> np.ndarray:
            base = np.polyval(p[: deg + 1], zz)
            osc = tau_sb_oscillatory_residual(
                ss,
                A=p[deg + 1],
                period=self.period,
                phase=p[deg + 2],
                amplitude_scale=amp_scale,
            )
            hier = tau_sb_hierarchical_step(zz, amplitude=p[deg + 3] * amp_scale * 0.01)
            return base + osc + hier

        def w0wa_model(p: np.ndarray, zz: np.ndarray, _ss: np.ndarray) -> np.ndarray:
            base = np.polyval(p[: deg + 1], zz)
            de = w0wa_dark_energy_modulation(
                zz,
                w0=p[deg + 1],
                wa=p[deg + 2],
                amplitude_scale=amp_scale,
            )
            return base + de

        poly_bounds = [(None, None)] * (deg + 1)
        lcdm = _fit(
            lcdm_model,
            [float(np.mean(observed)), 0.0, 0.0][: deg + 1],
            deg + 1,
            "ΛCDM (poly)",
        )
        a_lo, a_hi = ade_amplitude_bounds()
        o_lo, o_hi = ade_omega_bounds()
        z_lo, z_hi = ade_z_star_bounds()
        ade_bounds = (
            poly_bounds
            + [
                (a_lo, a_hi),
                (o_lo, o_hi),
                (-np.pi, np.pi),
                (z_lo, z_hi),
            ]
        )
        ade = _fit(
            ade_model,
            list(lcdm.params[: deg + 1]) + [0.01, 2.0, 0.0, 0.5],
            deg + 5,
            "aDE (axion+Λ)",
            bounds=ade_bounds,
        )
        tau_sb = _fit(
            tau_sb_model,
            list(lcdm.params[: deg + 1]) + [0.01, 0.0, 0.005],
            deg + 4,
            "Tau-SB (1/7 + hier)",
        )
        w0wa_bounds = (
            poly_bounds
            + [
                (-1.15, -0.55),  # DESI DR2 quadrant: w₀ > −1
                (-1.8, -0.05),   # DESI DR2 quadrant: w_a < 0
            ]
        )
        w0wa = _fit(
            w0wa_model,
            list(lcdm.params[: deg + 1]) + [-0.85, -0.6],
            deg + 3,
            "w0waCDM (CPL DE)",
            bounds=w0wa_bounds,
        )

        fits = [lcdm, ade, w0wa, tau_sb]
        ranking = sorted(fits, key=lambda f: f.aic)
        delta_aic = {f.name: f.aic - ranking[0].aic for f in fits}
        delta_bic = {f.name: f.bic - ranking[0].bic for f in fits}

        bayes = {
            "vs_lcdm": _bayes_factor_from_delta_bic(tau_sb.bic - lcdm.bic),
            "vs_ade": _bayes_factor_from_delta_bic(tau_sb.bic - ade.bic),
            "vs_w0wa": _bayes_factor_from_delta_bic(tau_sb.bic - w0wa.bic),
            "w0wa_vs_lcdm": _bayes_factor_from_delta_bic(w0wa.bic - lcdm.bic),
            "note": "BF>1 favors first-named model (BIC approximation)",
        }

        self.results["model_comparison"] = {
            "gamma_used": gamma_used,
            "ranking": [f.to_dict() for f in ranking],
            "delta_aic": delta_aic,
            "delta_bic": delta_bic,
            "bayes_factors_vs_tau_sb": bayes,
            "best_model": ranking[0].name,
            "lcdm": lcdm,
            "ade": ade,
            "w0wa": w0wa,
            "tau_sb": tau_sb,
            "w0wa_params": {
                "w0": float(w0wa.params[deg + 1]),
                "wa": float(w0wa.params[deg + 2]),
            },
        }
        return self.results["model_comparison"]

    def fit_tau_sb_model(
        self,
        z: np.ndarray,
        observable: np.ndarray,
        err: np.ndarray,
        baseline_params: tuple[float, float, float] | None = None,
        *,
        cov: np.ndarray | None = None,
        fit_A: bool = True,
        fit_phase: bool = True,
        fit_hier: bool | None = None,
        fixed_A_frac: float = 0.01,
        hier_bounds: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        from menus.astronomical.desi.stats import (
            compute_model_comparison,
            fit_null_model,
            gamma_jackknife_std,
        )

        gamma_used = self._gamma_for(z)
        s = s_from_z(z, gamma=gamma_used)
        amp_scale = _oscillation_scale(observable)
        n_pts = len(z)
        # Small vectors: drop hierarchical step to avoid saturating dof.
        use_hier = (n_pts >= 8) if fit_hier is None else bool(fit_hier)
        extra = _oscillation_param_count(fit_A=fit_A, fit_phase=fit_phase, use_hier=use_hier)
        deg = _baseline_degree(n_pts, extra)

        def model_func(params: np.ndarray, zz: np.ndarray, ss: np.ndarray) -> np.ndarray:
            baseline = np.polyval(params[: deg + 1], zz)
            idx = deg + 1
            a_frac = float(params[idx]) if fit_A else float(fixed_A_frac)
            if fit_A:
                idx += 1
            phase = float(params[idx]) if fit_phase else 0.0
            if fit_phase:
                idx += 1
            hier_frac = float(params[idx]) if use_hier and len(params) > idx else 0.0
            osc = tau_sb_oscillatory_residual(
                ss, A=a_frac, period=self.period, phase=phase, amplitude_scale=amp_scale
            )
            hier = (
                tau_sb_hierarchical_step(zz, amplitude=hier_frac * amp_scale * 0.01)
                if use_hier
                else 0.0
            )
            return baseline + osc + hier

        if baseline_params is None:
            baseline_params = tuple(
                [float(np.mean(observable))] + [0.0] * deg
            )
        elif len(baseline_params) < deg + 1:
            baseline_params = tuple(
                list(baseline_params) + [0.0] * (deg + 1 - len(baseline_params))
            )
        osc_p0: list[float] = []
        osc_bounds: list[tuple[float | None, float | None]] = []
        from menus.astronomical.desi.prior_bounds import (
            hier_frac_bounds as _hier_bounds_default,
            tau_amplitude_frac_bounds,
        )

        a_lo, a_hi = tau_amplitude_frac_bounds(mode="fit")
        if fit_A:
            osc_p0.append(0.01)
            osc_bounds.append((a_lo, a_hi))
        if fit_phase:
            osc_p0.append(0.0)
            osc_bounds.append((-np.pi, np.pi))
        if use_hier:
            osc_p0.append(0.001)
            hb = (
                hier_bounds
                if hier_bounds is not None
                else _hier_bounds_default(mode="fit")
            )
            osc_bounds.append((float(hb[0]), float(hb[1])))
        p0 = list(baseline_params[: deg + 1]) + osc_p0
        bounds: list[tuple[float | None, float | None]] = (
            [(None, None)] * (deg + 1) + osc_bounds
        )

        def chi2_func(p: np.ndarray) -> float:
            return gaussian_chi2(observable, model_func(p, z, s), err=err, cov=cov)

        res = optimize.minimize(chi2_func, p0, method="L-BFGS-B", bounds=bounds)
        best_p = res.x
        chi2_min = float(res.fun)
        n_params = len(best_p)
        dof = len(z) - n_params

        def baseline_only(p: np.ndarray) -> float:
            pred = np.polyval(p[: deg + 1], z)
            return gaussian_chi2(observable, pred, err=err, cov=cov)

        res_base = optimize.minimize(
            baseline_only,
            list(baseline_params[: deg + 1]),
            method="L-BFGS-B",
            bounds=[(None, None)] * (deg + 1),
        )
        chi2_base = float(res_base.fun)
        delta_chi2 = chi2_base - chi2_min
        pval_improvement = float(1 - chi2.cdf(delta_chi2, df=max(extra, 1)))

        null_comparison = None
        if cov is not None:
            _null_val, chi2_null = fit_null_model(observable, cov)
            null_comparison = compute_model_comparison(
                chi2_min, chi2_null, n_data=len(z), n_params_model=n_params
            )
            null_comparison["chi2_poly_baseline"] = float(chi2_base)
            null_comparison["delta_chi2_vs_poly_baseline"] = float(chi2_base - chi2_min)
            null_comparison["null_model_note"] = (
                "Weighted-mean null is a poor reference for redshift-structured BAO; "
                "prefer delta_chi2_vs_poly_baseline."
            )
            if chi2_null > 10.0 * max(chi2_base, 1.0):
                null_comparison["null_unreliable"] = True

        gamma_jk = None
        if self.auto_calibrate_gamma:

            def _cal(zz: np.ndarray) -> float:
                return calibrate_gamma_from_hierarchy(
                    z_min=float(np.min(zz)),
                    z_max=float(np.max(zz)),
                    n_hier=self.n_hier,
                    delta_n=self.delta_n,
                    period=self.period,
                )

            gamma_jk = gamma_jackknife_std(z, n_hier=self.n_hier, delta_n=self.delta_n, period=self.period, calibrate_fn=_cal)

        param_labels = _fit_parameter_labels(
            deg, fit_A=fit_A, fit_phase=fit_phase, use_hier=use_hier
        )
        a_idx = deg + 1 if fit_A else None
        a_frac = float(best_p[a_idx]) if a_idx is not None else float(fixed_A_frac)
        desc_parts = [f"{deg + 1} poly baseline"]
        if fit_A:
            desc_parts.append("A_osc_frac")
        if fit_phase:
            desc_parts.append("phase")
        if use_hier:
            desc_parts.append("hier_frac")
        self.results["fit"] = {
            "best_params": best_p,
            "parameter_labels": param_labels,
            "fit_options": {
                "fit_A": fit_A,
                "fit_phase": fit_phase,
                "fit_hier": use_hier,
                "fixed_A_frac": None if fit_A else float(fixed_A_frac),
            },
            "model_complexity": {
                "n_params": n_params,
                "n_data": len(z),
                "dof": dof,
                "baseline_degree": deg,
                "parameter_labels": param_labels,
                "description": f"{' + '.join(desc_parts)} = {n_params} parameters",
            },
            "baseline_degree": deg,
            "A_osc_frac": a_frac,
            "A_osc": a_frac,
            "A_osc_physical": a_frac * amp_scale,
            "amplitude_scale": amp_scale,
            "chi2": chi2_min,
            "reduced_chi2": chi2_min / max(dof, 1),
            "aic": _aic(chi2_min, n_params),
            "bic": _bic(chi2_min, n_params, len(z)),
            "dof": dof,
            "n_data": len(z),
            "n_params": n_params,
            "chi2_baseline": chi2_base,
            "delta_chi2": delta_chi2,
            "pval_improvement": pval_improvement,
            "null_comparison": null_comparison,
            "gamma_jackknife": gamma_jk,
            "gamma_used": gamma_used,
            "used_full_covariance": cov is not None,
            "model_func": lambda zz: model_func(best_p, zz, s_from_z(zz, gamma_used)),
        }
        return self.results["fit"]

    def scan_dipole(
        self,
        dipole_amplitude: float = 0.015,
        seed: int = 123,
    ):
        from menus.astronomical.desi.production import scan_dipole_healpy

        return scan_dipole_healpy(self, dipole_amplitude=dipole_amplitude, seed=seed)

    def plot_all(self, data_dict: dict[str, Any], save_prefix: str = "tau_sb_desi_scan") -> Path:
        _ensure_artifacts()
        z = data_dict["z"]
        obs = data_dict["observable"]
        err = data_dict.get("err", np.ones_like(z) * 0.01)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        best_fit = None
        if "model_comparison" in self.results:
            tau_row = self.results["model_comparison"]["tau_sb"]
            best_fit = ("Tau-SB", tau_row.model_at_z(z))
            ade_row = self.results["model_comparison"]["ade"]
            axes[0, 0].plot(z, ade_row.model_at_z(z), "b:", label="aDE best-fit", lw=1.5)
            if "w0wa" in self.results["model_comparison"]:
                w0wa_row = self.results["model_comparison"]["w0wa"]
                axes[0, 0].plot(
                    z, w0wa_row.model_at_z(z), "m-.", label="w0waCDM best-fit", lw=1.5
                )
        elif "fit" in self.results:
            best_fit = ("Tau-SB", self.results["fit"]["model_func"](z))

        if best_fit:
            axes[0, 0].plot(z, best_fit[1], "r--", label=f"{best_fit[0]} model", lw=2)

        axes[0, 0].errorbar(
            z, obs, yerr=err, fmt="o", color="blue", label=data_dict.get("label", "Data"), capsize=3
        )
        axes[0, 0].set_xlabel("Redshift z")
        axes[0, 0].set_ylabel("BAO observable")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].set_title("DESI BAO + Model Fits")
        axes[0, 0].grid(True, alpha=0.3)

        if "periodicity" in self.results:
            resids = self.results["periodicity"]["residuals"]
            s = self.results["periodicity"]["s"]
            osc_track = tau_sb_oscillatory_residual(
                s, A=0.02, amplitude_scale=_oscillation_scale(obs)
            )
            axes[0, 1].plot(z, resids, "o-", color="purple", label="Residuals")
            axes[0, 1].plot(z, osc_track, "g--", label="Example 1/7 track")
            axes[0, 1].axhline(0, color="gray", ls=":")
            axes[0, 1].set_xlabel("z")
            axes[0, 1].set_ylabel("Residuals")
            axes[0, 1].legend()
            axes[0, 1].set_title("Residuals vs 1/7 Template")

        if "periodicity" in self.results:
            freqs = self.results["periodicity"]["freqs"]
            power = self.results["periodicity"]["power"]
            exp_f = self.results["periodicity"]["expected_f"]
            axes[1, 0].plot(freqs, power, "k-")
            axes[1, 0].axvline(exp_f, color="red", ls="--", label=f"Expected f=1/{self.period}")
            axes[1, 0].axvline(
                self.results["periodicity"]["max_freq"], color="green", ls=":", label="Max power"
            )
            axes[1, 0].set_xlabel("Frequency (cycles per s-unit)")
            axes[1, 0].set_ylabel("Lomb-Scargle Power")
            axes[1, 0].legend()
            axes[1, 0].set_title("Periodicity Scan (s-space)")
            axes[1, 0].grid(True, alpha=0.3)

        if "model_comparison" in self.results:
            names = [r["name"] for r in self.results["model_comparison"]["ranking"]]
            aics = [r["aic"] for r in self.results["model_comparison"]["ranking"]]
            colors = ["#4c72b0", "#55a868", "#9467bd", "#c44e52"][: len(names)]
            axes[1, 1].barh(names, aics, color=colors)
            axes[1, 1].set_xlabel("AIC (lower is better)")
            axes[1, 1].set_title("ΛCDM vs aDE vs w0waCDM vs Tau-SB")
        elif "dipole" in self.results:
            dip = self.results["dipole"]
            method = dip.get("method", "healpy_l1")
            axes[1, 1].axis("off")
            summary = (
                f"Dipole ({method})\n"
                f"Amplitude: {dip.get('fitted_amplitude', 0.0):.4f}\n"
                f"l1 RMS: {dip.get('dipole_rms_l1', 0.0):.4f}\n"
                f"Axis RA: {dip.get('dipole_axis_ra_deg', 0.0):.1f}°\n"
                f"Axis Dec: {dip.get('dipole_axis_dec_deg', 0.0):.1f}°\n"
                f"Patches: {dip.get('n_patches', 'n/a')}"
            )
            axes[1, 1].text(0.05, 0.95, summary, va="top", fontsize=10, family="monospace")
            axes[1, 1].set_title("Healpy Dipole Fit")

        plt.tight_layout()
        out = artifact_path(
            TestSlug.TAU_SB_DESI,
            compose_dataset_slug(save_prefix),
            "summary_plot",
            "png",
        )
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out


def _load_scan_data(
    *,
    cobaya_path: str | Path | None = None,
    tracer: str = "ALL_GCcomb",
    quantity_filter: str | None = "DM_over_rs",
    data_mode: str | None = None,
    gamma: float = 10.0,
    include_legacy_bao: bool = False,
    combine_tracers: bool = False,
    high_power_stack: bool = False,
) -> dict[str, Any]:
    """Load live DESI DR2 Gaussian BAO tables (optional legacy / combined stacks)."""
    mode: str | None = data_mode
    if mode is None:
        norm = normalize_quantity_filter(quantity_filter)
        if norm in set(DATA_MODE_TO_QUANTITY.values()):
            mode = next(k for k, v in DATA_MODE_TO_QUANTITY.items() if v == norm)

    if combine_tracers or tracer == COMBINED_DR2_TRACER:
        if mode is None:
            mode = DATA_MODE
        return load_combined_dr2_data(mode, local_path=cobaya_path, gamma=gamma)

    if include_legacy_bao or tracer == EXTENDED_BAO_TRACER:
        if mode is None:
            mode = DATA_MODE
        return load_extended_bao_data(
            mode,
            local_path=cobaya_path,
            tracer=TRACER if tracer in {EXTENDED_BAO_TRACER, COMBINED_DR2_TRACER} else tracer,
            gamma=gamma,
            high_power_stack=high_power_stack,
        )

    if mode is not None:
        return load_scan_data_by_data_mode(
            mode,
            local_path=cobaya_path,
            tracer=tracer,
            gamma=gamma,
        )
    return load_desi_from_cobaya_repo(
        cobaya_path,
        tracer=tracer,
        quantity_filter=quantity_filter,
    )


def ensure_adequate_scan_data(
    data: dict[str, Any],
    *,
    cobaya_path: str | Path | None,
    tracer: str,
    quantity_filter: str | None,
    action_name: str = "",
    run_changepoints: bool = False,
) -> tuple[dict[str, Any], str | None, list[str]]:
    """
    Upgrade sparse quantity filters when possible and emit power warnings.

    For Full Production Pipeline, auto-upgrades e.g. DV_over_rs (n=1) to the
    full published vector (quantity=all) when that yields more points.
    """
    notes: list[str] = []
    n = int(data.get("n_data", 0))
    min_n = PRODUCTION_PIPELINE_MIN_N if action_name == "Full Production Pipeline" else (
        CHANGPOINTS_MIN_N if run_changepoints else 1
    )
    if n >= min_n:
        return data, quantity_filter, notes

    if quantity_filter is not None:
        # Never auto-upgrade into mixed DH+DM or quantity=all — that breaks χ² fits.
        safe_fallbacks = ("DH_over_rs", "DM_over_rs")
        for fallback in safe_fallbacks:
            if quantity_filter == fallback:
                continue
            try:
                data_fb = _load_scan_data(
                    cobaya_path=cobaya_path,
                    tracer=tracer,
                    quantity_filter=fallback,
                )
                n_fb = int(data_fb.get("n_data", 0))
                if n_fb > n:
                    notes.append(
                        f"quantity={quantity_filter!r} yields n={n}; "
                        f"auto-upgraded to {fallback} (n={n_fb})"
                    )
                    return data_fb, fallback, notes
            except (FileNotFoundError, ValueError):
                pass

    if n < CHANGPOINTS_MIN_N and run_changepoints:
        notes.append(
            f"n={n} too few for change-point detection (need n≥{CHANGPOINTS_MIN_N}); "
            "changepoints step will be skipped"
        )
    if action_name == "Full Production Pipeline" and n < PRODUCTION_PIPELINE_MIN_N:
        notes.append(
            f"n={n} underpowered for Full Production Pipeline "
            f"(need n≥{PRODUCTION_PIPELINE_MIN_N} for changepoints/MCMC; "
            f"keep DH_only or DM_only — do not mix channels to inflate n)"
        )
    return data, quantity_filter, notes


def print_data_vector_summary(data_vector: np.ndarray) -> None:
    """Log BAO data vector size and head/tail values for quick sanity checks."""
    obs = np.asarray(data_vector, dtype=float).ravel()
    print(f"Data vector length: {len(obs)}")
    if len(obs) >= 5:
        print(f"First 5 values: {obs[:5]}")
        print(f"Last 5 values : {obs[-5:]}")
    elif len(obs):
        print(f"Values: {obs}")


def _print_run_manifest(
    *,
    action_name: str,
    pipelines_enabled: str,
    data: dict[str, Any],
    tracer: str,
    quantity_filter: str | None,
    auto_calibrate_gamma: bool,
    use_covariance: bool,
    run_mcmc: bool,
    mcmc_steps: int,
) -> None:
    print("[Tau-SB DESI] Run manifest")
    print(f"  action: {action_name or 'desi_scan'}")
    print(f"  data: live DR2 Cobaya tables ({data.get('label', 'unknown')})")
    print(f"  source file: {data.get('source', 'n/a')}")
    mode_label = data.get("data_mode") or quantity_filter or "all"
    print(f"  tracer: {tracer} | data_mode: {mode_label} | n_data: {data.get('n_data', 0)}")
    if data.get("high_power_stack"):
        print(f"  stack: high-power (dedupe_z_tol={data.get('dedupe_z_tol', 0)})")
    print(f"  pipelines: {pipelines_enabled or 'periodicity_scan'}")
    print(f"  auto_calibrate_gamma: {auto_calibrate_gamma} | full_covariance: {use_covariance}")
    if run_mcmc:
        print(f"  mcmc_steps: {mcmc_steps}")
    else:
        print("  mcmc: not enabled for this action")
    print("  note: Cobaya here = bao_data repo path, not the Cobaya MCMC sampler")
    obs = data.get("observable")
    z_arr = data.get("z")
    quants = data.get("quantity")
    if obs is not None and z_arr is not None and quants:
        print_data_vector_summary(obs)
        print_quantity_composition(z_arr, obs, quants)
    elif obs is not None:
        print_data_vector_summary(obs)
    from menus.astronomical.desi.stats import diagnose_frequency_grid

    z_arr = data.get("z")
    if z_arr is not None and len(z_arr) >= 2:
        g = calibrate_gamma_from_hierarchy(
            z_min=float(np.min(z_arr)),
            z_max=float(np.max(z_arr)),
            n_hier=N_HIER_BINDING,
        )
        s_arr = s_from_z(z_arr, gamma=g)
        diag = diagnose_frequency_grid(s_arr)
        print(f"  s-grid: Δs={diag['s_range']:.3f}, cycles@1/7={diag['cycles_possible']:.2f}")
        if diag["warning"]:
            print(f"  ⚠ {diag['warning_message']}")
    n_pts = int(data.get("n_data", 0))
    mix_warn = mixed_quantity_warning(
        data.get("quantity") or [],
        data_mode=data.get("data_mode"),
    )
    if mix_warn:
        print(f"  ⚠ {mix_warn}")
    if n_pts < 8:
        print(f"  ⚠ n={n_pts}: underpowered for tav_resonance periodogram (need n≥8)")
    cov_health = data.get("cov_health")
    if isinstance(cov_health, dict) and cov_health.get("n"):
        kappa = cov_health.get("condition_number")
        reg = cov_health.get("regularization_applied", False)
        print(
            f"  cov: κ={kappa:.2e}, λ_min={cov_health.get('min_eigenvalue', float('nan')):.2e}"
            + (f", Tikhonov ε={cov_health.get('regularization_strength', 0):.2e}" if reg else "")
        )
        for warn in cov_health.get("warnings") or []:
            print(f"  ⚠ cov: {warn}")


def _tau_sb_prediction_from_fit(
    fit: dict[str, Any],
    *,
    z: np.ndarray,
    gamma: float,
    period: float,
    amp_scale: float | None = None,
) -> np.ndarray:
    """Rebuild Tau-SB model curve from a :func:`fit_tau_sb_model` result."""
    zz = np.asarray(z, dtype=float)
    best_p = np.asarray(fit["best_params"], dtype=float)
    deg = int(fit["baseline_degree"])
    opts = fit.get("fit_options") or {}
    fit_A = bool(opts.get("fit_A", True))
    fit_phase = bool(opts.get("fit_phase", True))
    use_hier = bool(opts.get("fit_hier", len(best_p) > deg + 2))
    fixed_A = float(opts.get("fixed_A_frac") or 0.01)
    scale = float(fit.get("amplitude_scale") if amp_scale is None else amp_scale)
    s_fix = s_from_z(zz, gamma=gamma)
    base = np.polyval(best_p[: deg + 1], zz)
    idx = deg + 1
    a_frac = float(best_p[idx]) if fit_A else fixed_A
    if fit_A:
        idx += 1
    phase = float(best_p[idx]) if fit_phase else 0.0
    if fit_phase:
        idx += 1
    hier_frac = float(best_p[idx]) if use_hier and len(best_p) > idx else 0.0
    osc = tau_sb_oscillatory_residual(
        s_fix, A=a_frac, period=period, phase=phase, amplitude_scale=scale
    )
    hier = tau_sb_hierarchical_step(zz, amplitude=hier_frac * scale * 0.01) if use_hier else 0.0
    return base + osc + hier


def _tau_sb_prediction_from_fit_whim(
    fit: dict[str, Any],
    *,
    z: np.ndarray,
    gamma: float,
    period: float,
    amp_scale: float | None = None,
    whim_strength: float = WEB_1D_CONSTRAINT_STRENGTH,
    use_tsb_rd: bool = True,
) -> np.ndarray:
    """
    Tau-SB fit prediction with WHIM / web 1D correction via :func:`tau_sb_model_with_whim`.

    Maps fitted oscillation parameters onto the WHIM model; keeps the fitted
    polynomial baseline in ``z`` and adds the s-space WHIM oscillation + web terms.
    """
    zz = np.asarray(z, dtype=float)
    best_p = np.asarray(fit["best_params"], dtype=float)
    deg = int(fit["baseline_degree"])
    opts = fit.get("fit_options") or {}
    fit_A = bool(opts.get("fit_A", True))
    fit_phase = bool(opts.get("fit_phase", True))
    use_hier = bool(opts.get("fit_hier", len(best_p) > deg + 2))
    fixed_A = float(opts.get("fixed_A_frac") or 0.01)
    scale = float(fit.get("amplitude_scale") if amp_scale is None else amp_scale)
    s_fix = s_from_z(zz, gamma=gamma)
    base_z = np.polyval(best_p[: deg + 1], zz)
    idx = deg + 1
    a_frac = float(best_p[idx]) if fit_A else fixed_A
    if fit_A:
        idx += 1
    phase = float(best_p[idx]) if fit_phase else 0.0
    if fit_phase:
        idx += 1
    hier_frac = float(best_p[idx]) if use_hier and len(best_p) > idx else 0.0

    whim_core = tau_sb_model_with_whim(
        s_fix,
        A_osc_frac=a_frac * scale,
        phase_rad=phase,
        gamma=gamma,
        hier_frac=hier_frac,
        baseline_params=[0.0, 0.0, 0.0],
        whim_strength=whim_strength,
        use_tsb_rd=use_tsb_rd,
        period=period,
    )
    return base_z + whim_core


def run_desi_scan(
    *,
    cobaya_path: str | Path | None = None,
    tracer: str = "ALL_GCcomb",
    quantity_filter: str | None = "DM_over_rs",
    gamma: float = 10.0,
    period: float = TAU_RESONANCE_PERIOD,
    auto_calibrate_gamma: bool = False,
    n_hier: float = N_HIER_BINDING,
    use_covariance: bool = True,
    run_fit: bool = True,
    compare_models: bool = False,
    run_dipole: bool = False,
    run_mcmc: bool = False,
    run_changepoints: bool = False,
    run_binding_kit: bool = False,
    run_injection_recovery: bool = False,
    run_joint_fit: bool = False,
    mcmc_steps: int = MCMC_STEPS_DEFAULT,
    plot: bool = True,
    output_prefix: str = "tau_sb_desi",
    action_name: str = "",
    pipelines_enabled: str = "",
    include_legacy_bao: bool = False,
    combine_tracers: bool = False,
    fit_A: bool = True,
    fit_phase: bool = True,
    fit_hier: bool | None = None,
    run_residual_diagnostics: bool | None = None,
    n_freq: int | None = None,
    n_injection_trials: int | None = None,
    max_sne: int | None = None,
    high_power_stack: bool = False,
    augment_covariance: bool = False,
    run_nested_sampling: bool = False,
    nested_nlive: int | None = None,
    nested_max_samples: int | None = None,
    use_jax_geometric: bool = True,
) -> DesiScanResult:
    scanner = TauSBScanner(
        gamma=gamma,
        period=period,
        auto_calibrate_gamma=auto_calibrate_gamma,
        n_hier=n_hier,
    )

    data = _load_scan_data(
        cobaya_path=cobaya_path,
        tracer=tracer,
        quantity_filter=quantity_filter,
        include_legacy_bao=include_legacy_bao,
        combine_tracers=combine_tracers,
        high_power_stack=high_power_stack,
    )
    data, quantity_filter, power_notes = ensure_adequate_scan_data(
        data,
        cobaya_path=cobaya_path,
        tracer=tracer,
        quantity_filter=quantity_filter,
        action_name=action_name,
        run_changepoints=run_changepoints,
    )
    _print_run_manifest(
        action_name=action_name,
        pipelines_enabled=pipelines_enabled,
        data=data,
        tracer=tracer,
        quantity_filter=quantity_filter,
        auto_calibrate_gamma=auto_calibrate_gamma,
        use_covariance=use_covariance,
        run_mcmc=run_mcmc,
        mcmc_steps=mcmc_steps,
    )
    for note in power_notes:
        print(f"  ⚠ {note}")

    if run_fit or compare_models:
        validate_single_channel_fit_data(data, action_name=action_name)

    z = data["z"]
    obs = data["observable"]
    err = data["err"]
    cov = data.get("cov") if use_covariance else None

    if augment_covariance and cov is not None:
        from menus.astronomical.desi.analysis import apply_augmented_covariance

        cov, aug_meta = apply_augmented_covariance(
            cov,
            z,
            name=str(data.get("label", "bao")),
        )
        data["cov"] = cov
        data["cov_augmented"] = aug_meta
        err = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        data["err"] = err
        print(
            f"  [Covariance] Augmented with domain drift (Δγ={aug_meta['delta_gamma']:.2f}) "
            f"+ friction P(k)∝k^{aug_meta['friction_exponent']}"
        )

    geometric_likelihood: dict[str, Any] | None = None
    if cov is not None and len(obs) >= 3:
        from menus.astronomical.desi.theory_likelihood import geometric_likelihood_summary

        qtypes = data.get("quantity_types") or []
        quantity = qtypes[0] if len(qtypes) == 1 else (data.get("quantity_filter") or "DH_over_rs")
        try:
            geometric_likelihood = geometric_likelihood_summary(
                obs,
                z,
                cov,
                quantity=str(quantity),
            )
            print(
                f"  [Geometry μ] fixed-prior logL={geometric_likelihood['log_likelihood']:.2f}, "
                f"χ²={geometric_likelihood['chi2']:.2f} "
                f"(reduced={geometric_likelihood['reduced_chi2']:.2f})"
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            print(f"  [Geometry μ] skipped: {exc}")

    if auto_calibrate_gamma and cov is not None:
        from menus.astronomical.desi.analysis import run_auto_diagnostics

        s_grid = data.get("s")
        if s_grid is None:
            s_grid = s_from_z(z, gamma=scanner.gamma)
        auto_diag = run_auto_diagnostics(
            s_grid,
            obs,
            cov,
            model_func=None,
            auto_calibrate_func=None,
            z=z,
            period=scanner.period,
            n_hier=scanner.n_hier,
            delta_n=scanner.delta_n,
            run_injection=not run_injection_recovery,
            print_summary=True,
        )
        gamma_diagnostics = auto_diag["gamma_results"]
        scanner.results["gamma_diagnostics"] = gamma_diagnostics
        scanner.results["auto_diagnostics"] = auto_diag
        data["gamma_diagnostics"] = gamma_diagnostics
        data["auto_diagnostics"] = auto_diag
        if auto_diag.get("inj_results"):
            scanner.results["injection_recovery_preview"] = auto_diag["inj_results"]
        if gamma_diagnostics.get("tsb_sound_horizon"):
            scanner.results["tsb_sound_horizon"] = gamma_diagnostics["tsb_sound_horizon"]
            scanner.results["rd_residual"] = gamma_diagnostics.get("rd_residual")

    per = scanner.scan_periodicity(
        z,
        obs,
        err,
        cov=cov,
        n_freq=int(n_freq) if n_freq is not None else 500,
    )
    gamma_used = per.get("gamma_used", gamma)

    fit = (
        scanner.fit_tau_sb_model(
            z,
            obs,
            err,
            cov=cov,
            fit_A=fit_A,
            fit_phase=fit_phase,
            fit_hier=fit_hier,
        )
        if run_fit
        else None
    )

    residual_diag = None
    if run_residual_diagnostics is None:
        run_residual_diagnostics = bool(run_fit and cov is not None)
    if run_fit and fit and run_residual_diagnostics and cov is not None:
        from menus.astronomical.desi.analysis import run_master_diagnostics

        s_grid = data.get("s")
        if s_grid is None:
            s_grid = s_from_z(z, gamma=scanner.gamma)
        # Prefer auto-calibrated γ from the fit (≈8.851) over fixed-γ sensitivity minimum.
        gamma_for_rd = float(gamma_used) if np.isfinite(float(gamma_used)) else None
        if gamma_for_rd is None and np.isfinite(float(scanner.gamma)):
            gamma_for_rd = float(scanner.gamma)
        gamma_pred = float(gamma_for_rd) if gamma_for_rd is not None else float(scanner.gamma)
        y_model = _tau_sb_prediction_from_fit_whim(
            fit, z=z, gamma=gamma_pred, period=scanner.period
        )
        residual_diag = run_master_diagnostics(
            s_grid,
            obs,
            y_model,
            cov,
            save_plots=plot,
            plot_dir=str(ARTIFACTS_DIR / TestSlug.TAU_SB_DESI / TestSlug.RESIDUAL_DIAGNOSTICS),
            verbose=True,
            show_plots=False,
            best_gamma=gamma_for_rd,
            include_binding_kit=run_binding_kit,
            z=z,
            n_hier=n_hier,
        )
        scanner.results["residual_diagnostics"] = residual_diag
        data["residual_diagnostics"] = residual_diag

    binding_kit = None
    if residual_diag is not None:
        binding_kit = residual_diag.get("harmonic_binding_kit")
    if run_binding_kit and binding_kit is None and fit is not None:
        from menus.astronomical.desi.harmonic_binding import run_harmonic_binding_kit

        s_bind = data.get("s") or s_from_z(z, gamma=scanner.gamma)
        y_bind = _tau_sb_prediction_from_fit_whim(
            fit, z=z, gamma=float(gamma_used), period=scanner.period
        )
        resid_bind = np.asarray(obs, dtype=float) - np.asarray(y_bind, dtype=float)
        binding_kit = run_harmonic_binding_kit(
            z,
            obs,
            resid_bind,
            s_bind,
            n_hier=n_hier,
            gamma=float(gamma_used),
            desi_tracers=[str(data.get("tracer", tracer))],
            verbose=True,
        )
    if binding_kit is not None:
        scanner.results["harmonic_binding_kit"] = binding_kit
        data["harmonic_binding_kit"] = binding_kit

    if fit and auto_calibrate_gamma:
        from menus.astronomical.desi.analysis import run_non_circular_gamma_diagnostic

        def _tau_sb_fit_model(*, y, z=None, cov=None, err=None, **_: Any) -> np.ndarray:
            return _tau_sb_prediction_from_fit(
                fit, z=np.asarray(z, dtype=float), gamma=scanner.gamma, period=scanner.period
            )

        tau_sb_nc = run_non_circular_gamma_diagnostic(
            z=z,
            y=obs,
            cov=cov,
            err=err,
            model_func=_tau_sb_fit_model,
            fixed_gamma=scanner.gamma,
            period=period,
        )
        if isinstance(per.get("fixed_gamma_diagnostic"), dict):
            per["fixed_gamma_diagnostic"]["tau_sb_fit"] = tau_sb_nc
        per["non_circular_tau_sb_fit"] = tau_sb_nc
        scanner.results["periodicity"] = per
    model_cmp = scanner.compare_models(z, obs, err=err, cov=cov) if compare_models else None
    dip = scanner.scan_dipole() if run_dipole else None

    if run_mcmc and mcmc_steps < MCMC_STEPS_DEFAULT:
        print(
            f"[Tau-SB DESI] Note: mcmc_steps={mcmc_steps} < {MCMC_STEPS_DEFAULT}; "
            f"use {MCMC_STEPS_DEFAULT}+ for publication-grade posteriors."
        )

    mcmc = changepoints = injection = joint = nested_evidence = None
    if any(
        [
            run_mcmc,
            run_changepoints,
            run_injection_recovery,
            run_joint_fit,
            run_nested_sampling,
        ]
    ):
        from menus.astronomical.desi.production import run_production_pipeline

        if run_fit and fit is not None:
            scanner.results["fit"] = fit
        prod = run_production_pipeline(
            scanner,
            data,
            run_mcmc=run_mcmc,
            run_healpy_dipole=False,
            run_changepoints=run_changepoints,
            run_injection=run_injection_recovery,
            run_joint=run_joint_fit,
            run_nested_sampling=run_nested_sampling,
            mcmc_steps=mcmc_steps,
            use_covariance=use_covariance,
            n_injection_trials=n_injection_trials,
            max_sne=max_sne,
            nested_nlive=nested_nlive,
            nested_max_samples=nested_max_samples,
            use_jax_geometric=use_jax_geometric,
        )
        mcmc = prod.get("mcmc")
        changepoints = prod.get("changepoints")
        injection = prod.get("injection_recovery")
        joint = prod.get("joint_fit")
        nested_evidence = prod.get("nested_evidence")

    plot_path = str(scanner.plot_all(data, save_prefix=output_prefix)) if plot else None
    tav_harmonics = per.get("tav_harmonics")
    from menus.astronomical.desi.stats import assess_scan_power

    grid_diag = per.get("s_grid_diagnostics") or {}
    n_params_fit = int(fit["n_params"]) if fit else None
    power_assessment = assess_scan_power(
        n_data=int(data.get("n_data", len(z))),
        cycles_possible=float(grid_diag.get("cycles_possible", 0.0)),
        auto_calibrate_gamma=auto_calibrate_gamma,
        n_params=n_params_fit,
        tav_skipped=bool(tav_harmonics.get("tav_resonance_skipped")) if tav_harmonics else False,
    )

    from menus.astronomical.desi.stats import build_recommended_next_steps

    auto_diag = data.get("auto_diagnostics")
    next_steps = build_recommended_next_steps(
        power_assessment=power_assessment,
        fit=fit,
        auto_diagnostics=auto_diag,
        residual_diagnostics=residual_diag,
        n_data=int(data.get("n_data", len(z))),
        include_legacy_bao=include_legacy_bao or bool(data.get("legacy_included")),
        fit_phase=fit_phase,
        fit_hier=fit_hier if fit_hier is not None else (int(data.get("n_data", len(z))) >= 8),
    )
    if next_steps:
        print("\nRecommended next steps:")
        for line in next_steps:
            print(f"  {line}")

    result = DesiScanResult(
        data_label=str(data.get("label", "unknown")),
        tracer=str(data.get("tracer", "")),
        gamma_used=float(gamma_used),
        periodicity=per,
        fit=fit,
        model_comparison=model_cmp,
        dipole=dip,
        mcmc=mcmc,
        changepoints=changepoints,
        injection_recovery=injection,
        joint_fit=joint,
        nested_evidence=nested_evidence,
        tav_harmonics=tav_harmonics,
        power_assessment=power_assessment,
        gamma_diagnostics=data.get("gamma_diagnostics") or per.get("gamma_diagnostics"),
        auto_diagnostics=auto_diag,
        residual_diagnostics=residual_diag,
        harmonic_binding_kit=binding_kit,
        recommended_next_steps=next_steps,
        tsb_sound_horizon=(
            (data.get("gamma_diagnostics") or {}).get("tsb_sound_horizon")
            or scanner.results.get("tsb_sound_horizon")
        ),
        rd_residual=(
            (data.get("gamma_diagnostics") or {}).get("rd_residual")
            or scanner.results.get("rd_residual")
        ),
        plot_path=plot_path,
    )

    _ensure_artifacts()
    report_path = artifact_path(
        TestSlug.TAU_SB_DESI,
        compose_dataset_slug(result.tracer, quantity_filter, output_prefix),
        "report",
        "json",
    )
    payload: dict[str, Any] = {
        "timestamp_utc": _utc_iso(),
        "action": action_name or None,
        "data_label": result.data_label,
        "data_source": "cobaya_dr2",
        "data_source_file": data.get("source"),
        "tracer": result.tracer,
        "quantity_filter": quantity_filter,
        "n_data": data.get("n_data"),
        "gamma_used": result.gamma_used,
        "n_hier": n_hier,
        "tau_period": period,
        "auto_calibrate_gamma": auto_calibrate_gamma,
        "use_covariance": use_covariance and data.get("has_full_covariance", False),
        "cov_health": data.get("cov_health"),
        "pipelines_enabled": pipelines_enabled or None,
        "mcmc_enabled": run_mcmc,
        "mcmc_steps": mcmc_steps if run_mcmc else None,
        "periodicity": {
            "expected_f": per["expected_f"],
            "power_at_expected": per["power_at_expected"],
            "max_power": per["max_power"],
            "max_freq": per["max_freq"],
            "pval_bootstrap": per.get("pval_bootstrap", per["pval_approx"]),
            "pval_approx": per["pval_approx"],
            "s_grid_diagnostics": per.get("s_grid_diagnostics"),
            "lomb_detected": bool(per.get("tav_harmonics", {}).get("lomb_detected", False)),
        },
        "tav_harmonics": tav_harmonics,
        "power_assessment": power_assessment,
        "fixed_gamma_diagnostic": per.get("fixed_gamma_diagnostic"),
        "gamma_diagnostics": data.get("gamma_diagnostics"),
        "tsb_sound_horizon": (
            (data.get("gamma_diagnostics") or {}).get("tsb_sound_horizon")
            or scanner.results.get("tsb_sound_horizon")
        ),
        "rd_residual": (
            (data.get("gamma_diagnostics") or {}).get("rd_residual")
            or scanner.results.get("rd_residual")
        ),
        "auto_diagnostics": auto_diag,
        "residual_diagnostics": residual_diag,
        "harmonic_binding_kit": binding_kit,
        "recommended_next_steps": next_steps,
        "data_stack": {
            "include_legacy_bao": include_legacy_bao,
            "combine_tracers": combine_tracers,
            "high_power_stack": high_power_stack or bool(data.get("high_power_stack")),
            "dedupe_z_tol": data.get("dedupe_z_tol"),
            "legacy_included": data.get("legacy_included"),
            "surveys": data.get("surveys"),
            "cov_augmented": data.get("cov_augmented"),
        },
        "geometric_likelihood": geometric_likelihood,
        "non_circular_tau_sb_fit": per.get("non_circular_tau_sb_fit"),
        "fit": (
            {
                "chi2": fit["chi2"],
                "reduced_chi2": fit.get("reduced_chi2"),
                "dof": fit.get("dof"),
                "n_data": fit.get("n_data"),
                "n_params": fit.get("n_params"),
                "parameter_labels": fit.get("parameter_labels"),
                "fit_options": fit.get("fit_options"),
                "model_complexity": fit.get("model_complexity"),
                "aic": fit["aic"],
                "bic": fit["bic"],
                "delta_chi2": fit["delta_chi2"],
                "A_osc_frac": float(fit.get("A_osc_frac", fit.get("A_osc", 0.0))),
                "A_osc_physical": fit.get("A_osc_physical"),
                "null_comparison": fit.get("null_comparison"),
                "gamma_jackknife": fit.get("gamma_jackknife"),
                "used_full_covariance": fit["used_full_covariance"],
            }
            if fit
            else None
        ),
        "model_comparison": (
            {
                "best_model": model_cmp["best_model"],
                "ranking": model_cmp["ranking"],
                "delta_aic": model_cmp["delta_aic"],
                "delta_bic": model_cmp.get("delta_bic"),
                "bayes_factors_vs_tau_sb": model_cmp["bayes_factors_vs_tau_sb"],
                "gamma_used": model_cmp.get("gamma_used"),
            }
            if model_cmp
            else None
        ),
        "dipole": (
            {"fitted_amplitude": dip["fitted_amplitude"], "method": dip.get("method")}
            if dip
            else None
        ),
        "mcmc": mcmc,
        "changepoints": changepoints,
        "injection_recovery": injection,
        "joint_fit": joint,
        "nested_evidence": nested_evidence,
        "plot_path": plot_path,
    }
    from menus.astronomical.desi.json_util import write_json

    write_json(report_path, payload, indent=2, sort_keys=True)
    result.report_path = str(report_path)
    return result


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tau-SB DESI BAO scanner — 1/7 tracks, aDE comparison, Cobaya DR2 loader"
    )
    parser.add_argument(
        "--data",
        choices=["cobaya"],
        default="cobaya",
        help="Live DESI DR2 Cobaya bao_data tables (only supported source)",
    )
    parser.add_argument(
        "--dipole",
        action="store_true",
        help="Run healpy l=1 dipole fit on patch catalog",
    )
    parser.add_argument(
        "--cobaya-path",
        default=None,
        help="Path to bao_data repo or desi_bao_dr2/ (default: datasets/desi/bao_data)",
    )
    parser.add_argument(
        "--tracer",
        default="ALL_GCcomb",
        help=f"DESI DR2 tracer key. Known: {', '.join(DESI_TRACERS)}",
    )
    parser.add_argument(
        "--quantity",
        default="DM_over_rs",
        help="Quantity filter for cobaya loads (DM_over_rs, DH_over_rs, DV_over_rs, or 'all')",
    )
    parser.add_argument("--gamma", type=float, default=10.0, help="Cylinder stretch γ")
    parser.add_argument(
        "--auto-gamma",
        action="store_true",
        help="Calibrate γ from n_hier and z-range (expected f=1/7 in s-space)",
    )
    parser.add_argument("--n-hier", type=float, default=N_HIER_BINDING, help="Hierarchical binding n_hier")
    parser.add_argument("--no-cov", action="store_true", help="Use diagonal errors only")
    parser.add_argument("--compare-models", action="store_true", help="ΛCDM vs aDE vs Tau-SB AIC/BIC")
    parser.add_argument("--mcmc", action="store_true", help="Run emcee MCMC posteriors")
    parser.add_argument(
        "--mcmc-steps",
        type=int,
        default=MCMC_STEPS_DEFAULT,
        metavar="N",
        help=f"emcee steps per walker (default {MCMC_STEPS_DEFAULT}; burn-in=N//3)",
    )
    parser.add_argument("--changepoints", action="store_true", help="ruptures hierarchical change-points")
    parser.add_argument("--injection-recovery", action="store_true", help="Injection/recovery test suite")
    parser.add_argument("--joint-fit", action="store_true", help="Joint DESI + Pantheon+ + Planck r_d")
    parser.add_argument("--batch-tracers", action="store_true", help="Scan pending DR2 tracers and exit")
    parser.add_argument(
        "--batch-limit",
        type=int,
        default=0,
        metavar="N",
        help="Max pending tracers per batch run (0 = all pending)",
    )
    parser.add_argument(
        "--force-rescan",
        action="store_true",
        help="Ignore datasets/desi/batch_done.txt and scan from the first tracers",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip PNG output")
    parser.add_argument("--list-tracers", action="store_true", help="List available DR2 tracers and exit")
    parser.add_argument("--output-prefix", default="tau_sb_desi", help="Artifact filename prefix")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_cli()
    args = parser.parse_args(argv)

    if args.list_tracers:
        tracers = list_desi_tracers(args.cobaya_path)
        print("Available DESI DR2 tracers:")
        for key in tracers:
            desc = DESI_TRACERS.get(key, "")
            print(f"  {key}: {desc}")
        return 0

    if args.batch_tracers:
        from menus.astronomical.desi.production import run_batch_tracer_scan

        batch = run_batch_tracer_scan(
            cobaya_path=args.cobaya_path,
            quantity_filter=None if str(args.quantity).lower() in {"all", "none", "*"} else args.quantity,
            auto_calibrate_gamma=args.auto_gamma,
            compare_models=args.compare_models,
            use_covariance=not args.no_cov,
            output_prefix=args.output_prefix,
            max_tracers=args.batch_limit,
            force_rescan=args.force_rescan,
        )
        print(f"Batch summary: {batch['summary_path']}")
        for tracer, info in batch["summaries"].items():
            print(f"  {tracer}: {info}")
        return 0

    quantity = None if str(args.quantity).lower() in {"all", "none", "*"} else args.quantity
    pipelines: list[str] = ["periodicity_scan"]
    if args.compare_models:
        pipelines.append("model_compare")
    if args.dipole:
        pipelines.append("healpy_dipole")
    if args.mcmc:
        pipelines.append("emcee_mcmc")
    if args.changepoints:
        pipelines.append("changepoints")
    if args.injection_recovery:
        pipelines.append("injection_recovery")
    if args.joint_fit:
        pipelines.append("joint_desi_sn_planck")

    result = run_desi_scan(
        cobaya_path=args.cobaya_path,
        tracer=args.tracer,
        quantity_filter=quantity,
        gamma=args.gamma,
        auto_calibrate_gamma=args.auto_gamma,
        n_hier=args.n_hier,
        use_covariance=not args.no_cov,
        compare_models=args.compare_models,
        run_dipole=args.dipole,
        run_mcmc=args.mcmc,
        run_changepoints=args.changepoints,
        run_injection_recovery=args.injection_recovery,
        run_joint_fit=args.joint_fit,
        mcmc_steps=args.mcmc_steps,
        plot=not args.no_plot,
        output_prefix=args.output_prefix,
        action_name="cli",
        pipelines_enabled=",".join(pipelines),
    )
    print("\n".join(result.summary_lines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())