"""
Configuration-backed tolerances and scaler defaults for dataset comparison.

Override via ``TAV_COMPARISON_CONFIG`` JSON file path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_DEFAULT_COMPARISON: dict[str, Any] = {
    "tolerances": {
        "absolute": 1.0e-9,
        "relative": 1.0e-6,
        "mass_gap_mev": 1.0e-3,
        "kinematic_gev": 1.0e-4,
        "angular_rad": 1.0e-5,
        "histogram_fraction": 1.0e-3,
        "amplitude": 1.0e-6,
        "chi2": 1.0e-4,
    },
    "scaler": "standard",
    "histogram_scaler": "minmax",
    "fit_reference": "combined",
    "cms_dimuon_primary_keys": [
        "leading_pt",
        "subleading_pt",
        "delta_phi",
        "delta_r",
    ],
    "cms_dimuon_features": [
        "leading_pt",
        "subleading_pt",
        "delta_phi",
        "delta_r",
        "system_pt",
    ],
    "key_round_digits": {
        "leading_pt": 3,
        "subleading_pt": 3,
        "delta_phi": 4,
        "delta_r": 3,
        "system_pt": 3,
    },
}


def load_comparison_config() -> dict[str, Any]:
    """Load comparison config from ``TAV_COMPARISON_CONFIG`` or defaults."""
    raw_path = (os.environ.get("TAV_COMPARISON_CONFIG") or "").strip()
    if not raw_path:
        return dict(_DEFAULT_COMPARISON)
    path = Path(raw_path).expanduser()
    if not path.is_file():
        return dict(_DEFAULT_COMPARISON)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        merged = dict(_DEFAULT_COMPARISON)
        for key, value in payload.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        return merged
    except (OSError, json.JSONDecodeError):
        return dict(_DEFAULT_COMPARISON)


def tolerance_for(name: str, *, config: dict[str, Any] | None = None) -> float:
    cfg = config or load_comparison_config()
    tol = cfg.get("tolerances", {})
    return float(tol.get(name, tol.get("absolute", 1.0e-9)))