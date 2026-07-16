"""
Configuration-backed physical bounds for ingestion validation.

Override ranges by setting TAV_INGESTION_CONFIG to a JSON file path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_DEFAULT_BOUNDS: dict[str, Any] = {
    "cms_dimuon": {
        "leading_pt_gev": [0.0, 2000.0],
        "subleading_pt_gev": [0.0, 2000.0],
        "delta_phi_rad": [0.0, 3.2],
        "delta_r": [0.0, 10.0],
        "system_pt_gev": [0.0, 500.0],
        "n_muon": [2, 64],
        "n_true_int": [0, 100],
        "min_events_after_filter": 10,
    },
    "bbn": {
        "tension_sigma_max": 50.0,
        "abundance_min": 1.0e-12,
        "abundance_max": 1.0,
        "uncertainty_min": 1.0e-15,
        "neutron_lifetime_s": [800.0, 920.0],
    },
    "fractal_tau": {
        "winding_density": [0.0, 10.0],
        "fractal_level": [0.0, 120.0],
        "phase_slip_alpha": [0.0, 7.0],
        "mass_gap_mev": [200.0, 500.0],
        "spectral_dim": [1.5, 4.5],
    },
}


def load_ingestion_bounds() -> dict[str, Any]:
    """Load bounds from TAV_INGESTION_CONFIG JSON or return defaults."""
    raw_path = (os.environ.get("TAV_INGESTION_CONFIG") or "").strip()
    if not raw_path:
        return dict(_DEFAULT_BOUNDS)
    path = Path(raw_path).expanduser()
    if not path.is_file():
        return dict(_DEFAULT_BOUNDS)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        merged = dict(_DEFAULT_BOUNDS)
        for key, value in payload.items():
            if isinstance(value, dict) and key in merged:
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        return merged
    except (OSError, json.JSONDecodeError):
        return dict(_DEFAULT_BOUNDS)