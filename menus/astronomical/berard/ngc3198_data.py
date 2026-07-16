"""
Frozen NGC 3198 rotation-curve ingestion (tau-cosmology pre-registration).

Primary source: SPARC MassModels_Lelli2016c (Lelli, McGaugh & Schombert 2016).
Once frozen, the CSV is not modified except via explicit re-ingest with new hash.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tav_shared.tav_project_paths import TAU_SUPERBLOCK_ROOT

BERARD_DATA_DIR = TAU_SUPERBLOCK_ROOT / "datasets" / "berard_framework"
FROZEN_CSV = BERARD_DATA_DIR / "data" / "ngc3198_rotation_curve.csv"
FROZEN_META = BERARD_DATA_DIR / "data" / "ngc3198_frozen_meta.json"
PREREG_PATH = BERARD_DATA_DIR / "text" / "PREREGISTRATION.md"

QUALITY_RULE = "include all points with eVobs > 0; no residual-based outlier rejection"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ingest_ngc3198_from_sparc(
    *,
    force: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Build frozen ``ngc3198_rotation_curve.csv`` from SPARC catalog.

    Columns: r_kpc, Vobs, eVobs, Vgas, Vdisk, Vbulge, D_mpc, quality_flag
    """
    from menus.astronomical.sparc.fetcher import fetch_mass_models_table

    FROZEN_CSV.parent.mkdir(parents=True, exist_ok=True)
    if FROZEN_CSV.is_file() and not force:
        meta = load_frozen_meta()
        if verbose:
            print(f"[NGC3198] Using frozen CSV: {FROZEN_CSV}")
            print(f"[NGC3198] SHA256: {meta.get('sha256', 'n/a')}")
        return meta

    table = fetch_mass_models_table()
    subset = table[table["ID"] == "NGC3198"].copy()
    if subset.empty:
        raise KeyError("NGC3198 not found in SPARC MassModels_Lelli2016c.mrt")

    frame = pd.DataFrame(
        {
            "r_kpc": subset["R"].astype(float),
            "Vobs": subset["Vobs"].astype(float),
            "eVobs": subset["e_Vobs"].astype(float),
            "Vgas": subset["Vgas"].fillna(0.0).astype(float),
            "Vdisk": subset["Vdisk"].fillna(0.0).astype(float),
            "Vbulge": subset["Vbul"].fillna(0.0).astype(float),
            "D_mpc": subset["D"].astype(float),
        }
    )
    frame = frame[frame["eVobs"] > 0].sort_values("r_kpc").reset_index(drop=True)
    frame["quality_flag"] = "sparc_published"

    frame.to_csv(FROZEN_CSV, index=False)
    sha = _sha256_file(FROZEN_CSV)
    meta: dict[str, Any] = {
        "galaxy": "NGC3198",
        "source": "SPARC MassModels_Lelli2016c.mrt",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "sha256": sha,
        "n_points": int(len(frame)),
        "r_kpc_range": [float(frame["r_kpc"].min()), float(frame["r_kpc"].max())],
        "quality_cut": QUALITY_RULE,
        "csv_path": str(FROZEN_CSV),
        "preregistration": str(PREREG_PATH) if PREREG_PATH.is_file() else None,
        "crosscheck": [
            "Begeman 1989",
            "Karukes, Salucci & Gentile 2015 (extended HI to ~48 kpc)",
        ],
    }
    FROZEN_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if verbose:
        print(f"[NGC3198] Frozen {len(frame)} points → {FROZEN_CSV}")
        print(f"[NGC3198] SHA256: {sha}")
    return meta


def load_frozen_meta() -> dict[str, Any]:
    if FROZEN_META.is_file():
        return json.loads(FROZEN_META.read_text(encoding="utf-8"))
    if FROZEN_CSV.is_file():
        return {
            "csv_path": str(FROZEN_CSV),
            "sha256": _sha256_file(FROZEN_CSV),
            "n_points": sum(1 for _ in FROZEN_CSV.open()),
        }
    return {}


def load_ngc3198_curve() -> dict[str, np.ndarray]:
    """Load frozen rotation curve arrays."""
    if not FROZEN_CSV.is_file():
        ingest_ngc3198_from_sparc(verbose=False)
    frame = pd.read_csv(FROZEN_CSV)
    return {
        "r_kpc": frame["r_kpc"].to_numpy(dtype=float),
        "Vobs": frame["Vobs"].to_numpy(dtype=float),
        "eVobs": frame["eVobs"].to_numpy(dtype=float),
        "Vgas": frame["Vgas"].to_numpy(dtype=float),
        "Vdisk": frame["Vdisk"].to_numpy(dtype=float),
        "Vbulge": frame["Vbulge"].to_numpy(dtype=float),
    }