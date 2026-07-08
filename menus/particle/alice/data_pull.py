"""
ALICE Heavy-Ion (Thermal Cooling) — O2 ROOT tracks or HEPData fallback.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import uproot

from menus.particle.hepdata.data_pull import fetch_and_graph as hepdata_fetch
from tav_research.data_pull_common import COMMON_BATCH_HINT, standard_entry_fields

MENU_LABEL = "ALICE Heavy-Ion (Thermal Cooling)"


def entry_instructions() -> list[str]:
    return [
        "Query: local AO2D.root path OR numeric HEPData/INSPIRE record ID.",
        "Files: ALICE O2 .root with DF_*/O2track;1 trees (fSigned1Pt, fTgl).",
        "URLs (HEPData mode): https://www.hepdata.net/record/ins{ID}?format=json",
        "Batch .txt lists of ROOT paths or record IDs supported.",
    ]


def entry_fields() -> list[dict]:
    return standard_entry_fields(
        MENU_LABEL,
        query_hint="AO2D.root or HEPData record ID",
    )


def load_alice_o2_root(path: str, entry_stop: int = 50000) -> tuple[pd.DataFrame, str]:
    """Load ALICE O2 AO2D files (DF_*/O2track trees)."""
    with uproot.open(path) as file:
        track_keys = [k for k in file.keys() if k.endswith("/O2track;1")]
        if not track_keys:
            raise KeyError(
                "no O2track tree found; "
                f"available keys include: {', '.join(list(file.keys())[:8])}..."
            )
        track_key = track_keys[0]
        print(f"[TAV ENGINE] Using tree: {track_key}")
        tree = file[track_key]
        data = tree.arrays(["fSigned1Pt", "fTgl"], library="np", entry_stop=entry_stop)
        signed_1pt = data["fSigned1Pt"]
        tgl = data["fTgl"]
        valid = np.abs(signed_1pt) > 1e-6
        if not np.any(valid):
            raise ValueError("O2track table has no valid fSigned1Pt entries")
        pt = np.zeros_like(signed_1pt, dtype=float)
        eta = np.zeros_like(signed_1pt, dtype=float)
        pt[valid] = 1.0 / np.abs(signed_1pt[valid])
        dip = np.arctan(tgl[valid])
        eta[valid] = -np.log(np.tan(np.pi / 4 - dip / 2))
        df = pd.DataFrame({"Pt": pt[valid], "Eta": eta[valid]})
        return df, track_key


def fetch_and_graph(query: str, params: dict[str, Any] | None = None) -> None:
    params = params or {}
    if query.endswith(".root"):
        try:
            entry_stop = int(params.get("root_entry_limit") or 50000)
        except ValueError:
            entry_stop = 50000
        print(f"[TAV ENGINE] Piercing local ROOT substrate: {query}")
        df, track_key = load_alice_o2_root(query, entry_stop=entry_stop)
        print(f"[TAV ENGINE] Loaded {len(df)} tracks from {track_key}")
        return
    hepdata_fetch(query, params)