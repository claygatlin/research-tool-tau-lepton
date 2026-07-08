"""
SPARC rotation-curve analyzer for Tav-Superblock conformal shadow fits.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THEORY_RATIO_TARGET = 5.2
COLUMN_ALIASES = {
    "R": ["R", "Radius", "radius", "r", "kpc"],
    "Vobs": ["Vobs", "Velocity", "velocity", "V_obs", "vobs"],
    "Vgas": ["Vgas", "V_gas", "vgas"],
    "Vdisk": ["Vdisk", "V_disk", "vdisk"],
    "Vbulge": ["Vbulge", "V_bulge", "vbulge"],
    "name": ["name", "Name", "galaxy", "Galaxy", "id", "ID"],
}


def resolve_sparc_file(query: str, params: Optional[dict] = None) -> str:
    """Resolve a SPARC CSV from galaxy ID, explicit path, or entry-form override."""
    params = params or {}
    override = (params.get("sparc_file") or params.get("data_file") or "").strip()
    if override and os.path.isfile(override):
        return override
    if query.endswith(".csv") and os.path.isfile(query):
        return query
    if os.path.isabs(query) and os.path.isfile(query):
        return query
    try:
        from menus.astronomical.sparc.fetcher import DEFAULT_DATA_DIR

        return str(DEFAULT_DATA_DIR / f"{query}.csv")
    except ImportError:
        return str(Path("datasets/sparc") / f"{query}.csv")


class SPARCAnalyzer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = self._load_and_normalize(file_path)

    @staticmethod
    def _pick_column(frame: pd.DataFrame, aliases: list[str]) -> Optional[str]:
        for alias in aliases:
            if alias in frame.columns:
                return alias
        return None

    def _load_and_normalize(self, file_path: str) -> pd.DataFrame:
        frame = pd.read_csv(file_path)
        normalized = pd.DataFrame()
        for canonical, aliases in COLUMN_ALIASES.items():
            source = self._pick_column(frame, aliases)
            if source is not None:
                normalized[canonical] = pd.to_numeric(frame[source], errors="coerce")

        if "R" not in normalized or "Vobs" not in normalized:
            raise ValueError(
                "SPARC CSV must include radius and observed velocity columns "
                "(R/Radius and Vobs/Velocity)."
            )

        for component in ("Vgas", "Vdisk", "Vbulge"):
            if component not in normalized:
                normalized[component] = 0.0

        if "name" not in normalized:
            stem = Path(file_path).stem
            normalized["name"] = stem

        normalized = normalized.dropna(subset=["R", "Vobs"]).sort_values("R")
        return normalized.reset_index(drop=True)

    def has_rotation_components(self) -> bool:
        baryon = self.data["Vgas"] ** 2 + self.data["Vdisk"] ** 2 + self.data["Vbulge"] ** 2
        return bool((baryon > 0).any())

    def get_galaxy_frame(self, galaxy_name: str) -> pd.DataFrame:
        if "name" in self.data.columns and self.data["name"].nunique() > 1:
            subset = self.data[self.data["name"].astype(str).str.upper() == galaxy_name.upper()]
            if subset.empty:
                available = ", ".join(sorted(self.data["name"].astype(str).unique())[:8])
                raise ValueError(f"Galaxy '{galaxy_name}' not found. Available: {available}")
            return subset.reset_index(drop=True)
        return self.data.copy()

    def plot_rotation_curve(self, galaxy_name: str, show: bool = True, save_path: Optional[str] = None):
        gal = self.get_galaxy_frame(galaxy_name)

        v_baryonic = np.sqrt(gal["Vgas"] ** 2 + gal["Vdisk"] ** 2 + gal["Vbulge"] ** 2)
        v_shadow = np.sqrt(np.maximum(0.0, gal["Vobs"] ** 2 - v_baryonic ** 2))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(gal["R"], gal["Vobs"], "k.", label="Observed (Vobs)")
        if self.has_rotation_components():
            ax.plot(gal["R"], gal["Vgas"], "b--", label="Gas")
            ax.plot(gal["R"], gal["Vdisk"], "g--", label="Disk")
            if (gal["Vbulge"] > 0).any():
                ax.plot(gal["R"], gal["Vbulge"], "m--", label="Bulge")
            ax.plot(gal["R"], v_shadow, "r-", linewidth=2, label="Predicted Shadow (SBT)")
        else:
            ax.plot(gal["R"], gal["Vobs"], "r-", linewidth=1.5, label="Observed trend (no baryon split)")

        ax.set_title(f"Rotation Curve: {galaxy_name} - Tav-Superblock Fit")
        ax.set_xlabel("Radius (kpc)")
        ax.set_ylabel("Velocity (km/s)")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    def calculate_mass_ratio(self, galaxy_name: Optional[str] = None) -> float:
        gal = self.get_galaxy_frame(galaxy_name) if galaxy_name else self.data
        v_sq_obs = gal["Vobs"] ** 2
        v_sq_baryon = gal["Vgas"] ** 2 + gal["Vdisk"] ** 2 + gal["Vbulge"] ** 2
        v_sq_shadow = np.maximum(0.0, v_sq_obs - v_sq_baryon)
        ratio = v_sq_shadow / np.maximum(v_sq_baryon, 1e-6)
        return float(ratio.mean())

    def analysis_report(self, galaxy_name: str) -> dict:
        ratio = self.calculate_mass_ratio(galaxy_name)
        delta = abs(ratio - THEORY_RATIO_TARGET)
        return {
            "galaxy": galaxy_name,
            "file": self.file_path,
            "mean_shadow_baryon_ratio": ratio,
            "theory_target_ratio": THEORY_RATIO_TARGET,
            "delta_from_theory": delta,
            "within_10_percent": delta <= THEORY_RATIO_TARGET * 0.1,
        }