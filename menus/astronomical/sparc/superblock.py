#!/usr/bin/env python3
"""
sparc_superblock.py  (Extended v2)

Dedicated module for SPARC + Superblock Theory.

Extensions implemented (as requested):
1. Replaced toy model in `predict_emergent_dm_velocity()` with a more physical mapping
   from domain parameters (β², phase_offset) to effective emergent dark matter velocity.
2. Added real rotation curve loading (supports per-galaxy files in a subfolder
   or combined mass model tables).
3. Added fitting routines using least-squares (`scipy.optimize.curve_fit`) for
   emergent DM parameters (β² and phase offset).
4. Added `run_sparc_test_suite()` for systematic analysis across many galaxies,
   plus clear integration example with the empirical tests pipeline.

This module is designed to work alongside:
- `tav_superblock_prime_past_harmonic.py`
- `superblock_empirical_tests.py`
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not found — fitting will be disabled. pip install scipy")

# =============================================================================
# CONFIG
# =============================================================================
_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = _PROJECT_ROOT / "datasets" / "sparc"
ARTIFACTS_DIR = _PROJECT_ROOT / "artifacts"
SPARC_BASE_URL = "https://astroweb.case.edu/SPARC/"

COLUMN_ALIASES = {
    "R": ["R", "Radius", "radius", "r", "kpc", "Rad"],
    "V_obs": ["V_obs", "Vobs", "Velocity", "velocity", "vobs", "Vrot"],
    "V_obs_err": ["V_obs_err", "e_Vrot", "Verr", "verr"],
    "V_gas": ["V_gas", "Vgas", "vgas"],
    "V_disk": ["V_disk", "Vdisk", "vdisk"],
    "V_bul": ["V_bul", "Vbulge", "Vbul", "vbulge"],
    "Name": ["Name", "name", "galaxy", "Galaxy"],
}


def _pick_column(frame: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    for alias in aliases:
        if alias in frame.columns:
            return alias
    return None


def _normalize_rotation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = pd.DataFrame()
    for canonical, aliases in COLUMN_ALIASES.items():
        source = _pick_column(frame, aliases)
        if source is not None:
            normalized[canonical] = pd.to_numeric(frame[source], errors="coerce")

    if "R" not in normalized or "V_obs" not in normalized:
        raise ValueError("Rotation curve must include radius and observed velocity columns.")

    for component in ("V_gas", "V_disk", "V_bul"):
        if component not in normalized:
            normalized[component] = 0.0
    if "V_obs_err" not in normalized:
        normalized["V_obs_err"] = 5.0

    if "V_bar" not in normalized.columns:
        normalized["V_bar"] = np.sqrt(
            normalized["V_gas"].fillna(0) ** 2
            + normalized["V_disk"].fillna(0) ** 2
            + normalized["V_bul"].fillna(0) ** 2
        )

    return normalized.dropna(subset=["R", "V_obs"]).sort_values("R").reset_index(drop=True)


# =============================================================================
# MAIN CLASS
# =============================================================================
class SPARCData:
    """SPARC data handler with strong Superblock Theory integration."""

    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.galaxy_table: Optional[pd.DataFrame] = None
        self._rotation_curves: Dict[str, pd.DataFrame] = {}

    # -------------------------------------------------------------------------
    # DATA ACCESS
    # -------------------------------------------------------------------------
    def download_data(self, force: bool = False, limit: int = 100):
        """
        Pull rotation-curve CSVs from the public SPARC repository.

        Uses sparc_fetcher.pull_sparc_batch(); completed targets are recorded in
        data/sparc/done.txt and skipped on subsequent runs.
        """
        from menus.astronomical.sparc.fetcher import DEFAULT_DONE_FILE, pull_sparc_batch

        done_file = self.data_dir / "done.txt"
        if done_file.resolve() != DEFAULT_DONE_FILE.resolve():
            done_file = DEFAULT_DONE_FILE

        summary = pull_sparc_batch(
            limit=limit,
            data_dir=self.data_dir,
            done_file=done_file,
            force_refresh_catalog=force,
        )
        print(
            f"[SPARC] download_data complete: {summary.downloaded_count} new CSVs, "
            f"{len(summary.failed)} failed."
        )
        return summary

    def load_galaxy_table(self) -> pd.DataFrame:
        if self.galaxy_table is not None:
            return self.galaxy_table

        for path in self.data_dir.glob("*.mrt"):
            try:
                df = pd.read_csv(path, comment="#", delim_whitespace=True, engine="python")
                self.galaxy_table = df
                print(f"[SPARC] Loaded galaxy table ({len(df)} galaxies) from {path.name}")
                return df
            except Exception:
                continue

        discovered = [{"Name": name} for name in self._discover_galaxy_names()]
        if discovered:
            self.galaxy_table = pd.DataFrame(discovered)
            print(f"[SPARC] Built galaxy index from local files ({len(discovered)} entries)")
            return self.galaxy_table

        raise FileNotFoundError(
            f"SPARC galaxy table (.mrt) not found in {self.data_dir} and no per-galaxy files detected."
        )

    def _discover_galaxy_names(self) -> List[str]:
        names = set()
        for path in self.data_dir.glob("*.csv"):
            names.add(path.stem)
        rc_dir = self.data_dir / "RotationCurves"
        if rc_dir.is_dir():
            for path in rc_dir.iterdir():
                if path.suffix.lower() in {".dat", ".txt", ".csv"}:
                    names.add(path.stem)
        return sorted(names)

    def load_rotation_curve(self, galaxy_name: str, subdir: str = "RotationCurves") -> pd.DataFrame:
        """
        Load rotation curve data.
        Priority:
        1. Per-galaxy file in data_dir: {galaxy_name}.csv
        2. Per-galaxy file: data_dir / subdir / {galaxy_name}.dat or .txt
        3. Combined mass models table (if present in data_dir)
        """
        key = galaxy_name.strip()
        if key in self._rotation_curves:
            return self._rotation_curves[key]

        candidates: List[Path] = [self.data_dir / f"{key}.csv"]
        for ext in [".dat", ".txt", ".csv"]:
            candidates.append(self.data_dir / subdir / f"{key}{ext}")

        for path in candidates:
            if not path.exists():
                continue
            try:
                if path.suffix.lower() == ".csv":
                    df = pd.read_csv(path)
                else:
                    df = pd.read_csv(path, comment="#", delim_whitespace=True)
                df = _normalize_rotation_frame(df)
                self._rotation_curves[key] = df
                return df
            except Exception as exc:
                print(f"[SPARC] Failed loading {path}: {exc}")

        combined = self.data_dir / "mass_models.txt"
        if combined.exists():
            try:
                df_all = pd.read_csv(combined, comment="#", delim_whitespace=True)
                name_col = _pick_column(df_all, COLUMN_ALIASES["Name"])
                if name_col:
                    subset = df_all[df_all[name_col].astype(str) == key].copy()
                    if len(subset) > 0:
                        df = _normalize_rotation_frame(subset)
                        self._rotation_curves[key] = df
                        return df
            except Exception:
                pass

        print(f"[SPARC] No rotation curve found for {key}. Returning empty frame.")
        empty = pd.DataFrame(columns=["R", "V_obs", "V_obs_err", "V_bar"])
        self._rotation_curves[key] = empty
        return empty

    # -------------------------------------------------------------------------
    # DERIVED QUANTITIES
    # -------------------------------------------------------------------------
    def compute_mass_discrepancy(self, rc_df: pd.DataFrame) -> pd.DataFrame:
        df = rc_df.copy()
        if {"V_obs", "V_bar"}.issubset(df.columns):
            df["V_ratio"] = df["V_obs"] / np.maximum(df["V_bar"], 1e-3)
            df["g_obs"] = df["V_obs"] ** 2 / np.maximum(df["R"], 0.1)
            df["g_bar"] = df["V_bar"] ** 2 / np.maximum(df["R"], 0.1)
            df["g_ratio"] = df["g_obs"] / np.maximum(df["g_bar"], 1e-3)
        return df

    # -------------------------------------------------------------------------
    # 1. IMPROVED EMERGENT DM MODEL
    # -------------------------------------------------------------------------
    def predict_emergent_dm_velocity(
        self,
        galaxy_name: str,
        beta2: float = 0.3,
        phase_offset: float = 1.0,
        theory: Any = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Predict emergent dark matter contribution using Superblock domain parameters."""
        rc = self.load_rotation_curve(galaxy_name)
        if len(rc) == 0:
            return rc

        try:
            info = self.get_galaxy_info(galaxy_name)
            logL = np.log10(max(info.get("L3.6", 1e9), 1e8))
        except Exception:
            logL = 9.5

        r_c = 2.0 + 4.0 * beta2 + 0.8 * np.sin(phase_offset)
        v_scale = 45 + 25 * (logL - 9) * (1 + 0.4 * beta2)

        radius = np.asarray(rc["R"])
        v_dm = v_scale * np.sqrt(radius / (radius + r_c)) * (1 + 0.12 * np.sin(phase_offset + radius / 6))

        if theory is not None:
            try:
                if hasattr(theory, "effective_mass"):
                    scale = np.sqrt(max(theory.effective_mass / 313.1, 0.6))
                    v_dm *= scale
            except Exception:
                pass

        v_bar = np.asarray(rc.get("V_bar", rc.get("V_disk", 0) + rc.get("V_gas", 0)))
        v_total = np.sqrt(v_bar ** 2 + v_dm ** 2)

        out = rc.copy()
        out["V_dm_emergent"] = v_dm
        out["V_theory"] = v_total
        out["beta2_used"] = beta2
        out["phase_offset_used"] = phase_offset
        return out

    # -------------------------------------------------------------------------
    # 3. FITTING ROUTINES
    # -------------------------------------------------------------------------
    def fit_emergent_dm(
        self,
        galaxy_name: str,
        initial_beta2: float = 0.3,
        initial_phase: float = 1.0,
    ) -> Dict[str, Any]:
        """Least-squares fit of emergent DM parameters to observed rotation curve."""
        if not HAS_SCIPY:
            return {"success": False, "message": "scipy required for fitting"}

        rc = self.load_rotation_curve(galaxy_name)
        if len(rc) < 5:
            return {"success": False, "message": "Not enough data points"}

        radius = np.array(rc["R"])
        v_obs = np.array(rc["V_obs"])
        v_obs_err = np.array(rc.get("V_obs_err", np.ones_like(v_obs) * 5))
        v_bar = np.array(rc.get("V_bar", np.zeros_like(radius)))

        def model(r_vals, beta2, phase):
            r_c = 2.0 + 4.0 * beta2 + 0.8 * np.sin(phase)
            v_scale = 45 + 20 * beta2
            v_dm = v_scale * np.sqrt(r_vals / (r_vals + r_c)) * (1 + 0.1 * np.sin(phase + r_vals / 6))
            return np.sqrt(v_bar ** 2 + v_dm ** 2)

        try:
            popt, pcov = curve_fit(
                model,
                radius,
                v_obs,
                p0=[initial_beta2, initial_phase],
                sigma=v_obs_err,
                bounds=([0.0, -np.pi], [0.95, np.pi]),
                maxfev=8000,
            )
            perr = np.sqrt(np.diag(pcov))
            chi2 = np.sum(((v_obs - model(radius, *popt)) / v_obs_err) ** 2)
            ndof = max(len(radius) - 2, 1)
            return {
                "success": True,
                "beta2": float(popt[0]),
                "phase_offset": float(popt[1]),
                "beta2_err": float(perr[0]),
                "phase_err": float(perr[1]),
                "chi2": float(chi2),
                "reduced_chi2": float(chi2 / ndof),
                "message": "Fit converged successfully",
            }
        except Exception as exc:
            return {"success": False, "message": str(exc)}

    # -------------------------------------------------------------------------
    # 4. SYSTEMATIC TEST SUITE + INTEGRATION
    # -------------------------------------------------------------------------
    def run_sparc_test_suite(
        self,
        galaxies: Optional[List[str]] = None,
        max_galaxies: int = 25,
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Systematic analysis across SPARC galaxies.
        Returns DataFrame with fit results, tensions, and statistics.
        """
        if galaxies is None:
            try:
                galaxies = self.list_galaxies()[:max_galaxies]
            except Exception:
                galaxies = ["NGC2403", "NGC3198", "UGC07261", "NGC5055"]

        results = []
        for gal in galaxies:
            try:
                rc = self.load_rotation_curve(gal)
                if len(rc) < 5:
                    continue

                row: Dict[str, Any] = {"galaxy": gal, "n_points": len(rc)}

                if fit and HAS_SCIPY:
                    fit_res = self.fit_emergent_dm(gal)
                    row.update(fit_res)
                else:
                    row.update({"success": False})

                rc_disc = self.compute_mass_discrepancy(rc)
                if "g_ratio" in rc_disc.columns and len(rc_disc) > 0:
                    row["median_g_ratio"] = float(np.median(rc_disc["g_ratio"]))
                    row["max_g_ratio"] = float(rc_disc["g_ratio"].max())

                results.append(row)
            except Exception as exc:
                results.append({"galaxy": gal, "success": False, "message": str(exc)})

        return pd.DataFrame(results)

    # -------------------------------------------------------------------------
    # UTILITIES
    # -------------------------------------------------------------------------
    def list_galaxies(self) -> List[str]:
        if self.galaxy_table is None:
            try:
                self.load_galaxy_table()
            except FileNotFoundError:
                return self._discover_galaxy_names()
        if self.galaxy_table is not None and "Name" in self.galaxy_table.columns:
            return self.galaxy_table["Name"].astype(str).tolist()
        return self._discover_galaxy_names()

    def get_galaxy_info(self, galaxy_name: str) -> Dict[str, Any]:
        if self.galaxy_table is None:
            try:
                self.load_galaxy_table()
            except FileNotFoundError:
                return {}
        if self.galaxy_table is not None:
            row = self.galaxy_table[self.galaxy_table["Name"] == galaxy_name]
            if len(row) > 0:
                return row.iloc[0].to_dict()
        return {}

    def plot_rotation_curve(
        self,
        galaxy_name: str,
        theory_curve: Optional[pd.DataFrame] = None,
        show: bool = True,
        save_path: Optional[str] = None,
    ):
        if not HAS_MPL:
            return None
        rc = self.load_rotation_curve(galaxy_name)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(
            rc["R"],
            rc["V_obs"],
            yerr=rc.get("V_obs_err", 5),
            fmt="o",
            label="V_obs (SPARC)",
            alpha=0.7,
        )
        if "V_bar" in rc.columns:
            ax.plot(rc["R"], rc["V_bar"], label="V_bar (baryons)", color="orange")
        if theory_curve is not None and "V_theory" in theory_curve.columns:
            ax.plot(
                theory_curve["R"],
                theory_curve["V_theory"],
                label="Superblock emergent DM",
                color="purple",
                linestyle="--",
                linewidth=2,
            )
        ax.set_xlabel("Radius (kpc)")
        ax.set_ylabel("Velocity (km/s)")
        ax.set_title(f"Rotation Curve — {galaxy_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig


# =============================================================================
# INTEGRATION WITH EMPIRICAL TESTS PIPELINE
# =============================================================================
def example_integration_with_empirical_tests():
    """How to use this module inside superblock_empirical_tests.py or analysis notebooks."""
    print("\n=== Integration Example ===")
    print("""
# In your empirical tests script or notebook:

from menus.astronomical.sparc.superblock import SPARCData
# from menus.prime_past.harmonic import TavSuperblockPrimePastHarmonic

sparc = SPARCData(data_dir="./data/sparc")
sparc.download_data()

results = sparc.run_sparc_test_suite(max_galaxies=30, fit=True)
print(results[["galaxy", "beta2", "reduced_chi2", "median_g_ratio"]].head(10))

good_fits = results[results.success == True]
mean_beta2 = good_fits["beta2"].mean()
print(f"Mean best-fit β² from SPARC sample: {mean_beta2:.3f}")
    """)


if __name__ == "__main__":
    print("SPARC + Superblock Theory Module — Extended Demo")
    sparc = SPARCData()

    try:
        print("\nFirst 8 galaxies:", sparc.list_galaxies()[:8])

        fit = sparc.fit_emergent_dm("NGC3198")
        print("\nFit result for NGC3198:")
        print(fit)

        suite = sparc.run_sparc_test_suite(max_galaxies=6, fit=True)
        print("\nTest suite results:")
        print(suite[["galaxy", "success", "beta2", "reduced_chi2"]])

    except Exception as exc:
        print(f"Demo limited (data files may be missing): {exc}")

    example_integration_with_empirical_tests()