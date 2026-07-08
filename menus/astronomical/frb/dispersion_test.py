#!/usr/bin/env python3
"""
FRB dispersion measure (DM) comparison near vs far from analytic Tav ladder nodes.

Uses healpy spherical geometry: ladder multipoles map to sky directions via the
7-fold cyclotomic + parity projector, and sightline classification uses
great-circle angular separation (rotator.angdist).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

try:
    import healpy as hp
except ImportError as exc:
    raise ImportError(
        "healpy is required for FRB dispersion Tav tests. "
        "Install with: ./venv/bin/pip install healpy"
    ) from exc

from menus.astronomical.frb.cosmic_web_tav import ARTIFACTS_DIR, load_frb_catalog
from menus.astronomical.frb.fetcher import resolve_frb_catalog_path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_NODE_THRESHOLD_DEG = 25.0
TAV_HARMONIC = 1.0 / 7.0
LADDER_ALPHA_DEFAULT = 41.341
ALLOWED_LADDER_RESIDUES = (0, 2, 3, 6)
GEOMETRY_MODEL = "healpy_spherical"


def default_tav_peaks(integrator: Any | None = None) -> List[int]:
    """Peaks from TavFrameworkIntegrator analytic ladder (α = 41.341)."""
    if integrator is None:
        from menus.integrator.integrator import TavFrameworkIntegrator

        integrator = TavFrameworkIntegrator()
    return integrator.analytic_tav_ladder()


def decompose_ladder_ell(
    ell: int,
    alpha: float = LADDER_ALPHA_DEFAULT,
) -> Tuple[int, int]:
    """Recover band index k and residue r from analytic ladder multipole ell."""
    k = max(1, int(round((float(ell) - 3.0) / (7.0 * alpha))))
    residue = int(round(float(ell) - 7.0 * k * alpha))
    if residue not in ALLOWED_LADDER_RESIDUES:
        residue = min(ALLOWED_LADDER_RESIDUES, key=lambda candidate: abs(candidate - residue))
    return k, residue


def ell_to_theta_phi(
    ell: int,
    alpha: float = LADDER_ALPHA_DEFAULT,
) -> Tuple[float, float]:
    """
    Map ladder multipole ell to healpy colatitude theta and azimuth phi [rad].

    phi encodes 7-fold Tav phasing; theta spreads nodes in declination by band.
    """
    k, residue = decompose_ladder_ell(ell, alpha)
    phi = (2.0 * np.pi) * (k * TAV_HARMONIC + residue / 7.0)
    phi = float(phi % (2.0 * np.pi))
    dec_deg = 75.0 * np.sin((k + residue / 7.0) * (2.0 * np.pi / 7.0))
    dec_deg = float(np.clip(dec_deg, -89.0, 89.0))
    theta = np.radians(90.0 - dec_deg)
    return float(theta), phi


def build_tav_node_directions(
    tav_peaks: List[int],
    alpha: float = LADDER_ALPHA_DEFAULT,
) -> np.ndarray:
    """Unit 3-vectors (N, 3) for Tav ladder nodes on the celestial sphere."""
    if not tav_peaks:
        return np.empty((0, 3), dtype=float)

    thetas = []
    phis = []
    for ell in tav_peaks:
        theta, phi = ell_to_theta_phi(int(ell), alpha=alpha)
        thetas.append(theta)
        phis.append(phi)

    return np.asarray(hp.ang2vec(np.asarray(thetas), np.asarray(phis)), dtype=float)


def radec_to_vec(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """FRB (RA, Dec) [deg] -> unit vectors (N, 3) for healpy angdist."""
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra % 360.0)
    return np.asarray(hp.ang2vec(theta, phi), dtype=float)


def min_angular_separations_deg(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    node_dirs: np.ndarray,
) -> np.ndarray:
    """
    Minimum great-circle separation [deg] from each sightline to any Tav node.

    Uses healpy unit vectors; separation via arccos(max dot product).
    """
    n_burst = len(ra_deg)
    if node_dirs.size == 0:
        return np.full(n_burst, 180.0, dtype=float)

    burst_dirs = radec_to_vec(ra_deg, dec_deg)
    node_dirs = np.asarray(node_dirs, dtype=float)
    if node_dirs.ndim == 1:
        node_dirs = node_dirs.reshape(1, 3)

    max_dot = np.max(burst_dirs @ node_dirs.T, axis=1)
    return np.degrees(np.arccos(np.clip(max_dot, -1.0, 1.0)))


def _yes(value: str) -> bool:
    return str(value or "").strip().lower() in {"yes", "y", "true", "1"}


class FRBDispersionTavTest:
    """
    Classify FRB sightlines as near/far from Tav harmonic nodes and compare DM.
    """

    def __init__(
        self,
        frb_catalog_path: Path | str,
        tav_peaks: List[int],
        *,
        node_threshold_deg: float = DEFAULT_NODE_THRESHOLD_DEG,
        ladder_alpha: float = LADDER_ALPHA_DEFAULT,
    ):
        self.catalog_path = Path(frb_catalog_path)
        self.tav_peaks = [int(p) for p in tav_peaks]
        self.node_threshold_deg = float(node_threshold_deg)
        self.ladder_alpha = float(ladder_alpha)
        self.frb = load_frb_catalog(self.catalog_path)
        self.node_directions = build_tav_node_directions(self.tav_peaks, alpha=self.ladder_alpha)
        self.node_sky_coords = self._node_sky_coords_table()
        self.results: Dict[str, Any] = {}

    @classmethod
    def from_defaults(
        cls,
        frb_catalog_path: Path | str | None = None,
        *,
        integrator: Any | None = None,
        node_threshold_deg: float = DEFAULT_NODE_THRESHOLD_DEG,
        force_refresh: bool = False,
        restore_archived: bool = True,
    ) -> "FRBDispersionTavTest":
        if integrator is None:
            from menus.integrator.integrator import TavFrameworkIntegrator

            integrator = TavFrameworkIntegrator()
        path = resolve_frb_catalog_path(
            frb_catalog_path,
            force_refresh=force_refresh,
            restore_archived=restore_archived,
            auto_fetch=True,
        )
        peaks = default_tav_peaks(integrator)
        alpha = float(integrator.config.get("analytic", {}).get("ladder_alpha", LADDER_ALPHA_DEFAULT))
        return cls(path, peaks, node_threshold_deg=node_threshold_deg, ladder_alpha=alpha)

    def _node_sky_coords_table(self) -> List[Dict[str, float]]:
        """RA/Dec [deg] for each Tav node (for reports / visualization)."""
        coords: List[Dict[str, float]] = []
        for ell in self.tav_peaks:
            theta, phi = ell_to_theta_phi(int(ell), alpha=self.ladder_alpha)
            coords.append(
                {
                    "ell": int(ell),
                    "ra_deg": round(float(np.degrees(phi) % 360.0), 3),
                    "dec_deg": round(float(90.0 - np.degrees(theta)), 3),
                }
            )
        return coords

    def classify_sightline(self, ra: float, dec: float) -> str:
        """Great-circle distance to nearest Tav node (healpy spherical geometry)."""
        sep = float(
            min_angular_separations_deg(
                np.asarray([ra]),
                np.asarray([dec]),
                self.node_directions,
            )[0]
        )
        return "near_node" if sep < self.node_threshold_deg else "far_from_node"

    def classify_all_sightlines(self) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized zone labels and minimum node separations [deg]."""
        separations = min_angular_separations_deg(
            self.frb["ra"].to_numpy(),
            self.frb["dec"].to_numpy(),
            self.node_directions,
        )
        zones = np.where(
            separations < self.node_threshold_deg,
            "near_node",
            "far_from_node",
        )
        return zones, separations

    def run_test(self) -> Dict[str, Any]:
        """KS test: DM distribution near Tav nodes vs far from nodes."""
        self.frb = self.frb.copy()
        zones, separations = self.classify_all_sightlines()
        self.frb["tav_zone"] = zones
        self.frb["node_sep_deg"] = separations

        near = self.frb.loc[self.frb["tav_zone"] == "near_node", "DM"].dropna()
        far = self.frb.loc[self.frb["tav_zone"] == "far_from_node", "DM"].dropna()

        n_near = int(len(near))
        n_far = int(len(far))

        if n_near < 2 or n_far < 2:
            self.results = {
                "ks_statistic": float("nan"),
                "p_value": float("nan"),
                "n_near_node": n_near,
                "n_far_from_node": n_far,
                "n_total": int(len(self.frb)),
                "node_threshold_deg": self.node_threshold_deg,
                "geometry_model": GEOMETRY_MODEL,
                "ladder_alpha": self.ladder_alpha,
                "tav_peak_count": len(self.tav_peaks),
                "node_sky_coords_sample": self.node_sky_coords[:8],
                "frb_catalog": str(self.catalog_path),
                "interpretation": (
                    "Insufficient bursts in near/far Tav zones for KS test "
                    f"(need ≥2 per group; got near={n_near}, far={n_far})."
                ),
                "significant": False,
            }
            return self.results

        stat, pval = ks_2samp(near, far)
        significant = bool(pval < 0.05)

        if significant:
            interpretation = (
                "Significant difference in DM between sightlines near vs far from "
                "Tav nodes (healpy great-circle geometry) supports topological impedance."
            )
        else:
            interpretation = (
                "No significant DM difference between near vs far Tav-node sightlines "
                "at current threshold (healpy spherical node model)."
            )

        self.results = {
            "ks_statistic": round(float(stat), 4),
            "p_value": round(float(pval), 6),
            "n_near_node": n_near,
            "n_far_from_node": n_far,
            "n_total": int(len(self.frb)),
            "dm_near_mean": round(float(near.mean()), 2),
            "dm_far_mean": round(float(far.mean()), 2),
            "mean_node_sep_near_deg": round(
                float(self.frb.loc[self.frb["tav_zone"] == "near_node", "node_sep_deg"].mean()),
                2,
            ),
            "mean_node_sep_far_deg": round(
                float(self.frb.loc[self.frb["tav_zone"] == "far_from_node", "node_sep_deg"].mean()),
                2,
            ),
            "node_threshold_deg": self.node_threshold_deg,
            "geometry_model": GEOMETRY_MODEL,
            "ladder_alpha": self.ladder_alpha,
            "tav_peaks_sample": self.tav_peaks[:12],
            "tav_peak_count": len(self.tav_peaks),
            "node_sky_coords_sample": self.node_sky_coords[:8],
            "frb_catalog": str(self.catalog_path),
            "interpretation": interpretation,
            "significant": significant,
        }
        return self.results

    def save_report(self, output_path: Path | str | None = None) -> Path:
        """Write JSON results under artifacts/."""
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        if output_path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = ARTIFACTS_DIR / f"frb_dispersion_tav_test_{stamp}.json"
        path = Path(output_path)
        payload = {
            "timestamp": datetime.now().isoformat(),
            "geometry_model": GEOMETRY_MODEL,
            "results": self.results,
            "node_sky_coords": self.node_sky_coords,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[TAV ENGINE] FRB dispersion test report: {path}")
        return path


def run_frb_dispersion_test(
    *,
    frb_catalog_path: Path | str | None = None,
    tav_peaks: Optional[List[int]] = None,
    node_threshold_deg: float = DEFAULT_NODE_THRESHOLD_DEG,
    force_refresh: bool = False,
    restore_archived: bool = True,
    save_report: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end FRB DM × Tav-node KS test (healpy spherical geometry).

    Auto-pulls, restores from finishedA/, and respects datasets/processed.txt
    unless force_refresh is True.
    """
    if tav_peaks is None:
        test = FRBDispersionTavTest.from_defaults(
            frb_catalog_path,
            node_threshold_deg=node_threshold_deg,
            force_refresh=force_refresh,
            restore_archived=restore_archived,
        )
    else:
        path = resolve_frb_catalog_path(
            frb_catalog_path,
            force_refresh=force_refresh,
            restore_archived=restore_archived,
            auto_fetch=True,
        )
        test = FRBDispersionTavTest(
            path,
            tav_peaks,
            node_threshold_deg=node_threshold_deg,
        )

    print(f"\n[TAV ENGINE] FRB Dispersion Tav Test — {test.catalog_path.name}")
    print(
        f"[TAV ENGINE] Geometry: {GEOMETRY_MODEL} | "
        f"{len(test.tav_peaks)} Tav nodes | threshold: {node_threshold_deg}° | "
        f"α={test.ladder_alpha}"
    )

    results = test.run_test()
    print(f"[TAV ENGINE] KS statistic: {results.get('ks_statistic')} | p-value: {results.get('p_value')}")
    print(f"[TAV ENGINE] Near node: {results.get('n_near_node')} | Far: {results.get('n_far_from_node')}")
    if results.get("mean_node_sep_near_deg") is not None:
        print(
            f"[TAV ENGINE] Mean sep (near/far): "
            f"{results.get('mean_node_sep_near_deg')}° / {results.get('mean_node_sep_far_deg')}°"
        )
    print(f"[TAV ENGINE] {results.get('interpretation')}")

    if save_report:
        test.save_report()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FRB DM dispersion Tav-node KS test.")
    parser.add_argument("--frb-catalog", default="", help="Path to FRB CSV")
    parser.add_argument("--threshold", type=float, default=DEFAULT_NODE_THRESHOLD_DEG)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--no-restore", action="store_true", help="Skip finishedA/ restore")
    args = parser.parse_args()

    catalog = args.frb_catalog or None
    run_frb_dispersion_test(
        frb_catalog_path=catalog,
        node_threshold_deg=args.threshold,
        force_refresh=args.force_refresh,
        restore_archived=not args.no_restore,
    )