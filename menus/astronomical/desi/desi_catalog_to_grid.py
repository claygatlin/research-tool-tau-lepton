"""
DESI / BAO catalog → gridded density contrast δ(x,y,z).

Builds a comoving galaxy catalog from DR2 BAO tracer redshift shells (or a
user-supplied CSV), deposits with Cloud-In-Cell (CIC), and returns δ on a
cubic FFT-ready grid for LSS falsification Methods 1, 4, 9, and 10.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from menus.astronomical.desi.fetcher import DEFAULT_COBAYA_ROOT
from tav_shared.artifact_paths import TestSlug, artifact_path, artifact_timestamp, compose_dataset_slug
from menus.astronomical.desi.scanner import (
    H0_H_UNITS,
    OMEGA_M_FIDUCIAL,
    R_TAU_MPC,
    load_desi_from_cobaya_repo,
    s_from_z,
)

H0_KM_S_MPC: float = H0_H_UNITS  # h=1 units → distances in h⁻¹ Mpc


def _comoving_distance_mpc(
    z: float,
    *,
    h0: float = H0_KM_S_MPC,
    om: float = OMEGA_M_FIDUCIAL,
) -> float:
    """Flat ΛCDM comoving distance [h⁻¹ Mpc] via trapezoid integral."""
    z = float(z)
    if z <= 0:
        return 0.0
    n = 128
    zg = np.linspace(0.0, z, n)
    ez = np.sqrt(om * (1 + zg) ** 3 + (1 - om))
    integrand = 1.0 / ez
    dz = np.diff(zg, prepend=0.0)
    dc = (299792.458 / h0) * np.cumsum(0.5 * (integrand + np.roll(integrand, 1)) * dz)
    return float(dc[-1])


def load_bao_shell_catalog(
    *,
    cobaya_path: str | Path | None = None,
    tracer: str = "ALL_GCcomb",
    galaxies_per_shell: int = 800,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Synthetic galaxy catalog anchored to DESI DR2 BAO effective redshifts.

    Each BAO measurement defines a shell; galaxies are drawn uniformly in
    comoving volume within ±Δz of the reported z_eff.
    """
    cobaya_path = Path(cobaya_path) if cobaya_path else DEFAULT_COBAYA_ROOT
    bao = load_desi_from_cobaya_repo(
        cobaya_path,
        tracer=tracer,
        quantity_filter="DM_over_rs",
    )
    z_eff = np.asarray(bao["z"], dtype=float)
    rng = np.random.default_rng(seed)

    ra_list: list[float] = []
    dec_list: list[float] = []
    z_list: list[float] = []
    weight_list: list[float] = []
    shell_meta: list[dict[str, Any]] = []

    for i, zc in enumerate(z_eff):
        dz = 0.04 if len(z_eff) < 8 else 0.03
        n_gal = max(50, int(galaxies_per_shell))
        z_draw = rng.uniform(max(0.01, zc - dz), zc + dz, n_gal)
        ra = rng.uniform(0.0, 360.0, n_gal)
        dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n_gal)))
        ra_list.extend(ra.tolist())
        dec_list.extend(dec.tolist())
        z_list.extend(z_draw.tolist())
        weight_list.extend(np.ones(n_gal, dtype=float).tolist())
        shell_meta.append(
            {
                "shell_index": int(i),
                "z_eff_bao": float(zc),
                "n_galaxies": n_gal,
                "z_span": [float(zc - dz), float(zc + dz)],
            }
        )

    catalog = {
        "ra_deg": np.asarray(ra_list, dtype=float),
        "dec_deg": np.asarray(dec_list, dtype=float),
        "z": np.asarray(z_list, dtype=float),
        "weight": np.asarray(weight_list, dtype=float),
        "n_galaxies": len(z_list),
        "source": "bao_shell_synthetic",
        "tracer": tracer,
        "cobaya_path": str(cobaya_path),
        "shells": shell_meta,
        "seed": int(seed),
    }
    return catalog


def load_catalog_csv(
    path: str | Path,
    *,
    ra_col: str = "RA",
    dec_col: str = "DEC",
    z_col: str = "Z",
    weight_col: str | None = "WEIGHT",
) -> dict[str, Any]:
    """Load a simple whitespace/CSV galaxy catalog."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Catalog not found: {path}")

    rows_ra: list[float] = []
    rows_dec: list[float] = []
    rows_z: list[float] = []
    rows_w: list[float] = []
    header: list[str] = []

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",") if "," in line else line.split()
        if not header:
            header = [p.strip().upper() for p in parts]
            continue
        row = {header[j]: parts[j] for j in range(min(len(header), len(parts)))}
        try:
            ra = float(row.get(ra_col.upper(), row.get("RA", "nan")))
            dec = float(row.get(dec_col.upper(), row.get("DEC", "nan")))
            z = float(row.get(z_col.upper(), row.get("Z", "nan")))
            w = float(row.get((weight_col or "WEIGHT").upper(), "1.0"))
        except ValueError:
            continue
        if np.isfinite(ra) and np.isfinite(dec) and np.isfinite(z) and z > 0:
            rows_ra.append(ra)
            rows_dec.append(dec)
            rows_z.append(z)
            rows_w.append(w if np.isfinite(w) and w > 0 else 1.0)

    if not rows_z:
        raise ValueError(f"No valid rows parsed from {path}")

    return {
        "ra_deg": np.asarray(rows_ra, dtype=float),
        "dec_deg": np.asarray(rows_dec, dtype=float),
        "z": np.asarray(rows_z, dtype=float),
        "weight": np.asarray(rows_w, dtype=float),
        "n_galaxies": len(rows_z),
        "source": str(path),
        "tracer": "user_csv",
        "shells": [],
    }


def catalog_to_comoving_positions(
    catalog: dict[str, Any],
    *,
    box_size_mpc: float,
    h0: float = H0_KM_S_MPC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Map (RA, Dec, z) → comoving Cartesian coords in a periodic cube."""
    ra = np.radians(np.asarray(catalog["ra_deg"], dtype=float))
    dec = np.radians(np.asarray(catalog["dec_deg"], dtype=float))
    z = np.asarray(catalog["z"], dtype=float)
    weight = np.asarray(catalog.get("weight", np.ones_like(z)), dtype=float)

    chi = np.array([_comoving_distance_mpc(zi, h0=h0) for zi in z], dtype=float)
    x = chi * np.cos(dec) * np.cos(ra)
    y = chi * np.cos(dec) * np.sin(ra)
    zc = chi * np.sin(dec)

    # Centre and wrap into periodic box
    box = float(box_size_mpc)
    for arr in (x, y, zc):
        arr -= float(np.median(arr))
        arr[:] = np.mod(arr, box)

    meta = {
        "box_size_mpc": box,
        "chi_min_mpc": float(np.min(chi)),
        "chi_max_mpc": float(np.max(chi)),
        "n_galaxies": int(len(z)),
        "mean_weight": float(np.mean(weight)),
    }
    return x, y, zc, meta


def cic_deposit(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    weight: np.ndarray,
    *,
    n_per_axis: int,
    box_size_mpc: float,
) -> np.ndarray:
    """Cloud-In-Cell deposit of weighted galaxies onto a cubic grid."""
    n = max(8, int(n_per_axis))
    box = float(box_size_mpc)
    cell = box / n
    rho = np.zeros((n, n, n), dtype=float)
    w = np.asarray(weight, dtype=float)

    for xi, yi, zi, wi in zip(x, y, z, w, strict=False):
        gx = xi / cell
        gy = yi / cell
        gz = zi / cell
        i0 = int(np.floor(gx)) % n
        j0 = int(np.floor(gy)) % n
        k0 = int(np.floor(gz)) % n
        fx = gx - np.floor(gx)
        fy = gy - np.floor(gy)
        fz = gz - np.floor(gz)
        for di, wx in ((0, 1 - fx), (1, fx)):
            for dj, wy in ((0, 1 - fy), (1, fy)):
                for dk, wz in ((0, 1 - fz), (1, fz)):
                    rho[(i0 + di) % n, (j0 + dj) % n, (k0 + dk) % n] += wi * wx * wy * wz

    return rho


def catalog_to_delta_field(
    catalog: dict[str, Any],
    *,
    n_per_axis: int = 64,
    box_size_mpc: float = 2000.0,
    h0: float = H0_KM_S_MPC,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Full pipeline: catalog → ρ grid → δ = (ρ − ρ̄)/ρ̄."""
    x, y, zc, pos_meta = catalog_to_comoving_positions(
        catalog, box_size_mpc=box_size_mpc, h0=h0
    )
    weight = np.asarray(catalog.get("weight", np.ones(len(x))), dtype=float)
    rho = cic_deposit(
        x, y, zc, weight, n_per_axis=n_per_axis, box_size_mpc=box_size_mpc
    )
    rho_bar = float(np.mean(rho)) or 1.0
    delta = (rho - rho_bar) / rho_bar

    meta = {
        **pos_meta,
        "n_per_axis": int(n_per_axis),
        "catalog_source": catalog.get("source", "unknown"),
        "tracer": catalog.get("tracer", "unknown"),
        "n_galaxies": int(catalog.get("n_galaxies", len(x))),
        "rho_mean": rho_bar,
        "delta_std": float(np.std(delta)),
        "delta_min": float(np.min(delta)),
        "delta_max": float(np.max(delta)),
        "R_tau_mpc": R_TAU_MPC,
        "dimension": 3,
    }
    return delta, meta


def run_desi_catalog_to_grid(
    *,
    catalog_path: str | Path | None = None,
    cobaya_path: str | Path | None = None,
    tracer: str = "ALL_GCcomb",
    galaxies_per_shell: int = 800,
    n_per_axis: int = 64,
    box_size_mpc: float = 2000.0,
    seed: int = 42,
    output_prefix: str = "desi_catalog_grid",
    verbose: bool = True,
) -> dict[str, Any]:
    """Menu entry: build δ grid and persist NPZ + JSON metadata."""
    if catalog_path:
        catalog = load_catalog_csv(catalog_path)
    else:
        catalog = load_bao_shell_catalog(
            cobaya_path=cobaya_path,
            tracer=tracer,
            galaxies_per_shell=galaxies_per_shell,
            seed=seed,
        )

    delta, meta = catalog_to_delta_field(
        catalog,
        n_per_axis=n_per_axis,
        box_size_mpc=box_size_mpc,
    )

    dataset = compose_dataset_slug(tracer, output_prefix)
    npz_path = artifact_path(TestSlug.DESI_CATALOG_GRID, dataset, "delta_grid", "npz")
    np.savez_compressed(
        npz_path,
        delta=delta,
        box_size_mpc=box_size_mpc,
        n_per_axis=n_per_axis,
    )

    from menus.astronomical.desi.json_util import write_json

    report = {
        "action": "DESI Catalog → LSS Grid",
        "npz_path": str(npz_path),
        "metadata": meta,
        "catalog_summary": {
            "n_galaxies": int(catalog["n_galaxies"]),
            "source": catalog.get("source"),
            "tracer": catalog.get("tracer"),
            "z_range": [
                float(np.min(catalog["z"])),
                float(np.max(catalog["z"])),
            ],
            "s_range": [
                float(np.min(s_from_z(catalog["z"]))),
                float(np.max(s_from_z(catalog["z"]))),
            ],
        },
        "timestamp": artifact_timestamp(),
    }
    json_path = artifact_path(TestSlug.DESI_CATALOG_GRID, dataset, "report", "json")
    write_json(json_path, report, indent=2, sort_keys=True)
    report["report_path"] = str(json_path)

    if verbose:
        print("=" * 70)
        print("DESI CATALOG → LSS GRID")
        print(f"  Galaxies : {catalog['n_galaxies']}")
        print(f"  Grid     : {n_per_axis}³ | box = {box_size_mpc:.0f} h⁻¹ Mpc")
        print(f"  δ std    : {meta['delta_std']:.4f}")
        print(f"  NPZ      : {npz_path}")
        print("=" * 70)

    return report


def load_delta_grid(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a saved δ NPZ from a prior catalog-to-grid run."""
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    delta = np.asarray(data["delta"], dtype=float)
    meta = {
        "box_size_mpc": float(data.get("box_size_mpc", 2000.0)),
        "n_per_axis": int(data.get("n_per_axis", delta.shape[0])),
        "npz_path": str(path),
    }
    return delta, meta