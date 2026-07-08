#!/usr/bin/env python3
"""
FRB × cosmic-web Tav scan.

Cross-matches CHIME-style FRB catalogs with SDSS void catalogs, classifies
burst sightlines (void_center / filament / boundary_cross), analyzes DM
residuals against a standard IGM model, and searches for 1/7 Tav harmonics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.spatial import cKDTree
from scipy.signal import find_peaks, periodogram

from menus.astronomical.frb.fetcher import DATASETS_DIR, archive_used_datasets, resolve_dataset_paths

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TAV_HARMONIC = 1.0 / 7.0
IGM_DM_PER_Z = 855.0  # pc/cm^3 per unit z (Macquart-style mean relation)
PATH_TYPES = ("void_center", "filament", "boundary_cross")


@dataclass
class FrbWebResult:
    frame: pd.DataFrame
    group_stats: pd.DataFrame
    freqs: np.ndarray
    power: np.ndarray
    harmonic_peaks: list[int]
    frb_path: str
    void_path: str
    output_prefix: str
    batch_tag: str
    tav_resonance_detected: bool

    def summary_lines(self) -> list[str]:
        lines = [
            "Source: CHIME FRB catalog × SDSS void catalog",
            f"FRB catalog: {self.frb_path}",
            f"Void catalog: {self.void_path}",
            f"Dataset batch tag: {self.batch_tag}",
            f"Artifact prefix: {self.output_prefix}",
            f"Bursts analyzed: {len(self.frame)}",
            f"Path mix: {self.frame['path_type'].value_counts().to_dict()}",
            f"Harmonic peaks (periodogram modes): {self.harmonic_peaks or 'none'}",
        ]
        if self.tav_resonance_detected:
            lines.append(
                "[TAV RESONANCE] 7-fold periodic structure detected in "
                "DM residual periodogram."
            )
        else:
            lines.append(
                "[INCONCLUSIVE] No strong 7-fold harmonic peak in DM residuals."
            )
        return lines


def ensure_output_dirs() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def dataset_batch_tag(frb_path: Path | str) -> str:
    stem = Path(frb_path).stem
    match = re.search(r"(catalog\d+|chime|frb)", stem, re.IGNORECASE)
    if match:
        return re.sub(r"[^\w.-]+", "_", match.group(0)).strip("_").lower()
    slug = re.sub(r"[^\w.-]+", "_", stem).strip("_").lower()
    return slug[:48] or "frb"


def build_output_prefix(frb_path: Path | str, base_prefix: str = "frb_cosmic_web_tav") -> str:
    base = re.sub(r"[^\w.-]+", "_", base_prefix.strip()).strip("_") or "frb_cosmic_web_tav"
    tag = dataset_batch_tag(frb_path)
    if tag in base.lower().split("_"):
        return base
    return f"{base}_{tag}"


def load_frb_catalog(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    lower = {col: col.lower() for col in frame.columns}
    frame = frame.rename(columns=lower)

    if "ra" not in frame.columns or "dec" not in frame.columns:
        raise ValueError(f"{path} must contain ra and dec columns.")

    dm_col = None
    for candidate in ("dm_exc_ne2001", "bonsai_dm", "dm_fitb", "dm"):
        if candidate in frame.columns:
            dm_col = candidate
            break
    if dm_col is None:
        raise ValueError(f"{path} must contain a DM column.")

    frame = frame.copy()
    frame["DM"] = pd.to_numeric(frame[dm_col], errors="coerce")
    frame["ra"] = pd.to_numeric(frame["ra"], errors="coerce")
    frame["dec"] = pd.to_numeric(frame["dec"], errors="coerce")

    if "excluded_flag" in frame.columns:
        flag = pd.to_numeric(frame["excluded_flag"], errors="coerce").fillna(0)
        frame = frame[flag == 0]

    frame = frame.dropna(subset=["ra", "dec", "DM"])
    frame = frame[(frame["DM"] > 0) & (frame["DM"] < 5000)]
    frame = frame[(frame["dec"] >= -90.0) & (frame["dec"] <= 90.0)]
    frame["ra"] = frame["ra"] % 360.0
    return frame.reset_index(drop=True)


def load_void_catalog(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    lower = {col: col.lower() for col in frame.columns}
    frame = frame.rename(columns=lower)

    ra_col = "ra" if "ra" in frame.columns else "radeg"
    dec_col = "dec" if "dec" in frame.columns else "dedeg"
    if ra_col not in frame.columns or dec_col not in frame.columns:
        raise ValueError(f"{path} must contain RA/Dec columns.")

    radius_col = None
    for candidate in ("reff_mpc", "reff", "radius_mpc", "radius"):
        if candidate in frame.columns:
            radius_col = candidate
            break
    if radius_col is None:
        frame["reff_mpc"] = 15.0
        radius_col = "reff_mpc"

    z_col = "z_void" if "z_void" in frame.columns else "z"
    if z_col not in frame.columns:
        frame["z_void"] = 0.05

    frame = frame.copy()
    frame["ra"] = pd.to_numeric(frame[ra_col], errors="coerce")
    frame["dec"] = pd.to_numeric(frame[dec_col], errors="coerce")
    frame["reff_mpc"] = pd.to_numeric(frame[radius_col], errors="coerce")
    frame["z_void"] = pd.to_numeric(frame[z_col], errors="coerce")
    frame = frame.dropna(subset=["ra", "dec", "reff_mpc"])
    frame = frame[frame["reff_mpc"] > 0]
    return frame.reset_index(drop=True)


def estimate_redshift(dm_pc_cm3: np.ndarray) -> np.ndarray:
    """Approximate redshift from excess DM (MW contribution already removed)."""
    dm = np.asarray(dm_pc_cm3, dtype=float)
    return np.clip((dm - 50.0) / IGM_DM_PER_Z, 0.001, 3.0)


def expected_dm_igm(z: np.ndarray) -> np.ndarray:
    return IGM_DM_PER_Z * np.asarray(z, dtype=float) + 50.0


def _cartesian_unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    return coord.cartesian.xyz.value.T


def _angular_diameter_distance_mpc(z: np.ndarray) -> np.ndarray:
    """Low-z analytic approximation (flat LambdaCDM, H0=70)."""
    z = np.asarray(z, dtype=float)
    h0 = 70.0
    c_km_s = 299792.458
    return (c_km_s / h0) * z / (1.0 + z)


def classify_paths(
    frb_ra: np.ndarray,
    frb_dec: np.ndarray,
    voids: pd.DataFrame,
    *,
    void_center_fraction: float = 0.35,
    boundary_shell_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Classify each FRB sightline relative to the nearest cosmic void.

    Returns path_type labels and angular separation to nearest void (deg).
    """
    void_coords = SkyCoord(
        ra=voids["ra"].to_numpy() * u.deg,
        dec=voids["dec"].to_numpy() * u.deg,
    )
    frb_coords = SkyCoord(ra=frb_ra * u.deg, dec=frb_dec * u.deg)

    void_xyz = _cartesian_unit_vectors(voids["ra"].to_numpy(), voids["dec"].to_numpy())
    frb_xyz = _cartesian_unit_vectors(frb_ra, frb_dec)
    tree = cKDTree(void_xyz)
    dist, idx = tree.query(frb_xyz, k=1)

    sep = frb_coords.separation(void_coords[idx]).deg
    z_void = voids["z_void"].to_numpy()[idx]
    reff_mpc = voids["reff_mpc"].to_numpy()[idx]
    da_mpc = _angular_diameter_distance_mpc(z_void)
    theta_void_deg = np.degrees(np.arctan2(reff_mpc, da_mpc))

    labels = np.full(len(frb_ra), "filament", dtype=object)
    center_mask = sep <= void_center_fraction * theta_void_deg
    boundary_mask = (~center_mask) & (
        sep <= (1.0 + boundary_shell_fraction) * theta_void_deg
    )
    labels[center_mask] = "void_center"
    labels[boundary_mask] = "boundary_cross"
    return labels, sep


def analyze_dm_residuals(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "z" not in result.columns or result["z"].isna().all():
        result["z"] = estimate_redshift(result["DM"].to_numpy())
    else:
        result["z"] = pd.to_numeric(result["z"], errors="coerce")
        missing = result["z"].isna()
        result.loc[missing, "z"] = estimate_redshift(result.loc[missing, "DM"].to_numpy())

    result["expected_dm"] = expected_dm_igm(result["z"].to_numpy())
    result["dm_residual"] = result["DM"] - result["expected_dm"]
    return result


def group_dm_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    stats = (
        frame.groupby("path_type")["dm_residual"]
        .agg(["mean", "std", "count"])
        .reindex(PATH_TYPES)
    )
    return stats


def analyze_tav_harmonics(residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int], bool]:
    clean = residuals[np.isfinite(residuals)]
    if len(clean) < 8:
        return np.array([]), np.array([]), [], False

    freqs, power = periodogram(clean, detrend="linear")
    peak_idx, _props = find_peaks(power, height=np.percentile(power, 90), distance=2)
    harmonic_peaks: list[int] = []
    n = len(power)
    for idx in peak_idx:
        if n <= 1:
            continue
        phase = idx / (n - 1)
        nearest_seventh = round(phase / TAV_HARMONIC) * TAV_HARMONIC
        delta = abs(phase - nearest_seventh)
        delta = min(delta, abs(phase - nearest_seventh - 1.0), abs(phase - nearest_seventh + 1.0))
        if delta < 0.04:
            harmonic_peaks.append(int(idx))

    detected = len(harmonic_peaks) >= 2
    return freqs, power, harmonic_peaks, detected


def run_frb_web_analysis(
    frb_frame: pd.DataFrame,
    void_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[int], bool]:
    classified = classify_paths(
        frb_frame["ra"].to_numpy(),
        frb_frame["dec"].to_numpy(),
        void_frame,
    )
    frb_frame = frb_frame.copy()
    frb_frame["path_type"] = classified[0]
    frb_frame["void_sep_deg"] = classified[1]

    frb_frame = analyze_dm_residuals(frb_frame)
    group_stats = group_dm_statistics(frb_frame)

    print("\n[TAV ENGINE] DM residual statistics by path type:")
    print(group_stats.to_string())

    freqs, power, harmonic_peaks, detected = analyze_tav_harmonics(
        frb_frame["dm_residual"].to_numpy()
    )
    return frb_frame, group_stats, freqs, power, harmonic_peaks, detected


def plot_frb_web_analysis(
    result: FrbWebResult,
    *,
    show: bool = False,
) -> list[Path]:
    ensure_output_dirs()
    prefix = result.output_prefix
    saved: list[Path] = []

    fig1, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="#050510")
    colors = {
        "void_center": "#00d2ff",
        "filament": "#ff0055",
        "boundary_cross": "#ffaa00",
    }

    ax0 = axes[0]
    for path_type, group in result.frame.groupby("path_type"):
        ax0.scatter(
            group["ra"],
            group["dec"],
            s=28,
            alpha=0.85,
            c=colors.get(path_type, "#cccccc"),
            label=path_type,
        )
    ax0.set_xlabel("RA (deg)")
    ax0.set_ylabel("Dec (deg)")
    ax0.set_title("FRB Sightlines by Cosmic-Web Path", color="#ddeeff")
    ax0.legend(fontsize=8)
    ax0.grid(True, linestyle=":", alpha=0.35)
    ax0.set_facecolor("#0a0a18")

    ax1 = axes[1]
    stats = result.group_stats.dropna(how="all")
    x = np.arange(len(stats))
    ax1.bar(x - 0.15, stats["mean"], width=0.3, color="#00d2ff", label="mean")
    ax1.bar(x + 0.15, stats["std"], width=0.3, color="#ff0055", label="std")
    ax1.set_xticks(x)
    ax1.set_xticklabels(stats.index, rotation=15)
    ax1.set_ylabel("DM residual (pc/cm³)")
    ax1.set_title("DM Residuals by Path Type", color="#ddeeff")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.35)
    ax1.set_facecolor("#0a0a18")

    for ax in axes:
        ax.tick_params(colors="#ccddee")

    fig1.tight_layout()
    path1 = ARTIFACTS_DIR / f"{prefix}_paths.png"
    fig1.savefig(path1, dpi=150, facecolor=fig1.get_facecolor())
    saved.append(path1)
    if show:
        plt.show()
    else:
        plt.close(fig1)

    if len(result.freqs):
        fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor="#050510")
        ax2.plot(result.freqs, result.power, color="#ffaa00", linewidth=1.2)
        for peak in result.harmonic_peaks:
            ax2.axvline(
                result.freqs[peak],
                color="#00d2ff",
                linestyle="--",
                alpha=0.75,
                linewidth=0.9,
            )
        ax2.set_title("DM Residual Periodogram (1/7 Tav search)", color="#ddeeff")
        ax2.set_xlabel("Frequency")
        ax2.set_ylabel("Power")
        ax2.grid(True, linestyle=":", alpha=0.35)
        ax2.set_facecolor("#0a0a18")
        ax2.tick_params(colors="#ccddee")
        fig2.tight_layout()
        path2 = ARTIFACTS_DIR / f"{prefix}_periodogram.png"
        fig2.savefig(path2, dpi=150, facecolor=fig2.get_facecolor())
        saved.append(path2)
        if show:
            plt.show()
        else:
            plt.close(fig2)

    for path in saved:
        print(f"[TAV ENGINE] Plot saved: {path}")
    return saved


def run_pipeline(
    *,
    frb_path: Path | str | None = None,
    void_path: Path | str | None = None,
    plot: bool = True,
    show_plot: bool = False,
    output_prefix: str = "frb_cosmic_web_tav",
    archive: bool = True,
    force_refresh: bool = False,
    restore_archived: bool = True,
    max_frbs: int | None = None,
) -> FrbWebResult:
    ensure_output_dirs()
    resolved_frb, resolved_void = resolve_dataset_paths(
        frb_path,
        void_path,
        force_refresh=force_refresh,
        restore_archived=restore_archived,
    )

    frb_frame = load_frb_catalog(resolved_frb)
    void_frame = load_void_catalog(resolved_void)

    if max_frbs is not None and len(frb_frame) > int(max_frbs):
        from tav_shared.n_selector_registry import subsample_rows

        n_use = max(3, int(max_frbs))
        print(f"[TAV ENGINE] Subsampling FRB catalog: {len(frb_frame)} → {n_use}")
        frb_frame = subsample_rows(frb_frame, n_use)

    (
        analyzed,
        group_stats,
        freqs,
        power,
        harmonic_peaks,
        detected,
    ) = run_frb_web_analysis(frb_frame, void_frame)

    batch_tag = dataset_batch_tag(resolved_frb)
    effective_prefix = build_output_prefix(resolved_frb, output_prefix)

    result = FrbWebResult(
        frame=analyzed,
        group_stats=group_stats,
        freqs=freqs,
        power=power,
        harmonic_peaks=harmonic_peaks,
        frb_path=str(resolved_frb),
        void_path=str(resolved_void),
        output_prefix=effective_prefix,
        batch_tag=batch_tag,
        tav_resonance_detected=detected,
    )

    print("\n[TAV ENGINE] FRB Cosmic-Web Tav Scan")
    for line in result.summary_lines():
        print(line)

    if plot:
        plot_frb_web_analysis(result, show=show_plot)

    if archive:
        archive_used_datasets([resolved_frb, resolved_void])
        from tav_shared.dataset_ledger import mark_processed

        mark_processed(
            [
                f"frb:{resolved_frb.stem}",
                f"frb:{resolved_void.stem}",
            ],
            note="frb pipeline archive",
        )

    print(
        "\n[TOPOLOGICAL ANCHOR] 313.1 MeV mass-gap signal is independent of "
        "FRB DM harmonic search parameters."
    )
    return result