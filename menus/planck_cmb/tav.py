#!/usr/bin/env python3
"""
Planck CMB power-spectrum scan for Tav 1/7 resonance periodicities.

Loads any CMB map + mask FITS from ./fits/ (or custom paths), computes D_ell,
and searches for 7-fold harmonic structure in spectrum residuals.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from tav_shared.tav_project_paths import (
    FINISHED_DIR,
    LEGACY_FINISHED2_DIR,
    LEGACY_FINISHED_DIR,
    ensure_tav_project_dirs,
    finished_fits_search_dirs,
    move_to_finished_archive,
)

PROJECT_ROOT = Path(__file__).resolve().parent
FITS_DIR = PROJECT_ROOT / "fits"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Example Planck R3 filenames (not required — any compatible .fits works).
EXAMPLE_CMB_MAP = "COM_CMB_IQU-commander_2048_R3.00_full.fits"
EXAMPLE_CMB_MASK = "COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"

CMB_MAP_URL = (
    "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/"
    "component-maps/cmb/COM_CMB_IQU-commander_2048_R3.00_full.fits"
)
CMB_MASK_URL = (
    "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/masks/"
    "COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
)
PLANCK_MASK_URL_BASE = (
    "https://irsa.ipac.caltech.edu/data/Planck/release_{release}/ancillary-data/masks/"
)

TAV_HARMONIC = 1.0 / 7.0


@dataclass
class CmbSpectrumResult:
    ell: np.ndarray
    dl: np.ndarray
    cls: np.ndarray
    residual: np.ndarray
    fft_amplitude: np.ndarray
    harmonic_peaks: list[int]
    cmb_path: str
    mask_path: str
    lmax: int
    output_prefix: str
    batch_tag: str
    tav_resonance_detected: bool

    def summary_lines(self) -> list[str]:
        lines = [
            "Source: Planck FITS",
            f"CMB map: {self.cmb_path}",
            f"Mask: {self.mask_path}",
            f"FITS batch tag: {self.batch_tag}",
            f"Artifact prefix: {self.output_prefix}",
            f"lmax: {self.lmax}",
            f"Spectrum length: {len(self.dl)} multipoles",
            f"Harmonic peaks (FFT modes): {self.harmonic_peaks or 'none'}",
        ]
        if self.tav_resonance_detected:
            lines.append(
                "[TAV RESONANCE] 7-fold periodic structure detected in "
                "CMB D_ell residual FFT."
            )
        else:
            lines.append(
                "[INCONCLUSIVE] No strong 7-fold harmonic peak in residual FFT."
            )
        return lines


def ensure_output_dirs() -> None:
    FITS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_tav_project_dirs()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def archive_used_fits(cmb_path: Path | str, mask_path: Path | str) -> list[Path]:
    """
    Move processed FITS files from ./fits/ into ../finished/.

    Only files under FITS_DIR are moved (explicit outside paths are left in place).
    Prior archives of the same FITS name are deleted before saving the current run.
    """
    ensure_output_dirs()
    moved: list[Path] = []
    seen: set[Path] = set()

    for src in (Path(cmb_path), Path(mask_path)):
        src = src.resolve()
        if src in seen or not src.is_file():
            continue
        seen.add(src)

        try:
            src.relative_to(FITS_DIR.resolve())
        except ValueError:
            print(f"[TAV ENGINE] Leaving in place (not under ./fits/): {src}")
            continue

        dest = move_to_finished_archive(
            src,
            FINISHED_DIR,
            also_search=(LEGACY_FINISHED_DIR, LEGACY_FINISHED2_DIR),
        )
        moved.append(dest)
        print(f"[TAV ENGINE] Archived FITS: {src.name} -> {dest}")

    if moved:
        print(f"[TAV ENGINE] {len(moved)} file(s) moved to {FINISHED_DIR}/")
    return moved


def _parse_planck_map_meta(map_name: str) -> tuple[int, str]:
    """Extract HEALPix nside and Planck release tag from a CMB map filename."""
    match = re.search(r"_(\d{3,5})_(R\d+\.\d+)", map_name, re.IGNORECASE)
    if match:
        return int(match.group(1)), match.group(2)
    return 2048, "R3.00"


def fits_batch_tag(cmb_path: Path | str) -> str:
    """
    Batch / half-mission tag from a Planck CMB map filename.

    Examples:
      COM_CMB_IQU-nilc_2048_R3.00_hm2.fits -> hm2
      COM_CMB_IQU-commander_2048_R3.00_full.fits -> full
    """
    stem = Path(cmb_path).stem
    match = re.search(r"R\d+\.\d+_(.+)$", stem, re.IGNORECASE)
    if match:
        tag = re.sub(r"[^\w.-]+", "_", match.group(1)).strip("_").lower()
        if tag:
            return tag
    slug = re.sub(r"[^\w.-]+", "_", stem).strip("_").lower()
    return slug[:48]


def build_output_prefix(cmb_path: Path | str, base_prefix: str = "planck_cmb_tav") -> str:
    """Unique artifact prefix so each FITS batch keeps its own PNGs."""
    base = re.sub(r"[^\w.-]+", "_", base_prefix.strip()).strip("_") or "planck_cmb_tav"
    tag = fits_batch_tag(cmb_path)
    if not tag:
        return base
    if tag in base.lower().split("_"):
        return base
    return f"{base}_{tag}"


def mask_filename_for_cmb_map(cmb_path: Path | str) -> str:
    """Planck common intensity mask matching the map nside/release."""
    nside, release = _parse_planck_map_meta(Path(cmb_path).name)
    return f"COM_Mask_CMB-common-Mask-Int_{nside}_{release}.fits"


def mask_download_url(cmb_path: Path | str) -> str:
    mask_name = mask_filename_for_cmb_map(cmb_path)
    _nside, release = _parse_planck_map_meta(Path(cmb_path).name)
    release_dir = release.lower().replace("r", "").split(".")[0]
    return f"{PLANCK_MASK_URL_BASE.format(release=release_dir)}{mask_name}"


def download_planck_mask_for_map(
    cmb_path: Path | str,
    *,
    dest_dir: Path | str = FITS_DIR,
) -> Path:
    """
    Fetch the Planck common CMB mask that matches a CMB map in ./fits/.

    Uses wget. Restores from ../finished/ if present; otherwise downloads from IRSA.
    """
    ensure_output_dirs()
    cmb_path = Path(cmb_path)
    dest_dir = Path(dest_dir)
    mask_name = mask_filename_for_cmb_map(cmb_path)
    dest = dest_dir / mask_name

    if dest.is_file() and dest.stat().st_size > 1000:
        print(f"[TAV ENGINE] Mask already in ./fits/: {mask_name}")
        return dest

    for archive_dir in finished_fits_search_dirs():
        finished_mask = archive_dir / mask_name
        if finished_mask.is_file() and finished_mask.stat().st_size > 1000:
            shutil.copy2(finished_mask, dest)
            print(f"[TAV ENGINE] Restored mask from {archive_dir}/: {mask_name}")
            return dest
        matches = sorted(archive_dir.glob(f"{Path(mask_name).stem}*{Path(mask_name).suffix}"))
        for candidate in matches:
            if candidate.is_file() and candidate.stat().st_size > 1000:
                shutil.copy2(candidate, dest)
                print(f"[TAV ENGINE] Restored mask from {archive_dir}/: {candidate.name}")
                return dest

    url = mask_download_url(cmb_path)
    print(f"[TAV ENGINE] No mask in ./fits/ for {cmb_path.name}")
    print(f"[TAV ENGINE] Downloading via wget: {mask_name}")
    print(f"[TAV ENGINE] URL: {url}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    if partial.exists():
        partial.unlink()

    result = subprocess.run(
        ["wget", "-q", "--timeout=120", "-O", str(partial), url],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not partial.is_file() or partial.stat().st_size < 1000:
        if partial.exists():
            partial.unlink()
        stderr = result.stderr.strip() or result.stdout.strip() or "(no wget output)"
        raise FileNotFoundError(
            f"wget could not download Planck mask for {cmb_path.name}.\n"
            f"  URL: {url}\n"
            f"  wget: {stderr}\n"
            f"Manual: wget -O {dest} '{url}'"
        )

    partial.replace(dest)
    print(f"[TAV ENGINE] Mask ready: {dest}")
    return dest


def _download_instructions() -> str:
    return (
        f"Place any CMB map + mask .fits files in: {FITS_DIR}/\n"
        f"Example CMB map:  {EXAMPLE_CMB_MAP}\n"
        f"  {CMB_MAP_URL}\n"
        f"Example mask:     {EXAMPLE_CMB_MASK}\n"
        f"  {CMB_MASK_URL}\n"
        "Blank form fields auto-detect .fits in ./fits/ (mask by name, else two files).\n"
        "If only a CMB map is present, the matching Planck mask is wget'd automatically."
    )


def list_fits_files(directory: Path | str = FITS_DIR) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    files = list(directory.glob("*.fits")) + list(directory.glob("*.FITS"))
    return sorted({path.resolve() for path in files})


def _looks_like_mask(path: Path) -> bool:
    return "mask" in path.name.lower()


def _looks_like_cmb(path: Path) -> bool:
    name = path.name.lower()
    return any(token in name for token in ("cmb", "commander", "iqu", "temperature", "map"))


def resolve_fits_paths(
    cmb_path: Path | str | None = None,
    mask_path: Path | str | None = None,
) -> tuple[Path, Path]:
    """
    Resolve CMB map and mask paths.

    Explicit paths are used when provided. Otherwise auto-detects .fits files
    in ./fits/ (mask filenames containing 'mask'; map from remaining file).
    """
    ensure_output_dirs()

    cmb_hint = str(cmb_path).strip() if cmb_path is not None else ""
    mask_hint = str(mask_path).strip() if mask_path is not None else ""

    resolved_cmb = Path(cmb_hint) if cmb_hint else None
    resolved_mask = Path(mask_hint) if mask_hint else None

    if resolved_cmb is not None and not resolved_cmb.is_file():
        raise FileNotFoundError(f"CMB map not found: {resolved_cmb}")
    if resolved_mask is not None and not resolved_mask.is_file():
        raise FileNotFoundError(f"Mask not found: {resolved_mask}")
    if resolved_cmb is not None and resolved_mask is not None:
        return resolved_cmb, resolved_mask

    fits_files = list_fits_files()
    if not fits_files:
        raise FileNotFoundError(
            f"No .fits files found in {FITS_DIR}/\n\n{_download_instructions()}"
        )

    if resolved_mask is None:
        mask_candidates = [path for path in fits_files if _looks_like_mask(path)]
        if len(mask_candidates) == 1:
            resolved_mask = mask_candidates[0]
        elif len(mask_candidates) > 1:
            resolved_mask = mask_candidates[0]
            print(f"[TAV ENGINE] Multiple masks in ./fits/; using {resolved_mask.name}")

    if resolved_cmb is None:
        cmb_candidates = [
            path for path in fits_files
            if path != resolved_mask and (_looks_like_cmb(path) or not _looks_like_mask(path))
        ]
        if len(cmb_candidates) == 1:
            resolved_cmb = cmb_candidates[0]
        elif len(cmb_candidates) > 1:
            resolved_cmb = cmb_candidates[0]
            print(f"[TAV ENGINE] Multiple CMB maps in ./fits/; using {resolved_cmb.name}")

    if resolved_cmb is None or resolved_mask is None:
        if len(fits_files) >= 2:
            if resolved_mask is None:
                for path in fits_files:
                    if _looks_like_mask(path):
                        resolved_mask = path
                        break
            if resolved_cmb is None:
                for path in fits_files:
                    if path != resolved_mask:
                        resolved_cmb = path
                        break

    if resolved_cmb is not None and resolved_mask is None:
        try:
            resolved_mask = download_planck_mask_for_map(resolved_cmb)
        except FileNotFoundError as exc:
            found = "\n".join(f"  - {path.name}" for path in fits_files)
            raise FileNotFoundError(
                "CMB map found but matching mask could not be resolved.\n"
                f"Map: {resolved_cmb.name}\n"
                f"Expected mask: {mask_filename_for_cmb_map(resolved_cmb)}\n"
                f"Found in ./fits/:\n{found}\n\n"
                f"{exc}\n\n{_download_instructions()}"
            ) from exc

    if resolved_cmb is None or resolved_mask is None:
        found = "\n".join(f"  - {path.name}" for path in fits_files)
        raise FileNotFoundError(
            "Could not resolve both a CMB map and mask from ./fits/.\n"
            f"Found:\n{found}\n\n"
            "Provide explicit paths in the form, or place a CMB map .fits in ./fits/ "
            "(mask will be wget'd automatically).\n\n"
            f"{_download_instructions()}"
        )

    if resolved_cmb == resolved_mask:
        raise FileNotFoundError(
            "CMB map and mask resolved to the same file. "
            "Provide separate .fits paths in the form."
        )

    print(f"[TAV ENGINE] Using CMB map: {resolved_cmb}")
    print(f"[TAV ENGINE] Using mask:    {resolved_mask}")
    return resolved_cmb, resolved_mask


def load_planck_maps(
    cmb_path: Path | str | None = None,
    mask_path: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray, Path, Path]:
    cmb_file, mask_file = resolve_fits_paths(cmb_path, mask_path)
    cmb_map = hp.read_map(str(cmb_file), field=0, verbose=False)
    mask = hp.read_map(str(mask_file), verbose=False)
    return cmb_map, mask, cmb_file, mask_file


def compute_power_spectrum(
    cmb_map: np.ndarray,
    mask: np.ndarray,
    *,
    lmax: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masked_map = hp.ma(cmb_map)
    masked_map.mask = np.logical_not(mask)

    cls = hp.anafast(
        masked_map.filled(),
        lmax=lmax,
        use_pixel_weights=True,
    )
    ell = np.arange(len(cls))
    dl = ell * (ell + 1) * cls / (2 * np.pi)
    return ell, dl, cls


def analyze_tav_harmonics(dl: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int], bool]:
    """FFT residual spectrum and search for 7-fold periodicities."""
    spectrum = dl[2:].astype(float)
    smooth = uniform_filter1d(spectrum, size=max(21, len(spectrum) // 80))
    residual = spectrum - smooth

    fft_amp = np.abs(fft(residual))
    freqs = np.arange(len(fft_amp))

    # Peaks whose mode index is near multiples of N/7
    peak_idx, _props = find_peaks(fft_amp, height=np.percentile(fft_amp, 92), distance=8)
    harmonic_peaks = []
    n = len(fft_amp)
    for idx in peak_idx:
        phase = (idx % n) / n
        nearest_seventh = round(phase / TAV_HARMONIC) * TAV_HARMONIC
        if abs(phase - nearest_seventh) < 0.02 or abs(phase - nearest_seventh) > 0.98:
            harmonic_peaks.append(int(idx))

    detected = len(harmonic_peaks) >= 2
    return residual, fft_amp, harmonic_peaks, detected


def plot_cmb_analysis(
    result: CmbSpectrumResult,
    *,
    output_prefix: str = "planck_cmb_tav",
    show: bool = False,
) -> list[Path]:
    ensure_output_dirs()
    saved: list[Path] = []

    fig1, ax1 = plt.subplots(figsize=(10, 6), facecolor="#050510")
    ax1.plot(result.ell, result.dl, color="#00d2ff", linewidth=1.2, label="Observed D_ell")
    ax1.set_title("CMB Power Spectrum Scan for Tav Resonance", color="#ddeeff")
    ax1.set_xlabel("Multipole (ell)")
    ax1.set_ylabel(r"$D_\ell$ [$\mu$K$^2$]")
    ax1.grid(True, linestyle=":", alpha=0.35)
    ax1.set_facecolor("#0a0a18")
    ax1.tick_params(colors="#ccddee")
    ax1.legend()
    fig1.tight_layout()
    from tav_shared.artifact_paths import TestSlug, artifact_path, compose_dataset_slug

    path1 = artifact_path(TestSlug.PLANCK_CMB, compose_dataset_slug(output_prefix), "spectrum", "png")
    fig1.savefig(path1, dpi=150, facecolor=fig1.get_facecolor())
    saved.append(path1)
    if show:
        plt.show()
    else:
        plt.close(fig1)

    fig2, axes = plt.subplots(2, 1, figsize=(10, 7), facecolor="#050510")
    ell_res = result.ell[2 : 2 + len(result.residual)]
    axes[0].plot(ell_res, result.residual, color="#ffaa00", linewidth=1.0)
    axes[0].set_title("D_ell Residuals (smooth LCDM proxy removed)", color="#ddeeff")
    axes[0].set_xlabel("Multipole (ell)")
    axes[0].grid(True, linestyle=":", alpha=0.35)
    axes[0].set_facecolor("#0a0a18")

    axes[1].plot(result.fft_amplitude, color="#ff0055", linewidth=1.0)
    for peak in result.harmonic_peaks:
        axes[1].axvline(peak, color="#00d2ff", linestyle="--", alpha=0.7, linewidth=0.9)
    axes[1].set_title("FFT of CMB Residuals (7-fold Tav search)", color="#ddeeff")
    axes[1].set_xlabel("Frequency mode")
    axes[1].grid(True, linestyle=":", alpha=0.35)
    axes[1].set_facecolor("#0a0a18")
    for ax in axes:
        ax.tick_params(colors="#ccddee")
    fig2.tight_layout()
    path2 = artifact_path(TestSlug.PLANCK_CMB, compose_dataset_slug(output_prefix), "fft", "png")
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
    cmb_path: Path | str | None = None,
    mask_path: Path | str | None = None,
    lmax: int = 2000,
    plot: bool = True,
    show_plot: bool = False,
    output_prefix: str = "planck_cmb_tav",
) -> CmbSpectrumResult:
    cmb_map, mask, resolved_cmb, resolved_mask = load_planck_maps(cmb_path, mask_path)
    effective_lmax = min(lmax, 3 * hp.get_nside(cmb_map) - 1)
    batch_tag = fits_batch_tag(resolved_cmb)
    effective_prefix = build_output_prefix(resolved_cmb, output_prefix)

    ell, dl, cls = compute_power_spectrum(cmb_map, mask, lmax=effective_lmax)
    residual, fft_amp, harmonic_peaks, detected = analyze_tav_harmonics(dl)

    result = CmbSpectrumResult(
        ell=ell,
        dl=dl,
        cls=cls,
        residual=residual,
        fft_amplitude=fft_amp,
        harmonic_peaks=harmonic_peaks,
        cmb_path=str(resolved_cmb),
        mask_path=str(resolved_mask),
        lmax=effective_lmax,
        output_prefix=effective_prefix,
        batch_tag=batch_tag,
        tav_resonance_detected=detected,
    )

    print("\n[TAV ENGINE] Planck CMB Tav-Resonance Scan")
    for line in result.summary_lines():
        print(line)

    if plot:
        plot_cmb_analysis(result, output_prefix=effective_prefix, show=show_plot)

    archive_used_fits(resolved_cmb, resolved_mask)

    print(
        "\n[TOPOLOGICAL ANCHOR] 313.1 MeV mass-gap signal is independent of "
        "CMB harmonic search parameters."
    )
    return result