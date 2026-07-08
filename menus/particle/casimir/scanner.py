#!/usr/bin/env python3
"""
TSB Tau Resonance Signatures Search in Casimir Data
====================================================

Searches experimental/simulation Casimir datasets for features consistent with
Tau Cylinder / Topological Exclusion predictions:

- 7-phase / 1/7 harmonic periodicity or steps
- Discrete thresholds / hysteresis (magnetic tuning)
- Log-scale modulations (hierarchical Δn steps)
- Dynamic asymmetry or resonant peaks
- Non-local / synchronized features (if multi-channel data available)

Adapts to common public datasets (Zenodo, GitHub magnetic-fluid data, etc.).
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, periodogram

from tav_shared.batch_ledger import (
    CASIMIR_BATCH_DONE,
    append_done_entries,
    batch_status,
    format_batch_banner,
    read_done_list,
    select_batch_items,
)

from tav_shared.tav_project_paths import TAU_SUPERBLOCK_ROOT as PROJECT_ROOT

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATASETS_DIR = PROJECT_ROOT / "datasets" / "casimir"
DEFAULT_DONE_FILE = CASIMIR_BATCH_DONE

TAU_RESONANCE_PERIOD: float = 7.0
ONE_SEVENTH: float = 1.0 / 7.0
TAU_CYCLE_142857: float = 142857.0  # 1/7 repeating-decimal resonance anchor

GITHUB_MF_API = (
    "https://api.github.com/repos/malong201408/"
    "Dataset-of-Prediction-of-a-measurable-sign-change-in-the-Casimir-force-using-a-magnetic-fluid/contents"
)
GITHUB_MF_RAW = (
    "https://raw.githubusercontent.com/malong201408/"
    "Dataset-of-Prediction-of-a-measurable-sign-change-in-the-Casimir-force-using-a-magnetic-fluid/main"
)

EXAMPLE_DATASETS: dict[str, dict[str, str]] = {
    "zenodo_superconducting_casimir": {
        "url": "https://zenodo.org/records/14700381/files/Casimir%20drums%20data.zip?download=1",
        "zip_name": "casimir_drums_data.zip",
        "description": "Superconducting Casimir force measurements (~780 MB zip with CSVs/HDF5)",
        "size_warning_mb": "780",
    },
    "github_magnetic_fluid_prediction": {
        "description": "Magnetic-fluid Casimir sign-change dielectric spectra (Fig2/Fig34 on GitHub)",
        "subdirs": "Fig2,Fig34",
    },
}

COLUMN_ALIASES: dict[str, str] = {
    "B": "magnetic_field_mT",
    "B_mT": "magnetic_field_mT",
    "field": "magnetic_field_mT",
    "magnetic_field": "magnetic_field_mT",
    "F": "force_pN",
    "Force": "force_pN",
    "force": "force_pN",
    "d": "separation_nm",
    "distance": "separation_nm",
    "sep": "separation_nm",
    "time": "time_s",
    "t": "time_s",
}


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _ensure_artifacts() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


def _ensure_datasets() -> Path:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    return DATASETS_DIR


def casimir_done_key(path: Path | str, *, data_root: Path | None = None) -> str:
    """Stable ledger key: path relative to datasets/casimir when possible."""
    root = (data_root or DATASETS_DIR).resolve()
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def casimir_batch_status(
    directory: str | Path,
    *,
    done_file: Path | str = DEFAULT_DONE_FILE,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Summarize how many tabular files are scanned, pending, or complete."""
    directory = Path(directory)
    root = data_root or DATASETS_DIR
    all_files = list_casimir_data_files(directory)
    return batch_status(
        all_files,
        done_file,
        key_fn=lambda file_path: casimir_done_key(file_path, data_root=root),
    )


def _try_tav_harmonic_check(residuals: np.ndarray) -> dict[str, Any] | None:
    try:
        from tav_resonance import analyze_tav_harmonics
    except ImportError:
        return None
    freqs, power, peaks, detected = analyze_tav_harmonics(
        np.asarray(residuals, dtype=float),
        phase_tolerance=0.05,
        min_peaks=1,
    )
    return {
        "harmonic_peaks": peaks,
        "tav_resonance_detected": detected,
        "n_freq_bins": int(freqs.size),
    }


def download_file(url: str, save_path: str | Path, *, chunk_size: int = 8192) -> Path:
    """Download a file with progress."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from {url} ...")
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    downloaded = 0
    with open(save_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            handle.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\rProgress: {percent:.1f}%", end="", flush=True)
    print(f"\nSaved to {save_path}")
    return save_path


def extract_zip(zip_path: str | Path, extract_to: str | Path) -> list[str]:
    """Extract ZIP and return list of tabular data files (CSV, TSV, TXT, DAT)."""
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    tabular_ext = {".csv", ".tsv", ".txt", ".dat", ".lst"}
    data_files: list[str] = []
    for root, _, files in os.walk(extract_to):
        for name in files:
            if Path(name).suffix.lower() in tabular_ext:
                data_files.append(os.path.join(root, name))
    print(f"Found {len(data_files)} tabular files.")
    return data_files


def _read_tabular_file(file_path: Path) -> pd.DataFrame:
    """Load CSV/TSV/whitespace tables; magnetic-fluid GitHub files are tab-separated."""
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".tsv", ".txt", ".dat", ".lst"}:
        try:
            return pd.read_csv(file_path, sep=r"\s+", engine="python")
        except pd.errors.ParserError:
            return pd.read_csv(file_path, sep="\t")
    # Extensionless numeric tables (GitHub Fig2/Fig34)
    try:
        return pd.read_csv(file_path, sep="\t", header=None)
    except pd.errors.ParserError:
        return pd.read_csv(file_path, sep=r"\s+", engine="python", header=None)


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Load tabular Casimir data into DataFrame and standardize common column names."""
    file_path = Path(file_path)
    df = _read_tabular_file(file_path)

    if df.shape[1] >= 2 and all(str(c).isdigit() for c in df.columns):
        rename: dict[Any, str] = {df.columns[0]: "matsubara_frequency"}
        if df.shape[1] == 2:
            rename[df.columns[1]] = "dielectric_response"
        else:
            rename[df.columns[1]] = "dielectric_response_primary"
        df = df.rename(columns=rename)

    print(f"Loaded {file_path} with shape {df.shape} and columns: {list(df.columns)}")
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})
    return df


def download_github_magnetic_fluid(
    *,
    subdirs: tuple[str, ...] = ("Fig2", "Fig34"),
    force: bool = False,
) -> list[Path]:
    """Download magnetic-fluid Casimir prediction spectra from GitHub."""
    out_root = _ensure_datasets() / "github_magnetic_fluid"
    out_root.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for subdir in subdirs:
        api_url = f"{GITHUB_MF_API}/{subdir}?ref=main"
        print(f"Listing GitHub {subdir} ...")
        response = requests.get(api_url, timeout=60)
        response.raise_for_status()
        entries = response.json()
        if not isinstance(entries, list):
            continue

        target_dir = out_root / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        skip_tokens = ("readme", "mathematica", "license", ".md", ".m", ".nb", ".pdf")
        for entry in entries:
            if entry.get("type") != "file":
                continue
            name = entry.get("name", "")
            lower = name.lower()
            if any(token in lower for token in skip_tokens):
                continue
            dest = target_dir / name
            if dest.is_file() and not force:
                saved.append(dest)
                continue
            raw_url = f"{GITHUB_MF_RAW}/{subdir}/{name}"
            download_file(raw_url, dest)
            saved.append(dest)

    print(f"GitHub magnetic-fluid cache: {len(saved)} files under {out_root}")
    return saved


def list_casimir_data_files(directory: str | Path) -> list[Path]:
    """Return sorted tabular files under a directory tree."""
    directory = Path(directory)
    tabular_ext = {".csv", ".tsv", ".txt", ".dat", ".lst", ""}
    files = [
        p
        for p in directory.rglob("*")
        if p.is_file() and (p.suffix.lower() in tabular_ext or p.suffix == "")
    ]
    return sorted(files)


def _seven_phase_positions(x: np.ndarray) -> np.ndarray:
    """Expected 7-phase clockwork marker positions (k/7 of parameter span)."""
    if len(x) < 2:
        return np.array([])
    x_min, x_max = float(np.min(x)), float(np.max(x))
    span = x_max - x_min
    if span <= 0:
        return np.array([])
    return np.array([x_min + span * k / TAU_RESONANCE_PERIOD for k in range(1, int(TAU_RESONANCE_PERIOD))])


def _normalized_autocorrelation(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return lags and normalized autocorrelation of a 1D series."""
    y0 = np.asarray(y, dtype=float)
    y0 = y0 - np.mean(y0)
    if len(y0) < 4 or np.std(y0) < 1e-12:
        return np.arange(len(y0)), np.zeros(len(y0))
    ac = np.correlate(y0, y0, mode="full")
    ac = ac[ac.size // 2 :]
    if ac[0] > 0:
        ac = ac / ac[0]
    lags = np.arange(ac.size)
    return lags, ac


def _detect_changepoints(y: np.ndarray, x: np.ndarray) -> dict[str, Any]:
    """Change-point detection on sorted sweep (ruptures with derivative fallback)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(y) < 8:
        return {"method": "none", "locations": [], "n_breaks": 0}

    try:
        import ruptures as rpt

        signal_2d = y.reshape(-1, 1)
        algo = rpt.Pelt(model="l2", min_size=3).fit(signal_2d)
        bkps = [b for b in algo.predict(pen=3.0) if b < len(y)]
        locations = [float(x[max(0, idx - 1)]) for idx in bkps if idx > 0]
        return {
            "method": "ruptures_pelt",
            "locations": locations[:10],
            "n_breaks": len(locations),
            "note": "Hierarchical level/slope transitions (ruptures PELT)",
        }
    except ImportError:
        dy = np.diff(y)
        threshold = np.std(dy) * 2.5
        idx = np.where(np.abs(dy) > threshold)[0]
        locations = [float(x[i + 1]) for i in idx[:10]]
        return {
            "method": "derivative_fallback",
            "locations": locations,
            "n_breaks": len(idx),
            "note": "Derivative-threshold change points (ruptures not installed)",
        }


def _auto_select_columns(df: pd.DataFrame) -> tuple[str, str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found in dataset")

    param_col = next(
        (
            c
            for c in numeric_cols
            if any(
                token in c.lower()
                for token in (
                    "magnetic",
                    "matsubara",
                    "frequency",
                    "field_mt",
                    "field",
                    "separation",
                    "distance",
                    "time_s",
                    "time",
                )
            )
            or c.lower() in {"b", "d", "sep"}
        ),
        numeric_cols[0],
    )
    value_candidates = [c for c in numeric_cols if c != param_col]
    value_col = next(
        (
            c
            for c in value_candidates
            if any(
                token in c.lower()
                for token in ("force", "energy", "pressure", "signal", "dielectric", "response")
            )
        ),
        value_candidates[0] if value_candidates else numeric_cols[0],
    )
    return param_col, value_col


def detect_tau_resonances(
    df: pd.DataFrame,
    param_col: str | None = None,
    value_col: str | None = None,
) -> dict[str, Any]:
    """
    Core search for TSB tau resonance signatures.

    Looks for periodic 1/7 harmonics, discrete steps, hysteresis, and log-scale
    modulations in separation sweeps.
    """
    results: dict[str, Any] = {
        "detected_features": [],
        "param_col": None,
        "value_col": None,
        "diagnostics": {},
    }

    if param_col is None or value_col is None:
        param_col, value_col = _auto_select_columns(df)
        print(f"Auto-selected param={param_col}, value={value_col}")

    results["param_col"] = param_col
    results["value_col"] = value_col

    x = df[param_col].values.astype(float)
    y = df[value_col].values.astype(float)

    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean, y_clean = x[mask], y[mask]
    param_span = float(np.max(x_clean) - np.min(x_clean)) if len(x_clean) else 0.0
    seven_phase_x = _seven_phase_positions(x_clean)
    results["diagnostics"]["seven_phase_positions"] = seven_phase_x.tolist()

    # 1. Fourier / periodogram — 1/7 harmonics and clockwork
    if len(x_clean) > 10:
        dx = np.mean(np.diff(x_clean)) if len(x_clean) > 1 else 1.0
        fs = 1.0 / dx if dx > 0 else 1.0
        freqs, power = periodogram(y_clean, fs=fs)
        peaks, peak_props = find_peaks(power, height=np.max(power) * 0.1, distance=5)
        results["diagnostics"]["freqs"] = freqs.tolist()
        results["diagnostics"]["power"] = power.tolist()

        if len(peaks) > 0:
            dominant_freqs = freqs[peaks]
            positive = dominant_freqs[dominant_freqs > 0]
            periods = 1.0 / positive if positive.size else np.array([])
            expected_freq = 1.0 / (param_span / TAU_RESONANCE_PERIOD) if param_span > 0 else ONE_SEVENTH
            nearest_idx = int(np.argmin(np.abs(freqs - expected_freq))) if freqs.size else 0
            harmonic_match = any(abs(f - expected_freq) / max(expected_freq, 1e-9) < 0.15 for f in positive)
            results["detected_features"].append(
                {
                    "type": "seven_phase_clockwork",
                    "periods": periods[:5].tolist(),
                    "dominant_freqs": dominant_freqs[:5].tolist(),
                    "expected_freq_1_over_7": float(expected_freq),
                    "power_at_one_seventh_proxy": float(power[nearest_idx]),
                    "harmonic_match": bool(harmonic_match),
                    "note": "7-phase / 1/7 harmonic periodicity in parameter sweep",
                }
            )

    # 2. 142857-cycle periodicity (repeating 1/7 resonance anchor)
    if len(y_clean) > 14:
        lags, ac = _normalized_autocorrelation(y_clean)
        results["diagnostics"]["autocorr_lags"] = lags[: min(80, len(lags))].tolist()
        results["diagnostics"]["autocorr"] = ac[: min(80, len(ac))].tolist()
        n = len(y_clean)
        seventh_lags = [int(round(n * k / TAU_RESONANCE_PERIOD)) for k in range(1, int(TAU_RESONANCE_PERIOD))]
        seventh_lags = [lag for lag in seventh_lags if 0 < lag < len(ac)]
        ac_at_seventh = [float(ac[lag]) for lag in seventh_lags]
        cycle_score = float(np.mean(ac_at_seventh)) if ac_at_seventh else 0.0
        residue_pattern = [int(round(val * 1e6)) % int(TAU_CYCLE_142857) for val in y_clean[:7]]
        if cycle_score > 0.25 or (ac_at_seventh and max(ac_at_seventh) > 0.4):
            results["detected_features"].append(
                {
                    "type": "tau_cycle_142857",
                    "autocorr_at_seventh_lags": ac_at_seventh,
                    "cycle_score": cycle_score,
                    "residue_pattern_head": residue_pattern,
                    "note": "142857-cycle / 1/7 repeating-decimal resonance structure",
                }
            )

    # 3. Discrete steps / thresholds
    if len(y_clean) > 5:
        dy = np.diff(y_clean)
        step_threshold = np.std(dy) * 2
        step_indices = np.where(np.abs(dy) > step_threshold)[0]
        if len(step_indices) > 0:
            step_locations = x_clean[step_indices]
            aligned = 0
            if param_span > 0 and len(seven_phase_x):
                tol = param_span / 14.0
                aligned = sum(
                    1
                    for loc in step_locations
                    if any(abs(loc - mark) < tol for mark in seven_phase_x)
                )
            results["detected_features"].append(
                {
                    "type": "discrete_steps_thresholds",
                    "locations": step_locations[:10].tolist(),
                    "n_steps": int(len(step_indices)),
                    "aligned_with_seven_phase": int(aligned),
                    "note": "Discrete thresholds / hysteresis jumps (magnetic tuning)",
                }
            )

    # 4. Hysteresis / dynamic asymmetry
    if "time_s" in df.columns or len(y_clean) > 20:
        mid = len(y_clean) // 2
        if mid > 0:
            diff_up_down = float(np.mean(np.abs(y_clean[:mid] - y_clean[mid:][::-1])))
            if diff_up_down > np.std(y_clean) * 0.5:
                results["detected_features"].append(
                    {
                        "type": "hysteresis_asymmetry",
                        "magnitude": diff_up_down,
                        "note": "Asymmetric up/down sweep — dynamic phase advancement + exclusion",
                    }
                )

    # 5. Log-scale modulation (hierarchical Δn steps in separation data)
    param_lower = param_col.lower()
    if any(token in param_lower for token in ("sep", "distance", "separation")) or param_lower in {"d"}:
        positive_x = x_clean[x_clean > 0]
        positive_y = y_clean[x_clean > 0]
        if len(positive_x) > 10:

            def sin_mod(t: np.ndarray, a: float, f: float, p: float, c: float) -> np.ndarray:
                return a * np.sin(2 * np.pi * f * t + p) + c

            log_x = np.log10(positive_x)
            try:
                popt, _ = curve_fit(
                    sin_mod,
                    log_x,
                    positive_y[: len(log_x)],
                    p0=[np.std(positive_y), ONE_SEVENTH, 0.0, np.mean(positive_y)],
                    maxfev=5000,
                )
                results["detected_features"].append(
                    {
                        "type": "log_scale_modulation",
                        "frequency": float(popt[1]),
                        "amplitude": float(popt[0]),
                        "note": f"Log-separation modulation f≈{popt[1]:.3f} (compare to 1/7≈0.143)",
                    }
                )
                results["diagnostics"]["log_fit_params"] = {
                    "amplitude": float(popt[0]),
                    "frequency": float(popt[1]),
                    "phase": float(popt[2]),
                    "offset": float(popt[3]),
                }
            except (RuntimeError, ValueError):
                pass

    # 6. Resonant peaks in observable
    if len(y_clean) > 8:
        y_std = float(np.std(y_clean)) or 1.0
        peak_idx, props = find_peaks(y_clean, prominence=0.25 * y_std, distance=3)
        if len(peak_idx) > 0:
            peak_locs = x_clean[peak_idx]
            peak_heights = y_clean[peak_idx]
            results["detected_features"].append(
                {
                    "type": "resonant_peaks",
                    "locations": peak_locs[:8].tolist(),
                    "heights": peak_heights[:8].tolist(),
                    "n_peaks": int(len(peak_idx)),
                    "note": "Resonant peaks in Casimir observable",
                }
            )
            results["diagnostics"]["resonant_peak_idx"] = peak_idx.tolist()

    # 7. Change-point detection (hierarchical transitions)
    cp = _detect_changepoints(y_clean, x_clean)
    if cp.get("n_breaks", 0) > 0:
        results["detected_features"].append(
            {
                "type": "changepoints",
                "locations": cp.get("locations", []),
                "n_breaks": cp.get("n_breaks", 0),
                "method": cp.get("method"),
                "note": cp.get("note", "Change-point transitions"),
            }
        )
    results["diagnostics"]["changepoints"] = cp

    # 8. tav-resonance cross-check on detrended signal
    if len(y_clean) > 12:
        smooth = pd.Series(y_clean).rolling(window=5, center=True, min_periods=1).mean().values
        residuals = y_clean - smooth
        tav = _try_tav_harmonic_check(residuals)
        if tav is not None:
            results["tav_harmonics"] = tav

    results["signature_summary"] = {
        feat["type"]: True for feat in results["detected_features"]
    }
    return results


def _expected_tsb_modulation(x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """Overlay: sinusoidal 1/7 clockwork modulation scaled to data."""
    if len(x) < 4:
        return None
    span = float(np.max(x) - np.min(x))
    if span <= 0:
        return None
    amp = 0.5 * float(np.std(y))
    offset = float(np.mean(y))
    phase = 2 * np.pi * (x - np.min(x)) / span * TAU_RESONANCE_PERIOD
    return offset + amp * np.sin(phase)


def plot_and_save(
    df: pd.DataFrame,
    param_col: str,
    value_col: str,
    results: dict[str, Any],
    prefix: str = "tau_search",
    *,
    output_dir: Path | None = None,
) -> str:
    """Generate diagnostic plots comparing data to expected TSB tau signatures."""
    out_dir = output_dir or _ensure_artifacts()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    diag = results.get("diagnostics", {})

    mask = df[param_col].notna() & df[value_col].notna()
    x = df.loc[mask, param_col].values.astype(float)
    y = df.loc[mask, value_col].values.astype(float)
    sort_idx = np.argsort(x) if len(x) else np.array([], dtype=int)
    x_s, y_s = x[sort_idx], y[sort_idx]

    # Raw data + 7-phase markers + expected 1/7 modulation
    ax = axes[0, 0]
    ax.plot(x_s, y_s, "b.-", alpha=0.75, label="data")
    for mark in diag.get("seven_phase_positions", []):
        ax.axvline(mark, color="orange", linestyle=":", alpha=0.6, linewidth=0.9)
    overlay = _expected_tsb_modulation(x_s, y_s)
    if overlay is not None:
        ax.plot(x_s, overlay, "g--", alpha=0.7, label="1/7 clockwork template")
    ax.set_xlabel(param_col)
    ax.set_ylabel(value_col)
    ax.set_title("Data vs 7-phase markers")
    ax.legend(fontsize=8)
    ax.grid(True)

    # Steps / change-points on derivative
    ax = axes[0, 1]
    if len(y_s) > 1:
        dy = np.diff(y_s)
        ax.plot(x_s[1:], dy, "r.-", alpha=0.8, label="dy/dparam")
        for feat in results.get("detected_features", []):
            if feat["type"] in {"discrete_steps_thresholds", "changepoints"}:
                for loc in feat.get("locations", [])[:6]:
                    ax.axvline(loc, color="purple", linestyle="--", alpha=0.5)
        ax.set_title("Steps & change-points")
        ax.grid(True)
    else:
        ax.set_axis_off()

    # Resonant peaks
    ax = axes[0, 2]
    ax.plot(x_s, y_s, "b.-", alpha=0.6)
    peak_idx = diag.get("resonant_peak_idx", [])
    if peak_idx:
        ax.plot(x_s[peak_idx], y_s[peak_idx], "ro", label="resonant peaks")
        ax.legend(fontsize=8)
    ax.set_title("Resonant peaks")
    ax.grid(True)

    # Periodogram
    ax = axes[1, 0]
    if len(x_s) > 10:
        freqs = np.asarray(diag.get("freqs") or [])
        power = np.asarray(diag.get("power") or [])
        if freqs.size == 0:
            freqs, power = periodogram(y_s)
        ax.semilogy(freqs, power)
        ax.axvline(ONE_SEVENTH, color="green", linestyle="--", alpha=0.8, label="f=1/7")
        for harmonic in (2 * ONE_SEVENTH, 3 * ONE_SEVENTH):
            ax.axvline(harmonic, color="lime", linestyle=":", alpha=0.5)
        ax.set_title("Periodogram (Fourier)")
        ax.set_xlabel("frequency")
        ax.legend(fontsize=8)
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        ax.set_axis_off()

    # Autocorrelation (142857-cycle lags)
    ax = axes[1, 1]
    ac_lags = np.asarray(diag.get("autocorr_lags") or [])
    ac_vals = np.asarray(diag.get("autocorr") or [])
    if ac_lags.size and ac_vals.size:
        ax.plot(ac_lags, ac_vals, "k-")
        n = len(y_s)
        for k in range(1, int(TAU_RESONANCE_PERIOD)):
            lag = int(round(n * k / TAU_RESONANCE_PERIOD))
            if 0 < lag < len(ac_vals):
                ax.axvline(lag, color="magenta", linestyle=":", alpha=0.5)
        ax.set_title("Autocorrelation (1/7 lag markers)")
        ax.set_xlabel("lag")
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        ax.set_axis_off()

    # Summary panel
    ax = axes[1, 2]
    summary_lines = [f"Param: {param_col} | Value: {value_col}", ""]
    for feat in results.get("detected_features", []):
        summary_lines.append(f"• {feat['type']}")
        if feat.get("note"):
            summary_lines.append(f"  {feat['note']}")
    if results.get("tav_harmonics"):
        th = results["tav_harmonics"]
        summary_lines.append(f"• tav-resonance 1/7: {th.get('tav_resonance_detected', False)}")
    ax.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
        transform=ax.transAxes,
    )
    ax.set_title("Signature summary")
    ax.set_axis_off()

    fig.suptitle("TSB Casimir Tau Resonance Search", fontsize=12)
    fig.tight_layout()
    plot_path = out_dir / f"{prefix}_{_utc_stamp()}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    results.setdefault("plots", []).append(str(plot_path))
    print(f"Plot saved: {plot_path}")
    return str(plot_path)


def generate_mock_casimir_data(n_points: int = 140, seed: int = 42) -> pd.DataFrame:
    """Synthetic magnetic-field Casimir sweep with 1/7 ripple and discrete steps."""
    rng = np.random.default_rng(seed)
    b_field = np.linspace(0.0, 70.0, n_points)
    base = -12.0 + 0.08 * b_field
    ripple = 0.35 * np.sin(2 * np.pi * b_field / TAU_RESONANCE_PERIOD)
    steps = np.zeros_like(b_field)
    for threshold in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]:
        steps += np.where(b_field >= threshold, 0.15, 0.0)
    noise = rng.normal(0.0, 0.05, size=n_points)
    force = base + ripple + steps + noise
    return pd.DataFrame({"magnetic_field_mT": b_field, "force_pN": force})


@dataclass
class CasimirScanResult:
    data_label: str
    file_path: str | None = None
    param_col: str = ""
    value_col: str = ""
    detected_features: list[dict[str, Any]] = field(default_factory=list)
    signature_summary: dict[str, bool] = field(default_factory=dict)
    tav_harmonics: dict[str, Any] | None = None
    plot_path: str | None = None
    report_path: str | None = None
    text_report_path: str | None = None

    def summary_lines(self) -> list[str]:
        lines = [
            "TSB Casimir tau resonance scan",
            f"Dataset: {self.data_label}",
        ]
        if self.file_path:
            lines.append(f"File: {self.file_path}")
        if self.param_col:
            lines.append(f"Columns: {self.param_col} → {self.value_col}")
        lines.append(f"Signatures detected: {len(self.detected_features)}")
        for feat in self.detected_features:
            lines.append(f"  • {feat['type']}: {feat.get('note', '')}")
        if self.signature_summary:
            lines.append(f"Signature flags: {', '.join(sorted(self.signature_summary))}")
        if self.tav_harmonics:
            lines.append(
                f"tav-resonance 1/7 check: detected={self.tav_harmonics.get('tav_resonance_detected')}"
            )
        if self.plot_path:
            lines.append(f"Plot: {self.plot_path}")
        if self.report_path:
            lines.append(f"JSON report: {self.report_path}")
        if self.text_report_path:
            lines.append(f"Text report: {self.text_report_path}")
        return lines


def _write_report(result: CasimirScanResult, output_prefix: str, diagnostics: dict | None = None) -> tuple[str, str]:
    out_dir = _ensure_artifacts()
    stamp = _utc_stamp()
    report_path = out_dir / f"{output_prefix}_report_{stamp}.json"
    text_path = out_dir / f"{output_prefix}_summary_{stamp}.txt"
    payload = {
        "data_label": result.data_label,
        "file_path": result.file_path,
        "param_col": result.param_col,
        "value_col": result.value_col,
        "detected_features": result.detected_features,
        "signature_summary": result.signature_summary,
        "tav_harmonics": result.tav_harmonics,
        "plot_path": result.plot_path,
        "diagnostics": diagnostics or {},
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    text_path.write_text("\n".join(result.summary_lines()) + "\n", encoding="utf-8")
    return str(report_path), str(text_path)


def run_casimir_scan(
    *,
    data_source: str = "mock",
    local_csv: str | Path | None = None,
    param_col: str | None = None,
    value_col: str | None = None,
    dataset_key: str = "zenodo_superconducting_casimir",
    download: bool = False,
    extract_dir: str | Path | None = None,
    plot: bool = True,
    output_prefix: str = "tsb_casimir",
) -> CasimirScanResult:
    """High-level Casimir tau-resonance scan entry point."""
    data_source = (data_source or "mock").strip().lower()
    df: pd.DataFrame
    label: str
    file_path: str | None = None

    if data_source == "mock":
        df = generate_mock_casimir_data()
        label = "mock magnetic-field Casimir sweep"
    elif data_source in {"local", "csv", "file"}:
        if not local_csv:
            raise ValueError("local_csv path required for data_source=local")
        file_path = str(Path(local_csv).resolve())
        df = load_data(file_path)
        label = Path(file_path).name
    elif data_source in {"zenodo", "download"}:
        meta = EXAMPLE_DATASETS.get(dataset_key, EXAMPLE_DATASETS["zenodo_superconducting_casimir"])
        ds_root = _ensure_datasets()
        zip_path = ds_root / meta["zip_name"]
        if download or not zip_path.is_file():
            download_file(meta["url"], zip_path)
        extract_to = Path(extract_dir) if extract_dir else ds_root / "casimir_drums_extracted"
        data_files = extract_zip(zip_path, extract_to)
        if not data_files:
            raise FileNotFoundError(f"No tabular files found under {extract_to}")
        file_path = data_files[0]
        df = load_data(file_path)
        label = f"zenodo extract ({Path(file_path).name})"
    elif data_source in {"github", "github_magnetic_fluid", "magnetic_fluid"}:
        cache = _ensure_datasets() / "github_magnetic_fluid"
        if download or not cache.is_dir() or not any(cache.rglob("*")):
            download_github_magnetic_fluid(force=download)
        candidates = list_casimir_data_files(cache)
        if not candidates:
            raise FileNotFoundError(f"No GitHub magnetic-fluid files under {cache}")
        if local_csv:
            file_path = str(Path(local_csv).resolve())
            if not Path(file_path).is_file():
                raise FileNotFoundError(file_path)
        else:
            file_path = str(candidates[0])
        df = load_data(file_path)
        label = f"github magnetic-fluid ({Path(file_path).name})"
    else:
        raise ValueError(f"Unknown data_source: {data_source}")

    analysis = detect_tau_resonances(df, param_col=param_col, value_col=value_col)
    pcol = analysis["param_col"] or param_col or ""
    vcol = analysis["value_col"] or value_col or ""

    result = CasimirScanResult(
        data_label=label,
        file_path=file_path,
        param_col=pcol,
        value_col=vcol,
        detected_features=analysis.get("detected_features", []),
        signature_summary=analysis.get("signature_summary", {}),
        tav_harmonics=analysis.get("tav_harmonics"),
    )

    if plot and pcol and vcol:
        result.plot_path = plot_and_save(df, pcol, vcol, analysis, prefix=output_prefix)

    json_path, text_path = _write_report(result, output_prefix, diagnostics=analysis.get("diagnostics"))
    result.report_path = json_path
    result.text_report_path = text_path
    return result


def run_batch_casimir_scan(
    directory: str | Path,
    *,
    max_files: int = 10,
    plot: bool = False,
    output_prefix: str = "tsb_casimir_batch",
    done_file: Path | str = DEFAULT_DONE_FILE,
    force_rescan: bool = False,
) -> dict[str, Any]:
    """
    Scan tabular files under a directory, resuming from datasets/casimir/done.txt.

    Each run processes the next pending slice (up to ``max_files``); successful
    scans are appended to done.txt so the following run continues where the last
    one stopped. Set ``max_files`` to 0 to scan all pending files in one run.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    _ensure_datasets()
    all_files = list_casimir_data_files(directory)
    if not all_files:
        raise FileNotFoundError(f"No tabular files under {directory}")

    selected, status = select_batch_items(
        all_files,
        done_file,
        key_fn=lambda file_path: casimir_done_key(file_path),
        limit=max_files,
        force_rescan=force_rescan,
    )
    print(format_batch_banner("TSB Casimir", status))
    if not selected:
        if status["done_count"] >= status["catalog_total"]:
            print(
                f"[TSB Casimir] All {status['catalog_total']} files already scanned. "
                "Use force_rescan=yes to re-run from the start."
            )
        return {
            "summary_path": None,
            "summaries": {},
            "scanned": 0,
            "done_count": status["done_count"],
            "pending_count": 0,
            "catalog_total": status["catalog_total"],
        }

    summaries: dict[str, str] = {}
    completed_keys: list[str] = []
    for csv_path in selected:
        key = casimir_done_key(csv_path)
        try:
            result = run_casimir_scan(
                data_source="local",
                local_csv=csv_path,
                plot=plot,
                output_prefix=f"{output_prefix}_{csv_path.stem}",
            )
            summaries[str(csv_path)] = (
                f"{len(result.detected_features)} features"
                + (
                    f", tav={result.tav_harmonics.get('tav_resonance_detected')}"
                    if result.tav_harmonics
                    else ""
                )
            )
            completed_keys.append(key)
        except Exception as exc:
            summaries[str(csv_path)] = f"error: {exc}"

    if completed_keys:
        append_done_entries(done_file, completed_keys)
        print(f"[TSB Casimir] Recorded {len(completed_keys)} file(s) in {done_file}")

    out_dir = _ensure_artifacts()
    summary_path = out_dir / f"{output_prefix}_summary_{_utc_stamp()}.json"
    payload = {
        "directory": str(directory),
        "done_file": str(done_file),
        "catalog_total": status["catalog_total"],
        "done_count": status["done_count"] + len(completed_keys),
        "pending_before": status["pending_count"],
        "n_files": len(selected),
        "force_rescan": force_rescan,
        "summaries": summaries,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Batch summary: {summary_path}")
    return {
        "summary_path": str(summary_path),
        "summaries": summaries,
        "scanned": len(completed_keys),
        "done_count": status["done_count"] + len(completed_keys),
        "pending_count": max(0, status["pending_count"] - len(completed_keys)),
        "catalog_total": status["catalog_total"],
    }


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TSB Casimir tau resonance signature search")
    parser.add_argument(
        "--data",
        choices=["mock", "local", "zenodo", "github"],
        default="mock",
        help="Data source: mock, local file, Zenodo Casimir drums, or GitHub magnetic-fluid",
    )
    parser.add_argument(
        "--download-github",
        action="store_true",
        help="Fetch GitHub magnetic-fluid Fig2/Fig34 spectra and exit",
    )
    parser.add_argument("--local-csv", help="Path to local CSV when --data local")
    parser.add_argument("--param-col", help="Parameter column (x-axis)")
    parser.add_argument("--value-col", help="Observable column (y-axis)")
    parser.add_argument("--download", action="store_true", help="Force download Zenodo zip")
    parser.add_argument("--extract-dir", help="Extraction directory for Zenodo zip")
    parser.add_argument("--batch-dir", help="Scan all CSVs in directory and exit")
    parser.add_argument("--max-files", type=int, default=10, help="Max pending files for --batch-dir (0=all)")
    parser.add_argument(
        "--done-file",
        default=str(DEFAULT_DONE_FILE),
        help="Ledger of scanned files (default: datasets/casimir/done.txt)",
    )
    parser.add_argument(
        "--force-rescan",
        action="store_true",
        help="Ignore done.txt and scan from the first files again",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip PNG output")
    parser.add_argument("--output-prefix", default="tsb_casimir", help="Artifact filename prefix")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_cli()
    args = parser.parse_args(argv)

    if args.download_github:
        paths = download_github_magnetic_fluid(force=args.download)
        print(f"Downloaded {len(paths)} files.")
        return 0

    if args.batch_dir:
        run_batch_casimir_scan(
            args.batch_dir,
            max_files=args.max_files,
            plot=not args.no_plot,
            output_prefix=args.output_prefix,
            done_file=args.done_file,
            force_rescan=args.force_rescan,
        )
        return 0

    result = run_casimir_scan(
        data_source=args.data,
        local_csv=args.local_csv,
        param_col=args.param_col,
        value_col=args.value_col,
        download=args.download,
        extract_dir=args.extract_dir,
        plot=not args.no_plot,
        output_prefix=args.output_prefix,
    )
    print("\n".join(result.summary_lines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())