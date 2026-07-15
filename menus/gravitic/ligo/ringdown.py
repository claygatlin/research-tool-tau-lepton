"""Ringdown harmonic scan on cached GWOSC strain data."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from menus.gravitic.common.gwosc_api import fetch_event_detail
from menus.gravitic.ligo.strain_io import StrainSegment, list_event_strain_files, load_strain_file
from tav_shared.tav_project_paths import ARTIFACTS_ROOT

TAV_HARMONIC_RATIO = 1.0 / 7.0
DEFAULT_RINGDOWN_START = 0.01
DEFAULT_RINGDOWN_END = 0.25
DEFAULT_BANDPASS = (20.0, 512.0)


@dataclass
class HarmonicPeak:
    frequency_hz: float
    power: float
    harmonic_index: int


@dataclass
class RingdownReport:
    event: str
    detector: str
    strain_file: str
    merger_gps: float | None
    sample_rate_hz: float
    ringdown_window_s: tuple[float, float]
    fundamental_hz: float | None
    harmonic_peaks: list[HarmonicPeak] = field(default_factory=list)
    subharmonic_excess: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _bandpass(strain: np.ndarray, sample_rate: float, band: tuple[float, float]) -> np.ndarray:
    low, high = band
    nyquist = 0.5 * sample_rate
    if high >= nyquist:
        high = nyquist * 0.98
    if low <= 0 or low >= high:
        return strain - np.mean(strain)
    sos = signal.butter(4, [low / nyquist, high / nyquist], btype="band", output="sos")
    return signal.sosfiltfilt(sos, strain)


def _ringdown_slice(
    segment: StrainSegment,
    *,
    merger_gps: float | None,
    start_offset_s: float,
    end_offset_s: float,
) -> tuple[np.ndarray, float]:
    if merger_gps is None:
        center = segment.duration * 0.5
    else:
        center = merger_gps - segment.gps_start
    start = max(0.0, center + start_offset_s)
    end = min(segment.duration, center + end_offset_s)
    if end <= start:
        raise ValueError("Ringdown window is empty for this segment")
    i0 = int(start * segment.sample_rate)
    i1 = int(end * segment.sample_rate)
    return segment.strain[i0:i1], segment.sample_rate


def _fundamental_peak(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float | None:
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return None
    subset_f = freqs[mask]
    subset_p = psd[mask]
    return float(subset_f[int(np.argmax(subset_p))])


def _peak_power_near(freqs: np.ndarray, psd: np.ndarray, target_hz: float, width_hz: float = 8.0) -> float:
    mask = (freqs >= target_hz - width_hz) & (freqs <= target_hz + width_hz)
    if not np.any(mask):
        return 0.0
    return float(np.max(psd[mask]))


def analyze_ringdown_segment(
    segment: StrainSegment,
    *,
    event: str,
    merger_gps: float | None = None,
    ringdown_start_s: float = DEFAULT_RINGDOWN_START,
    ringdown_end_s: float = DEFAULT_RINGDOWN_END,
    bandpass_hz: tuple[float, float] = DEFAULT_BANDPASS,
) -> RingdownReport:
    window, sample_rate = _ringdown_slice(
        segment,
        merger_gps=merger_gps,
        start_offset_s=ringdown_start_s,
        end_offset_s=ringdown_end_s,
    )
    filtered = _bandpass(window, sample_rate, bandpass_hz)
    freqs, psd = signal.welch(filtered, fs=sample_rate, nperseg=min(2048, max(256, filtered.size // 4)))
    fundamental = _fundamental_peak(freqs, psd, bandpass_hz)

    harmonic_peaks: list[HarmonicPeak] = []
    subharmonic_excess: dict[str, float] = {}
    notes: list[str] = []

    if fundamental:
        for index in range(1, 8):
            target = fundamental * index * TAV_HARMONIC_RATIO
            if target > bandpass_hz[1]:
                break
            power = _peak_power_near(freqs, psd, target)
            harmonic_peaks.append(
                HarmonicPeak(frequency_hz=target, power=power, harmonic_index=index)
            )
        f7 = fundamental * TAV_HARMONIC_RATIO
        baseline_mask = (freqs >= bandpass_hz[0]) & (freqs <= bandpass_hz[1])
        baseline = float(np.median(psd[baseline_mask])) if np.any(baseline_mask) else 0.0
        f7_power = _peak_power_near(freqs, psd, f7)
        subharmonic_excess["f_over_7"] = (f7_power / baseline) if baseline > 0 else 0.0
        notes.append(
            f"1/7 sub-harmonic excess at {f7:.1f} Hz: {subharmonic_excess['f_over_7']:.3f}× median PSD"
        )
    else:
        notes.append("No stable ringdown fundamental identified in the selected band.")

    return RingdownReport(
        event=event,
        detector=segment.detector,
        strain_file=str(segment.path),
        merger_gps=merger_gps,
        sample_rate_hz=sample_rate,
        ringdown_window_s=(ringdown_start_s, ringdown_end_s),
        fundamental_hz=fundamental,
        harmonic_peaks=harmonic_peaks,
        subharmonic_excess=subharmonic_excess,
        notes=notes,
    )


def _plot_report(report: RingdownReport, segment: StrainSegment, psd_freqs: np.ndarray, psd_vals: np.ndarray) -> Path:
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    png_path = ARTIFACTS_ROOT / f"gwosc_ringdown_{report.event}_{report.detector}_{stamp}.png"
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    time_axis = np.arange(segment.strain.size) / segment.sample_rate
    axes[0].plot(time_axis, segment.strain, color="0.35", linewidth=0.6)
    axes[0].set_title(f"{report.event} {report.detector} strain ({segment.path.name})")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Strain")
    axes[0].grid(True, linestyle=":", alpha=0.5)

    axes[1].semilogy(psd_freqs, psd_vals, color="tab:blue", linewidth=0.9, label="Ringdown PSD")
    if report.fundamental_hz:
        axes[1].axvline(report.fundamental_hz, color="tab:red", linestyle="--", label=f"f₀≈{report.fundamental_hz:.1f} Hz")
    for peak in report.harmonic_peaks:
        axes[1].axvline(
            peak.frequency_hz,
            color="tab:orange",
            linestyle=":",
            alpha=0.8,
            label=f"{peak.harmonic_index}/7 f₀" if peak.harmonic_index == 1 else None,
        )
    axes[1].set_xlim(DEFAULT_BANDPASS)
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("PSD")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend(loc="upper right")
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    return png_path


def run_ringdown_scan(
    event: str,
    *,
    strain_path: str | None = None,
    show_graphics: str = "artifacts",
) -> dict[str, Any]:
    """Analyze cached strain for one event (all detectors unless strain_path set)."""
    event_clean = event.strip().upper()
    merger_gps: float | None = None
    try:
        detail = fetch_event_detail(event_clean)
        raw_gps = detail.get("GPS")
        if raw_gps is not None:
            merger_gps = float(raw_gps)
    except Exception:
        merger_gps = None

    paths = [Path(strain_path)] if strain_path else list_event_strain_files(event_clean)
    reports: list[RingdownReport] = []
    artifact_paths: list[str] = []

    for path in paths:
        segment = load_strain_file(path)
        report = analyze_ringdown_segment(segment, event=event_clean, merger_gps=merger_gps)
        reports.append(report)

        window, sample_rate = _ringdown_slice(
            segment,
            merger_gps=merger_gps,
            start_offset_s=DEFAULT_RINGDOWN_START,
            end_offset_s=DEFAULT_RINGDOWN_END,
        )
        filtered = _bandpass(window, sample_rate, DEFAULT_BANDPASS)
        freqs, psd = signal.welch(filtered, fs=sample_rate, nperseg=min(2048, max(256, filtered.size // 4)))

        if show_graphics in {"artifacts", "popup", "yes", "y", "true", "1"}:
            artifact_paths.append(str(_plot_report(report, segment, freqs, psd)))

        print(f"[GWOSC RINGDOWN] {report.event} {report.detector}")
        for note in report.notes:
            print(f"  {note}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = ARTIFACTS_ROOT / f"gwosc_ringdown_{event_clean}_{stamp}.json"
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "event": event_clean,
        "merger_gps": merger_gps,
        "reports": [
            {
                **asdict(report),
                "harmonic_peaks": [asdict(peak) for peak in report.harmonic_peaks],
            }
            for report in reports
        ],
        "plots": artifact_paths,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[GWOSC RINGDOWN] Report: {json_path}")
    return payload