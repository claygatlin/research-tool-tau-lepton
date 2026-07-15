"""Load GWOSC strain segments from cached HDF5 or text exports."""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StrainSegment:
    path: Path
    detector: str
    gps_start: float
    duration: float
    sample_rate: float
    strain: np.ndarray


def _read_hdf5(path: Path) -> StrainSegment:
    import h5py

    with h5py.File(path, "r") as handle:
        strain = np.asarray(handle["strain/Strain"][...], dtype=np.float64)
        gps_start = float(handle["meta/GPSstart"][()])
        duration = float(handle["meta/Duration"][()])
        raw_detector = handle["meta/Detector"][()]
        detector = raw_detector.decode() if isinstance(raw_detector, bytes) else str(raw_detector)
        detector = detector.strip()
    sample_rate = float(strain.size) / duration if duration else 0.0
    return StrainSegment(
        path=path,
        detector=detector,
        gps_start=gps_start,
        duration=duration,
        sample_rate=sample_rate,
        strain=strain,
    )


def _read_txt_gz(path: Path) -> StrainSegment:
    stem = path.name
    parts = stem.replace(".txt.gz", "").split("-")
    gps_start = float(parts[-2]) if len(parts) >= 2 else 0.0
    duration = float(parts[-1].split(".")[0]) if parts else 0.0
    detector = "H1" if stem.startswith("H-") else "L1" if stem.startswith("L-") else "UNK"
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        values = [float(line.strip()) for line in handle if line.strip()]
    strain = np.asarray(values, dtype=np.float64)
    sample_rate = float(strain.size) / duration if duration else 0.0
    return StrainSegment(
        path=path,
        detector=detector,
        gps_start=gps_start,
        duration=duration,
        sample_rate=sample_rate,
        strain=strain,
    )


def load_strain_file(path: Path | str) -> StrainSegment:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    suffix = "".join(target.suffixes).lower()
    if suffix == ".hdf5" or target.suffix.lower() == ".h5":
        return _read_hdf5(target)
    if suffix == ".txt.gz":
        return _read_txt_gz(target)
    raise ValueError(f"Unsupported strain format: {target}")


def list_event_strain_files(event: str, *, datasets_root: Path | None = None) -> list[Path]:
    root = datasets_root or Path(__file__).resolve().parents[3] / "datasets" / "gwosc"
    event_dir = root / "events" / event.strip().upper()
    if not event_dir.is_dir():
        raise FileNotFoundError(f"No cached strain directory for {event!r}: {event_dir}")
    files = sorted(event_dir.glob("*.hdf5")) + sorted(event_dir.glob("*.txt.gz"))
    if not files:
        raise FileNotFoundError(f"No strain files cached under {event_dir}")
    return files