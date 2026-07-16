"""
Canonical normalization for heterogeneous ingestion sources.

Maps CSV, JSON, dict, and analyzer-native field names into the internal
format expected by Pydantic schemas and downstream solvers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

# Analyzer / NanoAOD aliases → canonical names
_CMS_FIELD_ALIASES: dict[str, str] = {
    "leading_pt": "leading_pt_gev",
    "leading_pt_gev": "leading_pt_gev",
    "subleading_pt": "subleading_pt_gev",
    "subleading_pt_gev": "subleading_pt_gev",
    "delta_phi": "delta_phi_rad",
    "delta_phi_rad": "delta_phi_rad",
    "delta_r": "delta_r",
    "system_pt": "system_pt_gev",
    "system_pt_gev": "system_pt_gev",
    "n_muon_per_event": "n_muon",
    "n_muon": "n_muon",
    "n_true_int": "n_true_int",
    "pileup": "n_true_int",
    "nTrueInt": "n_true_int",
}

_BBN_ABUNDANCE_ALIASES: dict[str, str] = {
    "D/H": "D_H",
    "D_H": "D_H",
    "Y_p": "Y_p_He4",
    "Y_p_He4": "Y_p_He4",
    "Li7/H": "Li7_H_obs",
    "Li7_H": "Li7_H_obs",
    "Li7_H_obs": "Li7_H_obs",
    "Li7_H_std_theory": "Li7_H_std_theory",
    "theory_Li7_H_standard": "Li7_H_std_theory",
}

_FRACTAL_PARAM_ALIASES: dict[str, str] = {
    "winding_density": "winding_density",
    "fractal_level": "fractal_level",
    "phase_slip_alpha": "phase_slip_alpha",
    "n_hier": "fractal_level",
}


def detect_source_format(data: Any, *, path: str | Path | None = None) -> str:
    """Infer source format from path extension or payload shape."""
    if path is not None:
        suffix = Path(path).suffix.lower()
        if suffix in {".csv", ".tsv"}:
            return "csv"
        if suffix == ".json":
            return "json"
        if suffix in {".root", ".ROOT"}:
            return "root"
    if isinstance(data, (str, Path)):
        return detect_source_format(None, path=data)
    if isinstance(data, Mapping):
        if "events" in data or "leading_pt" in data or "leading_pt_gev" in data:
            if any(
                isinstance(data.get(k), (list, np.ndarray)) and len(data.get(k, [])) > 1
                for k in ("leading_pt", "leading_pt_gev")
            ):
                return "cms_batch"
            if "events" in data:
                return "json"
        if "bbn_abundances" in data or "neutron_lifetime" in data:
            return "bbn"
        if "winding_density" in data or "fractal_level" in data:
            return "fractal_tau"
        return "dict"
    if isinstance(data, list):
        return "json"
    return "unknown"


def load_payload_from_path(path: str | Path) -> Any:
    """Load CSV or JSON from disk."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Ingestion source not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    if suffix in {".csv", ".tsv"}:
        import pandas as pd

        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(p, sep=sep).to_dict(orient="records")
    raise ValueError(f"Unsupported ingestion file type: {p.suffix}")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 0 or not np.isfinite(arr[0]):
            return None
        return float(arr[0])
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    f = _as_float(value)
    if f is None:
        return None
    return int(np.rint(f))


def _rename_keys(raw: Mapping[str, Any], aliases: Mapping[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in raw.items():
        canon = aliases.get(str(key), str(key))
        out[canon] = val
    return out


def normalize_abundance_pair(raw: Any) -> dict[str, float]:
    """Normalize (value, unc) tuples, dicts, or scalar pairs."""
    if isinstance(raw, Mapping):
        value = _as_float(raw.get("value", raw.get("val")))
        unc = _as_float(raw.get("uncertainty", raw.get("unc", raw.get("sigma"))))
        if value is not None and unc is not None:
            return {"value": value, "uncertainty": unc}
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        value = _as_float(raw[0])
        unc = _as_float(raw[1])
        if value is not None and unc is not None:
            return {"value": value, "uncertainty": abs(unc)}
    raise ValueError(f"Cannot normalize abundance pair: {raw!r}")


def normalize_bbn_payload(raw: Mapping[str, Any] | Any) -> dict[str, Any]:
    """BBN empirical dict → canonical nested structure."""
    if not isinstance(raw, Mapping):
        raise TypeError("BBN payload must be a mapping")
    payload = dict(raw)
    nl_raw = payload.get("neutron_lifetime", {})
    if not isinstance(nl_raw, Mapping):
        raise ValueError("neutron_lifetime must be a mapping")
    neutron = {
        "bottle": _as_float(nl_raw.get("bottle")),
        "beam": _as_float(nl_raw.get("beam")),
        "pdg": _as_float(nl_raw.get("pdg")),
        "unc": _as_float(nl_raw.get("unc")),
        "tension_sigma": _as_float(nl_raw.get("tension_sigma")),
    }
    if any(v is None for v in neutron.values()):
        raise ValueError(f"neutron_lifetime missing fields: {neutron}")

    abund_raw = payload.get("bbn_abundances", {})
    if not isinstance(abund_raw, Mapping):
        raise ValueError("bbn_abundances must be a mapping")
    abundances: dict[str, dict[str, float]] = {}
    for key, val in abund_raw.items():
        canon = _BBN_ABUNDANCE_ALIASES.get(str(key), str(key))
        abundances[canon] = normalize_abundance_pair(val)

    return {
        "neutron_lifetime": neutron,
        "bbn_abundances": abundances,
    }


def normalize_fractal_params(raw: Mapping[str, Any] | Any) -> dict[str, float]:
    if isinstance(raw, (list, tuple, np.ndarray)):
        arr = np.asarray(raw, dtype=float).ravel()
        if arr.size != 3:
            raise ValueError("fractal params vector must have length 3")
        return {
            "winding_density": float(arr[0]),
            "fractal_level": float(arr[1]),
            "phase_slip_alpha": float(arr[2]),
        }
    if not isinstance(raw, Mapping):
        raise TypeError("fractal params must be a mapping or length-3 sequence")
    renamed = _rename_keys(raw, _FRACTAL_PARAM_ALIASES)
    out: dict[str, float] = {}
    for key in ("winding_density", "fractal_level", "phase_slip_alpha"):
        val = _as_float(renamed.get(key))
        if val is None:
            raise ValueError(f"fractal param missing: {key}")
        out[key] = val
    return out


def normalize_fractal_observed(
    raw: Mapping[str, Any] | None,
    *,
    defaults: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    base = dict(defaults or {})
    if raw is None:
        return base
    if hasattr(raw, "as_dict"):
        raw = raw.as_dict()  # type: ignore[assignment]
    if not isinstance(raw, Mapping):
        raise TypeError("fractal observed data must be a mapping")
    merged = {**base, **dict(raw)}
    out: dict[str, Any] = {}
    for key in (
        "mass_gap_mev",
        "sigma_mass_gap_mev",
        "sigma_qcd_mev",
        "spectral_dim",
        "sigma_spectral_dim",
        "n_hier",
        "torsion_factor",
        "likelihood_mode",
    ):
        if key in merged:
            if key == "likelihood_mode":
                out[key] = str(merged[key])
            else:
                val = _as_float(merged[key])
                if val is not None:
                    out[key] = val
    return out


def normalize_cms_dimuon_batch(
    raw: Mapping[str, Any] | list[Mapping[str, Any]] | Any,
    *,
    source_format: str | None = None,
) -> dict[str, Any]:
    """
    Normalize CMS dimuon data to a list of per-event dicts with canonical keys.

    Accepts:
    - Analyzer output (column arrays under leading_pt, …)
    - List of per-event records (JSON/CSV rows)
    - Dict with ``events`` list
    """
    fmt = source_format or detect_source_format(raw)
    events: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}

    if isinstance(raw, list):
        for row in raw:
            if isinstance(row, Mapping):
                events.append(_normalize_cms_event_row(_rename_keys(row, _CMS_FIELD_ALIASES)))
        return {"events": events, "source_format": fmt or "json"}

    if not isinstance(raw, Mapping):
        raise TypeError("CMS dimuon batch must be a mapping or record list")

    if "events" in raw and isinstance(raw["events"], list):
        for row in raw["events"]:
            if isinstance(row, Mapping):
                events.append(
                    _normalize_cms_event_row(_rename_keys(row, _CMS_FIELD_ALIASES))
                )
        meta = {k: v for k, v in raw.items() if k != "events"}
        meta["source_format"] = fmt or "json"
        meta["events"] = events
        return meta

    # Columnar analyzer layout
    arrays: dict[str, Any] = {}
    for src_key, canon in _CMS_FIELD_ALIASES.items():
        if src_key in raw and canon not in arrays:
            arrays[canon] = raw[src_key]

    required = (
        "leading_pt_gev",
        "subleading_pt_gev",
        "delta_phi_rad",
        "delta_r",
        "system_pt_gev",
    )
    n_events = 0
    for key in required:
        arr = arrays.get(key)
        if arr is not None:
            n_events = max(n_events, len(np.asarray(arr).reshape(-1)))

    if n_events == 0:
        return {"events": [], "source_format": fmt or "cms_batch", **{
            k: v for k, v in raw.items()
            if k not in _CMS_FIELD_ALIASES and k not in _CMS_FIELD_ALIASES.values()
        }}

    def _col(name: str, default: Any = None) -> Any:
        if name not in arrays:
            return default
        return np.asarray(arrays[name]).reshape(-1)

    n_muon_col = _col("n_muon", np.full(n_events, 2, dtype=int))
    n_true_col = _col("n_true_int")

    for i in range(n_events):
        row: dict[str, Any] = {
            "leading_pt_gev": float(_col("leading_pt_gev")[i]),
            "subleading_pt_gev": float(_col("subleading_pt_gev")[i]),
            "delta_phi_rad": float(_col("delta_phi_rad")[i]),
            "delta_r": float(_col("delta_r")[i]),
            "system_pt_gev": float(_col("system_pt_gev")[i]),
            "n_muon": int(n_muon_col[i]) if n_muon_col is not None else 2,
        }
        if n_true_col is not None and i < len(n_true_col):
            row["n_true_int"] = int(np.rint(n_true_col[i]))
        events.append(_normalize_cms_event_row(row))

    meta = {
        k: v
        for k, v in raw.items()
        if k not in _CMS_FIELD_ALIASES and k not in arrays
    }
    meta["events"] = events
    meta["source_format"] = fmt or "cms_batch"
    return meta


def _normalize_cms_event_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "leading_pt_gev",
        "subleading_pt_gev",
        "delta_phi_rad",
        "delta_r",
        "system_pt_gev",
    ):
        val = _as_float(row.get(key))
        if val is None:
            raise ValueError(f"CMS event missing {key}")
        out[key] = val
    n_muon = _as_int(row.get("n_muon"))
    out["n_muon"] = n_muon if n_muon is not None else 2
    n_true = _as_int(row.get("n_true_int"))
    if n_true is not None:
        out["n_true_int"] = n_true
    return out


def cms_batch_to_columnar(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Convert validated event list back to analyzer-style column arrays."""
    events = batch.get("events", [])
    if not events:
        return dict(batch)

    columnar: dict[str, Any] = {k: v for k, v in batch.items() if k != "events"}
    keys = (
        "leading_pt_gev",
        "subleading_pt_gev",
        "delta_phi_rad",
        "delta_r",
        "system_pt_gev",
        "n_muon",
        "n_true_int",
    )
    for key in keys:
        vals = [ev.get(key) for ev in events if ev.get(key) is not None]
        if not vals:
            continue
        if key in {"n_muon", "n_true_int"}:
            columnar[_reverse_alias(key)] = np.asarray(vals, dtype=int)
        else:
            columnar[_reverse_alias(key)] = np.asarray(vals, dtype=float)
    columnar["n_dimuon_events"] = len(events)
    return columnar


def _reverse_alias(canon: str) -> str:
    """Map canonical name back to analyzer-native column name."""
    reverse = {
        "leading_pt_gev": "leading_pt",
        "subleading_pt_gev": "subleading_pt",
        "delta_phi_rad": "delta_phi",
        "system_pt_gev": "system_pt",
        "n_muon": "n_muon_per_event",
    }
    return reverse.get(canon, canon)