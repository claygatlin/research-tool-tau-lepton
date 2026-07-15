"""
Option 4 — photon-enriched crosscheck pipeline for Tav CMS validation.

Maps a secondary gamma / EM vector-boson dataset into the execution
configuration so ``photon_enriched`` and ``photon_crosscheck`` are populated
when the primary dimu skim lacks photon branches.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from menus.particle.cern.manifest import OPTION4_PHOTON_CROSSCHECK_KEYS
from menus.particle.cern.mod7_phase import (
    mod7_phase_residues,
    mod7_phase_slot_masks,
    weighted_mod7_histogram,
)
from menus.particle.cms.tav_enhanced_validation_v2 import (
    photon_kinematics_winding_rate,
    tav_photon_enriched_7fold_analysis,
    vector_winding_rate_from_mod_excess,
)
from menus.particle.cern.tav_superblock_cms_muon_analyzer import (
    _photon_muon_crosscheck,
)
from tav_shared.artifact_paths import (
    TestSlug,
    artifact_path,
    artifact_timestamp,
    compose_dataset_slug,
)

OPTION4_LABEL = "Option 4 — Photon-Enriched Crosscheck Pipeline"
PHOTON_CROSSCHECK_JSON_BUFFER_BYTES = 1024 * 1024


def _json_default(obj: Any) -> Any:
    """JSON serializer safe for numpy scalars and arrays."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

DEFAULT_WINDING_RATE_PARAMS: dict[str, float] = {
    "omega_wind": 1.45e43,
    "kappa_leak": 0.01,
    "R": 7.0,
    "I": 1.0,
}


def _mod_excess_block_from_results(data_results: dict[str, Any]) -> dict[str, Any]:
    """Pull un-isolated mod-7 structure from primary lepton analysis."""
    mult = data_results.get("multiplicity_7fold") or {}
    scan = data_results.get("scan_summary") or {}
    muon_pt = data_results.get("muon_pt_7fold") or {}
    return {
        "mod7_histogram": mult.get("mod7_histogram") or scan.get("mod7_histogram"),
        "mod7_fractions": mult.get("mod7_fractions") or scan.get("mod7_fractions"),
        "subharmonic_excess": (
            muon_pt.get("subharmonic_excess")
            or scan.get("subharmonic_excess")
            or data_results.get("subharmonic_excess")
        ),
    }


def _compact_photon_scan_for_json(scan: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    Drop per-event arrays before JSON serialization.

    Full ``leading_pt_per_event`` lists from DoubleElectron-scale scans can exceed
    stream buffers and truncate ``photon_crosscheck.json`` mid-write.
    """
    if not scan:
        return scan
    compact = dict(scan)
    leading = compact.pop("leading_pt_per_event", None)
    if leading is not None:
        compact["leading_pt_per_event_omitted"] = True
        compact["n_leading_pt_events"] = int(len(leading))
    for key in list(compact):
        value = compact[key]
        if isinstance(value, np.ndarray):
            if value.size > 512:
                compact[key] = value[:512].tolist()
                compact[f"{key}_truncated"] = True
                compact[f"{key}_full_size"] = int(value.size)
            else:
                compact[key] = value.tolist()
    return compact


def _write_photon_crosscheck_json(path: Path, report: dict[str, Any]) -> None:
    """Buffered JSON write for large Option 4 crosscheck payloads."""
    payload = dict(report)
    payload["photon_scan"] = _compact_photon_scan_for_json(
        payload.get("photon_scan")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(
        "w",
        encoding="utf-8",
        buffering=PHOTON_CROSSCHECK_JSON_BUFFER_BYTES,
    ) as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
        handle.flush()


def _as_1d(arr: Any) -> np.ndarray:
    if arr is None:
        return np.array([], dtype=float)
    return np.asarray(arr, dtype=float).ravel()


def _muon_histogram_from_results(results: dict[str, Any]) -> np.ndarray:
    hist = results.get("muon_pt_histogram")
    if hist is None:
        scan = results.get("scan_summary") or {}
        hist = scan.get("primary_pt_hist") or scan.get("muon_pt_hist")
    return _as_1d(hist)


def _photon_histogram_from_results(results: dict[str, Any]) -> np.ndarray | None:
    for key in ("photon_pt_histogram", "electron_pt_histogram"):
        raw = results.get(key)
        if raw is not None:
            arr = _as_1d(raw)
            if arr.size and arr.sum() > 0:
                return arr
    scan = results.get("scan_summary") or {}
    for key in ("photon_pt_hist", "electron_pt_hist", "crosscheck_pt_hist", "primary_pt_hist"):
        raw = scan.get(key)
        if raw is not None:
            arr = _as_1d(raw)
            if arr.size and arr.sum() > 0:
                return arr
    block = results.get("photon_scan") or {}
    raw = block.get("photon_pt_hist")
    if raw is not None:
        arr = _as_1d(raw)
        if arr.size and arr.sum() > 0:
            return arr
    return None


def discover_option4_photon_datasets(
    cached_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve Option 4 secondary gamma / EM datasets present in the cache."""
    if cached_rows is None:
        from menus.particle.cern.fetcher import list_cached_targets

        cached_rows = [
            {
                "key": str(item["key"]),
                "label": str(item.get("label") or item["key"]),
                "path": str(item.get("sample_root") or ""),
                "cached": bool(item.get("cached")),
            }
            for item in list_cached_targets()
            if str(item.get("key", "")).startswith("cms_nanoaod")
        ]

    by_key = {str(row["key"]): row for row in cached_rows}
    resolved: list[str] = []
    missing: list[str] = []
    for key in OPTION4_PHOTON_CROSSCHECK_KEYS:
        if key in by_key and by_key[key].get("cached"):
            resolved.append(key)
        else:
            missing.append(key)

    return {
        "crosscheck_keys": list(OPTION4_PHOTON_CROSSCHECK_KEYS),
        "resolved_keys": resolved,
        "missing_keys": missing,
        "cached_rows": [by_key[k] for k in resolved if k in by_key],
        "preferred_key": resolved[0] if resolved else OPTION4_PHOTON_CROSSCHECK_KEYS[0],
    }


def resolve_photon_crosscheck_dataset(
    raw: str | None = None,
    *,
    fallback_to_higgs: bool = True,
) -> str:
    """
    Expand menu input into a manifest key or existing ROOT path.

    Accepts ``option4``, ``photon``, ``gamma``, ``doubleelectron``, or a
    comma-separated manifest key / file path.
    """
    text = (raw or "").strip()
    lowered = text.lower()
    aliases = {
        "option4",
        "photon",
        "gamma",
        "crosscheck",
        "doubleelectron",
        "double_electron",
        "em_crosscheck",
    }
    if not text or lowered in aliases:
        stack = discover_option4_photon_datasets()
        if stack["resolved_keys"]:
            return stack["resolved_keys"][0]
        return OPTION4_PHOTON_CROSSCHECK_KEYS[0]

    from menus.particle.cms.validation_extension import resolve_mc_validation_query

    resolved = resolve_mc_validation_query(text, field="photon")
    if resolved:
        return resolved

    if fallback_to_higgs and lowered in {"higgs", "zz", "zz4l", "cms_nanoaod_higgs_zz"}:
        return "cms_nanoaod_higgs_zz"

    return text


def _resolve_photon_file_path(source: str) -> Path:
    from menus.particle.cern.fetcher import resolve_root_path

    path = resolve_root_path(source)
    if path is not None:
        return Path(path)
    candidate = Path(source).expanduser()
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Photon crosscheck source not found: {source!r}. "
        f"Prefetch Option 4 datasets ({', '.join(OPTION4_PHOTON_CROSSCHECK_KEYS)})."
    )


def _load_photon_channel(
    photon_source: str | Path | dict[str, Any] | None,
    *,
    chunk_size: int = 500_000,
    entry_stop: int | None = None,
    pt_bins: int = 20,
    pt_max_gev: float = 200.0,
    pt_reduction: str = "flatten",
    verbose: bool = False,
) -> dict[str, Any]:
    """Return photon histogram metadata from a scan dict, lepton results, or ROOT file."""
    if photon_source is None:
        raise ValueError("photon_source is required for Option 4 crosscheck")

    if isinstance(photon_source, dict):
        hist = _photon_histogram_from_results(photon_source)
        if hist is None or hist.size < 4:
            raise ValueError("photon_source dict has no usable photon/electron pT histogram")
        return {
            "source_kind": "results_dict",
            "photon_pt_hist": hist,
            "bin_edges": photon_source.get("bin_edges"),
            "channel": photon_source.get("channel") or photon_source.get("lepton", "unknown"),
            "path": photon_source.get("file_path") or photon_source.get("path"),
            "n_events_processed": photon_source.get("n_events_processed"),
        }

    from menus.particle.cern.analyzer import extract_photon_pt_histogram_from_nanoaod

    path = _resolve_photon_file_path(str(photon_source))
    scan = extract_photon_pt_histogram_from_nanoaod(
        path,
        chunk_size=chunk_size,
        entry_stop=entry_stop,
        pt_bins=pt_bins,
        pt_max_gev=pt_max_gev,
        pt_reduction=pt_reduction,
        verbose=verbose,
    )
    scan["source_kind"] = "nanoaod_scan"
    return scan


def _phase_slot_overlap_metrics(
    muon_event_counts: np.ndarray | None,
    photon_event_counts: np.ndarray | None,
) -> dict[str, Any] | None:
    """Compare mod-7 phase slot occupancy using vector masks (not scalar int cast)."""
    if muon_event_counts is None or photon_event_counts is None:
        return None
    muon_arr = np.asarray(muon_event_counts, dtype=np.float64).ravel()
    photon_arr = np.asarray(photon_event_counts, dtype=np.float64).ravel()
    n = min(muon_arr.size, photon_arr.size)
    if n < 8:
        return None
    muon_residues = mod7_phase_residues(muon_arr[:n])
    photon_residues = mod7_phase_residues(photon_arr[:n])
    muon_masks = mod7_phase_slot_masks(muon_arr[:n])
    photon_masks = mod7_phase_slot_masks(photon_arr[:n])
    overlap = np.sum(muon_masks & photon_masks, axis=1)
    muon_hist = weighted_mod7_histogram(muon_arr[:n])
    photon_hist = weighted_mod7_histogram(photon_arr[:n])
    muon_total = float(muon_hist.sum()) or 1.0
    photon_total = float(photon_hist.sum()) or 1.0
    return {
        "n_events_aligned": int(n),
        "mod7_residues_muon_sample": muon_residues[:16].tolist(),
        "mod7_residues_photon_sample": photon_residues[:16].tolist(),
        "slot_overlap_counts": overlap.astype(int).tolist(),
        "muon_slot_fractions": (muon_hist / muon_total).tolist(),
        "photon_slot_fractions": (photon_hist / photon_total).tolist(),
        "slot_mask_method": "vector_masks (mod_residues = event_array % 7)",
    }


def assess_cross_channel_7fold_consistency(
    *,
    muon_7fold_amplitude: float,
    photon_7fold_amplitude: float,
    photon_muon_correlation: float | None = None,
    relative_tolerance: float = 0.35,
) -> dict[str, Any]:
    """
    Test whether the 7-fold periodic amplitude is shared across muon and EM channels.
    """
    muon_amp = float(muon_7fold_amplitude)
    photon_amp = float(photon_7fold_amplitude)
    if muon_amp == 0.0 and photon_amp == 0.0:
        return {
            "verdict": "UNDERPOWERED",
            "global_spacetime_imprint": False,
            "muon_sector_specific": False,
        }

    denom = max(abs(muon_amp), abs(photon_amp), 1e-9)
    rel_delta = abs(muon_amp - photon_amp) / denom
    aligned = rel_delta <= float(relative_tolerance)
    corr = float(photon_muon_correlation) if photon_muon_correlation is not None else 0.0
    corr_support = corr >= 0.25

    if aligned and (corr_support or rel_delta <= 0.2):
        verdict = "GLOBAL_7FOLD_IMPRINT"
        global_imprint = True
        muon_specific = False
    elif photon_amp > muon_amp * 1.25:
        verdict = "EM_DOMINANT_7FOLD_HINT"
        global_imprint = True
        muon_specific = False
    elif muon_amp > photon_amp * 1.25:
        verdict = "MUON_SECTOR_SPECIFIC"
        global_imprint = False
        muon_specific = True
    else:
        verdict = "INCONCLUSIVE_CROSS_CHANNEL"
        global_imprint = False
        muon_specific = False

    return {
        "verdict": verdict,
        "relative_amplitude_delta": float(rel_delta),
        "muon_7fold_amplitude": muon_amp,
        "photon_7fold_amplitude": photon_amp,
        "photon_muon_correlation": corr,
        "global_spacetime_imprint": global_imprint,
        "muon_sector_specific": muon_specific,
        "interpretation": (
            "Cross-channel 7-fold amplitude agreement tests whether the Tav cylinder "
            "signature is unique to the muon sector or a macroscopic spacetime geometry "
            "property visible in vector gauge boson (γ / e±) channels."
        ),
    }


def run_photon_enriched_crosscheck_pipeline(
    data_results: dict[str, Any],
    photon_source: str | Path | dict[str, Any] | None = None,
    *,
    enhanced_report: dict[str, Any] | None = None,
    winding_rate_params: dict[str, float] | None = None,
    photon_sf: float = 1.0,
    photon_pt_reduction: str = "flatten",
    chunk_size: int = 500_000,
    entry_stop: int | None = None,
    output_dir: str | Path = "photon_crosscheck",
    run_when: datetime | None = None,
    dataset_slug: str = "photon_crosscheck",
    save_report: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Activate photon kinematics winding-rate hooks and populate crosscheck blocks.

    Parameters
    ----------
    data_results
        Primary dimuon (or muon-primary) lepton analysis dict.
    photon_source
        Secondary gamma dataset: manifest key, ROOT path, or pre-built results dict.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_when = run_when or datetime.now(timezone.utc)
    wr_params = dict(DEFAULT_WINDING_RATE_PARAMS)
    if winding_rate_params:
        wr_params.update(winding_rate_params)

    muon_hist = _muon_histogram_from_results(data_results)
    if muon_hist.size < 4:
        return {
            "option": OPTION4_LABEL,
            "verdict": "UNDERPOWERED",
            "reason": "primary muon pT histogram too small",
            "output_dir": str(out_dir),
        }

    muon_pt_block = data_results.get("muon_pt_7fold") or {}
    muon_7fold_amp = float(muon_pt_block.get("subharmonic_excess") or 0.0)
    if enhanced_report:
        dvm = enhanced_report.get("data_vs_mc") or {}
        muon_7fold_amp = float(
            dvm.get("data_7fold_amplitude") or muon_7fold_amp
        )

    photon_scan: dict[str, Any] | None = None
    photon_hist: np.ndarray | None = None
    load_error: str | None = None
    if photon_source is not None:
        try:
            photon_scan = _load_photon_channel(
                photon_source,
                chunk_size=chunk_size,
                entry_stop=entry_stop,
                pt_reduction=photon_pt_reduction,
                verbose=verbose,
            )
            photon_hist = _as_1d(photon_scan.get("photon_pt_hist"))
            if photon_sf != 1.0:
                photon_hist = photon_hist * float(photon_sf)
        except Exception as exc:
            load_error = str(exc)

    if photon_hist is None or photon_hist.size < 4:
        fallback = _photon_histogram_from_results(data_results)
        if fallback is not None and fallback.size >= 4:
            photon_hist = fallback
            photon_scan = {
                "source_kind": "embedded_in_data_results",
                "channel": "embedded",
                "path": data_results.get("file_path"),
            }

    photon_enriched: dict[str, Any] | None = None
    photon_crosscheck: dict[str, Any] | None = None
    cross_channel: dict[str, Any] | None = None

    mod_block = _mod_excess_block_from_results(data_results)
    vector_winding = vector_winding_rate_from_mod_excess(
        mod7_histogram=mod_block.get("mod7_histogram"),
        mod7_fractions=mod_block.get("mod7_fractions"),
        subharmonic_excess=mod_block.get("subharmonic_excess"),
        winding_rate_params=wr_params,
    )

    if photon_hist is not None and photon_hist.size >= 4:
        photon_enriched = tav_photon_enriched_7fold_analysis(
            photon_hist,
            muon_pt_hist=muon_hist,
            bin_edges=(
                np.asarray(photon_scan.get("bin_edges"), dtype=float)
                if photon_scan and photon_scan.get("bin_edges") is not None
                else None
            ),
        )
        photon_enriched["photon_sf_applied"] = float(photon_sf)
        if vector_winding.get("available"):
            photon_enriched["effective_winding_rate_proxy"] = vector_winding.get(
                "winding_rate_proxy"
            )
            photon_enriched["vector_winding_baseline"] = vector_winding
        else:
            photon_enriched["effective_winding_rate_proxy"] = photon_kinematics_winding_rate(
                **wr_params
            )
        photon_enriched["winding_rate_params"] = wr_params
        photon_enriched["photon_channel"] = (photon_scan or {}).get("channel")
        photon_enriched["photon_branch"] = (photon_scan or {}).get("photon_branch")

        photon_crosscheck = _photon_muon_crosscheck(muon_hist, photon_hist)
        cross_channel = assess_cross_channel_7fold_consistency(
            muon_7fold_amplitude=muon_7fold_amp,
            photon_7fold_amplitude=float(
                photon_enriched.get("photon_7fold_amplitude") or 0.0
            ),
            photon_muon_correlation=photon_crosscheck.get("muon_photon_correlation"),
        )
        n_muon_per_event = (data_results.get("scan_summary") or {}).get(
            "n_primary_per_event"
        ) or (data_results.get("scan_summary") or {}).get("n_muon_per_event")
        n_photon_per_event = (photon_scan or {}).get("n_events_per_object")
        phase_slots = _phase_slot_overlap_metrics(
            np.asarray(n_muon_per_event) if n_muon_per_event is not None else None,
            np.asarray(n_photon_per_event) if n_photon_per_event is not None else None,
        )
        if phase_slots:
            photon_enriched["phase_slot_masks"] = phase_slots
            cross_channel["phase_slot_overlap"] = phase_slots

    cache = discover_option4_photon_datasets()
    report: dict[str, Any] = {
        "option": OPTION4_LABEL,
        "timestamp": artifact_timestamp(run_when),
        "objective": (
            "Cross-verify whether the 7-fold periodic amplitude holds across vector "
            "gauge boson (γ / e±) channels versus the muon sector."
        ),
        "photon_source": str(photon_source) if photon_source is not None else None,
        "photon_scan": photon_scan,
        "photon_enriched": photon_enriched,
        "photon_crosscheck": photon_crosscheck,
        "cross_channel_consistency": cross_channel,
        "muon_reference_7fold_amplitude": muon_7fold_amp,
        "crosscheck_cache_status": cache,
        "winding_rate_params": wr_params,
        "vector_winding_baseline": vector_winding,
        "photon_sf": float(photon_sf),
        "output_dir": str(out_dir),
    }

    if load_error and photon_enriched is None:
        report["verdict"] = "PHOTON_CROSSCHECK_FAILED"
        report["error"] = load_error
    elif photon_enriched is None:
        report["verdict"] = "UNDERPOWERED"
        report["reason"] = "no secondary gamma histogram (map photon dataset in config)"
    else:
        report["verdict"] = (cross_channel or {}).get("verdict", "PHOTON_CROSSCHECK_COMPLETE")

    if enhanced_report is not None:
        dvm = enhanced_report.get("data_vs_mc") or {}
        report["enhanced_validation"] = {
            "data_7fold_amplitude": dvm.get("data_7fold_amplitude"),
            "report_path": enhanced_report.get("report_path"),
        }
        if enhanced_report.get("photon_enriched") is None and photon_enriched is not None:
            enhanced_report["photon_enriched"] = photon_enriched
        if photon_crosscheck is not None:
            enhanced_report["photon_crosscheck"] = photon_crosscheck
        if cross_channel is not None:
            enhanced_report["cross_channel_consistency"] = cross_channel

    if save_report:
        path = artifact_path(
            TestSlug.CERN,
            compose_dataset_slug(dataset_slug, "photon_crosscheck"),
            "report",
            "json",
            when=run_when,
            run_dir=out_dir,
        )
        _write_photon_crosscheck_json(path, report)
        report["report_path"] = str(path)

    return report


def print_option4_photon_crosscheck_summary(report: dict[str, Any]) -> None:
    """Stdout banner for Option 4 photon-enriched crosscheck."""
    print("=== TAV OPTION 4: PHOTON-ENRICHED CROSSCHECK ===")
    if report.get("verdict") in {"UNDERPOWERED", "PHOTON_CROSSCHECK_FAILED"}:
        print(f"  Status            : {report.get('verdict')} ({report.get('reason') or report.get('error', 'n/a')})")
        print("=" * 46)
        return

    photon = report.get("photon_enriched") or {}
    cross = report.get("photon_crosscheck") or {}
    channel = report.get("cross_channel_consistency") or {}
    scan = report.get("photon_scan") or {}
    cache = report.get("crosscheck_cache_status") or {}

    if scan.get("photon_branch"):
        print(f"  Photon branch     : {scan.get('photon_branch')} ({scan.get('channel', 'n/a')})")
    elif scan.get("source_kind"):
        print(f"  Photon source     : {scan.get('source_kind')}")
    if cache.get("missing_keys"):
        print(f"  Missing from cache: {', '.join(cache.get('missing_keys') or [])}")
    print(f"  Muon 7-fold amp   : {report.get('muon_reference_7fold_amplitude', 0):.3f}")
    print(f"  Photon 7-fold amp : {photon.get('photon_7fold_amplitude', 0):.3f}")
    print(
        f"  Winding rate proxy: {photon.get('effective_winding_rate_proxy', 0):.2e}"
    )
    if cross:
        print(f"  μ–γ correlation   : {cross.get('muon_photon_correlation', 0):.3f}")
        print(f"  Ratio 7-fold      : {(cross.get('ratio_spectrum_7fold') or {}).get('verdict', 'n/a')}")
    if channel:
        print(f"  Cross-channel     : {channel.get('verdict', 'n/a')}")
        if channel.get("global_spacetime_imprint"):
            print("  Geometry verdict  : global spacetime imprint (not muon-only)")
        elif channel.get("muon_sector_specific"):
            print("  Geometry verdict  : muon-sector-specific signature")
    if report.get("report_path"):
        print(f"  Report            : {report['report_path']}")
    print("=" * 46)


def ensure_option4_photon_cached(
    *,
    verbose: bool = True,
    auto_fetch: bool = True,
) -> dict[str, Any]:
    """Prefetch DoubleElectron + ZZTo4e (+ Higgs fallback) for Option 4."""
    from menus.particle.cern.fetcher import ensure_cern_target

    status: dict[str, Any] = {"fetched": [], "cached": [], "failed": {}}
    for key in OPTION4_PHOTON_CROSSCHECK_KEYS:
        try:
            path = ensure_cern_target(key, auto_fetch=auto_fetch)
            status["cached"].append({"key": key, "path": str(path)})
            if verbose:
                print(f"[Option 4] Ready: {key} → {path}")
        except Exception as exc:
            status["failed"][key] = str(exc)
            if verbose:
                print(f"[Option 4] Missing/failed: {key} — {exc}")
    return status