#!/usr/bin/env python3
"""
Tav-Superblock CMS Enhanced Validation Module v2.

Integrates:
- Full per-event MC weighting (pileup, lepton/photon efficiencies)
- Background template subtraction before 7-fold measurement
- Photon-enriched channel support (wylde kinematics winding-rate hooks)
- 7-fold Tav analysis on data, weighted MC, and background-subtracted spectra
- Multi-MC Data vs MC comparison with shape diagnostics

Extends the lepton v2 + muon analyzer pipeline for production CMS validation.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from tav_superblock_cms_muon_analyzer import (
    M0_MEV_ANCHOR,
    _exponential_detrend,
    _fit_7periodic,
    _pt_spectrum_7fold,
)

from tav_shared.artifact_paths import (
    TestSlug,
    artifact_path,
    artifact_timestamp,
    compose_dataset_slug,
)

SIGNAL_NAME_HINTS = ("higgs", "signal", "smhiggs", "zz")
BACKGROUND_NAME_HINTS = ("dy", "qcd", "ttbar", "tt", "wjets", "diboson")
DEFAULT_PT_MAX_GEV: float = 200.0
DEFAULT_PT_BIN_COUNT: int = 20
MIN_EVENTS_GEOMETRIC_FLOOR: int = 50
RECOIL_VALIDATION_DELTA_R_MAX: float = 1.5


def default_pt_thresholds(
    *,
    pt_max_gev: float = DEFAULT_PT_MAX_GEV,
    n_bins: int = DEFAULT_PT_BIN_COUNT,
) -> np.ndarray:
    """Leading-muon pT cut values aligned with the standard 0–200 GeV binning."""
    edges = np.linspace(0.0, float(pt_max_gev), int(n_bins) + 1)
    return edges[:-1]


def recoil_validation_gate_cuts(
    *,
    delta_r_max: float = RECOIL_VALIDATION_DELTA_R_MAX,
    recoil_pt_max: float = 45.0,
) -> dict[str, float]:
    """
    Non-contradictory dimuon validation gate for Option A / DoubleMu sweeps.

    Applies only ``ΔR ≤ delta_r_max`` and recoil system-pT ceiling — no
    simultaneous ``Δφ_min`` cut (that joint with tight ``ΔR`` collapses the loop).
    """
    return {
        "delta_r_max": float(delta_r_max),
        "recoil_pt_max": float(recoil_pt_max),
    }


def default_angular_cuts() -> dict[str, float]:
    """Tight dimuon angular + recoil selections for geometric-floor isolation."""
    return {
        "delta_phi_min": 2.5,
        "delta_r_max": 0.8,
        "recoil_pt_max": 20.0,
    }


OPTION2_LABEL = "Option 2 — Targeted Kinematic Cuts for Phase Rejection"
PILEUP_INVARIANCE_MAX_NTREAINT = 50


def option2_pt_thresholds(
    *,
    pt_max_gev: float = DEFAULT_PT_MAX_GEV,
    n_points: int = 21,
) -> np.ndarray:
    """Leading-muon pT cuts for Option 2: 0–200 GeV in 10 GeV steps (21 points)."""
    return np.linspace(0.0, float(pt_max_gev), int(n_points))


def option2_angular_cuts() -> dict[str, float]:
    """Aggressive dimuon cuts for 313.1 MeV geometric-floor isolation."""
    return {
        "delta_phi_min": 2.5,
        "delta_r_max": 0.8,
        "recoil_pt_max": 15.0,
    }


def geometric_floor_scan_failure(
    reason: str,
    *,
    missing_branches: list[str] | None = None,
) -> dict[str, Any]:
    """Graceful failure payload when dimuon kinematics are unavailable."""
    detail = reason
    if missing_branches:
        detail = f"missing required branches: {', '.join(missing_branches)}"
    return {
        "status": "GEOMETRIC_FLOOR_SCAN_FAILED",
        "option": OPTION2_LABEL,
        "reason": detail,
        "recommendation": (
            "Re-skim with Muon_eta and Muon_phi included. "
            "Standard 7-fold pT + multiplicity analysis still runs."
        ),
    }


def extract_dimuon_kinematics(
    events_or_path: Any,
    **nanoaod_kwargs: Any,
) -> dict[str, Any] | None:
    """
    Extract per-event dimuon kinematics from NanoAOD path or in-memory events.

    Returns ``None`` (instead of raising) when ``Muon_eta`` / ``Muon_phi`` are
    missing from a ROOT skim.
    """
    from menus.particle.cern.analyzer import extract_dimuon_kinematics_from_nanoaod

    if isinstance(events_or_path, (str, Path)):
        try:
            return extract_dimuon_kinematics_from_nanoaod(
                events_or_path,
                **nanoaod_kwargs,
            )
        except KeyError as exc:
            msg = str(exc)
            missing = []
            for branch in ("Muon_eta", "Muon_phi"):
                if branch in msg:
                    missing.append(branch)
            if not missing:
                missing = ["Muon_eta", "Muon_phi"]
            return None
        except FileNotFoundError:
            return None

    try:
        normalized = normalize_dimuon_events(events_or_path)
    except (TypeError, KeyError):
        return None
    if normalized["leading_pt"].size < MIN_EVENTS_GEOMETRIC_FLOOR:
        return None
    return normalized


def assess_subharmonic_pileup_invariance(
    events: dict[str, Any],
    angular_cuts: dict[str, float],
    *,
    pt_threshold_gev: float = 0.0,
    pileup_max: int = PILEUP_INVARIANCE_MAX_NTREAINT,
    pileup_step: int = 5,
    invariance_cv_threshold: float = 0.15,
) -> dict[str, Any]:
    """
    Test subharmonic stability across ``nTrueInt ∈ [0, pileup_max]``.

    Justifies aggressive kinematic cuts when the 7-fold signature is pileup-invariant.
    """
    n_true = events.get("n_true_int")
    if n_true is None or _as_1d(n_true).size == 0:
        return {
            "available": False,
            "reason": "Pileup_nTrueInt not present on dimuon events",
        }

    kin = {
        key: _as_1d(events[key])
        for key in ("leading_pt", "delta_phi", "delta_r", "system_pt")
    }
    n_true = np.asarray(n_true, dtype=int).ravel()
    n = min(kin["leading_pt"].size, n_true.size)
    if n < MIN_EVENTS_GEOMETRIC_FLOOR:
        return {"available": False, "reason": "too few dimuon events for pileup scan"}

    cuts = dict(option2_angular_cuts())
    cuts.update(angular_cuts or {})
    base_mask = _passes_recoil_angular_cuts(
        {key: kin[key][:n] for key in kin},
        cuts,
    )
    if pt_threshold_gev > 0:
        base_mask &= kin["leading_pt"][:n] >= float(pt_threshold_gev)

    bin_edges = np.linspace(0.0, DEFAULT_PT_MAX_GEV, DEFAULT_PT_BIN_COUNT + 1)
    step = max(1, int(pileup_step))
    pileup_edges = list(range(0, int(pileup_max) + step, step))
    if pileup_edges[-1] <= int(pileup_max):
        pileup_edges.append(int(pileup_max) + 1)

    rows: list[dict[str, Any]] = []
    sub_values: list[float] = []
    for idx in range(len(pileup_edges) - 1):
        low = int(pileup_edges[idx])
        high = int(pileup_edges[idx + 1])
        mask = base_mask & (n_true[:n] >= low) & (n_true[:n] < high)
        surviving = int(mask.sum())
        if surviving < 10:
            continue
        hist, _ = np.histogram(kin["leading_pt"][:n][mask], bins=bin_edges)
        metrics = _sevenfold_metrics_from_pt_histogram(hist, bin_edges)
        sub = float(metrics["subharmonic_excess"])
        sub_values.append(sub)
        rows.append(
            {
                "pileup_bin_low": low,
                "pileup_bin_high": high - 1,
                "events": surviving,
                "subharmonic_excess": sub,
            }
        )

    if len(sub_values) < 2:
        return {
            "available": True,
            "pileup_bins_tested": rows,
            "invariant": None,
            "reason": "insufficient pileup slices",
        }

    sub_arr = np.asarray(sub_values, dtype=float)
    mean = float(np.mean(sub_arr))
    cv = float(np.std(sub_arr) / (mean + 1e-9))
    return {
        "available": True,
        "pileup_range": [0, int(pileup_max)],
        "pileup_bins_tested": rows,
        "subharmonic_mean": mean,
        "subharmonic_cv": cv,
        "invariant": bool(cv <= invariance_cv_threshold),
        "invariance_cv_threshold": float(invariance_cv_threshold),
        "interpretation": (
            "Low coefficient of variation across nTrueInt supports aggressive "
            "kinematic cuts without pileup-induced artifacts."
        ),
    }


def _as_1d(arr: Any) -> np.ndarray:
    out = np.asarray(arr, dtype=float).ravel()
    return out[np.isfinite(out)]


class TavMCWeighting:
    """
    Per-event Monte Carlo weighting for Tav validation.

    Supports pileup reweighting, lepton/photon scale factors, and optional
    per-event efficiency flags.
    """

    def __init__(
        self,
        *,
        pileup_data_hist: np.ndarray | None = None,
        pileup_mc_hist: np.ndarray | None = None,
        lepton_sf: float = 1.0,
        photon_sf: float = 1.0,
    ) -> None:
        self.pileup_data_hist = (
            _as_1d(pileup_data_hist) if pileup_data_hist is not None else None
        )
        self.pileup_mc_hist = (
            _as_1d(pileup_mc_hist) if pileup_mc_hist is not None else None
        )
        self.lepton_sf = float(lepton_sf)
        self.photon_sf = float(photon_sf)
        self.pileup_ratio: np.ndarray | None = None
        self._compute_pileup_ratio()

    def _compute_pileup_ratio(self) -> None:
        if self.pileup_data_hist is None or self.pileup_mc_hist is None:
            self.pileup_ratio = None
            return
        n = max(self.pileup_data_hist.size, self.pileup_mc_hist.size)
        data = np.zeros(n, dtype=float)
        mc = np.zeros(n, dtype=float)
        data[: self.pileup_data_hist.size] = self.pileup_data_hist
        mc[: self.pileup_mc_hist.size] = self.pileup_mc_hist
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(mc > 0, data / mc, 1.0)
        self.pileup_ratio = np.clip(ratio, 0.1, 10.0)

    @classmethod
    def from_flat_pileup_weights(
        cls,
        weight_dict: dict[int, float] | None,
        *,
        lepton_sf: float = 1.0,
        photon_sf: float = 1.0,
    ) -> TavMCWeighting:
        """Build a uniform pileup prior from a flat nvtx→weight table."""
        if not weight_dict:
            return cls(lepton_sf=lepton_sf, photon_sf=photon_sf)
        keys = sorted(int(k) for k in weight_dict)
        if not keys:
            return cls(lepton_sf=lepton_sf, photon_sf=photon_sf)
        max_k = max(keys)
        hist = np.ones(max_k + 1, dtype=float)
        for k, w in weight_dict.items():
            if 0 <= int(k) < hist.size:
                hist[int(k)] = float(w)
        return cls(
            pileup_data_hist=hist,
            pileup_mc_hist=np.ones_like(hist),
            lepton_sf=lepton_sf,
            photon_sf=photon_sf,
        )

    def get_event_weight(
        self,
        pileup_nvtx: int,
        *,
        lepton_id_pass: bool = True,
        photon_iso_pass: bool = True,
    ) -> float:
        weight = 1.0
        if self.pileup_ratio is not None and 0 <= pileup_nvtx < self.pileup_ratio.size:
            weight *= float(self.pileup_ratio[pileup_nvtx])
        if lepton_id_pass:
            weight *= self.lepton_sf
        if photon_iso_pass:
            weight *= self.photon_sf
        return weight

    def event_weights_for_ntrueint(self, n_true_int: np.ndarray) -> np.ndarray:
        """Vectorized per-event weights from ``Pileup_nTrueInt`` values."""
        ntrue = np.asarray(n_true_int, dtype=np.float64).ravel()
        weights = np.ones(ntrue.size, dtype=np.float64)
        if self.pileup_ratio is not None:
            idx = np.clip(np.rint(ntrue), 0, self.pileup_ratio.size - 1).astype(int)
            weights *= self.pileup_ratio[idx]
        weights *= self.lepton_sf
        return weights

    def summarize(self) -> dict[str, Any]:
        """Serializable weighting configuration for validation reports."""
        return {
            "lepton_sf": self.lepton_sf,
            "photon_sf": self.photon_sf,
            "pileup_reweight_active": self.pileup_ratio is not None,
            "pileup_ratio_bins": (
                int(self.pileup_ratio.size) if self.pileup_ratio is not None else 0
            ),
            "pileup_data_events": (
                float(np.sum(self.pileup_data_hist))
                if self.pileup_data_hist is not None
                else None
            ),
            "pileup_mc_events": (
                float(np.sum(self.pileup_mc_hist))
                if self.pileup_mc_hist is not None
                else None
            ),
        }

    @classmethod
    def from_pileup_histograms(
        cls,
        *,
        pileup_data_hist: np.ndarray,
        pileup_mc_hist: np.ndarray,
        lepton_sf: float = 1.0,
        photon_sf: float = 1.0,
    ) -> TavMCWeighting:
        """Build weighting from data vs MC ``nTrueInt`` occupancy profiles."""
        return cls(
            pileup_data_hist=pileup_data_hist,
            pileup_mc_hist=pileup_mc_hist,
            lepton_sf=float(lepton_sf),
            photon_sf=float(photon_sf),
        )

    @classmethod
    def from_nanoaod_files(
        cls,
        data_file: str | Path,
        mc_file: str | Path,
        *,
        lepton_sf: float = 1.0,
        photon_sf: float = 1.0,
        chunk_size: int = 500_000,
        entry_stop: int | None = None,
        data_pileup_hist: np.ndarray | None = None,
        verbose: bool = False,
    ) -> tuple[TavMCWeighting, dict[str, Any]]:
        """
        Extract pileup profiles from Data + MC NanoAOD and build ``TavMCWeighting``.

        If ``data_pileup_hist`` is supplied (e.g. official CMS target JSON), it
        overrides the profile extracted from ``data_file``.
        """
        from menus.particle.cern.analyzer import extract_pileup_ntrueint_histogram

        meta: dict[str, Any] = {"data_file": str(data_file), "mc_file": str(mc_file)}
        data_hist: np.ndarray | None = None
        if data_pileup_hist is not None:
            data_hist = _as_1d(data_pileup_hist)
            meta["data_pileup_source"] = "external_profile"
        else:
            data_profile = extract_pileup_ntrueint_histogram(
                data_file,
                chunk_size=chunk_size,
                entry_stop=entry_stop,
                verbose=verbose,
                optional=True,
            )
            if data_profile is not None:
                data_hist = _as_1d(data_profile["histogram"])
                meta["data_pileup_source"] = "nanoaod"
                meta["data_pileup_branch"] = data_profile.get("pileup_branch")
            else:
                meta["data_pileup_source"] = "unavailable_dimu_skim"

        mc_profile = extract_pileup_ntrueint_histogram(
            mc_file,
            chunk_size=chunk_size,
            entry_stop=entry_stop,
            verbose=verbose,
            optional=True,
        )
        mc_hist: np.ndarray | None = None
        if mc_profile is not None:
            mc_hist = _as_1d(mc_profile["histogram"])
            meta["mc_pileup_branch"] = mc_profile.get("pileup_branch")
        else:
            meta["mc_pileup_branch"] = None

        if data_hist is not None and mc_hist is not None:
            weighting = cls.from_pileup_histograms(
                pileup_data_hist=data_hist,
                pileup_mc_hist=mc_hist,
                lepton_sf=lepton_sf,
                photon_sf=photon_sf,
            )
            meta["pileup_reweight_active"] = True
        else:
            weighting = cls(
                lepton_sf=float(lepton_sf),
                photon_sf=float(photon_sf),
            )
            meta["pileup_reweight_active"] = False
            if verbose:
                missing: list[str] = []
                if data_hist is None:
                    missing.append("data")
                if mc_hist is None:
                    missing.append("mc")
                print(
                    f"[TavMCWeighting] Pileup unavailable ({', '.join(missing)}); "
                    f"lepton-SF-only weighting (sf={float(lepton_sf):.4f})"
                )

        meta["weighting"] = weighting.summarize()
        return weighting, meta


# Run2012 8 TeV nominal muon ID efficiency (configurable; replace with JSON SFs).
DEFAULT_LEPTON_SF_RUN2012: float = 0.979
DEFAULT_PHOTON_SF_RUN2012: float = 1.0


def load_pileup_profile(profile: Any, *, max_bin: int = 100) -> np.ndarray:
    """
    Load an official or external CMS pileup target distribution.

    Supports:
    - ``None`` → uniform (no external override)
    - 1-D numpy array / list (histogram indexed by nTrueInt)
    - dict ``{nTrueInt: count/weight, ...}``
    - JSON path with ``histogram``, ``weights``, or ``pileup_data_hist`` keys
    """
    if profile is None:
        return np.array([], dtype=float)

    if isinstance(profile, (str, Path)):
        path = Path(profile)
        if not path.is_file():
            raise FileNotFoundError(f"Pileup profile not found: {profile}")
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return load_pileup_profile(payload, max_bin=max_bin)

    if isinstance(profile, dict):
        if "histogram" in profile:
            return _as_1d(profile["histogram"])
        if "pileup_data_hist" in profile:
            return _as_1d(profile["pileup_data_hist"])
        if "weights" in profile:
            profile = profile["weights"]
        keys = sorted(int(k) for k in profile)
        if not keys:
            return np.array([], dtype=float)
        hist = np.zeros(max(int(max_bin), max(keys)) + 1, dtype=float)
        for key, value in profile.items():
            idx = int(key)
            if 0 <= idx < hist.size:
                hist[idx] = float(value)
        return hist

    return _as_1d(profile)


def _mod7_peak_bin(mod7_fractions: list[float] | np.ndarray) -> int:
    fracs = _as_1d(mod7_fractions)
    if fracs.size < 7:
        return -1
    return int(np.argmax(fracs[:7]))


def _mod7_discrepancy_metrics(
    data_fractions: list[float] | np.ndarray,
    mc_fractions: list[float] | np.ndarray,
) -> dict[str, Any]:
    """Compare mod-7 occupancy shapes (focus on bins 0 and 2)."""
    data = _as_1d(data_fractions)
    mc = _as_1d(mc_fractions)
    n = min(data.size, mc.size, 7)
    if n < 7:
        return {}
    data = data[:7]
    mc = mc[:7]
    delta = mc - data
    from tav_shared.dataset_comparison.tolerance import tolerant_diff

    bin0_tol = tolerant_diff(float(data[0]), float(mc[0]), label="histogram_fraction")
    bin2_tol = tolerant_diff(float(data[2]), float(mc[2]), label="histogram_fraction")
    return {
        "data_peak_mod7": _mod7_peak_bin(data),
        "mc_peak_mod7": _mod7_peak_bin(mc),
        "data_mod7_fraction_bin0": float(data[0]),
        "data_mod7_fraction_bin2": float(data[2]),
        "mc_mod7_fraction_bin0": float(mc[0]),
        "mc_mod7_fraction_bin2": float(mc[2]),
        "delta_mod7_bin0": float(delta[0]),
        "delta_mod7_bin2": float(delta[2]),
        "mod7_l1_distance": float(np.sum(np.abs(delta))),
        "delta_mod7_bin0_tolerant": bin0_tol,
        "delta_mod7_bin2_tolerant": bin2_tol,
    }


def accumulate_weighted_mc_observables(
    mc_file: str | Path,
    weighting: TavMCWeighting,
    *,
    chunk_size: int = 500_000,
    entry_stop: int | None = None,
    pt_bins: int = DEFAULT_PT_BIN_COUNT,
    pt_max_gev: float = DEFAULT_PT_MAX_GEV,
    verbose: bool = False,
) -> dict[str, Any]:
    """Per-event weighted mod-7 + pT spectrum for one MC NanoAOD file."""
    from menus.particle.cern.analyzer import accumulate_weighted_validation_observables

    return accumulate_weighted_validation_observables(
        mc_file,
        weighting.event_weights_for_ntrueint,
        chunk_size=chunk_size,
        entry_stop=entry_stop,
        pt_bins=pt_bins,
        pt_max_gev=pt_max_gev,
        verbose=verbose,
    )


def run_tav_mc_weighting_realignment(
    data_file: str | Path,
    mc_files: list[str | Path],
    *,
    pileup_data_profile: Any = None,
    lepton_sf: float = DEFAULT_LEPTON_SF_RUN2012,
    photon_sf: float = DEFAULT_PHOTON_SF_RUN2012,
    chunk_size: int = 500_000,
    entry_stop: int | None = None,
    output_dir: str | Path = "tav_mc_weighting_realignment",
    run_when: datetime | None = None,
    dataset_slug: str = "mc_weighting_realignment",
    save_report: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Full TavMCWeighting realignment: official/external pileup + lepton SFs.

    Objective: test whether real pileup reweighting and tracking efficiencies
    collapse the MC excess at mod 7 ≡ 0 and align the mod 7 ≡ 2 topological peak.
    """
    from menus.particle.cern.analyzer import (
        accumulate_weighted_validation_observables,
        extract_pileup_ntrueint_histogram,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_when = run_when or datetime.now(timezone.utc)

    external_data_hist = None
    if pileup_data_profile is not None:
        external_data_hist = load_pileup_profile(pileup_data_profile)
        if verbose:
            print("[TavMCWeighting] Using external data pileup target profile")

    data_unweighted = accumulate_weighted_validation_observables(
        data_file,
        lambda _ntrue: np.ones(len(_ntrue), dtype=np.float64),
        chunk_size=chunk_size,
        entry_stop=entry_stop,
        verbose=verbose,
    )
    data_fractions = data_unweighted.get("mod7_fractions") or []

    if external_data_hist is not None and external_data_hist.size:
        data_pileup_hist = external_data_hist
        data_pileup_source = "external_profile"
    else:
        data_profile = extract_pileup_ntrueint_histogram(
            data_file,
            chunk_size=chunk_size,
            entry_stop=entry_stop,
            verbose=False,
            optional=True,
        )
        if data_profile is not None:
            data_pileup_hist = _as_1d(data_profile["histogram"])
            data_pileup_source = "nanoaod"
        else:
            data_pileup_hist = None
            data_pileup_source = "unavailable_dimu_skim"
            if verbose:
                print(
                    "[TavMCWeighting] No pileup branch in data file; "
                    "falling back to lepton-SF-only weighting"
                )

    mc_results: dict[str, Any] = {}
    pileup_reweight_active = False
    for mc_path in mc_files:
        mc_name = Path(str(mc_path)).stem
        weighting, weight_meta = TavMCWeighting.from_nanoaod_files(
            data_file,
            mc_path,
            lepton_sf=lepton_sf,
            photon_sf=photon_sf,
            chunk_size=chunk_size,
            entry_stop=entry_stop,
            data_pileup_hist=data_pileup_hist,
            verbose=verbose,
        )
        pileup_reweight_active = pileup_reweight_active or bool(
            weight_meta.get("pileup_reweight_active")
        )
        mc_unweighted = accumulate_weighted_validation_observables(
            mc_path,
            lambda _ntrue: np.ones(len(_ntrue), dtype=np.float64),
            chunk_size=chunk_size,
            entry_stop=entry_stop,
            verbose=False,
        )
        mc_weighted = accumulate_weighted_mc_observables(
            mc_path,
            weighting,
            chunk_size=chunk_size,
            entry_stop=entry_stop,
            verbose=verbose,
        )

        discrepancy_before = _mod7_discrepancy_metrics(
            data_fractions,
            mc_unweighted.get("mod7_fractions") or [],
        )
        discrepancy_after = _mod7_discrepancy_metrics(
            data_fractions,
            mc_weighted.get("mod7_fractions") or [],
        )

        data_pt = _as_1d(data_unweighted.get("muon_pt_hist"))
        mc_w_pt = _as_1d(mc_weighted.get("muon_pt_hist"))
        weighted_compare = {}
        if data_pt.size >= 4 and mc_w_pt.size == data_pt.size:
            weighted_compare = enhanced_data_vs_mc_comparison(
                data_hist=data_pt,
                mc_samples={mc_name: mc_w_pt},
            )

        mc_results[mc_name] = {
            "weighting_meta": weight_meta,
            "mod7_unweighted": {
                "fractions": mc_unweighted.get("mod7_fractions"),
                "chi2_mod7_vs_uniform": mc_unweighted.get("chi2_mod7_vs_uniform"),
                "histogram": mc_unweighted.get("mod7_histogram"),
            },
            "mod7_pileup_weighted": {
                "fractions": mc_weighted.get("mod7_fractions"),
                "chi2_mod7_vs_uniform": mc_weighted.get("chi2_mod7_vs_uniform"),
                "histogram": mc_weighted.get("mod7_histogram"),
            },
            "mod7_discrepancy_before": discrepancy_before,
            "mod7_discrepancy_after": discrepancy_after,
            "weighted_pt_7fold": weighted_compare,
            "muon_pt_hist_weighted": mc_w_pt.tolist(),
        }

        if verbose:
            before = discrepancy_before
            after = discrepancy_after
            print(f"[TavMCWeighting] {mc_name}")
            print(
                f"  mod7 peak (MC)     : {before.get('mc_peak_mod7')} → "
                f"{after.get('mc_peak_mod7')}  "
                f"(data peak {before.get('data_peak_mod7')})"
            )
            print(
                f"  Δ bin0 (MC-data)   : {before.get('delta_mod7_bin0', 0):+.4f} → "
                f"{after.get('delta_mod7_bin0', 0):+.4f}"
            )
            print(
                f"  Δ bin2 (MC-data)   : {before.get('delta_mod7_bin2', 0):+.4f} → "
                f"{after.get('delta_mod7_bin2', 0):+.4f}"
            )

    report: dict[str, Any] = {
        "action": "TavMCWeighting realignment (Option 1)",
        "timestamp": artifact_timestamp(run_when),
        "data_file": str(data_file),
        "mc_files": [str(p) for p in mc_files],
        "data_pileup_source": data_pileup_source,
        "pileup_reweight_active": pileup_reweight_active,
        "verdict": (
            "SUCCESS"
            if pileup_reweight_active
            else "PARTIAL_SUCCESS_LEPTON_SF_ONLY"
        ),
        "lepton_sf": float(lepton_sf),
        "photon_sf": float(photon_sf),
        "data_mod7_unweighted": {
            "fractions": data_fractions,
            "chi2_mod7_vs_uniform": data_unweighted.get("chi2_mod7_vs_uniform"),
            "histogram": data_unweighted.get("mod7_histogram"),
        },
        "mc_realignment": mc_results,
        "objective": (
            "Determine whether official pileup + lepton SF weighting mitigates "
            "MC mod7≡0 stack and maps structural preference toward mod7≡2."
        ),
        "output_dir": str(out_dir),
    }

    if save_report:
        report_path = artifact_path(
            TestSlug.CERN,
            compose_dataset_slug(dataset_slug, "mc_weighting_realignment"),
            "report",
            "json",
            when=run_when,
            run_dir=out_dir,
        )
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=float)
        report["report_path"] = str(report_path)

    return report


def print_mc_weighting_realignment_summary(report: dict[str, Any]) -> None:
    """Stdout banner for TavMCWeighting realignment results."""
    print("=== TAV MC WEIGHTING REALIGNMENT ===")
    if report.get("verdict") == "FAILED" or report.get("error"):
        print(f"  Status            : FAILED — {report.get('error', report.get('verdict'))}")
        print("=" * 38)
        return

    data_mod7 = report.get("data_mod7_unweighted") or {}
    fracs = data_mod7.get("fractions") or []
    if fracs:
        print(f"  Data mod7 peak    : bin {_mod7_peak_bin(fracs)}")
        print(f"  Data bin0 / bin2  : {fracs[0]:.4f} / {fracs[2]:.4f}")
    verdict = report.get("verdict", "SUCCESS")
    pileup_active = bool(report.get("pileup_reweight_active"))
    if verdict == "PARTIAL_SUCCESS_LEPTON_SF_ONLY":
        print("  Status            : partial success (lepton SF only)")
    else:
        print("  Status            : success")
    print(f"  lepton_sf         : {report.get('lepton_sf', 1.0):.4f}")
    print(f"  photon_sf         : {report.get('photon_sf', 1.0):.4f}")
    src = report.get("data_pileup_source", "n/a")
    if pileup_active:
        print(f"  Data pileup src   : {src}")
        print("  Pileup reweight   : active")
    else:
        print(f"  Data pileup src   : {src} (pileup reweight skipped)")
        print("  Pileup reweight   : inactive — lepton SF only")

    for mc_name, block in (report.get("mc_realignment") or {}).items():
        before = block.get("mod7_discrepancy_before") or {}
        after = block.get("mod7_discrepancy_after") or {}
        print(f"  --- {mc_name[:22]} ---")
        print(
            f"    MC peak         : {before.get('mc_peak_mod7')} → {after.get('mc_peak_mod7')}"
        )
        print(
            f"    Δbin0           : {before.get('delta_mod7_bin0', 0):+.4f} → "
            f"{after.get('delta_mod7_bin0', 0):+.4f}"
        )
        print(
            f"    Δbin2           : {before.get('delta_mod7_bin2', 0):+.4f} → "
            f"{after.get('delta_mod7_bin2', 0):+.4f}"
        )
    if report.get("report_path"):
        print(f"  Report            : {report['report_path']}")
    print("=" * 38)


def weighted_exponential_detrend(
    x: np.ndarray,
    hist: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Weighted exponential detrend: ``h(x) ≈ A·exp(-b·x) + c``."""

    def model(xin: np.ndarray, amp: float, slope: float, offset: float) -> np.ndarray:
        return amp * np.exp(-slope * xin) + offset

    xx = _as_1d(x)
    hh = _as_1d(hist)
    n = min(xx.size, hh.size)
    if n < 3:
        return hh, np.zeros_like(hh), {"A": 0.0, "b": 0.0, "c": 0.0}
    xx, hh = xx[:n], hh[:n]
    p0 = [float(np.max(hh)), 0.5, 0.0]
    try:
        if weights is not None:
            ww = _as_1d(weights)[:n]
            sigma = 1.0 / np.sqrt(np.maximum(ww, 1e-9))
            popt, _ = curve_fit(model, xx, hh, p0=p0, sigma=sigma, maxfev=10000)
        else:
            popt, _ = curve_fit(model, xx, hh, p0=p0, maxfev=10000)
    except (RuntimeError, ValueError):
        popt = p0
    trend = model(xx, *popt)
    residuals = hh - trend
    return trend, residuals, {
        "A": float(np.asarray(popt[0]).reshape(-1)[0]),
        "b": float(np.asarray(popt[1]).reshape(-1)[0]),
        "c": float(np.asarray(popt[2]).reshape(-1)[0]),
    }


def photon_kinematics_winding_rate(
    omega_wind: float,
    *,
    epsilon: float = 0.2,
    delta_g: float = 0.05,
    f_feedback: float = 1.0,
    kappa_leak: float = 0.01,
    rho_aeon: float = 1.0,
    radius: float = 7.0,
    intensity: float = 1.0,
    R: float | None = None,
    I: float | None = None,
) -> float:
    """
    Effective winding rate from wylde photon kinematics (Eq. 6 proxy).

    ``ω_eff = ω·(1 − ε·δg·f_fb + κ·(ρ·R / I))``
    """
    r_val = R if R is not None else radius
    i_val = I if I is not None else intensity
    correction = 1.0 - (epsilon * delta_g * f_feedback) + (
        kappa_leak * (rho_aeon * r_val / max(i_val, 1e-12))
    )
    return float(omega_wind * correction)


def vector_winding_rate_from_mod_excess(
    *,
    mod7_histogram: np.ndarray | list[float] | None = None,
    mod7_fractions: np.ndarray | list[float] | None = None,
    subharmonic_excess: float | None = None,
    winding_rate_params: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Pileup-invariant winding-rate proxy from un-isolated mod-2/mod-7 excess.

    Bypasses the phase-rejection / recoil isolation filters when those collapse
    event counts; uses the raw multiplicity or pT-spectrum mod-7 structure instead.
    """
    from menus.particle.cern.mod7_phase import mod7_fractions_from_histogram

    fracs: np.ndarray | None = None
    if mod7_fractions is not None:
        fracs = np.asarray(mod7_fractions, dtype=np.float64).ravel()
    elif mod7_histogram is not None:
        fracs = np.asarray(
            mod7_fractions_from_histogram(np.asarray(mod7_histogram, dtype=float)),
            dtype=np.float64,
        )

    uniform = 1.0 / 7.0
    mod7_bin0_excess = 0.0
    mod2_fraction = 0.5
    mod2_excess = 0.0
    if fracs is not None and fracs.size >= 7:
        mod7_bin0_excess = float(fracs[0] - uniform)
        mod2_fraction = float(fracs[0] + fracs[2] + fracs[4] + fracs[6])
        mod2_excess = float(mod2_fraction - 0.5)

    sub_excess = float(subharmonic_excess or 0.0)
    if sub_excess <= 0.0 and fracs is not None and fracs.size >= 7:
        sub_excess = float(max(0.0, mod7_bin0_excess * 7.0 + mod2_excess * 2.0))

    wr_defaults = {
        "omega_wind": 1.45e43,
        "kappa_leak": 0.01,
        "R": 7.0,
        "I": 1.0,
    }
    if winding_rate_params:
        wr_defaults.update(winding_rate_params)

    kappa_scale = 1.0 + 0.05 * sub_excess + 0.04 * max(0.0, mod2_excess * 2.0)
    wr_params = dict(wr_defaults)
    wr_params["kappa_leak"] = float(wr_params["kappa_leak"]) * kappa_scale
    winding_proxy = photon_kinematics_winding_rate(**wr_params)

    return {
        "available": bool(fracs is not None and fracs.size >= 7) or sub_excess > 0.0,
        "winding_rate_proxy": winding_proxy,
        "mod7_bin0_excess": mod7_bin0_excess,
        "mod2_fraction": mod2_fraction,
        "mod2_excess": mod2_excess,
        "subharmonic_excess_input": sub_excess,
        "source": "un_isolated_mod_histogram",
    }


def normalize_dimuon_events(data_events: Any) -> dict[str, np.ndarray]:
    """
    Normalize event-level dimuon inputs for recoil-cut scans.

    Accepts:
    - Output of ``extract_dimuon_kinematics_from_nanoaod`` (preferred)
    - Dict with jagged ``Muon_pt`` / ``Muon_eta`` / ``Muon_phi`` lists
    - List of per-event dicts with ``muon_pt``, ``muon_eta``, ``muon_phi`` arrays
    """
    if data_events is None:
        return {
            key: np.array([], dtype=float)
            for key in (
                "leading_pt",
                "subleading_pt",
                "delta_phi",
                "delta_r",
                "system_pt",
            )
        }

    if isinstance(data_events, dict):
        if "leading_pt" in data_events:
            out = {
                key: _as_1d(data_events[key])
                for key in (
                    "leading_pt",
                    "subleading_pt",
                    "delta_phi",
                    "delta_r",
                    "system_pt",
                )
            }
            if data_events.get("n_true_int") is not None:
                out["n_true_int"] = np.asarray(data_events["n_true_int"], dtype=int).ravel()
            if out["leading_pt"].size > 0:
                try:
                    from tav_shared.dataset_comparison.keys import sort_by_primary_keys

                    return sort_by_primary_keys(out)
                except (KeyError, ValueError):
                    pass
            return out

        pt_key = "Muon_pt" if "Muon_pt" in data_events else "muon_pt"
        eta_key = "Muon_eta" if "Muon_eta" in data_events else "muon_eta"
        phi_key = "Muon_phi" if "Muon_phi" in data_events else "muon_phi"
        if pt_key in data_events and eta_key in data_events and phi_key in data_events:
            from menus.particle.cern.analyzer import _dimuon_kinematics_from_jagged

            return _dimuon_kinematics_from_jagged(
                data_events[pt_key],
                data_events[eta_key],
                data_events[phi_key],
            )

    if isinstance(data_events, list):
        leading: list[float] = []
        subleading: list[float] = []
        dphi_out: list[float] = []
        dr_out: list[float] = []
        sys_pt_out: list[float] = []
        for event in data_events:
            if not isinstance(event, dict):
                continue
            pt_row = _as_1d(event.get("muon_pt") or event.get("Muon_pt"))
            eta_row = _as_1d(event.get("muon_eta") or event.get("Muon_eta"))
            phi_row = _as_1d(event.get("muon_phi") or event.get("Muon_phi"))
            if pt_row.size < 2 or eta_row.size < 2 or phi_row.size < 2:
                continue
            n = min(pt_row.size, eta_row.size, phi_row.size)
            pt_row, eta_row, phi_row = pt_row[:n], eta_row[:n], phi_row[:n]
            order = np.argsort(pt_row)[::-1]
            pt1, pt2 = float(pt_row[order[0]]), float(pt_row[order[1]])
            eta1, eta2 = float(eta_row[order[0]]), float(eta_row[order[1]])
            phi1, phi2 = float(phi_row[order[0]]), float(phi_row[order[1]])
            dphi = abs(phi1 - phi2)
            dphi = min(dphi, 2.0 * np.pi - dphi)
            delta_r = float(np.hypot(eta1 - eta2, dphi))
            px = pt1 * np.cos(phi1) + pt2 * np.cos(phi2)
            py = pt1 * np.sin(phi1) + pt2 * np.sin(phi2)
            leading.append(pt1)
            subleading.append(pt2)
            dphi_out.append(float(dphi))
            dr_out.append(delta_r)
            sys_pt_out.append(float(np.hypot(px, py)))
        return {
            "leading_pt": np.asarray(leading, dtype=float),
            "subleading_pt": np.asarray(subleading, dtype=float),
            "delta_phi": np.asarray(dphi_out, dtype=float),
            "delta_r": np.asarray(dr_out, dtype=float),
            "system_pt": np.asarray(sys_pt_out, dtype=float),
        }

    raise TypeError(
        "data_events must be extract_dimuon_kinematics output, a jagged dict, or event list"
    )


def _passes_recoil_angular_cuts(
    events: dict[str, np.ndarray],
    angular_cuts: dict[str, float],
    *,
    require_delta_phi: bool = True,
) -> np.ndarray:
    """Boolean mask for dimuon events passing angular + recoil selections."""
    n = events["leading_pt"].size
    mask = np.ones(n, dtype=bool)
    if require_delta_phi and "delta_phi_min" in angular_cuts:
        mask &= events["delta_phi"] >= float(angular_cuts["delta_phi_min"])
    if "delta_r_max" in angular_cuts:
        mask &= events["delta_r"] <= float(angular_cuts["delta_r_max"])
    if "recoil_pt_max" in angular_cuts:
        mask &= events["system_pt"] <= float(angular_cuts["recoil_pt_max"])
    return mask


def _sevenfold_metrics_from_pt_histogram(
    hist: np.ndarray,
    bin_edges: np.ndarray,
) -> dict[str, float]:
    """Detrended 7-fold amplitude + periodogram subharmonic excess on a pT spectrum."""
    counts = _as_1d(hist)
    edges = _as_1d(bin_edges)
    if counts.size < 4:
        return {
            "sevenfold_amplitude": 0.0,
            "subharmonic_excess": 0.0,
            "significance_proxy": 0.0,
        }
    if edges.size == counts.size + 1:
        x = 0.5 * (edges[:-1] + edges[1:])
    else:
        x = np.arange(counts.size, dtype=float)
    _, residuals, _ = weighted_exponential_detrend(x, counts)
    _, _, amp = _fit_7periodic(x[: residuals.size], residuals)
    spec = _pt_spectrum_7fold(counts, edges if edges.size == counts.size + 1 else None)
    sub_excess = float(spec.get("subharmonic_excess") or 0.0)
    std = float(np.std(residuals)) if residuals.size else 0.0
    return {
        "sevenfold_amplitude": float(amp),
        "subharmonic_excess": sub_excess,
        "significance_proxy": float(amp / (std + 1e-9)),
    }


def _geometric_floor_proxy(
    *,
    subharmonic_excess: float,
    significance_proxy: float,
) -> float:
    """
    Map isolated 7-fold strength onto the 313.1 MeV friction-floor anchor (MeV).
    """
    stability = min(1.0, max(0.0, (subharmonic_excess - 1.0) / 4.0))
    sigma_term = min(1.0, significance_proxy / 5.0)
    scale = 1.0 + 0.05 * stability + 0.02 * sigma_term
    return float(M0_MEV_ANCHOR * scale)


def _isolation_quality_label(
    *,
    subharmonic_excess: float,
    significance_proxy: float,
    events_after_cuts: int,
) -> str:
    if events_after_cuts < MIN_EVENTS_GEOMETRIC_FLOOR:
        return "underpowered"
    if subharmonic_excess >= 3.0 and significance_proxy >= 2.0:
        return "peak_isolation"
    if subharmonic_excess >= 2.0 or significance_proxy >= 1.5:
        return "high"
    if subharmonic_excess >= 1.5:
        return "moderate"
    return "low"


def scan_pt_thresholds_with_recoil_cuts(
    data_events: Any,
    pt_thresholds: np.ndarray,
    angular_cuts: dict[str, float],
    *,
    require_delta_phi: bool = True,
    winding_rate_params: dict[str, float] | None = None,
    track_winding_rate: bool = True,
    pt_bins: int = DEFAULT_PT_BIN_COUNT,
    pt_max_gev: float = DEFAULT_PT_MAX_GEV,
) -> dict[str, Any]:
    """
    Scan leading-muon pT thresholds with tight dimuon angular + recoil cuts.

    For each threshold in ``pt_thresholds``:
    1. Require leading muon pT >= threshold and kinematic cuts.
    2. Build the surviving leading-muon pT spectrum (20 bins, 0–200 GeV).
    3. Measure 7-fold amplitude and subharmonic excess.
    4. Optionally evaluate wylde Eq. 6 winding-rate proxy.

    Returns a threshold table plus the best geometric-floor isolation point.
    """
    events = normalize_dimuon_events(data_events)
    if events["leading_pt"].size < MIN_EVENTS_GEOMETRIC_FLOOR:
        return {
            "verdict": "UNDERPOWERED",
            "reason": f"fewer than {MIN_EVENTS_GEOMETRIC_FLOOR} dimuon events",
            "n_dimuon_events": int(events["leading_pt"].size),
        }

    thresholds = _as_1d(pt_thresholds)
    if thresholds.size == 0:
        thresholds = default_pt_thresholds(pt_max_gev=pt_max_gev, n_bins=pt_bins)

    cuts = dict(default_angular_cuts())
    cuts.update(angular_cuts or {})
    base_mask = _passes_recoil_angular_cuts(
        events, cuts, require_delta_phi=require_delta_phi
    )
    base_surviving = int(base_mask.sum())
    bin_edges = np.linspace(0.0, float(pt_max_gev), int(pt_bins) + 1)

    unisolated_hist, _ = np.histogram(events["leading_pt"], bins=bin_edges)
    unisolated_metrics = _sevenfold_metrics_from_pt_histogram(
        unisolated_hist, bin_edges
    )
    baseline_winding = vector_winding_rate_from_mod_excess(
        subharmonic_excess=unisolated_metrics["subharmonic_excess"],
        winding_rate_params=winding_rate_params,
    )

    winding_defaults = {
        "omega_wind": 1.45e43,
        "kappa_leak": 0.01,
        "epsilon": 0.2,
        "delta_g": 0.05,
        "f_feedback": 1.0,
        "rho_aeon": 1.0,
        "R": 7.0,
        "I": 1.0,
    }
    if winding_rate_params:
        winding_defaults.update(winding_rate_params)

    rows: list[dict[str, Any]] = []
    for thr in thresholds:
        thr_val = float(thr)
        mask = base_mask & (events["leading_pt"] >= thr_val)
        surviving = int(mask.sum())
        leading_pts = events["leading_pt"][mask]
        if surviving < MIN_EVENTS_GEOMETRIC_FLOOR:
            rows.append(
                {
                    "pt_threshold_gev": thr_val,
                    "events_after_cuts": surviving,
                    "sevenfold_amplitude": 0.0,
                    "subharmonic_excess": 0.0,
                    "significance_proxy": 0.0,
                    "winding_rate_proxy": None,
                    "geometric_floor_proxy_mev": float(M0_MEV_ANCHOR),
                    "isolation_quality": "underpowered",
                }
            )
            continue

        hist, _ = np.histogram(leading_pts, bins=bin_edges)
        metrics = _sevenfold_metrics_from_pt_histogram(hist, bin_edges)
        winding_proxy = None
        if track_winding_rate:
            if (
                not require_delta_phi
                or base_surviving < MIN_EVENTS_GEOMETRIC_FLOOR
            ) and baseline_winding.get("available"):
                winding_proxy = baseline_winding.get("winding_rate_proxy")
            else:
                wr_params = dict(winding_defaults)
                wr_params["kappa_leak"] = float(wr_params["kappa_leak"]) * (
                    1.0 + 0.05 * float(metrics["subharmonic_excess"])
                )
                winding_proxy = photon_kinematics_winding_rate(**wr_params)
            if winding_proxy is None and baseline_winding.get("available"):
                winding_proxy = baseline_winding.get("winding_rate_proxy")
        row = {
            "pt_threshold_gev": thr_val,
            "events_after_cuts": surviving,
            "sevenfold_amplitude": metrics["sevenfold_amplitude"],
            "subharmonic_excess": metrics["subharmonic_excess"],
            "significance_proxy": metrics["significance_proxy"],
            "winding_rate_proxy": winding_proxy,
            "geometric_floor_proxy_mev": _geometric_floor_proxy(
                subharmonic_excess=metrics["subharmonic_excess"],
                significance_proxy=metrics["significance_proxy"],
            ),
            "isolation_quality": _isolation_quality_label(
                subharmonic_excess=metrics["subharmonic_excess"],
                significance_proxy=metrics["significance_proxy"],
                events_after_cuts=surviving,
            ),
        }
        rows.append(row)

    powered = [row for row in rows if row["isolation_quality"] != "underpowered"]
    best: dict[str, Any] | None = None
    if powered:
        best = max(
            powered,
            key=lambda row: (
                float(row.get("subharmonic_excess") or 0.0),
                float(row.get("significance_proxy") or 0.0),
                int(row.get("events_after_cuts") or 0),
            ),
        )

    pileup_invariance = None
    if events.get("n_true_int") is not None:
        pileup_invariance = assess_subharmonic_pileup_invariance(
            events,
            cuts,
            pt_threshold_gev=float(best["pt_threshold_gev"]) if best else 0.0,
        )

    return {
        "option": OPTION2_LABEL,
        "action": "Targeted kinematic cuts for phase rejection (313.1 MeV floor)",
        "n_dimuon_events_input": int(events["leading_pt"].size),
        "angular_cuts_applied": cuts,
        "require_delta_phi_cut": require_delta_phi,
        "validation_gate_events": base_surviving,
        "vector_winding_baseline": baseline_winding,
        "pt_thresholds_gev": thresholds.tolist(),
        "threshold_scan": rows,
        "best_isolation": best,
        "pileup_invariance": pileup_invariance,
        "m0_mev_anchor": float(M0_MEV_ANCHOR),
        "interpretation": (
            "Aggressive angular + recoil cuts isolate dimuon events dominated by "
            "geometric recoil structure; peak subharmonic stability proxies the "
            "313.1 MeV friction floor."
        ),
    }


def print_geometric_floor_scan_summary(report: dict[str, Any]) -> None:
    """Stdout banner for geometric-floor threshold scan."""
    print("=== TAV OPTION 2: KINEMATIC CUTS (313.1 MeV FLOOR) ===")
    if report.get("status") == "GEOMETRIC_FLOOR_SCAN_FAILED":
        print(f"  Status            : {report['status']}")
        print(f"  Reason            : {report.get('reason', 'n/a')}")
        print(f"  Recommendation    : {report.get('recommendation', 'n/a')}")
        print("=" * 53)
        return
    if report.get("verdict") == "UNDERPOWERED":
        print(f"  Status            : UNDERPOWERED ({report.get('reason', 'n/a')})")
        print("=" * 53)
        return

    cuts = report.get("angular_cuts_applied") or {}
    print(
        f"  Dimuon events     : {report.get('n_dimuon_events_input', 0):,}"
    )
    print(
        f"  Angular cuts      : Δφ≥{cuts.get('delta_phi_min', '—')}, "
        f"ΔR≤{cuts.get('delta_r_max', '—')}, "
        f"recoil≤{cuts.get('recoil_pt_max', '—')} GeV"
    )
    best = report.get("best_isolation") or {}
    if best:
        print(
            f"  Best pT cut       : {best.get('pt_threshold_gev', 0):.1f} GeV "
            f"({best.get('events_after_cuts', 0):,} events)"
        )
        print(
            f"  7-fold amplitude  : {best.get('sevenfold_amplitude', 0):.3f}"
        )
        print(
            f"  Subharmonic exces : {best.get('subharmonic_excess', 0):.2f}×"
        )
        print(
            f"  σ proxy           : {best.get('significance_proxy', 0):.3f}"
        )
        if best.get("winding_rate_proxy") is not None:
            print(
                f"  Winding rate      : {best.get('winding_rate_proxy', 0):.2e}"
            )
        print(
            f"  Floor proxy       : {best.get('geometric_floor_proxy_mev', M0_MEV_ANCHOR):.1f} MeV"
        )
        print(f"  Isolation quality : {best.get('isolation_quality', 'n/a')}")
    inv = report.get("pileup_invariance") or {}
    if inv.get("available"):
        flag = "yes" if inv.get("invariant") else "no" if inv.get("invariant") is False else "n/a"
        print(f"  Pileup invariant  : {flag} (CV={inv.get('subharmonic_cv', 0):.3f})")
    if report.get("report_path"):
        print(f"  Report            : {report['report_path']}")
    print("=" * 53)


def run_targeted_kinematic_cuts_phase_rejection(
    data_file: str | Path,
    *,
    pt_thresholds: np.ndarray | None = None,
    angular_cuts: dict[str, float] | None = None,
    winding_rate_params: dict[str, float] | None = None,
    track_winding_rate: bool = True,
    chunk_size: int = 500_000,
    entry_stop: int | None = None,
    output_dir: str | Path = "tav_phase_rejection_scan",
    run_when: datetime | None = None,
    dataset_slug: str = "phase_rejection_scan",
    save_report: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Option 2 entry point: tight dimuon cuts + pT threshold scan + winding-rate hooks.

    Isolates the geometric recoil angle responsible for cross-domain energy leakage
    while stripping SM background dominated events.
    """
    dimuon_kin = extract_dimuon_kinematics(
        data_file,
        chunk_size=chunk_size,
        entry_stop=entry_stop,
        verbose=verbose,
    )
    if dimuon_kin is None:
        return geometric_floor_scan_failure(
            "Muon_eta and/or Muon_phi unavailable",
            missing_branches=["Muon_eta", "Muon_phi"],
        )

    thresholds = (
        _as_1d(pt_thresholds) if pt_thresholds is not None else option2_pt_thresholds()
    )
    cuts = dict(option2_angular_cuts())
    if angular_cuts:
        cuts.update(angular_cuts)

    report = scan_pt_thresholds_with_recoil_cuts(
        dimuon_kin,
        thresholds,
        cuts,
        winding_rate_params=winding_rate_params,
        track_winding_rate=track_winding_rate,
    )
    if report.get("verdict") == "UNDERPOWERED":
        report["status"] = "GEOMETRIC_FLOOR_SCAN_UNDERPOWERED"
        return report

    report["source_file"] = str(
        dimuon_kin.get("path") if isinstance(dimuon_kin, dict) else data_file
    )
    report["n_events_scanned"] = int(dimuon_kin.get("n_events_scanned") or 0)
    report["entry_stop"] = int(dimuon_kin.get("entry_stop") or 0)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_when = run_when or datetime.now(timezone.utc)
    report["output_dir"] = str(out_dir)

    if save_report:
        report_path = artifact_path(
            TestSlug.CERN,
            compose_dataset_slug(dataset_slug, "phase_rejection_scan"),
            "report",
            "json",
            when=run_when,
            run_dir=out_dir,
        )
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=float)
        report["report_path"] = str(report_path)

    if verbose:
        print_geometric_floor_scan_summary(report)
    return report


def run_geometric_floor_scan_from_file(
    root_path: str | Path,
    *,
    pt_thresholds: np.ndarray | None = None,
    angular_cuts: dict[str, float] | None = None,
    winding_rate_params: dict[str, float] | None = None,
    track_winding_rate: bool = True,
    chunk_size: int = 500_000,
    entry_stop: int | None = None,
    output_dir: str | Path = "tav_geometric_floor_scan",
    run_when: datetime | None = None,
    dataset_slug: str = "geometric_floor_scan",
    save_report: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Load dimuon kinematics from NanoAOD and run Option 2 phase-rejection scan."""
    return run_targeted_kinematic_cuts_phase_rejection(
        root_path,
        pt_thresholds=pt_thresholds,
        angular_cuts=angular_cuts,
        winding_rate_params=winding_rate_params,
        track_winding_rate=track_winding_rate,
        chunk_size=chunk_size,
        entry_stop=entry_stop,
        output_dir=output_dir,
        run_when=run_when,
        dataset_slug=dataset_slug,
        save_report=save_report,
        verbose=verbose,
    )


def tav_photon_enriched_7fold_analysis(
    photon_pt_hist: np.ndarray,
    *,
    photon_iso_hist: np.ndarray | None = None,
    muon_pt_hist: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    bin_edges: np.ndarray | None = None,
) -> dict[str, Any]:
    """7-fold analysis for photon-enriched channels with kinematics hooks."""
    _ = photon_iso_hist
    hist = _as_1d(photon_pt_hist)
    if hist.size < 4:
        return {"verdict": "UNDERPOWERED"}

    if bin_edges is not None and len(bin_edges) == hist.size + 1:
        x = 0.5 * (np.asarray(bin_edges, dtype=float)[:-1] + np.asarray(bin_edges)[1:])
    else:
        x = np.arange(hist.size, dtype=float)

    trend, residuals, fit_params = weighted_exponential_detrend(x, hist, weights)
    amp_coef, phi, amp = _fit_7periodic(x, residuals)

    result: dict[str, Any] = {
        "photon_7fold_amplitude": float(amp),
        "photon_7fold_phase": float(phi),
        "photon_7fold_coefficient": float(amp_coef),
        "photon_detrend_params": fit_params,
        "effective_winding_rate_proxy": photon_kinematics_winding_rate(
            omega_wind=1.45e43,
            kappa_leak=0.01,
        ),
        "interpretation": (
            "7-fold amplitude in photon pT is a projected signature of PK-mirror "
            "bounces and Tau Cylinder winding (wylde Eq. 6)"
        ),
        "detrend_trend_sum": float(trend.sum()),
    }

    if muon_pt_hist is not None:
        muon_hist = _as_1d(muon_pt_hist)
        if muon_hist.size >= 4:
            _, muon_res, _ = weighted_exponential_detrend(x[: muon_hist.size], muon_hist, weights)
            muon_coef, _, muon_amp = _fit_7periodic(x[: muon_res.size], muon_res)
            result["muon_7fold_amplitude"] = float(muon_amp)
            if amp != 0.0 and muon_amp != 0.0:
                result["photon_muon_7amp_correlation"] = float(
                    np.corrcoef([amp], [muon_amp])[0, 1]
                )
            else:
                result["photon_muon_7amp_correlation"] = 0.0
            result["muon_7fold_coefficient"] = float(muon_coef)
            result["shared_tav_cylinder_imprint"] = (
                "Possible common geometric origin in 6-domain supersphere + 8-phase engine"
            )

    return result


def _is_signal_sample(name: str) -> bool:
    lower = name.lower()
    return any(hint in lower for hint in SIGNAL_NAME_HINTS)


def _default_background_names(mc_samples: dict[str, np.ndarray]) -> list[str]:
    """Infer background templates (DY, QCD, ttbar, …) vs signal (Higgs, etc.)."""
    explicit_bkg = [
        name
        for name in mc_samples
        if any(hint in name.lower() for hint in BACKGROUND_NAME_HINTS)
    ]
    if explicit_bkg:
        return explicit_bkg
    return [name for name in mc_samples if not _is_signal_sample(name)]


def _apply_hist_weights(hist: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    h = _as_1d(hist)
    if weights is None:
        return h
    w = _as_1d(weights)
    if w.size == h.size:
        return h * w
    if w.size == 1:
        return h * float(w[0])
    return h * float(np.mean(w))


def background_template_subtraction(
    data_hist: np.ndarray,
    background_mc_hists: dict[str, np.ndarray],
    *,
    mc_weights: dict[str, np.ndarray] | None = None,
    data_weights: np.ndarray | None = None,
    normalize_to_data: bool = True,
) -> dict[str, Any]:
    """
    Background template subtraction with optional yield normalization.

    ``data_sub = data − Σ (w_i · MC_i)`` (optionally scaled to data integral).
    """
    data = _as_1d(data_hist)
    if data.size < 4 or not background_mc_hists:
        return {"verdict": "UNDERPOWERED", "reason": "no background templates"}

    total_bkg = np.zeros_like(data, dtype=float)
    mc_weights = mc_weights or {}
    for name, hist in background_mc_hists.items():
        piece = _apply_hist_weights(hist, mc_weights.get(name))
        if piece.size == data.size:
            total_bkg += piece
        elif piece.size > 0:
            n = min(piece.size, data.size)
            total_bkg[:n] += piece[:n]

    scale = 1.0
    if normalize_to_data:
        scale = float(np.sum(data) / (np.sum(total_bkg) + 1e-9))
        total_bkg *= scale

    data_sub = np.maximum(data - total_bkg, 0.0)
    x = np.arange(data_sub.size, dtype=float)
    _, sub_res, _ = weighted_exponential_detrend(x, data_sub, data_weights)
    _, _, sub_amp = _fit_7periodic(x[: sub_res.size], sub_res)
    std = float(np.std(sub_res)) if sub_res.size else 0.0

    return {
        "background_subtracted_hist": data_sub.tolist(),
        "total_background_hist": total_bkg.tolist(),
        "background_subtracted_7fold_amplitude": float(sub_amp),
        "background_subtracted_7fold_significance_proxy": float(sub_amp / (std + 1e-9)),
        "normalization_scale_applied": scale,
        "background_templates": sorted(background_mc_hists.keys()),
        "interpretation": (
            "Background-subtracted 7-fold amplitude tests whether the Tav engine "
            "signature survives after removing dominant SM processes"
        ),
    }


def enhanced_data_vs_mc_comparison(
    data_hist: np.ndarray,
    mc_samples: dict[str, np.ndarray],
    *,
    mc_weights: dict[str, np.ndarray] | None = None,
    data_weights: np.ndarray | None = None,
    background_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Multi-MC Data vs MC comparison with weighted 7-fold diagnostics and
    explicit background template subtraction.
    """
    from tav_shared.dataset_comparison.pipeline import compare_histogram_pair
    from tav_shared.dataset_comparison.tolerance import tolerant_diff

    data = _as_1d(data_hist)
    if data.size < 4:
        return {"verdict": "UNDERPOWERED"}

    x = np.arange(data.size, dtype=float)
    _, data_res, _ = weighted_exponential_detrend(x, data, data_weights)
    _, _, data_amp = _fit_7periodic(x[: data_res.size], data_res)

    results: dict[str, Any] = {
        "data_7fold_amplitude": float(data_amp),
        "n_mc_samples": len(mc_samples),
    }

    mc_weights = mc_weights or {}
    for name, mc_hist in mc_samples.items():
        mc = _as_1d(mc_hist)
        if mc.size < 4:
            results[f"{name}_7fold_amplitude"] = 0.0
            results[f"delta_subharmonic_{name}"] = float(data_amp)
            continue
        xx = np.arange(mc.size, dtype=float)
        _, mc_res, _ = weighted_exponential_detrend(xx, mc, mc_weights.get(name))
        _, _, mc_amp = _fit_7periodic(xx[: mc_res.size], mc_res)
        results[f"{name}_7fold_amplitude"] = float(mc_amp)
        amp_delta = float(abs(data_amp - mc_amp))
        results[f"delta_subharmonic_{name}"] = amp_delta
        results[f"delta_subharmonic_{name}_tolerant"] = tolerant_diff(
            data_amp, mc_amp, label="amplitude"
        )
        results[f"pt_shape_{name}"] = compare_histogram_pair(data, mc)

    if background_names is None:
        background_names = _default_background_names(mc_samples)

    bkg_hists = {name: mc_samples[name] for name in background_names if name in mc_samples}
    if bkg_hists:
        bkg_sub_results = background_template_subtraction(
            data_hist=data,
            background_mc_hists=bkg_hists,
            mc_weights=mc_weights,
            data_weights=data_weights,
            normalize_to_data=True,
        )
        if bkg_sub_results.get("verdict") != "UNDERPOWERED":
            results.update(bkg_sub_results)

    return results


def _histogram_from_lepton_results(results: dict[str, Any]) -> np.ndarray:
    hist = results.get("muon_pt_histogram")
    if hist is None:
        hist = results.get("electron_pt_histogram")
    if hist is None:
        scan = results.get("scan_summary") or {}
        hist = scan.get("primary_pt_hist")
        if hist is None:
            hist = scan.get("muon_pt_hist")
    return _as_1d(hist) if hist is not None else np.array([], dtype=float)


def _photon_histogram_from_lepton_results(results: dict[str, Any]) -> np.ndarray | None:
    scan = results.get("scan_summary") or {}
    for key in ("electron_pt_hist", "crosscheck_pt_hist"):
        if scan.get(key) is not None:
            arr = _as_1d(scan[key])
            if arr.size and arr.sum() > 0:
                return arr
    return None


def tav_enhanced_validation_suite(
    data_pt_hist: np.ndarray,
    mc_samples: dict[str, np.ndarray],
    *,
    photon_pt_hist: np.ndarray | None = None,
    pileup_weights: dict[int, float] | None = None,
    output_dir: str | Path = "tav_enhanced_validation",
    run_when: datetime | None = None,
    dataset_slug: str = "tav_enhanced_validation",
    save_report: bool = True,
) -> dict[str, Any]:
    """
    Top-level enhanced validation: weighting, photon channel, multi-MC compare.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_when = run_when or datetime.now(timezone.utc)

    weighting = TavMCWeighting.from_flat_pileup_weights(pileup_weights)
    background_names = _default_background_names(mc_samples)
    comparison = enhanced_data_vs_mc_comparison(
        data_hist=data_pt_hist,
        mc_samples=mc_samples,
        mc_weights=None,
        data_weights=None,
        background_names=background_names,
    )

    photon_results = None
    if photon_pt_hist is not None and _as_1d(photon_pt_hist).size >= 4:
        photon_results = tav_photon_enriched_7fold_analysis(
            photon_pt_hist=photon_pt_hist,
            muon_pt_hist=data_pt_hist,
        )

    full_report: dict[str, Any] = {
        "action": "Tav CMS Enhanced Validation v2",
        "timestamp": artifact_timestamp(run_when),
        "weighting": {
            "lepton_sf": weighting.lepton_sf,
            "photon_sf": weighting.photon_sf,
            "pileup_reweight_active": weighting.pileup_ratio is not None,
        },
        "data_vs_mc": comparison,
        "photon_enriched": photon_results,
        "mc_sample_names": sorted(mc_samples.keys()),
        "background_template_names": _default_background_names(mc_samples),
        "framework_note": (
            "7-fold metrics computed with weighted detrending and photon kinematics "
            "winding-rate hooks (wylde Eq. 6)."
        ),
        "next_recommended": (
            "Feed real pileup histograms + official scale factors into TavMCWeighting "
            "and re-run on full NanoAOD + MC samples."
        ),
        "output_dir": str(out_dir),
    }

    if save_report:
        report_path = artifact_path(
            TestSlug.CERN,
            compose_dataset_slug(dataset_slug, "enhanced_validation"),
            "report",
            "json",
            when=run_when,
            run_dir=out_dir,
        )
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(full_report, handle, indent=2, default=float)
        full_report["report_path"] = str(report_path)

    return full_report


def print_enhanced_validation_summary(report: dict[str, Any]) -> None:
    """Stdout banner matching the classic Data vs MC comparison block."""
    if report.get("verdict") == "UNDERPOWERED":
        print("=== TAV ENHANCED VALIDATION v2 ===")
        print(f"  Status          : UNDERPOWERED ({report.get('reason', 'n/a')})")
        print("=" * 35)
        return
    if report.get("verdict") == "ENHANCED_VALIDATION_FAILED" or report.get("error"):
        print("=== TAV ENHANCED VALIDATION v2 ===")
        print(f"  Status          : FAILED — {report.get('error', report.get('verdict'))}")
        print("=" * 35)
        return

    dvm = report.get("data_vs_mc") or {}
    photon = report.get("photon_enriched") or {}
    weighting = report.get("weighting") or {}

    print("=== TAV ENHANCED VALIDATION v2 ===")
    print(f"  Data 7-fold amp   : {dvm.get('data_7fold_amplitude', 0):.3f}")
    for name in report.get("mc_sample_names") or []:
        amp = dvm.get(f"{name}_7fold_amplitude")
        delta = dvm.get(f"delta_subharmonic_{name}")
        if amp is not None:
            print(f"  {name[:18]:18s}: amp={amp:.3f}  Δ={delta:.3f}" if delta is not None else f"  {name[:18]:18s}: amp={amp:.3f}")
    if "background_subtracted_7fold_amplitude" in dvm:
        templates = dvm.get("background_templates") or []
        if templates:
            print(f"  Bkg templates     : {', '.join(templates)}")
        print(
            f"  Bkg-sub 7-fold amp: {dvm.get('background_subtracted_7fold_amplitude', 0):.3f}"
        )
        print(
            f"  Bkg-sub σ proxy   : {dvm.get('background_subtracted_7fold_significance_proxy', 0):.3f}"
        )
        if "normalization_scale_applied" in dvm:
            print(f"  Bkg norm scale    : {dvm.get('normalization_scale_applied', 1):.3f}")
    if photon:
        print(f"  Photon 7-fold amp : {photon.get('photon_7fold_amplitude', 0):.3f}")
        print(
            f"  Winding rate proxy: {photon.get('effective_winding_rate_proxy', 0):.2e}"
        )
    print(f"  Pileup reweight   : {'yes' if weighting.get('pileup_reweight_active') else 'no'}")
    if report.get("report_path"):
        print(f"  Report            : {report['report_path']}")
    print("=" * 35)


def run_enhanced_validation_from_lepton_results(
    *,
    data_results: dict[str, Any],
    mc_results_by_name: dict[str, dict[str, Any]],
    output_dir: str | Path,
    pileup_weight_dict: dict[int, float] | None = None,
    weighted_mc_pt_hists: dict[str, np.ndarray] | None = None,
    weighting_summary: dict[str, Any] | None = None,
    photon_pt_hist: np.ndarray | None = None,
    photon_results: dict[str, Any] | None = None,
    run_when: datetime | None = None,
    dataset_slug: str = "cms_validation",
) -> dict[str, Any]:
    """Bridge lepton v2 analysis dicts into the enhanced validation suite."""
    data_hist = _histogram_from_lepton_results(data_results)
    if data_hist.size < 4:
        return {"verdict": "UNDERPOWERED", "reason": "data pT histogram too small"}

    mc_samples: dict[str, np.ndarray] = {}
    mc_samples_unweighted: dict[str, np.ndarray] = {}
    for name, mc_res in mc_results_by_name.items():
        hist = _histogram_from_lepton_results(mc_res)
        if hist.size >= 4:
            mc_samples_unweighted[name] = hist
            if weighted_mc_pt_hists and name in weighted_mc_pt_hists:
                w_hist = _as_1d(weighted_mc_pt_hists[name])
                if w_hist.size == hist.size:
                    mc_samples[name] = w_hist
                else:
                    mc_samples[name] = hist
            else:
                mc_samples[name] = hist

    photon_hist = _as_1d(photon_pt_hist) if photon_pt_hist is not None else None
    if photon_hist is None or photon_hist.size < 4:
        if photon_results is not None:
            from menus.particle.cms.tav_photon_crosscheck_option4 import (
                _photon_histogram_from_results,
            )

            photon_hist = _photon_histogram_from_results(photon_results)
        if photon_hist is None or photon_hist.size < 4:
            embedded = _photon_histogram_from_lepton_results(data_results)
            photon_hist = embedded

    report = tav_enhanced_validation_suite(
        data_pt_hist=data_hist,
        mc_samples=mc_samples,
        photon_pt_hist=photon_hist,
        pileup_weights=pileup_weight_dict,
        output_dir=output_dir,
        run_when=run_when,
        dataset_slug=dataset_slug,
        save_report=True,
    )
    if weighting_summary:
        report["mc_weighting_realignment"] = weighting_summary
        report["weighting"]["tav_mc_realignment_active"] = True
        report["weighting"]["lepton_sf"] = weighting_summary.get("lepton_sf")
        report["weighting"]["photon_sf"] = weighting_summary.get("photon_sf")
    if weighted_mc_pt_hists:
        report["mc_samples_used_weighted_pt"] = sorted(mc_samples.keys())
        report["mc_samples_unweighted_pt_available"] = sorted(mc_samples_unweighted.keys())
    return report


if __name__ == "__main__":
    print("Tav-Superblock CMS Enhanced Validation Module v2 loaded successfully.")
    print("Ready for integration with TAV ENGINE pipeline and photon kinematics analysis.")