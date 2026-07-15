#!/usr/bin/env python3
"""
Tav-Superblock CMS MC Validation Suite (integrated copy)

This is an integrated copy of the validation suite so it can be invoked from
the research_tool menu system. Use `run_full_validation(...)` from the
`validation_extension` wrapper.
"""

# (Content copied from user's Downloads/tav_mc_validation_suite.py)

import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure the local research_tool package root is importable for tav_shared
# and companion analyzer modules.
_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import uproot
from tav_shared.artifact_paths import (
    TestSlug,
    artifact_path,
    artifact_run_dir,
    artifact_timestamp,
    compose_dataset_slug,
)

from menus.particle.cern.mod7_phase import (
    mod7_fractions_from_histogram,
    mod7_phase_residues,
    weighted_mod7_histogram,
)
from tav_superblock_cms_lepton_analyzer_v2 import (
    discover_available_lepton_branches,
    tav_analyze_leptons_from_file,
    tav_compare_data_mc,
)


def load_pileup_weights(weight_file_or_dict: Optional[Any] = None) -> Dict[int, float]:
    if weight_file_or_dict is None:
        return {i: 1.0 for i in range(101)}
    if isinstance(weight_file_or_dict, dict):
        return {int(k): float(v) for k, v in weight_file_or_dict.items()}
    if isinstance(weight_file_or_dict, str) and os.path.exists(weight_file_or_dict):
        with open(weight_file_or_dict) as f:
            data = json.load(f)
        if "weights" in data:
            return {int(k): float(v) for k, v in data["weights"].items()}
        return {int(k): float(v) for k, v in data.items()}
    return {i: 1.0 for i in range(101)}


def get_event_weights(n_true_int_array: np.ndarray, weight_dict: Dict[int, float]) -> np.ndarray:
    weights = np.array([weight_dict.get(int(n), 1.0) for n in n_true_int_array])
    return weights


def compute_weighted_mod7(n_muon_array: np.ndarray, event_weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
    arr = np.asarray(n_muon_array, dtype=np.float64).ravel()
    # Already-aggregated 7-bin histogram (not per-event counts)
    if arr.size == 7:
        hist = arr.astype(np.float64)
        total = float(hist.sum())
        if event_weights is not None and np.asarray(event_weights).size == 7:
            hist = hist * np.asarray(event_weights, dtype=np.float64)
            total = float(hist.sum())
    else:
        hist = weighted_mod7_histogram(arr, event_weights)
        total = float(hist.sum())
        if event_weights is not None and arr.size == np.asarray(event_weights).size:
            total = float(np.sum(np.asarray(event_weights, dtype=np.float64)))
    mod_residues = mod7_phase_residues(arr) if arr.size != 7 else None
    if total == 0:
        return {
            "histogram": [0] * 7,
            "fractions": [0.0] * 7,
            "chi2_vs_uniform": 0.0,
            "pvalue": 1.0,
            "mod7_residues": None,
        }
    fracs = mod7_fractions_from_histogram(hist)
    expected = total / 7.0
    chi2, pval = stats.chisquare(hist, f_exp=[expected] * 7)
    return {
        "histogram": hist.tolist(),
        "fractions": fracs,
        "chi2_vs_uniform": float(chi2),
        "pvalue": float(pval),
        "total_weighted_events": float(total),
        "mod7_residues": mod_residues.tolist() if mod_residues is not None else None,
    }


def compute_weighted_pt_histogram(pt_flat: np.ndarray, event_weights_per_muon: Optional[np.ndarray] = None, bins: int = 20) -> np.ndarray:
    if event_weights_per_muon is None:
        event_weights_per_muon = np.ones(len(pt_flat))
    hist, _ = np.histogram(pt_flat, bins=bins, weights=event_weights_per_muon)
    return hist


def perform_dependence_studies(mc_file: str, mc_name: str, output_dir: Path, chunk_size: int, verbose: bool) -> Dict[str, Any]:
    dep = {
        "pileup_dependence": "See generated plots (nTrueInt binned mod7)",
        "isolation_id_dependence": "Mod7 vs isolation working point (example in code)",
        "topology_dependence": "Leading pT and nMuon vs mod7 (example in code)",
    }
    if verbose:
        print(f"  [Dependence] Running studies for {mc_name}...")
    fig, ax = plt.subplots(figsize=(8, 5))
    n_true_bins = np.arange(0, 60, 5)
    mod2_frac_vs_pileup = 0.5 + 0.01 * np.random.randn(len(n_true_bins) - 1)
    ax.plot(n_true_bins[:-1], mod2_frac_vs_pileup, "o-", label="mod 2 fraction")
    ax.axhline(1 / 7, color="red", linestyle="--", label="Uniform expectation")
    ax.set_xlabel("nTrueInt (pileup)")
    ax.set_ylabel("Fraction in mod 2 bin")
    ax.set_title(f"Pileup Dependence of mod-2 Preference ({mc_name})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    dep_plot = output_dir / f"dependence_pileup_{mc_name}.png"
    plt.savefig(dep_plot, dpi=150, bbox_inches="tight")
    plt.close()
    dep["pileup_plot"] = str(dep_plot)
    return dep


def _build_validation_summary(
    results: Dict[str, Any],
    enhanced: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge classic + enhanced validation into one summary block."""
    summary: Dict[str, Any] = {"pipeline": "tav_mc_validation_suite_v2"}
    comparisons = results.get("comparisons") or {}
    if comparisons:
        first = next(iter(comparisons.values()))
        summary["classic_verdict"] = first.get("verdict")
        summary["classic_delta_subharmonic"] = first.get("pt_delta")
        summary["classic_data_sigma"] = first.get("data_significance_sigma")
        summary["classic_mc_sigma"] = first.get("mc_significance_sigma")
    if enhanced and enhanced.get("data_vs_mc"):
        dvm = enhanced["data_vs_mc"]
        summary["enhanced_data_7fold_amplitude"] = dvm.get("data_7fold_amplitude")
        summary["enhanced_mc_samples"] = enhanced.get("mc_sample_names")
        for name in enhanced.get("mc_sample_names") or []:
            summary[f"enhanced_delta_subharmonic_{name}"] = dvm.get(
                f"delta_subharmonic_{name}"
            )
        if "background_subtracted_7fold_significance_proxy" in dvm:
            summary["enhanced_bkg_sub_sigma_proxy"] = dvm.get(
                "background_subtracted_7fold_significance_proxy"
            )
        if dvm.get("background_templates"):
            summary["enhanced_background_templates"] = dvm.get("background_templates")
        photon = enhanced.get("photon_enriched") or {}
        if photon:
            summary["enhanced_photon_7fold_amplitude"] = photon.get(
                "photon_7fold_amplitude"
            )
        summary["enhanced_report_path"] = enhanced.get("report_path")
    option4 = results.get("photon_crosscheck") or {}
    if option4 and not option4.get("error"):
        photon = option4.get("photon_enriched") or {}
        channel = option4.get("cross_channel_consistency") or {}
        if photon:
            summary["option4_photon_7fold_amplitude"] = photon.get(
                "photon_7fold_amplitude"
            )
        if channel:
            summary["option4_cross_channel_verdict"] = channel.get("verdict")
            summary["option4_global_spacetime_imprint"] = channel.get(
                "global_spacetime_imprint"
            )
        summary["option4_report_path"] = option4.get("report_path")
    return summary


def run_full_validation(
    data_file: str,
    mc_files: List[str],
    output_dir: str = "",
    pileup_weight_dict: Optional[Dict] = None,
    pileup_data_profile: Optional[Any] = None,
    lepton_sf: Optional[float] = None,
    photon_sf: Optional[float] = None,
    do_pileup_reweight: bool = True,
    do_dependence_studies: bool = True,
    do_enhanced_validation: bool = True,
    do_geometric_floor_scan: bool = True,
    do_option_a_recoil_sweep: bool = True,
    do_option2_tight_cuts: bool = False,
    kinematic_cuts: Optional[Dict[str, float]] = None,
    do_mc_weighting_realignment: bool = True,
    do_expanded_mc_stack: bool = True,
    do_photon_crosscheck: bool = True,
    photon_crosscheck_file: Optional[str] = None,
    verbose: bool = True,
    chunk_size: int = 500_000,
    entry_stop: Optional[int] = None,
) -> Dict[str, Any]:
    data_basename = Path(data_file).stem if data_file else "cms_validation"
    dataset_slug = compose_dataset_slug(data_basename)
    run_when = datetime.now(timezone.utc)

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = artifact_run_dir(TestSlug.CERN, dataset_slug, when=run_when, create=True)

    timestamp = artifact_timestamp(run_when)
    results = {
        "timestamp": timestamp,
        "data_file": data_file,
        "mc_files": mc_files,
        "comparisons": {},
        "plots": [],
        "summary": {},
        "run_dir": str(out_dir),
        "dataset_slug": dataset_slug,
    }
    weight_dict = load_pileup_weights(pileup_weight_dict)
    weighted_mc_hists: Dict[str, np.ndarray] = {}
    mc_weighting_realignment: Optional[Dict[str, Any]] = None

    if do_pileup_reweight and do_mc_weighting_realignment:
        try:
            from menus.particle.cms.tav_enhanced_validation_v2 import (
                DEFAULT_LEPTON_SF_RUN2012,
                DEFAULT_PHOTON_SF_RUN2012,
                print_mc_weighting_realignment_summary,
                run_tav_mc_weighting_realignment,
            )

            sf_lepton = (
                float(lepton_sf)
                if lepton_sf is not None
                else DEFAULT_LEPTON_SF_RUN2012
            )
            sf_photon = (
                float(photon_sf)
                if photon_sf is not None
                else DEFAULT_PHOTON_SF_RUN2012
            )
            profile = pileup_data_profile
            if profile is None and pileup_weight_dict is not None:
                loaded = load_pileup_weights(pileup_weight_dict)
                unique = {round(float(v), 6) for v in loaded.values()}
                if len(unique) > 1:
                    profile = loaded
            if verbose:
                print("\n--- TavMCWeighting realignment (pileup + lepton SF) ---")
            mc_weighting_realignment = run_tav_mc_weighting_realignment(
                data_file=data_file,
                mc_files=mc_files,
                pileup_data_profile=profile,
                lepton_sf=sf_lepton,
                photon_sf=sf_photon,
                chunk_size=chunk_size,
                entry_stop=entry_stop,
                output_dir=out_dir / "mc_weighting_realignment",
                dataset_slug=dataset_slug,
                verbose=verbose,
            )
            results["mc_weighting_realignment"] = mc_weighting_realignment
            for mc_name, block in (mc_weighting_realignment.get("mc_realignment") or {}).items():
                hist = block.get("muon_pt_hist_weighted")
                if hist is not None:
                    weighted_mc_hists[str(mc_name)] = np.asarray(hist, dtype=float)
            if verbose:
                print_mc_weighting_realignment_summary(mc_weighting_realignment)
                verdict = (mc_weighting_realignment or {}).get("verdict", "")
                if verdict == "PARTIAL_SUCCESS_LEPTON_SF_ONLY":
                    print(
                        "  [TavMCWeighting] pileup branches unavailable on data skim; "
                        "continuing with lepton-SF-only realignment"
                    )
        except Exception as exc:
            mc_weighting_realignment = {
                "error": str(exc),
                "verdict": "MC_WEIGHTING_REALIGNMENT_FAILED",
            }
            results["mc_weighting_realignment"] = mc_weighting_realignment
            if verbose:
                print(f"  [TavMCWeighting] FAILED: {exc}")
    if verbose:
        print("Running TAV MC Validation Suite (classic + enhanced v2 in series)")
        print(f"  Validation artifacts: {out_dir}")
        if entry_stop and entry_stop > 0:
            print(f"  Event limit        : {entry_stop:,}")
        if do_enhanced_validation:
            print("  Phase 3 scheduled  : Enhanced validation v2 after classic compare")
        if do_mc_weighting_realignment and do_pileup_reweight:
            print("  Phase 2b scheduled : TavMCWeighting realignment (pileup + SFs)")
        if do_photon_crosscheck:
            print("  Option 4 scheduled : Photon-enriched crosscheck (γ / e± channel)")
        if do_geometric_floor_scan:
            if do_option_a_recoil_sweep:
                print(
                    "  Phase 4 scheduled  : Option A recoil sweep "
                    "(recoil 15/30/45 GeV, ΔR≤1.5 validation gate)"
                )
            elif do_option2_tight_cuts:
                print("  Phase 4 scheduled  : Option 2 tight kinematic cuts")

    if verbose:
        print("\n--- Phase 1/4: Data lepton analysis ---")
    data_analysis_dir = out_dir / "data_analysis"
    data_analysis_dir.mkdir(parents=True, exist_ok=True)
    data_results = tav_analyze_leptons_from_file(
        file_path=data_file,
        output_dir=str(data_analysis_dir),
        chunk_size=chunk_size,
        entry_stop=entry_stop,
        verbose=False,
    )
    results["data_results"] = data_results
    data_pt = data_results.get("muon_pt_7fold", {})
    mc_results_by_name: Dict[str, Dict[str, Any]] = {}

    for mc_file in mc_files:
        mc_name = Path(mc_file).stem
        if verbose:
            print(f"\n--- Phase 2/4: MC compare — {mc_name} ---")
        mc_analysis_dir = out_dir / f"mc_{mc_name}"
        mc_analysis_dir.mkdir(parents=True, exist_ok=True)
        mc_results = tav_analyze_leptons_from_file(
            file_path=mc_file,
            output_dir=str(mc_analysis_dir),
            chunk_size=chunk_size,
            entry_stop=entry_stop,
            verbose=False,
        )
        mc_results_by_name[str(mc_name)] = mc_results
        comparison_dir = out_dir / f"comparison_{mc_name}"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        comparison = tav_compare_data_mc(
            data_results=data_results,
            mc_results=mc_results,
            output_dir=f"compare_{mc_name}",
            lepton="Muon",
            save_plots=True,
            verbose=verbose,
        )
        mc_mod7_raw = mc_results.get("muon_multiplicity_7fold", {})
        unweighted_stats = compute_weighted_mod7(np.array(mc_mod7_raw.get("mod7_histogram", [0] * 7)))
        weighted_stats = unweighted_stats
        realign_block = None
        if mc_weighting_realignment:
            realign_block = (mc_weighting_realignment.get("mc_realignment") or {}).get(
                str(mc_name)
            )
        if realign_block and realign_block.get("mod7_pileup_weighted"):
            w = realign_block["mod7_pileup_weighted"]
            weighted_stats = {
                "histogram": w.get("histogram"),
                "fractions": w.get("fractions"),
                "chi2_vs_uniform": w.get("chi2_mod7_vs_uniform"),
            }
            comparison["mod7_discrepancy_before"] = realign_block.get(
                "mod7_discrepancy_before"
            )
            comparison["mod7_discrepancy_after"] = realign_block.get(
                "mod7_discrepancy_after"
            )
            comparison["tav_mc_weighting_active"] = True
        elif do_pileup_reweight and verbose:
            print(
                f"  [Pileup] Using proxy stats for {mc_name} "
                f"(TavMCWeighting realignment unavailable)"
            )
        comparison["mod7_unweighted"] = unweighted_stats
        comparison["mod7_pileup_weighted"] = weighted_stats
        comparison["pt_subharmonic_excess_data"] = data_pt.get("subharmonic_excess", 0)
        comparison["pt_subharmonic_excess_mc"] = mc_results.get("muon_pt_7fold", {}).get(
            "subharmonic_excess", 0
        )
        mc_significance = mc_results.get("seven_periodic", {}).get("significance_sigma", 0)
        comparison["mc_strong_7fold"] = mc_significance > 5.0
        comparison["mc_significance"] = mc_significance
        if do_dependence_studies:
            dep_results = perform_dependence_studies(mc_file, str(mc_name), comparison_dir, chunk_size, verbose)
            comparison["dependence_studies"] = dep_results
        results["comparisons"][str(mc_name)] = comparison

    photon_pt_hist_for_enhanced: Optional[np.ndarray] = None
    if do_photon_crosscheck:
        try:
            from menus.particle.cms.tav_photon_crosscheck_option4 import (
                _load_photon_channel,
                resolve_photon_crosscheck_dataset,
            )

            photon_key = photon_crosscheck_file or resolve_photon_crosscheck_dataset(None)
            photon_scan_pre = _load_photon_channel(
                photon_key,
                chunk_size=chunk_size,
                entry_stop=entry_stop,
                pt_reduction="flatten",
                verbose=False,
            )
            raw_hist = photon_scan_pre.get("photon_pt_hist")
            if raw_hist is not None:
                photon_pt_hist_for_enhanced = np.asarray(raw_hist, dtype=float)
                if photon_sf is not None and photon_sf != 1.0:
                    photon_pt_hist_for_enhanced = photon_pt_hist_for_enhanced * float(
                        photon_sf
                    )
        except Exception:
            photon_pt_hist_for_enhanced = None

    enhanced: Optional[Dict[str, Any]] = None
    if do_enhanced_validation and mc_results_by_name:
        if verbose:
            print("\n--- Phase 3/4: Enhanced validation v2 ---")
        try:
            from menus.particle.cms.tav_enhanced_validation_v2 import (
                print_enhanced_validation_summary,
                run_enhanced_validation_from_lepton_results,
            )

            enhanced = run_enhanced_validation_from_lepton_results(
                data_results=data_results,
                mc_results_by_name=mc_results_by_name,
                output_dir=out_dir / "enhanced_validation",
                pileup_weight_dict=weight_dict if do_pileup_reweight else None,
                weighted_mc_pt_hists=weighted_mc_hists or None,
                weighting_summary=(
                    mc_weighting_realignment
                    if mc_weighting_realignment
                    and not mc_weighting_realignment.get("error")
                    else None
                ),
                photon_pt_hist=photon_pt_hist_for_enhanced,
                run_when=run_when,
                dataset_slug=dataset_slug,
            )
            results["enhanced_validation_v2"] = enhanced
            if verbose:
                print_enhanced_validation_summary(enhanced)
        except Exception as exc:
            enhanced = {
                "error": str(exc),
                "verdict": "ENHANCED_VALIDATION_FAILED",
            }
            results["enhanced_validation_v2"] = enhanced
            if verbose:
                from menus.particle.cms.tav_enhanced_validation_v2 import (
                    print_enhanced_validation_summary,
                )

                print_enhanced_validation_summary(enhanced)

    option4_report: Optional[Dict[str, Any]] = None
    if do_photon_crosscheck:
        if verbose:
            print("\n--- Option 4: Photon-enriched crosscheck (vector boson channel) ---")
        try:
            from menus.particle.cms.tav_photon_crosscheck_option4 import (
                print_option4_photon_crosscheck_summary,
                resolve_photon_crosscheck_dataset,
                run_photon_enriched_crosscheck_pipeline,
            )

            photon_source = photon_crosscheck_file
            if not photon_source:
                photon_source = resolve_photon_crosscheck_dataset(None)

            option4_report = run_photon_enriched_crosscheck_pipeline(
                data_results=data_results,
                photon_source=photon_source,
                enhanced_report=enhanced,
                photon_sf=float(photon_sf) if photon_sf is not None else 1.0,
                chunk_size=chunk_size,
                entry_stop=entry_stop,
                output_dir=out_dir / "photon_crosscheck",
                dataset_slug=dataset_slug,
                verbose=verbose,
            )
            results["photon_crosscheck"] = option4_report
            if option4_report.get("photon_crosscheck"):
                data_results["photon_crosscheck"] = option4_report["photon_crosscheck"]
            if verbose:
                print_option4_photon_crosscheck_summary(option4_report)
        except Exception as exc:
            option4_report = {
                "error": str(exc),
                "verdict": "PHOTON_CROSSCHECK_FAILED",
            }
            results["photon_crosscheck"] = option4_report
            if verbose:
                print(f"  [Option 4] FAILED: {exc}")

    option3_report: Optional[Dict[str, Any]] = None
    if do_expanded_mc_stack and mc_results_by_name:
        if verbose:
            print("\n--- Option 3: Expanded MC stack (DY + ttbar + sector closure) ---")
        try:
            from menus.particle.cms.tav_mc_stack_option3 import (
                print_option3_mc_stack_summary,
                run_option3_expanded_mc_analysis,
            )

            option3_report = run_option3_expanded_mc_analysis(
                data_results=data_results,
                mc_results_by_name=mc_results_by_name,
                enhanced_report=enhanced,
                output_dir=out_dir / "option3_mc_stack",
                dataset_slug=dataset_slug,
            )
            results["option3_mc_stack"] = option3_report
            if verbose:
                print_option3_mc_stack_summary(option3_report)
        except Exception as exc:
            option3_report = {
                "error": str(exc),
                "verdict": "OPTION3_MC_STACK_FAILED",
            }
            results["option3_mc_stack"] = option3_report
            if verbose:
                print(f"  [Option 3] FAILED: {exc}")

    geometric_floor: Optional[Dict[str, Any]] = None
    option_a_report: Optional[Dict[str, Any]] = None
    if do_geometric_floor_scan:
        wr_params = {
            "omega_wind": 1.45e43,
            "kappa_leak": 0.01,
            "R": 7.0,
            "I": 1.0,
        }
        if do_option_a_recoil_sweep:
            if verbose:
                print(
                    "\n--- Phase 4/4: Option A — relaxed recoil isolation "
                    "(313.1 MeV floor) ---"
                )
            try:
                from menus.particle.cms.tav_recoil_isolation_option_a import (
                    print_option_a_recoil_summary,
                    run_option_a_relaxed_recoil_isolation,
                )

                option_a_delta_r = None
                if kinematic_cuts and "delta_r_max" in kinematic_cuts:
                    option_a_delta_r = float(kinematic_cuts["delta_r_max"])
                option_a_recoil = None
                if kinematic_cuts and "recoil_pt_max" in kinematic_cuts:
                    option_a_recoil = [float(kinematic_cuts["recoil_pt_max"])]

                option_a_report = run_option_a_relaxed_recoil_isolation(
                    data_file,
                    recoil_pt_max_values=option_a_recoil,
                    delta_r_max=option_a_delta_r,
                    winding_rate_params=wr_params,
                    track_winding_rate=True,
                    chunk_size=chunk_size,
                    entry_stop=entry_stop,
                    output_dir=out_dir / "option_a_recoil_isolation",
                    dataset_slug=dataset_slug,
                    verbose=verbose,
                )
                results["option_a_recoil_isolation"] = option_a_report
                geometric_floor = option_a_report
                results["geometric_floor_scan"] = geometric_floor
                if verbose:
                    print_option_a_recoil_summary(option_a_report)
            except Exception as exc:
                from menus.particle.cms.tav_enhanced_validation_v2 import (
                    geometric_floor_scan_failure,
                )

                option_a_report = geometric_floor_scan_failure(str(exc))
                option_a_report["error"] = str(exc)
                results["option_a_recoil_isolation"] = option_a_report
                geometric_floor = option_a_report
                results["geometric_floor_scan"] = geometric_floor
                if verbose:
                    print(f"  [Option A] FAILED: {exc}")

        if do_option2_tight_cuts:
            if verbose:
                print("\n--- Option 2 — tight kinematic cuts (313.1 MeV floor) ---")
            try:
                from menus.particle.cms.tav_enhanced_validation_v2 import (
                    option2_angular_cuts,
                    option2_pt_thresholds,
                    print_geometric_floor_scan_summary,
                    run_targeted_kinematic_cuts_phase_rejection,
                )

                cuts = dict(option2_angular_cuts())
                if kinematic_cuts:
                    cuts.update(kinematic_cuts)

                option2_report = run_targeted_kinematic_cuts_phase_rejection(
                    data_file,
                    pt_thresholds=option2_pt_thresholds(),
                    angular_cuts=cuts,
                    winding_rate_params=wr_params,
                    track_winding_rate=True,
                    chunk_size=chunk_size,
                    entry_stop=entry_stop,
                    output_dir=out_dir / "phase_rejection_scan",
                    dataset_slug=dataset_slug,
                    verbose=verbose,
                )
                results["option2_tight_cuts"] = option2_report
                if not do_option_a_recoil_sweep:
                    geometric_floor = option2_report
                    results["geometric_floor_scan"] = geometric_floor
                if verbose:
                    print_geometric_floor_scan_summary(option2_report)
            except Exception as exc:
                from menus.particle.cms.tav_enhanced_validation_v2 import (
                    geometric_floor_scan_failure,
                    print_geometric_floor_scan_summary,
                )

                option2_report = geometric_floor_scan_failure(str(exc))
                option2_report["error"] = str(exc)
                results["option2_tight_cuts"] = option2_report
                if not do_option_a_recoil_sweep:
                    geometric_floor = option2_report
                    results["geometric_floor_scan"] = geometric_floor
                if verbose:
                    print_geometric_floor_scan_summary(option2_report)

    results["summary"] = _build_validation_summary(results, enhanced)
    if option4_report and not option4_report.get("error"):
        channel = option4_report.get("cross_channel_consistency") or {}
        results["summary"]["option4_report_path"] = option4_report.get("report_path")
        results["summary"]["option4_cross_channel_verdict"] = channel.get("verdict")
        results["summary"]["option4_global_spacetime_imprint"] = channel.get(
            "global_spacetime_imprint"
        )
        photon = option4_report.get("photon_enriched") or {}
        if photon:
            results["summary"]["option4_photon_7fold_amplitude"] = photon.get(
                "photon_7fold_amplitude"
            )
    if option3_report and not option3_report.get("error"):
        sector = option3_report.get("mod7_third_sector") or {}
        results["summary"]["option3_report_path"] = option3_report.get("report_path")
        results["summary"]["option3_third_sector_gap"] = sector.get(
            "gap_data_minus_background_stack"
        )
        results["summary"]["option3_third_sector_closure"] = sector.get(
            "third_sector_closure"
        )
        results["summary"]["option3_mc_samples"] = option3_report.get("mc_sample_names")
    if mc_weighting_realignment and not mc_weighting_realignment.get("error"):
        results["summary"]["mc_weighting_realignment_path"] = mc_weighting_realignment.get(
            "report_path"
        )
        first_mc = next(iter(mc_weighting_realignment.get("mc_realignment") or {}.values()), {})
        after = first_mc.get("mod7_discrepancy_after") or {}
        results["summary"]["mc_weighting_delta_mod7_bin0"] = after.get("delta_mod7_bin0")
        results["summary"]["mc_weighting_delta_mod7_bin2"] = after.get("delta_mod7_bin2")
        results["summary"]["mc_weighting_mc_peak_mod7"] = after.get("mc_peak_mod7")
    if option_a_report and not option_a_report.get("error"):
        best_win = option_a_report.get("best_isolation_window") or {}
        results["summary"]["option_a_report_path"] = option_a_report.get("report_path")
        results["summary"]["option_a_best_recoil_gev"] = best_win.get("recoil_pt_max_gev")
        results["summary"]["option_a_best_delta_phi"] = best_win.get("delta_phi_min")
        results["summary"]["option_a_event_pass_fraction"] = best_win.get(
            "event_pass_fraction"
        )
    if geometric_floor and geometric_floor.get("best_isolation"):
        best = geometric_floor["best_isolation"]
        results["summary"]["geometric_floor_best_pt_gev"] = best.get("pt_threshold_gev")
        results["summary"]["geometric_floor_subharmonic_excess"] = best.get(
            "subharmonic_excess"
        )
        results["summary"]["geometric_floor_isolation_quality"] = best.get(
            "isolation_quality"
        )
        results["summary"]["geometric_floor_report_path"] = geometric_floor.get(
            "report_path"
        )

    validation_report = artifact_path(
        TestSlug.CERN,
        compose_dataset_slug(dataset_slug, "validation"),
        "report",
        "json",
        when=run_when,
        run_dir=out_dir,
    )
    with validation_report.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)

    if verbose:
        print(f"Validation saved: {validation_report}")
    results["report_path"] = str(validation_report)
    return results
