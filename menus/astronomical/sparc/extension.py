"""
Unified SPARC module ↔ research_tool.py

All analysis actions batch-process ./data/sparc/*.csv via sparc_fetcher.
Repository pulls populate that directory; graphics display after curses exits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from menus.astronomical.sparc.analyzer import SPARCAnalyzer, THEORY_RATIO_TARGET
from tav_shared.batch_ledger import (
    SPARC_BATCH_DONE,
    append_done_entries,
    format_batch_banner,
    select_batch_items,
)
from menus.astronomical.sparc.fetcher import (
    DEFAULT_DATA_DIR,
    DEFAULT_DONE_FILE,
    archive_used_datasets,
    ensure_galaxy_csv,
    iter_local_csv_files,
    list_local_galaxies,
    pull_sparc_batch,
)
from tav_shared.run_output import parse_batch_limit, parse_show_graphics
from menus.astronomical.sparc.superblock import ARTIFACTS_DIR, SPARCData
from tav_shared.llm_analysis import analyze_run_output as query_llm_analysis

SUBMENU_TITLE = "SPARC"
MODULE_TAG = "SPARC"
LOCAL_DATA_GLOB = "./datasets/sparc/*.csv"

MENU_ACTIONS = [
    "Pull Batch from Repository",
    "Pull Single Galaxy CSV",
    "Empirical Plot + MCMC Sweep",
    "Rotation Curve + Shadow Fit",
    "Mass Ratio Report (5.2 check)",
    "Full SPARC Analysis",
    "Emergent DM Prediction",
    "Fit Emergent DM Parameters",
    "SPARC Test Suite (Systematic)",
    "Superblock SPARC Full Demo",
]

_ACTION_KEYS = {
    "Pull Batch from Repository": "pull_batch",
    "Pull Single Galaxy CSV": "pull_single",
    "Empirical Plot + MCMC Sweep": "empirical_mcmc",
    "Rotation Curve + Shadow Fit": "rotation_curve",
    "Mass Ratio Report (5.2 check)": "mass_ratio",
    "Full SPARC Analysis": "full_analysis",
    "Emergent DM Prediction": "predict",
    "Fit Emergent DM Parameters": "fit",
    "SPARC Test Suite (Systematic)": "suite",
    "Superblock SPARC Full Demo": "full_demo",
}

_BATCH_ACTIONS = {
    "empirical_mcmc",
    "rotation_curve",
    "mass_ratio",
    "full_analysis",
    "predict",
    "fit",
    "full_demo",
}

SPARC_RUN_FIELDS = [
    {
        "key": "batch_limit",
        "label": "Batch limit (0 = all pending)",
        "default": "10",
        "required": False,
        "hint": "Resumes via datasets/sparc/batch_done.txt; 0 = all pending CSVs",
    },
    {
        "key": "force_rescan",
        "label": "Force re-scan (ignore batch_done.txt)",
        "default": "no",
        "required": False,
        "hint": "Re-analyze from the first CSVs instead of resuming pending",
        "choices": ["no", "yes"],
    },
    {
        "key": "show_graphics",
        "label": "Graphics mode",
        "default": "artifacts",
        "required": False,
        "hint": "popup = Tk windows | artifacts = PNG only",
        "choices": ["artifacts", "popup"],
    },
]


def is_module_selection(repo: str | None) -> bool:
    return repo == MODULE_TAG


def _parse_float(value: str, default: float) -> float:
    try:
        return float((value or "").strip())
    except ValueError:
        return default


def _parse_int(value: str, default: int) -> int:
    try:
        return int((value or "").strip())
    except ValueError:
        return default


def _yes(value: str) -> bool:
    return (value or "").strip().lower() in {"yes", "y", "true", "1"}


def _parse_selected_datasets(options: dict) -> Optional[set[str]]:
    raw = (options.get("selected_datasets") or "").strip()
    if not raw:
        return None
    return {part.strip() for part in raw.split(",") if part.strip()}


def _iter_local_batch(
    limit: int = 0,
    selected: Optional[set[str]] = None,
    *,
    force_rescan: bool = False,
    done_file: Path | str = SPARC_BATCH_DONE,
) -> Iterator[Tuple[str, Path]]:
    """Yield (galaxy_name, csv_path) for pending ./datasets/sparc/*.csv files."""
    files = iter_local_csv_files(DEFAULT_DATA_DIR)
    if selected:
        files = [path for path in files if path.stem in selected]
    if not files:
        print(f"[TAV ENGINE] No CSV files found at {LOCAL_DATA_GLOB}")
        print("[TAV ENGINE] Run 'Pull Batch from Repository' or Dataset Manager fetch first.")
        return

    batch_files, status = select_batch_items(
        files,
        done_file,
        key_fn=lambda path: path.stem,
        limit=limit,
        force_rescan=force_rescan,
    )
    print(format_batch_banner("SPARC", status))
    if not batch_files:
        if status["done_count"] >= status["catalog_total"]:
            print(
                "[TAV ENGINE] All local SPARC CSVs already analyzed. "
                "Use force_rescan=yes to re-run from the start."
            )
        return
    print(f"[TAV ENGINE] Batch source: {LOCAL_DATA_GLOB}")
    for path in batch_files:
        yield path.stem, path


def _resolve_show_plots(options: dict, show_plots: bool) -> bool:
    if "show_graphics" in (options or {}):
        return parse_show_graphics(options, default="artifacts")
    return show_plots


def _display_plot(show_plots: bool) -> None:
    if show_plots:
        plt.show(block=True)
    else:
        plt.close("all")


def _load_theory(options: dict) -> Any:
    if not _yes(options.get("use_theory", "")):
        return None
    try:
        from menus.prime_past.harmonic import TavSuperblockPrimePastHarmonic

        return TavSuperblockPrimePastHarmonic()
    except Exception as exc:
        print(f"[TAV ENGINE] Theory object unavailable: {exc}")
        return None


def _print_fit_report(galaxy: str, fit: Dict[str, Any]) -> None:
    print(f"\n--- Emergent DM Fit: {galaxy} ---")
    if not fit.get("success"):
        print(f"[ERROR] Fit failed: {fit.get('message', 'unknown error')}")
        return
    print(f"beta2 = {fit['beta2']:.4f} ± {fit.get('beta2_err', 0):.4f}")
    print(f"phase_offset = {fit['phase_offset']:.4f} ± {fit.get('phase_err', 0):.4f}")
    print(f"chi2 = {fit['chi2']:.2f}  |  reduced chi2 = {fit['reduced_chi2']:.2f}")


def _run_analyzer_for_galaxy(
    selection: str,
    action: str,
    galaxy: str,
    csv_path: Path,
    show_plots: bool = True,
) -> None:
    analyzer = SPARCAnalyzer(str(csv_path))
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[TAV ENGINE] SPARC — {selection}")
    print(f"[TAV ENGINE] Galaxy: {galaxy}")
    print(f"[TAV ENGINE] Data file: {csv_path}")

    if action in {"rotation_curve", "full_analysis", "empirical_mcmc"}:
        from tav_shared.artifact_paths import TestSlug, artifact_path, compose_dataset_slug

        plot_path = artifact_path(TestSlug.SPARC, compose_dataset_slug(galaxy, "rotation"), "plot", "png")
        analyzer.plot_rotation_curve(galaxy, show=show_plots, save_path=str(plot_path))
        print(f"[TAV ENGINE] Rotation curve saved: {plot_path}")
        if show_plots:
            _display_plot(True)
        else:
            plt.close("all")

    if action in {"mass_ratio", "full_analysis"}:
        report = analyzer.analysis_report(galaxy)
        print("\n--- SPARC Mass Ratio Report ---")
        print(f"Mean shadow/baryon v^2 ratio: {report['mean_shadow_baryon_ratio']:.4f}")
        print(f"Theory target (Tav-Superblock): {report['theory_target_ratio']:.1f}")
        print(f"Delta from theory: {report['delta_from_theory']:.4f}")
        if report["within_10_percent"]:
            print("[STATISTICAL NOTE] Observed ratio within 10% of 5.2 expectation.")
        else:
            print("[STATISTICAL NOTE] Ratio deviates from 5.2 — inspect domain shear / phase alignment.")


def _run_empirical_mcmc_for_galaxy(galaxy: str, csv_path: Path, show_plots: bool = True) -> None:
    from tav_shared import mcmc_engine, require_topology

    analyzer = SPARCAnalyzer(str(csv_path))
    gal = analyzer.get_galaxy_frame(galaxy)
    x, y_total = gal["R"], gal["Vobs"]
    df = gal.rename(columns={"R": "Radius", "Vobs": "Velocity"})
    df["Kinematic_X"] = gal["R"]
    df["Observable_Y"] = gal["Vobs"]

    ratio = analyzer.calculate_mass_ratio(galaxy)
    print(f"[TAV ENGINE] SPARC shadow/baryon ratio: {ratio:.4f} (theory target {THEORY_RATIO_TARGET})")

    _run_analyzer_for_galaxy(
        "Empirical Plot + MCMC Sweep", "empirical_mcmc", galaxy, csv_path, show_plots
    )

    dev, is_anomalous = require_topology.calculate_topological_correlation(df, domain="SPARC")
    print(f"\n--- Analytical Report: {galaxy} ---")
    print(f"Mean Flux Intensity: {np.mean(y_total):.6f}")
    print(f"Max Deviation: {np.max(np.abs(dev)):.6f}")
    if is_anomalous:
        print("[CORRELATION FOUND] Anomalous phase-shift detected.")
    else:
        print("[SIGNAL STABLE] No TS dynamic deviations detected.")

    print("\n[TAV ENGINE] Initiating MCMC statistical sweep...")
    comparison = mcmc_engine.run_model_comparison(x, y_total)
    print(mcmc_engine.format_comparison_report(comparison))

    tav_row = next(row for row in comparison if row["model_type"] == "Tav")
    piso_row = next(row for row in comparison if row["model_type"] == "pISO")
    best = comparison[0]

    print(f"\n{mcmc_engine.describe_tav_fit(tav_row.get('params'))}")
    if np.isfinite(best["aic"]):
        print(f"Best global fit: {best['label']} (AIC = {best['aic']:.2f})")
    if np.isfinite(tav_row["aic"]):
        print(f"Tav-Superblock AIC: {tav_row['aic']:.2f}")
    if np.isfinite(piso_row["aic"]):
        print(f"pISO AIC: {piso_row['aic']:.2f}")


def _run_superblock_for_galaxy(
    action: str,
    galaxy: str,
    options: dict,
    show_plots: bool = True,
) -> None:
    sparc = SPARCData(data_dir=DEFAULT_DATA_DIR)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if action == "predict":
        beta2 = _parse_float(options.get("beta2", ""), 0.3)
        phase = _parse_float(options.get("phase_offset", ""), 1.0)
        theory = _load_theory(options)
        theory_curve = sparc.predict_emergent_dm_velocity(
            galaxy, beta2=beta2, phase_offset=phase, theory=theory
        )
        if theory_curve.empty:
            print(f"[ERROR] No rotation curve data for {galaxy}")
            return
        plot_path = artifact_path(TestSlug.SPARC, compose_dataset_slug(galaxy, "emergent_dm"), "plot", "png")
        sparc.plot_rotation_curve(
            galaxy, theory_curve=theory_curve, show=show_plots, save_path=str(plot_path)
        )
        print(f"[TAV ENGINE] Plot saved: {plot_path}")
        if show_plots:
            _display_plot(True)
        else:
            plt.close("all")
        return

    if action == "fit":
        fit = sparc.fit_emergent_dm(
            galaxy,
            initial_beta2=_parse_float(options.get("beta2", ""), 0.3),
            initial_phase=_parse_float(options.get("phase_offset", ""), 1.0),
        )
        _print_fit_report(galaxy, fit)
        if fit.get("success"):
            theory_curve = sparc.predict_emergent_dm_velocity(
                galaxy, beta2=fit["beta2"], phase_offset=fit["phase_offset"]
            )
            plot_path = artifact_path(TestSlug.SPARC, compose_dataset_slug(galaxy, "fit"), "plot", "png")
            sparc.plot_rotation_curve(
                galaxy, theory_curve=theory_curve, show=show_plots, save_path=str(plot_path)
            )
            print(f"[TAV ENGINE] Fit plot saved: {plot_path}")
            if show_plots:
                _display_plot(True)
            else:
                plt.close("all")
        return

    if action == "full_demo":
        fit = sparc.fit_emergent_dm(galaxy)
        _print_fit_report(galaxy, fit)
        beta2 = fit.get("beta2", 0.3) if fit.get("success") else 0.3
        phase = fit.get("phase_offset", 1.0) if fit.get("success") else 1.0
        theory = _load_theory(options)
        theory_curve = sparc.predict_emergent_dm_velocity(
            galaxy, beta2=beta2, phase_offset=phase, theory=theory
        )
        plot_path = artifact_path(TestSlug.SPARC, compose_dataset_slug(galaxy, "full_demo"), "plot", "png")
        sparc.plot_rotation_curve(
            galaxy, theory_curve=theory_curve, show=show_plots, save_path=str(plot_path)
        )
        print(f"[TAV ENGINE] Full demo plot saved: {plot_path}")
        if show_plots:
            _display_plot(True)
        else:
            plt.close("all")


def _run_batch(action: str, selection: str, options: dict, show_plots: bool = True) -> None:
    limit = parse_batch_limit(options, default=0)
    show_plots = _resolve_show_plots(options, show_plots)
    selected = _parse_selected_datasets(options)
    force_rescan = _yes(options.get("force_rescan", ""))
    print(
        f"[TAV ENGINE] Graphics: {'Tk popup' if show_plots else 'artifacts only (no popup)'}"
    )
    if limit > 0:
        print(f"[TAV ENGINE] Batch limit: {limit}")
    if selected:
        print(f"[TAV ENGINE] Selected datasets: {', '.join(sorted(selected))}")

    processed_paths: list[Path] = []
    completed_ids: list[str] = []
    processed = 0
    for galaxy, csv_path in (
        _iter_local_batch(limit=limit, selected=selected, force_rescan=force_rescan) or []
    ):
        processed += 1
        print(f"\n{'=' * 60}")
        print(f"[TAV ENGINE] Batch item {processed}: {galaxy}")
        print(f"{'=' * 60}")

        if action == "empirical_mcmc":
            _run_empirical_mcmc_for_galaxy(galaxy, csv_path, show_plots)
        elif action in {"rotation_curve", "mass_ratio", "full_analysis"}:
            _run_analyzer_for_galaxy(selection, action, galaxy, csv_path, show_plots)
        elif action in {"predict", "fit", "full_demo"}:
            print(f"\n[TAV ENGINE] SPARC Superblock — {selection}")
            _run_superblock_for_galaxy(action, galaxy, options, show_plots)
        else:
            break
        processed_paths.append(csv_path)
        completed_ids.append(galaxy)

    if completed_ids:
        append_done_entries(SPARC_BATCH_DONE, completed_ids)
        print(f"[TAV ENGINE] Recorded {len(completed_ids)} galaxy(ies) in {SPARC_BATCH_DONE}")

    if processed:
        print(f"\n[TAV ENGINE] Batch complete: {processed} galaxies from {LOCAL_DATA_GLOB}")
        if str(options.get("archive_after_run", "yes")).strip().lower() not in {
            "no",
            "n",
            "false",
            "0",
        }:
            archive_used_datasets(processed_paths)
            from tav_shared.dataset_ledger import mark_processed

            mark_processed(
                [f"sparc:{path.stem}" for path in processed_paths],
                note="sparc batch archive",
            )


def _run_test_suite(options: dict) -> None:
    sparc = SPARCData(data_dir=DEFAULT_DATA_DIR)
    galaxies = list_local_galaxies(DEFAULT_DATA_DIR)
    if not galaxies:
        print(f"[TAV ENGINE] No CSV files at {LOCAL_DATA_GLOB}. Run a repository pull first.")
        return

    limit = parse_batch_limit(options, default=0)
    force_rescan = _yes(options.get("force_rescan", ""))
    galaxies, status = select_batch_items(
        galaxies,
        SPARC_BATCH_DONE,
        key_fn=str,
        limit=limit,
        force_rescan=force_rescan,
    )
    print(format_batch_banner("SPARC test suite", status))
    if not galaxies:
        print("[TAV ENGINE] No pending galaxies for the test suite.")
        return
    max_galaxies = _parse_int(options.get("max_galaxies", ""), len(galaxies))
    max_galaxies = min(max_galaxies, len(galaxies))
    fit = _yes(options.get("run_fit", "yes"))
    run_galaxies = galaxies[:max_galaxies]
    results = sparc.run_sparc_test_suite(
        galaxies=run_galaxies, max_galaxies=max_galaxies, fit=fit
    )
    append_done_entries(SPARC_BATCH_DONE, run_galaxies)
    if results.empty:
        print("[TAV ENGINE] No galaxies met minimum data requirements.")
        return

    out_path = ARTIFACTS_DIR / "sparc_superblock_test_suite.csv"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"\n--- SPARC Test Suite ({len(results)} galaxies) ---")
    display_cols = [
        c for c in ["galaxy", "success", "beta2", "reduced_chi2", "median_g_ratio"] if c in results.columns
    ]
    print(results[display_cols].to_string(index=False))
    print(f"[TAV ENGINE] Suite report saved: {out_path}")


# =============================================================================
# BLOCK: Entry form — fields
# =============================================================================
def entry_fields(action: str) -> list[dict]:
    """Curses entry-form field specs for SPARC actions."""
    from tav_shared.tav_project_paths import FINISHED_A_DIR

    if action == "Pull Batch from Repository":
        return [
            {
                "key": "pull_limit",
                "label": "Galaxies to pull",
                "default": "100",
                "required": False,
                "hint": "Max new targets (skips cached CSVs)",
            },
            {
                "key": "refresh_catalog",
                "label": "Refresh catalog",
                "default": "no",
                "required": False,
                "hint": "Re-download MassModels .mrt from SPARC",
                "choices": ["no", "yes"],
            },
            {
                "key": "force_repull",
                "label": "Force re-pull",
                "default": "no",
                "required": False,
                "hint": "Overwrite existing galaxy CSVs from catalog",
                "choices": ["no", "yes"],
            },
            {
                "key": "restore_archived",
                "label": "Restore archived",
                "default": "no",
                "required": False,
                "hint": f"Copy missing CSVs from {FINISHED_A_DIR}/",
                "choices": ["no", "yes"],
            },
        ]
    if action == "Pull Single Galaxy CSV":
        return [
            {
                "key": "query",
                "label": "Galaxy name",
                "default": "",
                "required": True,
                "hint": "e.g. NGC3198",
            },
        ]
    return list(SPARC_RUN_FIELDS)


# =============================================================================
# BLOCK: Entry form — instructions
# =============================================================================
def entry_instructions(action: str) -> list[str]:
    """Short help bullets shown above the SPARC entry form."""
    from menus.astronomical.sparc.fetcher import DATASETS_DIR, FINISHED_A_DIR

    if action == "Pull Batch from Repository":
        return [
            "Fetches from astroweb.case.edu SPARC MassModels table.",
            f"Caches catalog + galaxy CSVs in {DATASETS_DIR}/.",
            f"Ledger: {DATASETS_DIR}/done.txt | processed CSVs -> {FINISHED_A_DIR}/.",
            "0 downloaded usually means all 175 galaxies are already cached locally.",
            f"force_repull=yes re-extracts CSVs; restore_archived=yes copies from {FINISHED_A_DIR}/.",
        ]
    if action == "Pull Single Galaxy CSV":
        return [
            f"Downloads one galaxy into {DATASETS_DIR}/{{name}}.csv.",
            "Examples: NGC3198, DDO154, CamB.",
        ]
    return [
        "Pull ledger: datasets/sparc/done.txt (repository fetch resume).",
        "Analysis ledger: datasets/sparc/batch_done.txt (batch analysis resume).",
        "batch_limit: 0 = all pending | force_rescan=yes ignores batch_done.txt.",
        "show_graphics: popup = Tk windows | artifacts = PNG only.",
        "Dataset Manager can pick specific galaxies to fetch/run.",
        "Text output saved to artifacts/run_*.txt automatically.",
    ]


def run_action(selection: str, show_plots: bool = True, options: dict | None = None) -> None:
    options = options or {}
    action = _ACTION_KEYS.get(selection)
    if action is None:
        print(f"[TAV ENGINE] Unknown SPARC action: {selection}")
        return
    _execute_action(selection, action, options, show_plots)


def _execute_action(selection: str, action: str, options: dict, show_plots: bool) -> None:
    if action == "pull_batch":
        summary = pull_sparc_batch(
            limit=_parse_int(options.get("pull_limit", ""), 100),
            data_dir=DEFAULT_DATA_DIR,
            done_file=DEFAULT_DONE_FILE,
            force_refresh_catalog=_yes(options.get("refresh_catalog", "")),
            force_repull=_yes(options.get("force_repull", "")),
            restore_archived=_yes(options.get("restore_archived", "")),
        )
        skipped = len(summary.skipped_done) + len(summary.skipped_existing)
        print(
            f"[TAV ENGINE] Batch pull complete: {summary.downloaded_count} downloaded, "
            f"{skipped} skipped, {len(summary.failed)} failed."
        )
        if summary.downloaded_count == 0 and skipped == 0 and not summary.failed:
            print(
                "[TAV ENGINE] Local cache is complete — run analysis actions "
                "(e.g. Full SPARC Analysis) on datasets/sparc/*.csv."
            )
        return

    if action == "pull_single":
        galaxy = (options.get("query") or "").strip()
        if not galaxy:
            print("[ERROR] Galaxy name is required for single pull.")
            return
        path = ensure_galaxy_csv(
            galaxy,
            data_dir=DEFAULT_DATA_DIR,
            done_file=DEFAULT_DONE_FILE,
            force_refresh_catalog=_yes(options.get("refresh_catalog", "")),
        )
        print(f"[TAV ENGINE] Single pull complete: {path}")
        return

    if action == "suite":
        _run_test_suite(options)
        return

    if action in _BATCH_ACTIONS:
        _run_batch(action, selection, options, show_plots=show_plots)