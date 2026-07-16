"""
TSB Casimir tau resonance scanner ↔ research_tool.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt

from tav_shared.llm_analysis import analyze_run_output as query_llm_analysis
from tav_shared.run_output import parse_show_graphics
from menus.particle.casimir.scanner import (
    ARTIFACTS_DIR,
    DATASETS_DIR,
    DEFAULT_DONE_FILE,
    EXAMPLE_DATASETS,
    run_batch_casimir_scan,
    run_casimir_scan,
)

MODULE_TAG = "TSB_CASIMIR"
SUBMENU_TITLE = "Casimir Tau-SB Scan"

MENU_ACTIONS = [
    "Example Casimir Tau-Search",
    "TSB Falsification Test Suite (5 predictions)",
    "Scan Local CSV",
    "Download Zenodo Casimir Drums",
    "Download GitHub Magnetic-Fluid Data",
    "Scan GitHub Magnetic-Fluid CSV",
    "Scan Extracted Dataset (batch)",
    "Full Casimir Tau-Search Pipeline",
]

_ACTION_CONFIG: dict[str, dict] = {
    "Example Casimir Tau-Search": {
        "data_source": "mock",
        "plot": True,
    },
    "TSB Falsification Test Suite (5 predictions)": {
        "falsification": True,
        "data_source": "mock",
        "plot": True,
    },
    "Scan Local CSV": {
        "data_source": "local",
        "plot": True,
    },
    "Download Zenodo Casimir Drums": {
        "download_only": True,
        "dataset": "zenodo",
    },
    "Download GitHub Magnetic-Fluid Data": {
        "download_only": True,
        "dataset": "github",
    },
    "Scan GitHub Magnetic-Fluid CSV": {
        "data_source": "github",
        "plot": True,
    },
    "Scan Extracted Dataset (batch)": {
        "batch": True,
        "plot": False,
    },
    "Full Casimir Tau-Search Pipeline": {
        "data_source": "github",
        "plot": True,
        "batch_if_extracted": True,
        "github_if_cached": True,
    },
}


def is_module_selection(repo: str | None) -> bool:
    return repo == MODULE_TAG


def _yes(value: str) -> bool:
    return str(value or "").strip().lower() in {"yes", "y", "true", "1"}


def handle_pre_form(stdscr, selection: str, params: dict[str, str]) -> str | None:
    """Auto-generate falsification fixture CSV and skip the entry form."""
    _ = stdscr
    if selection != "TSB Falsification Test Suite (5 predictions)":
        return None
    from menus.particle.casimir.test_framework import falsification_default_options

    params.update(falsification_default_options())
    return "skip_form"


def _show_saved_plot(plot_path: str | None) -> None:
    if not plot_path or not Path(plot_path).is_file():
        return
    image = plt.imread(plot_path)
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title(os.path.basename(plot_path))
    plt.tight_layout()
    plt.show()


# =============================================================================
# BLOCK: Entry form — fields
# =============================================================================
def entry_fields(action: str) -> list[dict]:
    """Curses entry-form field specs for TSB Casimir tau-resonance actions."""
    fields = [
        {
            "key": "local_csv",
            "label": "Local CSV path (optional)",
            "default": "",
            "required": False,
            "hint": "Required for Scan Local CSV; blank uses mock for example runs",
        },
        {
            "key": "param_col",
            "label": "Parameter column (x-axis)",
            "default": "",
            "required": False,
            "hint": "e.g. magnetic_field_mT, separation_nm — blank = auto-detect",
        },
        {
            "key": "value_col",
            "label": "Observable column (y-axis)",
            "default": "",
            "required": False,
            "hint": "e.g. force_pN — blank = auto-detect",
        },
        {
            "key": "extract_dir",
            "label": "Extracted dataset directory",
            "default": "",
            "required": False,
            "hint": "Blank = datasets/casimir/casimir_drums_extracted",
        },
        {
            "key": "max_files",
            "label": "Max CSV files (batch)",
            "default": "10",
            "required": False,
            "hint": "Pending files per run; 0 = all pending; resumes via datasets/casimir/done.txt",
        },
        {
            "key": "force_rescan",
            "label": "Force re-scan (ignore done.txt)",
            "default": "no",
            "required": False,
            "hint": "Re-scan from the first files instead of resuming pending",
            "choices": ["no", "yes"],
        },
        {
            "key": "confirm_large_download",
            "label": "Confirm large download (~780 MB)",
            "default": "no",
            "required": False,
            "hint": "Set yes to download Zenodo Casimir drums zip",
            "choices": ["no", "yes"],
        },
        {
            "key": "force_download",
            "label": "Force re-download",
            "default": "no",
            "required": False,
            "hint": "Re-fetch GitHub magnetic-fluid or Zenodo cache",
            "choices": ["no", "yes"],
        },
        {
            "key": "output_prefix",
            "label": "Plot/report prefix",
            "default": "tsb_casimir",
            "required": False,
            "hint": "Artifacts saved under artifacts/",
        },
        {
            "key": "show_graphics",
            "label": "Graphics mode",
            "default": "popup",
            "required": False,
            "hint": "popup = Tk window | artifacts = PNG only",
            "choices": ["artifacts", "popup"],
        },
    ]
    if action == "Download Zenodo Casimir Drums":
        return [f for f in fields if f["key"] in {"confirm_large_download", "output_prefix"}]
    if action == "Download GitHub Magnetic-Fluid Data":
        return [f for f in fields if f["key"] in {"force_download", "output_prefix"}]
    if action == "Scan GitHub Magnetic-Fluid CSV":
        return [
            f
            for f in fields
            if f["key"] in {"local_csv", "param_col", "value_col", "output_prefix", "show_graphics"}
        ]
    if action == "Scan Extracted Dataset (batch)":
        return [
            f
            for f in fields
            if f["key"] in {"extract_dir", "max_files", "force_rescan", "output_prefix"}
        ]
    if action == "Scan Local CSV":
        return [f for f in fields if f["key"] != "confirm_large_download"]
    if action == "TSB Falsification Test Suite (5 predictions)":
        return []
    return fields


# =============================================================================
# BLOCK: Entry form — instructions
# =============================================================================
def entry_instructions(action: str) -> list[str]:
    """Short help bullets shown above the Casimir scanner entry form."""
    if action == "TSB Falsification Test Suite (5 predictions)":
        return [
            "Five explicit TSB vs Lifshitz falsification tests with p-values.",
            "Predictions: discrete steps, hysteresis, log-Δn modulations, anisotropy, 1/7 periodicity.",
            "Fixture CSV auto-generated at datasets/casimir/fixtures/tsb_falsification_mock.csv.",
            "Outputs: artifacts/tsb_test_results/*_report.json, *_summary.txt, six-panel PNG.",
            "Complements the broader tau-resonance scan (periodogram, ruptures, tav-resonance cross-check).",
        ]
    return [
        "Downloads: Zenodo Casimir drums (~780 MB) or GitHub magnetic-fluid Fig2/Fig34 spectra.",
        "Extracts ZIP→CSV/TSV; loads direct CSV, TSV, whitespace, or extensionless tables.",
        "Signatures: 7-phase/1/7 harmonics, 142857-cycle, steps, hysteresis, log-Δn, peaks.",
        "Methods: periodogram (Fourier), scipy peak/step detection, autocorrelation, ruptures CP.",
        "Plots overlay 7-phase markers, 1/7 template, periodogram, autocorrelation, summary.",
        "Reports: artifacts/tsb_casimir_*_report.json and *_summary.txt",
        f"Cache: {DATASETS_DIR}/ | batch ledger: {DATASETS_DIR}/done.txt (resume next pending).",
        "Batch scans skip files in done.txt; force_rescan=yes re-runs from the start.",
        "CLI: python tsb_casimir_scanner.py --data github --download-github",
    ]


def run_action(selection: str, show_plots: bool = True, options: dict | None = None) -> None:
    options = options or {}
    config = _ACTION_CONFIG.get(selection)
    if config is None:
        print(f"[TAV ENGINE] Unknown Casimir Tau-SB action: {selection}")
        return

    output_prefix = (options.get("output_prefix") or "tsb_casimir").strip()
    show_popup = parse_show_graphics(options, default="popup") if show_plots else False

    if config.get("falsification"):
        from menus.particle.casimir.test_framework import run_falsification_from_source

        local_csv = (options.get("local_csv") or "").strip() or None
        param_col = (options.get("param_col") or "").strip() or None
        value_col = (options.get("value_col") or "").strip() or None
        data_source = "local" if local_csv else (config.get("data_source") or "mock")
        fals_prefix = output_prefix if output_prefix != "tsb_casimir" else "tsb_falsification"
        try:
            if local_csv and Path(local_csv).is_file():
                data_source = "local"
            elif not local_csv:
                gh_cache = DATASETS_DIR / "github_magnetic_fluid"
                if gh_cache.is_dir() and any(gh_cache.rglob("*")):
                    data_source = "github"
            report = run_falsification_from_source(
                data_source=data_source,
                local_csv=local_csv,
                param_col=param_col,
                value_col=value_col,
                output_prefix=fals_prefix,
                plot=config.get("plot", True),
                show_popup=show_popup,
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"[TAV ENGINE] TSB falsification suite failed: {exc}")
            return
        if show_popup and report.get("plot_path"):
            _show_saved_plot(report["plot_path"])
        elif report.get("plot_path"):
            print(f"[TAV ENGINE] Falsification plot: {report['plot_path']}")
        print(f"[TAV ENGINE] Report: {report.get('report_path')}")
        return

    if config.get("download_only"):
        if config.get("dataset") == "github":
            from menus.particle.casimir.scanner import download_github_magnetic_fluid

            paths = download_github_magnetic_fluid(force=_yes(options.get("force_download", "no")))
            print(f"[TAV ENGINE] Cached {len(paths)} GitHub magnetic-fluid files.")
            print("  Use 'Scan GitHub Magnetic-Fluid CSV' to analyze.")
            return

        meta = EXAMPLE_DATASETS["zenodo_superconducting_casimir"]
        size_mb = meta.get("size_warning_mb", "780")
        if not _yes(options.get("confirm_large_download", "no")):
            print(f"[TAV ENGINE] Zenodo Casimir drums zip is ~{size_mb} MB.")
            print("  Set confirm_large_download=yes to proceed.")
            print(f"  Or download manually to {DATASETS_DIR}/")
            return
        from menus.particle.casimir.scanner import download_file

        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = DATASETS_DIR / meta["zip_name"]
        download_file(meta["url"], zip_path)
        print(f"[TAV ENGINE] Saved: {zip_path}")
        print("  Use 'Scan Extracted Dataset (batch)' after extraction.")
        return

    if config.get("batch"):
        extract_dir = (options.get("extract_dir") or "").strip()
        if not extract_dir:
            extract_dir = str(DATASETS_DIR / "casimir_drums_extracted")
        if not Path(extract_dir).is_dir():
            print(f"[TAV ENGINE] Extract directory not found: {extract_dir}")
            print("  Run 'Download Zenodo Casimir Drums' first, or set extract_dir.")
            return
        try:
            max_files = int(options.get("max_files") or 10)
        except ValueError:
            max_files = 10
        batch = run_batch_casimir_scan(
            extract_dir,
            max_files=max_files,
            plot=_yes(options.get("batch_plots", "no")),
            output_prefix=output_prefix,
            done_file=DEFAULT_DONE_FILE,
            force_rescan=_yes(options.get("force_rescan", "no")),
        )
        for path, info in batch["summaries"].items():
            print(f"  {Path(path).name}: {info}")
        return

    data_source = (options.get("data_source") or config.get("data_source", "mock")).strip().lower()
    local_csv = (options.get("local_csv") or "").strip() or None
    param_col = (options.get("param_col") or "").strip() or None
    value_col = (options.get("value_col") or "").strip() or None

    if data_source == "local" and not local_csv:
        print("[TAV ENGINE] local_csv path required for Scan Local CSV.")
        return

    try:
        result = run_casimir_scan(
            data_source=data_source,
            local_csv=local_csv,
            param_col=param_col,
            value_col=value_col,
            download=_yes(options.get("force_download", "no")),
            extract_dir=(options.get("extract_dir") or "").strip() or None,
            plot=config.get("plot", True),
            output_prefix=output_prefix,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"[TAV ENGINE] Casimir scan failed: {exc}")
        return

    print("\n".join(result.summary_lines()))

    if config.get("plot") and show_popup:
        _show_saved_plot(result.plot_path)
    elif config.get("plot") and result.plot_path:
        print(f"[TAV ENGINE] Diagnostic plot: {result.plot_path}")

    if config.get("batch_if_extracted"):
        extract_dir = DATASETS_DIR / "casimir_drums_extracted"
        if extract_dir.is_dir() and any(extract_dir.rglob("*")):
            print("\n[TAV ENGINE] Running batch scan on extracted Zenodo data...")
            batch = run_batch_casimir_scan(
                extract_dir,
                max_files=int(options.get("max_files") or 5),
                plot=False,
                output_prefix=f"{output_prefix}_batch",
                done_file=DEFAULT_DONE_FILE,
                force_rescan=_yes(options.get("force_rescan", "no")),
            )
            for path, info in batch["summaries"].items():
                print(f"  {Path(path).name}: {info}")
        else:
            print(f"[TAV ENGINE] No extracted data at {extract_dir} — Zenodo batch skipped.")

    if config.get("github_if_cached"):
        gh_dir = DATASETS_DIR / "github_magnetic_fluid"
        if gh_dir.is_dir() and any(gh_dir.rglob("*")):
            print("\n[TAV ENGINE] Running GitHub magnetic-fluid batch scan...")
            batch = run_batch_casimir_scan(
                gh_dir,
                max_files=int(options.get("max_files") or 3),
                plot=False,
                output_prefix=f"{output_prefix}_github",
                done_file=DEFAULT_DONE_FILE,
                force_rescan=_yes(options.get("force_rescan", "no")),
            )
            for path, info in batch["summaries"].items():
                print(f"  {Path(path).name}: {info}")