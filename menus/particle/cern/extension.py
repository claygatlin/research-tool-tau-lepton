"""
CERN Open Data ↔ research_tool.py
"""

from __future__ import annotations

from typing import Any

from menus.particle.cern.analyzer import run_analysis
from menus.particle.cern.fetcher import (
    DATASETS_DIR,
    list_cached_targets,
    pull_open_archives,
    pull_selected_targets,
)
from menus.particle.cern.manifest import (
    ACTION_DEFAULT_TARGETS,
    CERN_MANIFEST,
)

MODULE_TAG = "CERN_OPENDATA"
SUBMENU_TITLE = "CERN Open Data"

MENU_ACTIONS = [
    "Pull Datasets from Open Archives",
    "Analyze CMS NanoAOD",
    "CMS 7-Fold Muon Tav Analysis",
    "CMS Full-Dataset 7-Fold Scan",
    "Run MC Validation Suite",
    "Analyze ALICE ROOT Sample",
    "List Cached Datasets",
]

_MANIFEST_CHOICES = list(CERN_MANIFEST.keys())


def is_module_selection(repo: str | None) -> bool:
    return repo == MODULE_TAG


def entry_fields(action: str) -> list[dict]:
    if action == "Pull Datasets from Open Archives":
        return [
            {
                "key": "targets",
                "label": "Targets (comma-separated)",
                "default": ",".join(_MANIFEST_CHOICES),
                "required": False,
                "hint": "Manifest keys or 'all'",
            },
            {
                "key": "force_refresh",
                "label": "Force refresh",
                "default": "no",
                "required": False,
                "hint": "yes = re-download even if cached",
                "choices": ["no", "yes"],
            },
            {
                "key": "restore_archived",
                "label": "Restore from finishedA/",
                "default": "yes",
                "required": False,
                "hint": "yes = restore archived copy before remote pull",
                "choices": ["yes", "no"],
            },
        ]
    if action == "List Cached Datasets":
        return []
    if action == "CMS Full-Dataset 7-Fold Scan":
        return [
            {
                "key": "target",
                "label": "Manifest target or .root path",
                "default": "cms_nanoaod_dimu",
                "required": False,
                "hint": "Run2012BC DoubleMuParked (~61M events) or Higgs→4L",
                "choices": _MANIFEST_CHOICES,
            },
            {
                "key": "chunk_size",
                "label": "Chunk size (events)",
                "default": "500000",
                "required": False,
                "hint": "Uproot iterate step_size; lower if RAM limited",
            },
            {
                "key": "entry_stop",
                "label": "Max events (0 = entire file)",
                "default": "0",
                "required": False,
                "hint": "0 processes all events in ROOT file",
            },
            {
                "key": "pt_bins",
                "label": "Muon pT histogram bins",
                "default": "20",
                "required": False,
                "hint": "Global histogram edges 0–200 GeV",
            },
            {
                "key": "output_prefix",
                "label": "Output dataset label",
                "default": "tav_full_61M_events_output",
                "required": False,
                "hint": "artifacts/cern_analysis/{label}__*",
            },
            {
                "key": "save_plots",
                "label": "Save plots",
                "default": "yes",
                "required": False,
                "choices": ["yes", "no"],
            },
        ]
    if action == "CMS 7-Fold Muon Tav Analysis":
        return [
            {
                "key": "target",
                "label": "Manifest target or .root path",
                "default": "cms_nanoaod_dimu",
                "required": False,
                "hint": "Dimuon or Higgs→4L NanoAOD (Muon_pt + nMuon)",
                "choices": _MANIFEST_CHOICES,
            },
            {
                "key": "entry_stop",
                "label": "Max events (0 = entire file)",
                "default": "50000",
                "required": False,
                "hint": "Default 50k quick sample; 0 = full chunked scan (see CMS Full-Dataset action)",
            },
            {
                "key": "output_prefix",
                "label": "Output dataset label",
                "default": "tav_cms_output",
                "required": False,
                "hint": "artifacts/cern_analysis/{label}__tav_7fold_muon__*",
            },
            {
                "key": "save_plots",
                "label": "Save plots",
                "default": "yes",
                "required": False,
                "choices": ["yes", "no"],
            },
        ]
    if action == "Run MC Validation Suite":
        from menus.particle.cms.validation_extension import entry_fields as validation_entry_fields

        return validation_entry_fields(action)
    return [
        {
            "key": "target",
            "label": "Manifest target or .root path",
            "default": "cms_nanoaod_dimu",
            "required": False,
            "hint": "Preset key, recid, or local ROOT path",
            "choices": _MANIFEST_CHOICES,
        },
        {
            "key": "entry_stop",
            "label": "Max events / tracks",
            "default": "50000",
            "required": False,
            "hint": "Uproot read limit",
        },
        {
            "key": "output_prefix",
            "label": "Report prefix",
            "default": "cern_analysis",
            "required": False,
            "hint": "JSON under artifacts/",
        },
    ]


def entry_instructions(action: str) -> list[str]:
    if action == "Pull Datasets from Open Archives":
        return [
            "Programmatic fetch via opendata.cern.ch API — no browser required.",
            f"Cache: {DATASETS_DIR}/<target>/recid_<N>/",
            "Ledger: datasets/cern/done.txt + global processed registry.",
            "Manifest: CMS NanoAOD education samples + ALICE ESD skim (1 file).",
        ]
    if action == "List Cached Datasets":
        return [
            "Shows manifest targets, recid, cache path, and sample ROOT files.",
        ]
    if action == "Analyze CMS NanoAOD":
        return [
            "Requires cached cms_nanoaod_* target (auto-fetched before run).",
            "Uses uproot + awkward (native venv) on Events tree.",
            "Runs Tav 7-fold muon analysis (pT + nMuon multiplicity) by default.",
            "Writes artifacts/cern_analysis/*__tav_7fold_muon__*.{json,png}.",
        ]
    if action == "CMS 7-Fold Muon Tav Analysis":
        return [
            "Quick Tav run — default reads first 50,000 events only (not the full 61M file).",
            "For entire Run2012BC_DoubleMuParked_Muons.root use CMS Full-Dataset 7-Fold Scan.",
            "Set Max events = 0 here to run the chunked full-file path from this menu.",
            "Optional Photon_pt / Electron_pt cross-check when branches exist.",
            f"Cache: {DATASETS_DIR}/cms_nanoaod_*/recid_*/",
        ]
    if action == "CMS Full-Dataset 7-Fold Scan":
        return [
            "Chunked uproot.iterate scan over entire NanoAOD file (61M+ events).",
            "Accumulates global Muon_pt histogram + per-event nMuon array.",
            "Feeds tav_7fold_muon_analysis; reports seven_periodic.significance_sigma.",
            "Default target: cms_nanoaod_dimu / Run2012BC_DoubleMuParked_Muons.root",
        ]
    if action == "Run MC Validation Suite":
        from menus.particle.cms.validation_extension import entry_instructions as validation_entry_instructions

        return validation_entry_instructions(action)
    return [
        "Requires cached alice_esd_sample (1 ESD ROOT file).",
        "Uses menus/particle/alice O2 track loader when applicable.",
        "Writes artifacts/cern_alice_root_*.json.",
    ]


def _yes(value: str) -> bool:
    return str(value or "").strip().lower() in {"yes", "y", "true", "1"}


def run_action(action: str, show_plots: bool = True, options: dict | None = None) -> str | None:
    _ = show_plots
    options = options or {}

    if action == "Pull Datasets from Open Archives":
        force = _yes(options.get("force_refresh", "no"))
        restore = _yes(options.get("restore_archived", "yes"))
        raw_targets = str(options.get("targets") or "all").strip()
        if raw_targets.lower() in {"all", "*"}:
            summary = pull_open_archives(force_refresh=force, restore_archived=restore)
        else:
            names = [t.strip() for t in raw_targets.split(",") if t.strip()]
            summary = pull_selected_targets(
                names,
                params={"restore_archived": "yes" if restore else "no"},
                force_refresh=force,
            )
        print(f"[CERN OPENDATA] Downloaded: {summary.downloaded}")
        print(f"[CERN OPENDATA] Skipped: {summary.skipped_done + summary.skipped_existing}")
        if summary.failed:
            for key, msg in summary.failed.items():
                print(f"[CERN OPENDATA] FAILED {key}: {msg}")
        return None

    if action == "List Cached Datasets":
        for row in list_cached_targets():
            status = "cached" if row["cached"] else "remote"
            print(
                f"  {row['key']:22} recid={row['recid']} [{status}] "
                f"root={row['n_root_files']} → {row['cache_dir']}"
            )
        return None

    if action == "Run MC Validation Suite":
        try:
            from menus.particle.cms.validation_extension import run_action as run_validation_action
        except Exception as exc:
            raise ImportError(f"Cannot load validation suite: {exc}")

        report_path = run_validation_action(action, show_plots=show_plots, options=options)
        if report_path is not None:
            print(f"[CERN OPENDATA] Validation report: {report_path}")
        return report_path

    target = str(options.get("target") or ACTION_DEFAULT_TARGETS.get(action, ["cms_nanoaod_dimu"])[0])

    def _parse_entry_stop(raw: Any, default: int) -> int:
        text = str(raw).strip() if raw is not None else ""
        if not text:
            return default
        try:
            return int(text)
        except ValueError:
            return default

    if action == "CMS Full-Dataset 7-Fold Scan":
        entry_default = 0
    elif action == "CMS 7-Fold Muon Tav Analysis":
        entry_default = 50000
    else:
        entry_default = 50000
    entry_stop = _parse_entry_stop(options.get("entry_stop"), entry_default)
    try:
        chunk_size = int(options.get("chunk_size") or 500_000)
    except ValueError:
        chunk_size = 500_000
    try:
        pt_bins = int(options.get("pt_bins") or 20)
    except ValueError:
        pt_bins = 20

    if action == "CMS Full-Dataset 7-Fold Scan":
        prefix = str(options.get("output_prefix") or "tav_full_61M_events_output").strip()
        save_plots = _yes(options.get("save_plots", "yes"))
        full_scan = True
    elif action == "CMS 7-Fold Muon Tav Analysis":
        prefix = str(options.get("output_prefix") or "tav_cms_output").strip()
        save_plots = _yes(options.get("save_plots", "yes"))
        full_scan = entry_stop <= 0
    else:
        prefix = str(options.get("output_prefix") or "cern_analysis").strip()
        save_plots = True
        full_scan = False

    spec = CERN_MANIFEST.get(target, {})
    analysis = spec.get("analyze") or (
        "cms_nanoaod"
        if action
        in {
            "Analyze CMS NanoAOD",
            "CMS 7-Fold Muon Tav Analysis",
            "CMS Full-Dataset 7-Fold Scan",
        }
        else "alice_esd"
    )
    run_tav = action in {
        "Analyze CMS NanoAOD",
        "CMS 7-Fold Muon Tav Analysis",
        "CMS Full-Dataset 7-Fold Scan",
    }
    report = run_analysis(
        target,
        analysis=analysis,
        entry_stop=entry_stop,
        output_prefix=prefix,
        run_tav_7fold=run_tav,
        save_tav_plots=save_plots,
        full_scan=full_scan,
        chunk_size=chunk_size,
        pt_bins=pt_bins,
        verbose=True,
    )
    verdict = report.get("tav_verdict") or report.get("verdict")
    sp = (report.get("seven_periodic") or report.get("tav_7fold_muon", {}).get("seven_periodic") or {})
    n_events = report.get("n_events_processed")
    if n_events is None:
        tav = report.get("tav_7fold_muon") or {}
        mult = tav.get("multiplicity_7fold") or {}
        n_events = mult.get("n_events") or tav.get("n_events")
    if n_events is None:
        n_events = report.get("entry_stop")
    file_total = (report.get("scan") or {}).get("n_entries_in_file")
    if n_events is not None:
        if file_total and int(n_events) < int(file_total):
            print(
                f"[CERN OPENDATA] Events analyzed: {int(n_events):,} "
                f"(file has {int(file_total):,}; not full dataset)"
            )
        else:
            print(f"[CERN OPENDATA] Events analyzed: {int(n_events):,}")
    print(f"[CERN OPENDATA] {action} → {verdict or 'complete'}")
    if sp.get("significance_sigma") is not None:
        print(f"[CERN OPENDATA] 7-periodic σ: {sp['significance_sigma']:.2f}")
    print(f"[CERN OPENDATA] Report: {report.get('report_path')}")
    if report.get("report_chunked"):
        for p in (report.get("report_paths") or [])[1:]:
            print(f"[CERN OPENDATA] Report part: {p}")
    elif (report.get("tav_7fold_muon") or {}).get("report_chunked"):
        tav = report["tav_7fold_muon"]
        for p in (tav.get("report_paths") or [])[1:]:
            print(f"[CERN OPENDATA] Report part: {p}")
    if report.get("tav_plot_paths"):
        for p in report["tav_plot_paths"]:
            print(f"[CERN OPENDATA] Plot: {p}")
    return str(report.get("report_path") or "")