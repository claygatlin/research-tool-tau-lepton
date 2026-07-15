"""CMS validation menu extension for research_tool

Adds a menu action to run the Tav MC Validation Suite from the CMS menu.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tav_shared.artifact_paths import ARTIFACTS_ROOT, TestSlug, compose_dataset_slug
from tav_shared.llm_analysis import append_llm_entry_fields

MODULE_TAG = "CMS_VALIDATION"
SUBMENU_TITLE = "CMS Validation"

MENU_ACTIONS = [
    "Run MC Validation Suite",
]

_ACTION_KEYS: dict[str, str] = {
    "Run MC Validation Suite": "run_validation",
}

_RUN_FOLDER_RE = re.compile(
    r"^(?P<day>\d{2})-(?P<month>\d{2})-(?P<year>\d{4})_(?P<hour>\d{2})-(?P<minute>\d{2})_"
)

_PREFERRED_DATA_KEY = "cms_nanoaod_dimu"
_PREFERRED_MC_KEY = "cms_nanoaod_higgs_zz"

try:
    from menus.particle.cern.manifest import OPTION3_MC_STACK_KEYS, OPTION4_PHOTON_CROSSCHECK_KEYS
except ImportError:
    OPTION3_MC_STACK_KEYS = (
        "cms_nanoaod_dy",
        "cms_nanoaod_ttbar",
        "cms_nanoaod_higgs_zz",
    )
    OPTION4_PHOTON_CROSSCHECK_KEYS = (
        "cms_nanoaod_doubleelectron",
        "cms_nanoaod_zz4e",
        "cms_nanoaod_higgs_zz",
    )


def is_module_selection(repo: str | None) -> bool:
    return repo == MODULE_TAG


def _parse_run_folder_stamp(folder_name: str) -> datetime | None:
    match = _RUN_FOLDER_RE.match(folder_name)
    if not match:
        return None
    parts = match.groupdict()
    try:
        return datetime(
            int(parts["year"]),
            int(parts["month"]),
            int(parts["day"]),
            int(parts["hour"]),
            int(parts["minute"]),
            tzinfo=timezone.utc,
        )
    except ValueError:
        return None


def _is_usable_dataset_ref(value: str) -> bool:
    text = (value or "").strip()
    if not text or text in {"*", "default", "data", "mc"}:
        return False
    if text.startswith("tmp_") or "/tmp/" in text:
        return False
    return True


def _resolve_dataset_ref(value: str) -> str | None:
    """Return manifest key or existing .root path, or None."""
    from menus.particle.cern.fetcher import resolve_root_path
    from menus.particle.cern.manifest import resolve_target_key

    text = (value or "").strip()
    if not _is_usable_dataset_ref(text):
        return None
    path = Path(text).expanduser()
    if path.is_file() and path.suffix.lower() == ".root":
        return str(path.resolve())
    resolved = resolve_root_path(text)
    if resolved is not None:
        key = resolve_target_key(text)
        return str(key) if key else str(resolved)
    key = resolve_target_key(text)
    if key:
        return key
    return text if path.exists() else None


def _read_json_report(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _latest_validation_from_artifacts() -> dict[str, Any] | None:
    """Read data/MC paths from the newest validation report under artifacts."""
    root = ARTIFACTS_ROOT / TestSlug.CERN
    if not root.is_dir():
        return None

    candidates: list[tuple[datetime, Path, dict[str, Any]]] = []
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        stamp = _parse_run_folder_stamp(run_dir.name)
        if stamp is None:
            stamp = datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc)
        for report in run_dir.glob("*validation*report*.json"):
            payload = _read_json_report(report)
            if not payload:
                continue
            if "data_file" not in payload or "mc_files" not in payload:
                continue
            candidates.append((stamp, run_dir, payload))

    if not candidates:
        return None

    candidates.sort(key=lambda row: row[0], reverse=True)
    _, run_dir, payload = candidates[0]
    data_ref = str(payload.get("data_file") or "")
    mc_refs = [str(item) for item in (payload.get("mc_files") or []) if _is_usable_dataset_ref(str(item))]
    data_resolved = _resolve_dataset_ref(data_ref)
    mc_resolved = [m for m in (_resolve_dataset_ref(item) for item in mc_refs) if m]
    if not data_resolved and not mc_resolved:
        return None
    return {
        "source": "latest_validation",
        "run_dir": str(run_dir),
        "data_file": data_resolved,
        "mc_files": mc_resolved,
        "data_label": data_ref,
        "mc_labels": mc_refs,
    }


def _cached_cms_dataset_rows() -> list[dict[str, Any]]:
    from menus.particle.cern.fetcher import list_cached_targets
    from menus.particle.cern.manifest import CERN_MANIFEST

    rows: list[dict[str, Any]] = []
    for item in list_cached_targets():
        key = str(item.get("key") or "")
        spec = CERN_MANIFEST.get(key, {})
        if spec.get("analyze") != "cms_nanoaod":
            continue
        if not item.get("cached") or not item.get("sample_root"):
            continue
        path = Path(str(item["sample_root"]))
        rows.append(
            {
                "key": key,
                "label": f"{key} — {path.name}",
                "path": str(path.resolve()),
            }
        )
    rows.sort(key=lambda row: str(row["label"]))
    return rows


def discover_mc_validation_defaults() -> dict[str, Any]:
    """
    Resolve default Data + MC inputs for the validation suite.

    Priority: latest validation artifact run → cached datasets/cern/.
    """
    cached = _cached_cms_dataset_rows()
    cached_by_key = {row["key"]: row for row in cached}

    latest = _latest_validation_from_artifacts()
    data_file: str | None = None
    mc_files: list[str] = []
    source = "cached_datasets"
    run_dir: str | None = None
    data_label = ""
    mc_labels: list[str] = []

    if latest:
        data_file = latest.get("data_file")
        mc_files = list(latest.get("mc_files") or [])
        source = str(latest.get("source") or source)
        run_dir = latest.get("run_dir")
        data_label = str(latest.get("data_label") or "")
        mc_labels = list(latest.get("mc_labels") or [])

    if not data_file:
        preferred = cached_by_key.get(_PREFERRED_DATA_KEY)
        if preferred:
            data_file = preferred["key"]
            data_label = preferred["label"]
        elif cached:
            data_file = cached[0]["key"]
            data_label = cached[0]["label"]

    if not mc_files:
        stack_keys = [k for k in OPTION3_MC_STACK_KEYS if k in cached_by_key]
        if len(stack_keys) >= 2:
            mc_files = stack_keys
            mc_labels = [cached_by_key[k]["label"] for k in stack_keys]
        else:
            preferred_mc = cached_by_key.get(_PREFERRED_MC_KEY)
            if preferred_mc:
                mc_files = [preferred_mc["key"]]
                mc_labels = [preferred_mc["label"]]
            else:
                mc_files = [
                    row["key"]
                    for row in cached
                    if row["key"] != data_file
                ][:3]
                mc_labels = [
                    cached_by_key[k]["label"] for k in mc_files if k in cached_by_key
                ]

    return {
        "source": source,
        "run_dir": run_dir,
        "data_file": data_file or _PREFERRED_DATA_KEY,
        "mc_files": mc_files or [_PREFERRED_MC_KEY],
        "data_label": data_label or _PREFERRED_DATA_KEY,
        "mc_labels": mc_labels,
        "cached_datasets": cached,
    }


def _dataset_choice_rows() -> list[dict[str, str]]:
    defaults = discover_mc_validation_defaults()
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    def _add(value: str, label: str) -> None:
        if value in seen:
            return
        seen.add(value)
        rows.append({"value": value, "label": label})

    if defaults.get("data_file"):
        _add(
            str(defaults["data_file"]),
            f"Latest default — {defaults.get('data_label') or defaults['data_file']}",
        )
    for row in defaults.get("cached_datasets") or []:
        _add(str(row["key"]), str(row["label"]))
    return rows


def _mc_choice_default(defaults: dict[str, Any]) -> str:
    mc_files = defaults.get("mc_files") or []
    cached = {row["key"]: row["label"] for row in defaults.get("cached_datasets") or []}
    labels = [cached.get(key, key) for key in mc_files]
    if labels:
        return ", ".join(labels)
    return ", ".join(OPTION3_MC_STACK_KEYS)


def resolve_mc_validation_query(query: str, *, field: str = "data") -> str:
    """Map picker label / manifest key / path to a validation input."""
    text = (query or "").strip()
    defaults = discover_mc_validation_defaults()
    if not text:
        if field == "data":
            return str(defaults.get("data_file") or _PREFERRED_DATA_KEY)
        return _mc_choice_default(defaults)

    for row in _dataset_choice_rows():
        if text == row["label"] or text == row["value"]:
            return row["value"]

    if " — " in text:
        head = text.split(" — ", 1)[0].strip()
        resolved = _resolve_dataset_ref(head)
        if resolved:
            return resolved

    resolved = _resolve_dataset_ref(text)
    return resolved or text


def resolve_mc_validation_mc_files(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        defaults = discover_mc_validation_defaults()
        return [str(item) for item in defaults.get("mc_files") or []]

    lowered = text.lower()
    if lowered in {"option3", "full_stack", "expanded", "default_stack"}:
        from menus.particle.cms.tav_mc_stack_option3 import resolve_option3_mc_files

        return resolve_option3_mc_files(text)

    parts = [part.strip() for part in text.split(",") if part.strip()]
    resolved: list[str] = []
    for part in parts:
        if " — " in part:
            part = part.split(" — ", 1)[0].strip()
        value = resolve_mc_validation_query(part, field="mc")
        if value:
            resolved.append(value)
    return resolved


def entry_fields(action: str) -> list[dict]:
    if action == "Run MC Validation Suite":
        defaults = discover_mc_validation_defaults()
        data_choices = [row["label"] for row in _dataset_choice_rows()]
        if not data_choices:
            data_choices = [_PREFERRED_DATA_KEY]

        default_data_label = data_choices[0]
        for row in _dataset_choice_rows():
            if row["value"] == defaults.get("data_file"):
                default_data_label = row["label"]
                break

        mc_default = _mc_choice_default(defaults)
        cached = defaults.get("cached_datasets") or []
        mc_hint_keys = ", ".join(row["key"] for row in cached if row["key"] != defaults.get("data_file"))

        return append_llm_entry_fields(
            [
                {
                    "key": "data_file",
                    "label": "Data file (auto-discovered)",
                    "default": default_data_label,
                    "required": False,
                    "choices": data_choices,
                    "hint": (
                        f"Defaults to latest run / cached data "
                        f"({defaults.get('source')})"
                    ),
                },
                {
                    "key": "mc_files",
                    "label": "MC files (comma-separated)",
                    "default": mc_default,
                    "required": False,
                    "hint": (
                        "Option 3 default: DY + ttbar + Higgs (cms_nanoaod_dy, "
                        "cms_nanoaod_ttbar, cms_nanoaod_higgs_zz). "
                        "Type option3 for full stack."
                        + (f" Cached: {mc_hint_keys}" if mc_hint_keys else "")
                    ),
                },
                {
                    "key": "prefetch_option3_mc",
                    "label": "Prefetch DY+ttbar+Higgs (yes/no)",
                    "default": "yes",
                    "required": False,
                    "choices": ["yes", "no"],
                    "hint": "Auto-download missing Option 3 MC from CERN Open Data",
                },
                {
                    "key": "photon_crosscheck_file",
                    "label": "Secondary gamma dataset (Option 4)",
                    "default": "option4",
                    "required": False,
                    "hint": (
                        "DoubleElectron / ZZTo4e / Higgs for γ–μ crosscheck. "
                        f"Keys: {', '.join(OPTION4_PHOTON_CROSSCHECK_KEYS)}. "
                        "Type option4 for auto-resolve."
                    ),
                },
                {
                    "key": "prefetch_option4_photon",
                    "label": "Prefetch Option 4 photon datasets (yes/no)",
                    "default": "yes",
                    "required": False,
                    "choices": ["yes", "no"],
                    "hint": "Auto-download DoubleElectron + ZZTo4e from CERN Open Data",
                },
                {
                    "key": "photon_crosscheck",
                    "label": "Option 4 photon crosscheck (yes/no)",
                    "default": "yes",
                    "required": False,
                    "choices": ["yes", "no"],
                    "hint": "Activate photon kinematics winding-rate hooks + μ–γ 7-fold compare",
                },
                {
                    "key": "entry_stop",
                    "label": "Max events per file (0 = all)",
                    "default": "0",
                    "required": False,
                    "hint": "Quick test: 50000; full dimu skim: 0",
                },
                {
                    "key": "output_dir",
                    "label": "Output directory",
                    "default": "",
                    "required": False,
                    "hint": "Leave blank to create a canonical artifacts/cern_analysis run directory.",
                },
                {
                    "key": "pileup_reweight",
                    "label": "Pileup reweight (yes/no)",
                    "default": "yes",
                    "required": False,
                    "choices": ["yes", "no"],
                },
                {
                    "key": "pileup_data_profile",
                    "label": "Official data pileup profile (JSON path)",
                    "default": "",
                    "required": False,
                    "hint": "Optional CMS nTrueInt target JSON; blank = extract from data NanoAOD",
                },
                {
                    "key": "lepton_sf",
                    "label": "Muon/lepton scale factor",
                    "default": "0.979",
                    "required": False,
                    "hint": "Run2012 nominal ~0.979; override with official JSON-derived SF",
                },
                {
                    "key": "photon_sf",
                    "label": "Photon scale factor",
                    "default": "1.0",
                    "required": False,
                },
                {
                    "key": "dependence_studies",
                    "label": "Dependence studies (yes/no)",
                    "default": "yes",
                    "required": False,
                    "choices": ["yes", "no"],
                },
                {
                    "key": "enhanced_validation",
                    "label": "Enhanced validation v2 (yes/no)",
                    "default": "yes",
                    "required": False,
                    "hint": "Weighted MC, photon channel, multi-MC background subtraction",
                    "choices": ["yes", "no"],
                },
                {
                    "key": "geometric_floor_scan",
                    "label": "Phase 4 geometric floor (yes/no)",
                    "default": "yes",
                    "required": False,
                    "hint": "313.1 MeV friction-floor isolation (Option A recommended)",
                    "choices": ["yes", "no"],
                },
                {
                    "key": "option_a_recoil_sweep",
                    "label": "Option A relaxed recoil sweep (yes/no)",
                    "default": "yes",
                    "required": False,
                    "hint": "Recoil 15/30/45 GeV + ΔR≤1.5 validation gate (no Δφ joint cut)",
                    "choices": ["yes", "no"],
                },
                {
                    "key": "option2_tight_cuts",
                    "label": "Option 2 tight cuts add-on (yes/no)",
                    "default": "no",
                    "required": False,
                    "hint": "Also run tight Δφ≥2.5, recoil≤15 GeV scan alongside Option A",
                    "choices": ["yes", "no"],
                },
                {
                    "key": "recoil_pt_max",
                    "label": "Recoil pT max override (GeV, Option 2 only)",
                    "default": "15",
                    "required": False,
                    "hint": "Dimuon system pT ceiling when Option 2 tight cuts enabled",
                },
                {
                    "key": "delta_phi_min",
                    "label": "Min Δφ override (radians, Option 2 only)",
                    "default": "2.5",
                    "required": False,
                },
                {
                    "key": "delta_r_max",
                    "label": "Max ΔR (Option A validation gate)",
                    "default": "1.5",
                    "required": False,
                    "hint": "Loose angular gate for DoubleMu; default 1.5 (no Δφ_min joint cut)",
                },
            ]
        )
    return []


def entry_instructions(action: str) -> list[str]:
    defaults = discover_mc_validation_defaults()
    lines = [
        "Run the full TAV CMS MC validation suite: plots, JSON summary, and tables.",
        f"Auto-discovered data default: {defaults.get('data_label') or defaults.get('data_file')}.",
        f"Auto-discovered MC default: {', '.join(defaults.get('mc_labels') or defaults.get('mc_files') or [])}.",
        "Leave fields blank to use latest validation run or cached datasets/cern/.",
        "Leave Output directory blank to create a canonical artifacts/cern_analysis run directory.",
        "Enhanced v2 adds weighted 7-fold diagnostics, photon kinematics hooks, and multi-MC shape tests.",
        "Option A (Phase 4, recommended) sweeps recoil 15/30/45 GeV with ΔR≤1.5 validation gate (no Δφ joint cut).",
        "Option 2 tight cuts (optional) adds aggressive Δφ≥2.5, recoil≤15 GeV pT scan.",
        "Option 3 expands MC to DYJetsToLL + TTbar + Higgs for background-subtracted shape tests.",
        "Option 4 maps a secondary gamma/EM dataset (DoubleElectron) for photon_enriched + photon_crosscheck.",
    ]
    if defaults.get("run_dir"):
        lines.append(f"Latest validation artifacts: {defaults['run_dir']}")
    return lines


def run_action(selection: str, show_plots: bool = True, options: dict | None = None) -> str | None:
    options = options or {}
    action = _ACTION_KEYS.get(selection)
    if action is None:
        return None

    if action == "run_validation":
        data_file = resolve_mc_validation_query(
            str(options.get("data_file") or ""),
            field="data",
        )
        mc_files = resolve_mc_validation_mc_files(str(options.get("mc_files") or ""))
        output_dir = (options.get("output_dir") or "").strip()
        pileup = (options.get("pileup_reweight") or "yes").strip().lower() in ("yes", "y", "true", "1")
        pileup_profile_raw = (options.get("pileup_data_profile") or "").strip()
        pileup_data_profile = pileup_profile_raw or None
        lepton_sf_raw = (options.get("lepton_sf") or "").strip()
        photon_sf_raw = (options.get("photon_sf") or "").strip()
        lepton_sf = float(lepton_sf_raw) if lepton_sf_raw else None
        photon_sf = float(photon_sf_raw) if photon_sf_raw else None
        depend = (options.get("dependence_studies") or "yes").strip().lower() in ("yes", "y", "true", "1")
        enhanced = (options.get("enhanced_validation") or "yes").strip().lower() in (
            "yes",
            "y",
            "true",
            "1",
        )
        geometric_floor = (options.get("geometric_floor_scan") or "yes").strip().lower() in (
            "yes",
            "y",
            "true",
            "1",
        )
        option_a_sweep = (options.get("option_a_recoil_sweep") or "yes").strip().lower() in (
            "yes",
            "y",
            "true",
            "1",
        )
        option2_tight = (options.get("option2_tight_cuts") or "no").strip().lower() in (
            "yes",
            "y",
            "true",
            "1",
        )
        kinematic_cuts: dict[str, float] = {}
        for key, field in (
            ("recoil_pt_max", "recoil_pt_max"),
            ("delta_phi_min", "delta_phi_min"),
            ("delta_r_max", "delta_r_max"),
        ):
            raw = (options.get(field) or "").strip()
            if raw:
                try:
                    kinematic_cuts[key] = float(raw)
                except ValueError:
                    pass
        try:
            entry_stop_raw = int(str(options.get("entry_stop") or "0").strip())
        except ValueError:
            entry_stop_raw = 0
        entry_stop = entry_stop_raw if entry_stop_raw > 0 else None

        prefetch = (options.get("prefetch_option3_mc") or "yes").strip().lower() in (
            "yes",
            "y",
            "true",
            "1",
        )
        if prefetch:
            try:
                from menus.particle.cms.tav_mc_stack_option3 import ensure_option3_mc_cached

                ensure_option3_mc_cached(verbose=True, auto_fetch=True)
            except Exception as exc:
                print(f"[MC VALIDATION] Option 3 prefetch warning: {exc}")

        prefetch_photon = (options.get("prefetch_option4_photon") or "yes").strip().lower() in (
            "yes",
            "y",
            "true",
            "1",
        )
        if prefetch_photon:
            try:
                from menus.particle.cms.tav_photon_crosscheck_option4 import (
                    ensure_option4_photon_cached,
                )

                ensure_option4_photon_cached(verbose=True, auto_fetch=True)
            except Exception as exc:
                print(f"[MC VALIDATION] Option 4 prefetch warning: {exc}")

        if not data_file or not mc_files:
            raise ValueError("data_file and mc_files are required (no cached datasets found)")

        try:
            from menus.particle.cms.tav_mc_validation_suite import run_full_validation
            from menus.particle.cern.fetcher import resolve_root_path
            from tav_shared.artifact_paths import artifact_run_dir
        except Exception as e:
            raise ImportError(f"Validation suite not available: {e}") from e

        resolved_path = resolve_root_path(data_file)
        if resolved_path is not None:
            resolved_data_file = str(resolved_path)
        elif os.path.exists(data_file) and data_file.lower().endswith(".root"):
            resolved_data_file = data_file
        else:
            resolved_data_file = data_file

        resolved_mc_files: list[str] = []
        for mc in mc_files:
            mc_path = resolve_root_path(mc)
            if mc_path is not None:
                resolved_mc_files.append(str(mc_path))
            elif os.path.exists(mc) and mc.lower().endswith(".root"):
                resolved_mc_files.append(mc)
            else:
                resolved_mc_files.append(mc)

        print(f"[MC VALIDATION] Data: {resolved_data_file}")
        print(f"[MC VALIDATION] MC: {', '.join(resolved_mc_files)}")

        dataset_hint = os.path.basename(resolved_data_file) or data_file or "cms_validation"
        dataset_slug = compose_dataset_slug(dataset_hint)

        if output_dir:
            final_output_dir = output_dir
        else:
            run_dir = artifact_run_dir(TestSlug.CERN, dataset_slug, create=True)
            final_output_dir = str(run_dir)

        photon_crosscheck = (options.get("photon_crosscheck") or "yes").strip().lower() in (
            "yes",
            "y",
            "true",
            "1",
        )
        photon_crosscheck_raw = (options.get("photon_crosscheck_file") or "option4").strip()
        photon_crosscheck_file: str | None = None
        if photon_crosscheck_raw and photon_crosscheck_raw.lower() not in {"", "none", "no"}:
            from menus.particle.cms.tav_photon_crosscheck_option4 import (
                resolve_photon_crosscheck_dataset,
            )

            photon_crosscheck_file = resolve_photon_crosscheck_dataset(photon_crosscheck_raw)

        res = run_full_validation(
            data_file=resolved_data_file,
            mc_files=resolved_mc_files,
            output_dir=final_output_dir,
            pileup_data_profile=pileup_data_profile,
            lepton_sf=lepton_sf,
            photon_sf=photon_sf,
            do_pileup_reweight=pileup,
            do_dependence_studies=depend,
            do_enhanced_validation=enhanced,
            do_geometric_floor_scan=geometric_floor,
            do_option_a_recoil_sweep=option_a_sweep,
            do_option2_tight_cuts=option2_tight,
            do_expanded_mc_stack=True,
            do_photon_crosscheck=photon_crosscheck,
            photon_crosscheck_file=photon_crosscheck_file,
            kinematic_cuts=kinematic_cuts or None,
            entry_stop=entry_stop,
            verbose=True,
        )
        if res.get("summary"):
            print("[MC VALIDATION] Summary:", res["summary"].get("classic_verdict"), end="")
            if res["summary"].get("enhanced_report_path"):
                print(f" | enhanced: {res['summary']['enhanced_report_path']}")
            else:
                print()
        try:
            return str(res.get("report_path") or res.get("timestamp") or os.path.basename(final_output_dir))
        except Exception:
            return os.path.basename(final_output_dir)

    return None