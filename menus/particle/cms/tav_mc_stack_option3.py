"""
Option 3 — expanded Monte Carlo sample base for Tav CMS validation.

Adds Drell-Yan + ttbar (+ optional Higgs signal) to the MC stack so shape
mismatch verdicts and background template subtraction use dominant SM processes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from menus.particle.cern.manifest import OPTION3_MC_STACK_KEYS
from tav_shared.artifact_paths import (
    TestSlug,
    artifact_path,
    artifact_timestamp,
    compose_dataset_slug,
)

OPTION3_LABEL = "Option 3 — Expanded Monte Carlo Sample Base"
MOD7_THIRD_SECTOR_BIN = 2  # mod 7 ≡ 2 topological peak (third sector, 0-indexed)


def _mod7_fractions_from_results(results: dict[str, Any]) -> list[float] | None:
    mult = results.get("muon_multiplicity_7fold") or results.get("multiplicity_7fold")
    if not mult:
        return None
    fracs = mult.get("mod7_fractions")
    if not fracs or len(fracs) < 7:
        return None
    return [float(x) for x in fracs[:7]]


def _event_weight(results: dict[str, Any]) -> float:
    n = results.get("n_events_processed")
    if n is None:
        mult = results.get("muon_multiplicity_7fold") or {}
        n = mult.get("n_events")
    try:
        w = float(n or 0)
    except (TypeError, ValueError):
        w = 0.0
    return max(w, 1.0)


def classify_mc_role(name: str) -> str:
    """Return ``background``, ``signal``, or ``other`` from manifest key / filename."""
    lower = str(name).lower()
    if any(h in lower for h in ("higgs", "smhiggs", "signal", "zz4l", "zz")):
        return "signal"
    if any(h in lower for h in ("dy", "drell", "dyjets", "ttbar", "tt_", "qcd", "wjets")):
        return "background"
    if "cms_nanoaod_dy" in lower or lower.endswith("_dy"):
        return "background"
    if "cms_nanoaod_ttbar" in lower or "ttbar" in lower:
        return "background"
    return "other"


def discover_option3_mc_stack(
    cached_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve DY + ttbar + Higgs keys that are present in the local cache."""
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
    for key in OPTION3_MC_STACK_KEYS:
        if key in by_key and by_key[key].get("cached"):
            resolved.append(key)
        else:
            missing.append(key)

    return {
        "stack_keys": list(OPTION3_MC_STACK_KEYS),
        "resolved_keys": resolved,
        "missing_keys": missing,
        "cached_rows": [by_key[k] for k in resolved if k in by_key],
    }


def resolve_option3_mc_files(
    raw: str | None = None,
    *,
    include_signal: bool = True,
) -> list[str]:
    """
    Expand menu input into manifest keys for the full Option 3 stack.

    Accepts ``option3``, ``full_stack``, ``expanded``, or comma-separated keys.
    """
    text = (raw or "").strip().lower()
    if not text or text in {"option3", "full_stack", "expanded", "default_stack"}:
        stack = discover_option3_mc_stack()
        keys = list(stack["resolved_keys"])
        if not include_signal:
            keys = [k for k in keys if classify_mc_role(k) != "signal"]
        if keys:
            return keys
        return [k for k in OPTION3_MC_STACK_KEYS if k != "cms_nanoaod_higgs_zz" or include_signal]

    from menus.particle.cms.validation_extension import resolve_mc_validation_mc_files

    return resolve_mc_validation_mc_files(raw or "")


def assess_mod7_third_sector_closure(
    data_results: dict[str, Any],
    mc_results_by_name: dict[str, dict[str, Any]],
    *,
    sector_bin: int = MOD7_THIRD_SECTOR_BIN,
) -> dict[str, Any]:
    """
    Test whether the expanded MC stack accounts for data occupancy in mod 7 sector 2.

    Uses event-count-weighted stacking of background MC (DY + ttbar) and full stack.
    """
    data_fracs = _mod7_fractions_from_results(data_results)
    if not data_fracs:
        return {"verdict": "UNDERPOWERED", "reason": "data mod7 fractions unavailable"}

    per_mc: dict[str, Any] = {}
    bkg_weights: list[float] = []
    bkg_frac2: list[float] = []
    stack_weights: list[float] = []
    stack_frac2: list[float] = []

    for name, mc_res in mc_results_by_name.items():
        fracs = _mod7_fractions_from_results(mc_res)
        if not fracs:
            continue
        w = _event_weight(mc_res)
        role = classify_mc_role(name)
        frac2 = float(fracs[sector_bin])
        per_mc[name] = {
            "role": role,
            "mod7_fraction_sector": frac2,
            "mod7_fractions": fracs,
            "n_events_weight": w,
        }
        stack_weights.append(w)
        stack_frac2.append(frac2)
        if role == "background":
            bkg_weights.append(w)
            bkg_frac2.append(frac2)

    if not stack_frac2:
        return {"verdict": "UNDERPOWERED", "reason": "no MC mod7 fractions"}

    data_sector = float(data_fracs[sector_bin])
    bkg_stacked = (
        float(np.average(bkg_frac2, weights=bkg_weights))
        if bkg_weights
        else 0.0
    )
    full_stacked = float(np.average(stack_frac2, weights=stack_weights))

    gap_vs_bkg = data_sector - bkg_stacked
    gap_vs_full = data_sector - full_stacked
    uniform = 1.0 / 7.0
    data_excess_vs_uniform = data_sector - uniform
    bkg_excess_vs_uniform = bkg_stacked - uniform

    closes_third_sector = abs(gap_vs_bkg) < 0.05 or abs(gap_vs_full) < 0.05
    sm_accounts_for_peak = bkg_stacked >= uniform * 0.9 and gap_vs_bkg <= 0.1

    return {
        "sector_bin": int(sector_bin),
        "sector_label": f"mod7≡{sector_bin}",
        "data_fraction_sector": data_sector,
        "background_stacked_fraction": bkg_stacked,
        "full_stack_fraction": full_stacked,
        "gap_data_minus_background_stack": gap_vs_bkg,
        "gap_data_minus_full_stack": gap_vs_full,
        "data_excess_vs_uniform": data_excess_vs_uniform,
        "background_excess_vs_uniform": bkg_excess_vs_uniform,
        "fraction_of_uniform_in_sector": data_sector / uniform if uniform else 0.0,
        "sm_stack_accounts_for_third_sector": sm_accounts_for_peak,
        "third_sector_closure": closes_third_sector,
        "per_mc": per_mc,
        "interpretation": (
            f"Compares data occupancy in mod-7 bin {sector_bin} against "
            "DY+ttbar backgrounds and the full MC stack to test whether "
            "electroweak + top production explain the third-sector accumulation."
        ),
    }


def run_option3_expanded_mc_analysis(
    data_results: dict[str, Any],
    mc_results_by_name: dict[str, dict[str, Any]],
    *,
    enhanced_report: dict[str, Any] | None = None,
    output_dir: str | Path = "option3_mc_stack",
    run_when: datetime | None = None,
    dataset_slug: str = "option3_mc_stack",
    save_report: bool = True,
) -> dict[str, Any]:
    """Option 3 analysis: multi-MC stack + third-sector closure + enhanced bkg-sub."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_when = run_when or datetime.now(timezone.utc)

    sector = assess_mod7_third_sector_closure(data_results, mc_results_by_name)
    stack = discover_option3_mc_stack()

    report: dict[str, Any] = {
        "option": OPTION3_LABEL,
        "timestamp": artifact_timestamp(run_when),
        "mc_sample_names": sorted(mc_results_by_name.keys()),
        "stack_manifest_keys": list(OPTION3_MC_STACK_KEYS),
        "stack_cache_status": stack,
        "mod7_third_sector": sector,
        "objective": (
            "Test whether Drell-Yan + ttbar (+ signal) account for missing event "
            "accumulation in the third mod-7 sector before attributing shape mismatch "
            "to the Tav engine."
        ),
        "output_dir": str(out_dir),
    }

    if enhanced_report:
        dvm = enhanced_report.get("data_vs_mc") or {}
        report["enhanced_validation"] = {
            "background_templates": dvm.get("background_templates"),
            "background_subtracted_7fold_amplitude": dvm.get(
                "background_subtracted_7fold_amplitude"
            ),
            "background_subtracted_7fold_significance_proxy": dvm.get(
                "background_subtracted_7fold_significance_proxy"
            ),
            "normalization_scale_applied": dvm.get("normalization_scale_applied"),
            "report_path": enhanced_report.get("report_path"),
        }

    if save_report:
        path = artifact_path(
            TestSlug.CERN,
            compose_dataset_slug(dataset_slug, "option3_mc_stack"),
            "report",
            "json",
            when=run_when,
            run_dir=out_dir,
        )
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=float)
        report["report_path"] = str(path)

    return report


def print_option3_mc_stack_summary(report: dict[str, Any]) -> None:
    """Stdout banner for Option 3 expanded MC stack analysis."""
    print("=== TAV OPTION 3: EXPANDED MC STACK ===")
    if report.get("verdict") == "UNDERPOWERED":
        print(f"  Status            : UNDERPOWERED ({report.get('reason', 'n/a')})")
        print("=" * 40)
        return

    sector = report.get("mod7_third_sector") or {}
    cache = report.get("stack_cache_status") or {}
    missing = cache.get("missing_keys") or []
    print(f"  MC samples        : {', '.join(report.get('mc_sample_names') or [])}")
    if missing:
        print(f"  Missing from cache: {', '.join(missing)}")
    if sector and sector.get("verdict") != "UNDERPOWERED":
        label = sector.get("sector_label", "mod7≡2")
        print(f"  Data {label} frac  : {sector.get('data_fraction_sector', 0):.4f}")
        print(
            f"  Bkg stack (DY+tt) : {sector.get('background_stacked_fraction', 0):.4f}"
        )
        print(
            f"  Gap (data−bkg)    : {sector.get('gap_data_minus_background_stack', 0):+.4f}"
        )
        print(
            f"  3rd-sector closes : "
            f"{'yes' if sector.get('third_sector_closure') else 'no'}"
        )
    enh = report.get("enhanced_validation") or {}
    if enh.get("background_subtracted_7fold_amplitude") is not None:
        templates = enh.get("background_templates") or []
        if templates:
            print(f"  Bkg templates     : {', '.join(templates)}")
        print(
            f"  Bkg-sub 7-fold amp: {enh.get('background_subtracted_7fold_amplitude', 0):.3f}"
        )
        print(
            f"  Bkg-sub σ proxy   : "
            f"{enh.get('background_subtracted_7fold_significance_proxy', 0):.3f}"
        )
    if report.get("report_path"):
        print(f"  Report            : {report['report_path']}")
    print("=" * 40)


def ensure_option3_mc_cached(
    *,
    verbose: bool = True,
    auto_fetch: bool = True,
) -> dict[str, Any]:
    """Prefetch DY + ttbar + Higgs manifests when not already cached."""
    from menus.particle.cern.fetcher import ensure_cern_target

    status: dict[str, Any] = {"fetched": [], "cached": [], "failed": {}}
    for key in OPTION3_MC_STACK_KEYS:
        try:
            path = ensure_cern_target(key, auto_fetch=auto_fetch)
            status["cached"].append({"key": key, "path": str(path)})
            if verbose:
                print(f"[Option 3] Ready: {key} → {path}")
        except Exception as exc:
            status["failed"][key] = str(exc)
            if verbose:
                print(f"[Option 3] Missing/failed: {key} — {exc}")
    return status