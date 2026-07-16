"""
NGC 3198 Phase 1 rotation-curve benchmark (tau-cosmology pre-registration).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy.optimize import minimize

from menus.astronomical.berard.constants import BERARD_CONSTANT
from menus.astronomical.berard.halo_models import (
    MODEL_SPECS,
    bic_verdict,
    chi2_reduced,
    information_criteria,
    model_velocity,
)
from menus.astronomical.berard.ngc3198_data import ingest_ngc3198_from_sparc, load_ngc3198_curve
from menus.astronomical.berard.phase2_residual import run_phase2_residual_analysis
from tav_shared.artifact_paths import TestSlug, artifact_path, artifact_timestamp


def _fit_model(
    kind: str,
    r: np.ndarray,
    v_obs: np.ndarray,
    e_vobs: np.ndarray,
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
    *,
    n_params: int,
) -> dict[str, Any]:
    """Nelder-Mead χ² fit with pre-registered prior ranges."""

    def objective(x: np.ndarray) -> float:
        params: dict[str, float] = {"upsilon_disk": float(x[0])}
        if kind in {"piso", "berard_piso", "burkert"}:
            params["rho0_pc3"] = float(x[1])
            params["rc_kpc"] = float(x[2])
        elif kind == "nfw":
            params["rho_s"] = float(x[1])
            params["r_s"] = float(x[2])
        if kind == "berard_piso":
            params["bc"] = BERARD_CONSTANT
        pred = model_velocity(kind, r, v_gas, v_disk, v_bulge, params)
        err = np.maximum(e_vobs, 1.0)
        return float(np.sum(((v_obs - pred) / err) ** 2))

    if kind == "baryons":
        x0 = np.array([0.5])
        bounds = [(0.1, 1.0)]
    elif kind in {"piso", "berard_piso", "burkert"}:
        x0 = np.array([0.5, 0.004, 8.0])
        bounds = [(0.1, 1.0), (1e-4, 5e-2), (0.5, 30.0)]
    elif kind == "nfw":
        x0 = np.array([0.5, 1e7, 5.0])
        bounds = [(0.1, 1.0), (1e5, 1e9), (0.5, 30.0)]
    else:
        raise ValueError(kind)

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
    )
    x = result.x
    params: dict[str, float] = {"upsilon_disk": float(x[0])}
    if kind in {"piso", "berard_piso", "burkert"}:
        params["rho0_pc3"] = float(x[1])
        params["rc_kpc"] = float(x[2])
    elif kind == "nfw":
        params["rho_s"] = float(x[1])
        params["r_s"] = float(x[2])
    if kind == "berard_piso":
        params["bc"] = BERARD_CONSTANT

    v_model = model_velocity(kind, r, v_gas, v_disk, v_bulge, params)
    chi2 = float(result.fun)
    ic = information_criteria(chi2, r.size, n_params)
    return {
        "kind": kind,
        "params": params,
        "v_model": v_model.tolist(),
        "chi2": chi2,
        "chi2_reduced": chi2_reduced(v_obs, v_model, e_vobs),
        **ic,
        "success": bool(result.success),
    }


def run_ngc3198_phase1_benchmark(*, verbose: bool = True) -> dict[str, Any]:
    """Phase 1 smooth-model comparison on frozen NGC 3198 data."""
    ingest_ngc3198_from_sparc(verbose=verbose)
    data = load_ngc3198_curve()
    r = data["r_kpc"]
    v_obs = data["Vobs"]
    e_vobs = data["eVobs"]

    models: dict[str, Any] = {}
    for key, spec in MODEL_SPECS.items():
        kind = spec.kind
        if key == "berard_hybrid_piso":
            kind = "berard_piso"
        fit = _fit_model(
            kind,
            r,
            v_obs,
            e_vobs,
            data["Vgas"],
            data["Vdisk"],
            data["Vbulge"],
            n_params=spec.n_params,
        )
        fit["model_name"] = key
        models[key] = fit

    baseline = models["baryons_only"]
    comparisons: list[dict[str, Any]] = []
    for name, fit in models.items():
        if name == "baryons_only":
            continue
        delta_bic = float(fit["bic"] - baseline["bic"])
        comparisons.append(
            {
                "model": name,
                "delta_bic_vs_baryons": delta_bic,
                "delta_aic_vs_baryons": float(fit["aic"] - baseline["aic"]),
                "verdict_vs_baryons": bic_verdict(delta_bic),
            }
        )

    best = min(models.values(), key=lambda m: m["bic"])
    phase1_verdict = "HONEST_TIE_EXPECTED"
    for comp in comparisons:
        if comp["verdict_vs_baryons"] == "strong_favored" and comp["model"] in {
            "plain_piso",
            "berard_hybrid_piso",
        }:
            phase1_verdict = "CORED_MODEL_FAVORED_OVER_BARYONS"
            break

    report: dict[str, Any] = {
        "action": "NGC 3198 Phase 1 benchmark",
        "timestamp": artifact_timestamp(datetime.now(timezone.utc)),
        "galaxy": "NGC3198",
        "n_points": int(r.size),
        "models": models,
        "comparisons_vs_baryons": comparisons,
        "best_model_by_bic": best["model_name"],
        "phase1_verdict": phase1_verdict,
        "preregistration_note": (
            "Plain pISO and berard_hybrid_piso are functionally identical at Phase 1; "
            "distinguishing power is Phase 2 only."
        ),
        "confidence": "exploratory",
    }

    if verbose:
        print("=" * 60)
        print("NGC 3198 — Phase 1 model comparison")
        for name, fit in models.items():
            print(
                f"  {name:22s} χ²_red={fit['chi2_reduced']:.2f}  "
                f"BIC={fit['bic']:.1f}"
            )
        print(f"  Best (BIC)         : {best['model_name']}")
        print(f"  Phase 1 verdict    : {phase1_verdict}")
        print("=" * 60)
    return report


def run_ngc3198_full_benchmark(
    *,
    verbose: bool = True,
    save_report: bool = True,
) -> dict[str, Any]:
    """Phase 1 + Phase 2 on frozen NGC 3198 with optional plot generation."""
    from menus.astronomical.berard.plotting import plot_ngc3198_benchmark
    from menus.astronomical.desi.json_util import write_json

    phase1 = run_ngc3198_phase1_benchmark(verbose=verbose)
    data = load_ngc3198_curve()
    best_name = phase1["best_model_by_bic"]
    best = phase1["models"][best_name]
    spec = MODEL_SPECS[best_name]
    kind = spec.kind if best_name != "berard_hybrid_piso" else "berard_piso"
    v_smooth = np.asarray(best["v_model"], dtype=float)

    phase2 = run_phase2_residual_analysis(
        data["r_kpc"],
        data["Vobs"],
        v_smooth,
        data["eVobs"],
        verbose=verbose,
    )

    plot_paths = plot_ngc3198_benchmark(
        data,
        phase1["models"],
        best_name,
        phase2,
    )

    report = {
        **phase1,
        "phase2_residual": phase2,
        "plots": plot_paths,
        "best_smooth_model": best_name,
    }

    if save_report:
        path = artifact_path(TestSlug.BERARD, "ngc3198_benchmark", "report", "json")
        write_json(path, report, indent=2, sort_keys=True)
        report["report_path"] = str(path)

    return report