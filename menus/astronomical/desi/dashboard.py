#!/usr/bin/env python3
"""Full DESI residual diagnostic dashboard (ACF, Q-Q, SoundHorizon, summary)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from menus.tsb_research.core import (
    ARTIFACTS_DIR,
    ResidualDiagnostics,
    SoundHorizon,
    TavSuperblockVariables,
    load_desi_fit_residuals,
)

DASHBOARD_ACTION = "Generate Full DESI Summary Dashboard"
DASHBOARD_DIR = ARTIFACTS_DIR / "desi_dashboard"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _resample_vectors(
    residuals: np.ndarray,
    fitted: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate residuals/fitted onto ``n`` evenly spaced points along the fit axis."""
    n = max(2, int(n))
    src = np.linspace(0.0, 1.0, len(residuals))
    dst = np.linspace(0.0, 1.0, n)
    return (
        np.interp(dst, src, np.asarray(residuals, dtype=float)),
        np.interp(dst, src, np.asarray(fitted, dtype=float)),
    )


def _load_dashboard_bundle(n: int) -> dict[str, Any]:
    """Prefer genuine DESI fit residuals; resize to requested n; fall back to synthetic."""
    n = max(2, int(n))
    try:
        bundle = load_desi_fit_residuals(auto_calibrate_gamma=True)
        residuals = np.asarray(bundle["residuals"], dtype=float).reshape(-1)
        fitted = np.asarray(bundle["y_model"], dtype=float).reshape(-1)
        n_source = int(len(residuals))
        if n_source != n:
            residuals, fitted = _resample_vectors(residuals, fitted, n)
        data_class = bundle.get("data_class", "real_cached")
        provenance = {
            "source": "desi_fit",
            "tracer": bundle.get("tracer"),
            "data_mode": bundle.get("data_mode"),
            "gamma": bundle.get("gamma"),
            "n_data": n_source,
            "n_requested": n,
            "resampled": n_source != n,
            "chi2_reduced": bundle.get("chi2_reduced"),
        }
        return {
            "residuals": residuals,
            "fitted": fitted,
            "data_class": data_class,
            "provenance": provenance,
        }
    except Exception as exc:
        raise RuntimeError(
            f"DESI dashboard requires live DR2 fit residuals; load failed: {exc}"
        ) from exc


def _plot_sound_horizon_curve(sh: SoundHorizon, save_path: Path) -> None:
    import matplotlib.pyplot as plt

    gammas = np.linspace(7.5, 12.5, 41)
    rds = np.array([sh.predict_rd(float(g)) for g in gammas], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gammas, rds, "b-", linewidth=2, label=r"$r_d^{\mathrm{pred}}(\gamma)$")
    ax.axhline(147.0, color="red", linestyle="--", alpha=0.7, label="Observed 147 Mpc")
    ax.set_xlabel("γ (cylinder stretch)")
    ax.set_ylabel(r"$r_d$ (Mpc)")
    ax.set_title("SoundHorizon Φ-anchored calibration")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_summary_panel(report: dict[str, Any], save_path: Path) -> None:
    import matplotlib.pyplot as plt

    lines = [
        "DESI Diagnostic Dashboard Summary",
        f"n_residuals : {report.get('n_residuals', 'N/A')}",
        f"data_class  : {report.get('data_class', 'N/A')}",
        f"Q-Q R²      : {report.get('qq_r2', 'N/A')}",
        f"BP p-value  : {report.get('breusch_pagan', {}).get('lm_pvalue', 'N/A')}",
        f"SoundHorizon γ=9.5 : {report.get('sound_horizon_mpc', 'N/A')} Mpc",
    ]
    prov = report.get("provenance") or {}
    if prov.get("tracer"):
        lines.append(f"tracer      : {prov.get('tracer')} | {prov.get('data_mode')}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    ax.text(0.03, 0.97, "\n".join(lines), va="top", family="monospace", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_full_desi_dashboard(*, n: int = 25) -> dict[str, Any]:
    """Generate dashboard artifacts (no curses). Used by CLI and curses wrapper."""
    bundle = _load_dashboard_bundle(int(n))
    residuals = bundle["residuals"]
    fitted = bundle["fitted"]
    tsv = TavSuperblockVariables()
    sh = SoundHorizon(tsv)

    if len(residuals) >= 2:
        exog = np.column_stack([np.ones(len(residuals)), np.arange(len(residuals), dtype=float)])
        diag = ResidualDiagnostics(residuals=residuals, exog=exog)
    else:
        diag = ResidualDiagnostics(residuals=residuals)

    stamp = _utc_stamp()
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"desi_dashboard_{stamp}_n{len(residuals)}"
    paths = {
        "acf": DASHBOARD_DIR / f"{tag}_acf.png",
        "residuals_vs_fitted": DASHBOARD_DIR / f"{tag}_residuals_vs_fitted.png",
        "qq": DASHBOARD_DIR / f"{tag}_qq.png",
        "sound_horizon": DASHBOARD_DIR / f"{tag}_sound_horizon.png",
        "summary": DASHBOARD_DIR / f"{tag}_summary.png",
        "report": DASHBOARD_DIR / f"{tag}_report.json",
    }

    diag.plot_acf(paths["acf"])
    if len(fitted) == len(residuals):
        diag.plot_residuals_vs_fitted(fitted, paths["residuals_vs_fitted"])
    diag.plot_qq(paths["qq"])
    _plot_sound_horizon_curve(sh, paths["sound_horizon"])

    diagnostics = diag.run_full_diagnostics(verbose=False, save_plots=False)
    report: dict[str, Any] = {
        "action": DASHBOARD_ACTION,
        "n_requested": int(n),
        "n_residuals": int(len(residuals)),
        "data_class": bundle["data_class"],
        "provenance": bundle["provenance"],
        "sound_horizon_mpc": float(sh.predict_rd(9.5)),
        "qq_r2": float(diag._qq_r2()),
        "diagnostics": diagnostics,
        "plots": {key: str(path) for key, path in paths.items() if key != "report"},
    }
    _plot_summary_panel(report, paths["summary"])
    paths["report"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(paths["report"])
    return report


def generate_full_desi_dashboard(stdscr, n: int = 25) -> dict[str, Any]:
    """
    Generate the full set of diagnostic plots (ACF, Residuals vs Fitted, Q-Q,
    SoundHorizon, Summary).
    """
    import curses

    from research_tool import _safe_addstr

    stdscr.clear()
    _safe_addstr(
        stdscr,
        2,
        2,
        f"Generating full DESI diagnostic dashboard (n={n})...",
        curses.A_BOLD,
    )
    stdscr.refresh()

    report = run_full_desi_dashboard(n=int(n))

    stdscr.clear()
    _safe_addstr(stdscr, 2, 2, "Dashboard generated successfully!", curses.A_BOLD)
    _safe_addstr(
        stdscr,
        4,
        2,
        f"Data class: {report['data_class']} | n = {report['n_residuals']}",
    )
    _safe_addstr(stdscr, 5, 2, f"Saved under {DASHBOARD_DIR}/")
    _safe_addstr(stdscr, 7, 2, f"ACF           : {Path(report['plots']['acf']).name}")
    _safe_addstr(
        stdscr,
        8,
        2,
        f"Residuals     : {Path(report['plots']['residuals_vs_fitted']).name}",
    )
    _safe_addstr(stdscr, 9, 2, f"Q-Q           : {Path(report['plots']['qq']).name}")
    _safe_addstr(
        stdscr,
        10,
        2,
        f"SoundHorizon  : {Path(report['plots']['sound_horizon']).name}",
    )
    _safe_addstr(stdscr, 11, 2, f"Summary       : {Path(report['plots']['summary']).name}")
    _safe_addstr(stdscr, 13, 2, "Press any key to return...")
    stdscr.refresh()
    stdscr.getch()
    return report