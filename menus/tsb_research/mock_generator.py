#!/usr/bin/env python3
"""
Interactive Tau-SB mock generator for the Research Engine.

Curses flow: cylinder oscillation template + SoundHorizon anchor + residual
diagnostics on synthetic residuals. Legacy BAO batch helpers remain for
injection/validation utilities.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tav_shared.tav_project_paths import TAU_SUPERBLOCK_ROOT

MOCKS_DIR = TAU_SUPERBLOCK_ROOT / "artifacts" / "mocks"

GENERATE_MOCKS_LABEL = "Generate Mocks"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _z_grid_template() -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Prefer genuine DESI DR2 z/errors as the mock lattice; fallback to uniform grid."""
    try:
        from menus.astronomical.desi.scanner import load_scan_data_by_data_mode

        data = load_scan_data_by_data_mode("DH_only")
        z = np.asarray(data["z"], dtype=float)
        err = np.asarray(data["err"], dtype=float)
        cov = np.asarray(data["cov"], dtype=float) if data.get("cov") is not None else None
        return z, err, cov
    except Exception:
        z = np.linspace(0.295, 2.33, 6, dtype=float)
        return z, None, None


def generate_research_mock_report(
    n: int,
    *,
    gamma: float = 9.5,
    residual_sigma: float = 0.08,
    seed: int | None = None,
) -> dict[str, Any]:
    """Build cylinder-template mocks via clockwork, operator algebra, and SoundHorizon."""
    from menus.tsb_research.core import (
        CylinderOperatorAlgebra,
        OctonionicClockwork,
        ResidualDiagnostics,
        SoundHorizon,
        TavSuperblockVariables,
        generate_cylinder_oscillation_template,
    )

    n = int(n)
    if n < 2:
        raise ValueError("n must be at least 2")

    rng = np.random.default_rng(seed)
    tsv = TavSuperblockVariables()
    sh = SoundHorizon(tsv)
    clock = OctonionicClockwork()
    alg = CylinderOperatorAlgebra()

    s_values = np.linspace(0, 45.8, n)
    template = generate_cylinder_oscillation_template(tsv.phi, s_values)
    rd_mpc = float(sh.predict_rd(gamma=float(gamma)))

    delta_s = float(s_values[-1] / max(n - 1, 1))
    operator_traj = alg.evolve_trajectory(
        alg.initial_state(active_index=0),
        n_steps=n - 1,
        delta_s=delta_s,
    )
    occupations = np.vstack(
        [np.asarray(row, dtype=float) for row in operator_traj["occupations"]]
    )
    active_flux = occupations[:, :7].sum(axis=1)
    operator_signal = np.diff(active_flux, prepend=active_flux[0])
    operator_signal -= operator_signal.mean()
    op_scale = float(np.std(operator_signal)) or 1.0
    operator_signal /= op_scale

    phi = 0.0
    phi_trace: list[float] = []
    ak, bk = clock.generate_fourier_coefficients()
    coeff_norm_trace: list[float] = []
    for i in range(n):
        phi_trace.append(float(phi))
        coeff_norm_trace.append(float(np.linalg.norm(ak) + np.linalg.norm(bk)))
        if i < n - 1:
            phi = clock.advance_phase(phi, steps=1)
            ak, bk = clock.evolve_phi_coefficients(ak, bk, steps=1, s=float(s_values[i]))

    template_centered = template - np.mean(template)
    residuals = template_centered + 0.15 * operator_signal
    residuals += rng.normal(0.0, float(residual_sigma), n)

    diag = ResidualDiagnostics(residuals=residuals, exog=np.column_stack([np.ones(n), s_values]))
    diagnostics = diag.run_full_diagnostics(verbose=False)

    return {
        "n": n,
        "gamma": float(gamma),
        "residual_sigma": float(residual_sigma),
        "seed": seed,
        "data_class": "synthetic",
        "sound_horizon_mpc": rd_mpc,
        "template_std": float(np.std(template)),
        "s_values": s_values.tolist(),
        "template": np.asarray(template, dtype=float).tolist(),
        "residuals": residuals.tolist(),
        "diagnostics": diagnostics,
        "clockwork": {
            "phi_trace": phi_trace,
            "coeff_norm_trace": coeff_norm_trace,
            "phi_final": float(phi_trace[-1]),
        },
        "operator_algebra": {
            "delta_s": delta_s,
            "reset_occupation_final": float(operator_traj["reset_occupation_final"]),
            "active_flux": active_flux.tolist(),
        },
    }


def save_research_mock_report(report: dict[str, Any], *, stamp: str | None = None) -> Path:
    """Write research mock JSON under artifacts/mocks/."""
    MOCKS_DIR.mkdir(parents=True, exist_ok=True)
    tag = stamp or _utc_stamp()
    path = MOCKS_DIR / f"tau_sb_research_mocks_{tag}_n{report['n']}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def generate_mock_batch(
    n: int,
    *,
    seed_start: int = 42,
    gamma: float = 10.0,
    A_osc_frac: float = 0.02,
    noise_level: float | None = None,
) -> dict[str, Any]:
    """Generate ``n`` independent Tau-SB mock BAO datasets (legacy batch path)."""
    from menus.astronomical.desi.scanner import generate_tau_sb_mock_data

    z_tpl, err_tpl, cov_tpl = _z_grid_template()
    mocks: list[dict[str, Any]] = []
    for i in range(int(n)):
        seed = int(seed_start) + i
        mock = generate_tau_sb_mock_data(
            z_tpl,
            gamma=float(gamma),
            A_osc_frac=float(A_osc_frac),
            noise_level=noise_level,
            err=err_tpl,
            cov=cov_tpl,
            seed=seed,
        )
        mocks.append(
            {
                "index": i,
                "seed": seed,
                "n_data": int(mock["n_data"]),
                "A_osc_frac": float(mock["A_osc_frac"]),
                "gamma": float(gamma),
                "z": np.asarray(mock["z"], dtype=float).tolist(),
                "observable": np.asarray(mock["observable"], dtype=float).tolist(),
                "err": np.asarray(mock["err"], dtype=float).tolist(),
                "true_model": np.asarray(mock["true_model"], dtype=float).tolist(),
            }
        )
    return {
        "n_mocks": int(n),
        "seed_start": int(seed_start),
        "gamma": float(gamma),
        "A_osc_frac": float(A_osc_frac),
        "z_template_source": "desi_dr2_dh_only" if err_tpl is not None else "synthetic_uniform",
        "data_class": "synthetic",
        "mocks": mocks,
    }


def save_mock_batch(batch: dict[str, Any], *, stamp: str | None = None) -> Path:
    """Write legacy mock batch JSON under artifacts/mocks/."""
    MOCKS_DIR.mkdir(parents=True, exist_ok=True)
    tag = stamp or _utc_stamp()
    path = MOCKS_DIR / f"tau_sb_mocks_{tag}_n{batch['n_mocks']}.json"
    path.write_text(json.dumps(batch, indent=2) + "\n", encoding="utf-8")
    return path


def run_mock_generator(stdscr, *, n: int) -> dict[str, Any]:
    """Generate mocks using Cylinder + SoundHorizon + OctonionicClockwork."""
    import curses

    from research_tool import _safe_addstr

    stdscr.clear()
    _safe_addstr(stdscr, 2, 2, f"Generating Tau-SB mocks with n = {n}...", curses.A_BOLD)
    stdscr.refresh()
    time.sleep(0.6)

    report: dict[str, Any]
    path: Path
    try:
        report = generate_research_mock_report(n)
        path = save_research_mock_report(report)
        report["saved_path"] = str(path)
        report["generator_mode"] = "research_cylinder"
    except Exception as exc:
        _safe_addstr(
            stdscr,
            4,
            2,
            f"Research mock failed ({exc}); using legacy BAO batch fallback...",
            curses.A_DIM,
        )
        stdscr.refresh()
        batch = generate_mock_batch(n)
        path = save_mock_batch(batch)
        report = {
            "n": n,
            "generator_mode": "legacy_bao_batch",
            "saved_path": str(path),
            "n_mocks": batch["n_mocks"],
            "diagnostics": {},
            "sound_horizon_mpc": None,
            "template_std": None,
            "clockwork": {},
            "operator_algebra": {},
        }

    stdscr.clear()
    mode = report.get("generator_mode", "research_cylinder")
    _safe_addstr(stdscr, 2, 2, f"Mocks generated successfully (n={n})", curses.A_BOLD)
    _safe_addstr(stdscr, 3, 2, f"Mode: {mode}", curses.A_DIM)
    row = 5
    rd = report.get("sound_horizon_mpc")
    if rd is not None:
        _safe_addstr(stdscr, row, 2, f"Sound horizon anchor: {float(rd):.2f} Mpc")
        row += 1
    tmpl_std = report.get("template_std")
    if tmpl_std is not None:
        _safe_addstr(stdscr, row, 2, f"Template std dev: {float(tmpl_std):.4f}")
        row += 1
    clock = report.get("clockwork", {})
    op = report.get("operator_algebra", {})
    if clock:
        _safe_addstr(stdscr, row, 2, f"Clockwork ϕ_final: {clock.get('phi_final', 0.0):.4f} rad")
        row += 1
    if op:
        _safe_addstr(
            stdscr,
            row,
            2,
            f"Reset occupation: {op.get('reset_occupation_final', 0.0):.4f}",
        )
        row += 1
    if report.get("n_mocks") is not None:
        _safe_addstr(stdscr, row, 2, f"Legacy BAO mocks: {report['n_mocks']}")
        row += 1
    _safe_addstr(stdscr, row, 2, f"Saved: {path}")
    row += 2

    bp = report.get("diagnostics", {}).get("breusch_pagan", {})
    if bp:
        _safe_addstr(stdscr, row, 2, "Diagnostics summary:", curses.A_DIM)
        row += 1
        _safe_addstr(stdscr, row, 4, f"Breusch-Pagan p-value: {bp.get('lm_pvalue', 'N/A')}")
        row += 2
    _safe_addstr(stdscr, row, 2, "Press any key to return to menu...")
    stdscr.refresh()
    stdscr.getch()
    return report