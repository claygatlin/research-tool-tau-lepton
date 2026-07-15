#!/usr/bin/env python3
"""
TSB Casimir Test Framework — statistical falsification tests for five TSB predictions.

Tests TSB (Tav/Superblock) signatures against smooth Lifshitz / symmetric nulls:
1. Discrete thresholds & hysteresis in tunable force
2. Arrow-of-time / dissipative asymmetry in dynamic Casimir
3. Hierarchical scale-dependent modulations vs separation
4. Domain multiplicity anisotropy + emergent gravity back-reaction
5. Non-local topological correlations (periodicity proxy on single-channel sweeps)

Complements detect_tau_resonances() in scanner.py with explicit p-values and a
six-panel falsification report.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit
from scipy.stats import chi2, norm

from menus.particle.casimir.scanner import (
    ARTIFACTS_DIR,
    _auto_select_columns,
    generate_mock_casimir_data,
    load_data,
)
from menus.particle.casimir.v_ppr import (
    DAMPING_V_PPR,
    evolve_v_ppr_trajectory,
    latency_to_pN_scale,
)

TSB_TEST_RESULTS_DIR = ARTIFACTS_DIR / "tsb_test_results"
TSB_EVIDENCE_ALPHA = 0.05


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _prepare_xy(
    df: pd.DataFrame,
    param_col: str | None,
    value_col: str | None,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    if param_col is None or value_col is None:
        param_col, value_col = _auto_select_columns(df)
    x = df[param_col].values.astype(float)
    y = df[value_col].values.astype(float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    sort_idx = np.argsort(x)
    return x[sort_idx], y[sort_idx], param_col, value_col


def detect_steps(
    x: np.ndarray,
    y: np.ndarray,
    *,
    threshold_sigma: float = 2.0,
) -> tuple[np.ndarray, float, int]:
    """Prediction 1: discrete steps vs smooth Lifshitz."""
    dy = np.diff(y)
    std_dy = float(np.std(dy)) if len(dy) else 0.0
    if std_dy <= 0:
        return np.array([], dtype=int), 1.0, 0
    step_mask = np.abs(dy) > threshold_sigma * std_dy
    step_indices = np.where(step_mask)[0]
    n_steps = int(len(step_indices))
    expected_steps = max(len(dy) * 0.05, 1.0)
    z_score = (n_steps - expected_steps) / np.sqrt(expected_steps)
    p_value = float(1.0 - norm.cdf(z_score))
    return step_indices, p_value, n_steps


def test_periodicity(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float]:
    """Predictions 1 & 5: 7-phase / 1/7 harmonic periodicity."""
    if len(x) < 20:
        return None, 1.0
    dx = np.mean(np.diff(x))
    fs = 1.0 / dx if dx > 0 else 1.0
    freqs, power = signal.periodogram(y, fs=fs)
    peaks, _ = signal.find_peaks(power, height=np.max(power) * 0.2)
    if len(peaks) == 0:
        return None, 1.0
    dominant_freq = float(freqs[peaks[np.argmax(power[peaks])]])
    period = 1.0 / dominant_freq if dominant_freq > 0 else float("inf")
    p_value = float(1.0 - chi2.cdf(float(np.max(power)), df=2))
    return period, p_value


def test_hysteresis(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Prediction 2: hysteresis / arrow-of-time asymmetry."""
    _ = x
    if len(y) < 10:
        return 0.0, 1.0
    mid = len(y) // 2
    diff = float(np.mean(np.abs(y[:mid] - y[mid:][::-1])))
    std_y = float(np.std(y))
    effect_size = diff / std_y if std_y > 0 else 0.0
    p_value = float(1.0 - norm.cdf(effect_size))
    return effect_size, p_value


def test_log_modulations(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float]:
    """Prediction 3: hierarchical Δn modulations in log-separation."""
    pos_mask = x > 0
    if int(np.sum(pos_mask)) < 10:
        return None, 1.0
    log_x = np.log10(x[pos_mask])
    y_pos = y[pos_mask]

    def sin_model(t: np.ndarray, a: float, f: float, p: float, c: float) -> np.ndarray:
        return a * np.sin(2 * np.pi * f * t + p) + c

    try:
        p0 = [float(np.std(y_pos)), 1.0 / 7.0, 0.0, float(np.mean(y_pos))]
        popt, _ = curve_fit(sin_model, log_x, y_pos, p0=p0, maxfev=5000)
        fitted_freq = float(popt[1])
        residuals = y_pos - sin_model(log_x, *popt)
        resid_std = float(np.std(residuals)) or 1.0
        chi2_val = float(np.sum((residuals / resid_std) ** 2))
        p_value = float(1.0 - chi2.cdf(chi2_val, df=max(len(y_pos) - 4, 1)))
        return fitted_freq, p_value
    except (RuntimeError, ValueError, TypeError):
        return None, 1.0


def test_anisotropy_or_backreaction(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Prediction 4: deviation from pure short-range power-law (simplified)."""
    if len(x) < 10:
        return 0.0, 1.0
    log_x = np.log(np.maximum(x, 1e-12))
    log_y = np.log(np.abs(y) + 1e-10)
    slope = float(np.polyfit(log_x, log_y, 1)[0])
    deviation = abs(slope + 4.0)
    p_value = float(1.0 - norm.cdf(deviation))
    return deviation, p_value


def run_full_test_suite(
    df: pd.DataFrame,
    *,
    param_col: str | None = None,
    value_col: str | None = None,
    output_prefix: str = "tsb_falsification",
    plot: bool = True,
    show_popup: bool = False,
) -> dict[str, Any]:
    """Run all five TSB falsification tests; write JSON, PNG, and TXT report."""
    TSB_TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _utc_stamp()

    x, y, pcol, vcol = _prepare_xy(df, param_col, value_col)

    step_idx, p_steps, n_steps = detect_steps(x, y)
    effect_hyst, p_hyst = test_hysteresis(x, y)
    v_ppr = evolve_v_ppr_trajectory(n_max=8, damping=DAMPING_V_PPR)
    v_ppr["predicted_hysteresis_pN"] = latency_to_pN_scale(
        v_ppr["topological_vacuum_latency"]
    )
    v_ppr["hysteresis_residual_pN"] = (
        float(effect_hyst) - v_ppr["predicted_hysteresis_pN"]
        if np.isfinite(v_ppr["predicted_hysteresis_pN"])
        else float("nan")
    )
    period, p_period = test_periodicity(x, y)
    freq_log, p_log = test_log_modulations(x, y)
    dev, p_dev = test_anisotropy_or_backreaction(x, y)

    def _result_block(
        *,
        p_value: float,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            **extra,
            "p_value": p_value,
            "evidence_for_TSB": p_value < TSB_EVIDENCE_ALPHA,
        }

    results: dict[str, dict[str, Any]] = {
        "discrete_steps": _result_block(
            p_value=p_steps,
            extra={"n_steps": n_steps, "step_indices": step_idx.tolist()},
        ),
        "hysteresis": _result_block(
            p_value=p_hyst,
            extra={
                "effect_size": effect_hyst,
                "v_ppr_damping": DAMPING_V_PPR,
                "v_ppr_topological_vacuum_latency": v_ppr["topological_vacuum_latency"],
                "v_ppr_predicted_hysteresis_pN": v_ppr["predicted_hysteresis_pN"],
                "v_ppr_hysteresis_residual_pN": v_ppr["hysteresis_residual_pN"],
                "v_ppr_a_terminal": {
                    "a_7": v_ppr["a_terminal_7"],
                    "a_8": v_ppr["a_terminal_8"],
                },
            },
        ),
        "periodicity": _result_block(
            p_value=p_period,
            extra={"dominant_period": period},
        ),
        "log_modulations": _result_block(
            p_value=p_log,
            extra={"fitted_frequency": freq_log},
        ),
        "anisotropy_backreaction": _result_block(
            p_value=p_dev,
            extra={"deviation_from_power_law_-4": dev},
        ),
    }

    plot_path: str | None = None
    if plot:
        plot_path = str(_save_falsification_plot(
            x,
            y,
            pcol,
            vcol,
            step_idx,
            effect_hyst,
            p_hyst,
            results,
            output_prefix=output_prefix,
            stamp=stamp,
        ))
        if show_popup and plot_path:
            img = plt.imread(plot_path)
            plt.figure(figsize=(15, 10))
            plt.imshow(img)
            plt.axis("off")
            plt.title(Path(plot_path).name)
            plt.show()

    json_path = TSB_TEST_RESULTS_DIR / f"{output_prefix}_report_{stamp}.json"
    txt_path = TSB_TEST_RESULTS_DIR / f"{output_prefix}_summary_{stamp}.txt"

    report = {
        "module": "tsb_casimir_test_framework",
        "timestamp": stamp,
        "param_col": pcol,
        "value_col": vcol,
        "n_points": int(len(x)),
        "alpha": TSB_EVIDENCE_ALPHA,
        "predictions": {
            "1_discrete_steps": "Discrete thresholds vs smooth Lifshitz",
            "2_hysteresis": "Arrow-of-time / dissipative asymmetry",
            "3_log_modulations": "Hierarchical Δn modulations in log-d",
            "4_anisotropy_backreaction": "Domain anisotropy / back-reaction",
            "5_periodicity": "1/7 harmonic / topological periodicity",
        },
        "results": results,
        "plot_path": plot_path,
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "=== TSB Casimir Falsification Test Report ===",
        f"timestamp: {stamp}",
        f"columns: {pcol} vs {vcol}",
        f"n_points: {len(x)}",
        "",
    ]
    for test, res in results.items():
        flag = "TSB evidence" if res["evidence_for_TSB"] else "null-consistent"
        lines.append(f"{test}: p={res['p_value']:.4g} ({flag})")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== TSB Casimir Falsification Test Report ===")
    for test, res in results.items():
        print(
            f"{test}: p={res['p_value']:.4g} | "
            f"Evidence for TSB: {res['evidence_for_TSB']}"
        )
    if plot_path:
        print(f"\nPlot saved to: {plot_path}")
    print(f"JSON: {json_path}")
    print(f"TXT:  {txt_path}")

    report["report_path"] = str(json_path)
    report["summary_path"] = str(txt_path)
    return report


def _save_falsification_plot(
    x: np.ndarray,
    y: np.ndarray,
    param_col: str,
    value_col: str,
    step_idx: np.ndarray,
    effect_hyst: float,
    p_hyst: float,
    results: dict[str, dict[str, Any]],
    *,
    output_prefix: str,
    stamp: str,
) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()

    axes_flat[0].plot(x, y, "b.-", alpha=0.7)
    axes_flat[0].set_title(f"Raw Data ({param_col} vs {value_col})")
    axes_flat[0].set_xlabel(param_col)
    axes_flat[0].set_ylabel(value_col)

    axes_flat[1].plot(x[1:], np.diff(y), "r.-")
    for idx in step_idx:
        axes_flat[1].axvline(x[idx], color="green", linestyle="--", alpha=0.5)
    axes_flat[1].set_title("Derivative + Detected Steps")

    if len(x) > 20:
        freqs, power = signal.periodogram(y)
        axes_flat[2].semilogy(freqs, power)
    axes_flat[2].set_title("Periodogram (1/7 harmonics)")

    axes_flat[3].text(
        0.5,
        0.5,
        f"Hysteresis effect: {effect_hyst:.3f}\np={p_hyst:.3g}",
        transform=axes_flat[3].transAxes,
        ha="center",
        va="center",
    )
    axes_flat[3].set_title("Hysteresis Test")
    axes_flat[3].axis("off")

    if len(x) > 10 and np.all(x > 0):
        axes_flat[4].plot(np.log10(x), y, "g.-")
    axes_flat[4].set_title("Log-scale Data (hierarchical modulations)")
    axes_flat[4].set_xlabel(f"log10({param_col})")

    summary_text = "\n".join(
        f"{k}: p={v['p_value']:.3g}"
        + (" (TSB)" if v["evidence_for_TSB"] else "")
        for k, v in results.items()
    )
    axes_flat[5].text(
        0.1,
        0.5,
        summary_text,
        transform=axes_flat[5].transAxes,
        fontsize=9,
        va="center",
    )
    axes_flat[5].set_title("TSB Falsification Summary")
    axes_flat[5].axis("off")

    fig.tight_layout()
    out = TSB_TEST_RESULTS_DIR / f"{output_prefix}_report_{stamp}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def run_falsification_from_source(
    *,
    data_source: str = "mock",
    local_csv: str | Path | None = None,
    param_col: str | None = None,
    value_col: str | None = None,
    output_prefix: str = "tsb_falsification",
    plot: bool = True,
    show_popup: bool = False,
) -> dict[str, Any]:
    """Load Casimir data (mock, local CSV, or GitHub cache) and run falsification suite."""
    data_source = (data_source or "mock").strip().lower()
    if data_source == "mock":
        df = generate_mock_casimir_data()
        label = "mock magnetic-field Casimir sweep"
    elif data_source in {"local", "csv", "file"}:
        if not local_csv:
            raise ValueError("local_csv required for data_source=local")
        df = load_data(local_csv)
        label = str(Path(local_csv).name)
    elif data_source in {"github", "github_magnetic_fluid"}:
        from menus.particle.casimir.scanner import DATASETS_DIR, list_casimir_data_files

        cache = DATASETS_DIR / "github_magnetic_fluid"
        if local_csv:
            path = Path(local_csv)
        else:
            candidates = list_casimir_data_files(cache)
            if not candidates:
                raise FileNotFoundError(
                    f"No GitHub magnetic-fluid files under {cache}. "
                    "Run Download GitHub Magnetic-Fluid Data first."
                )
            path = candidates[0]
        df = load_data(path)
        label = path.name
    else:
        raise ValueError(f"Unknown data_source: {data_source}")

    report = run_full_test_suite(
        df,
        param_col=param_col,
        value_col=value_col,
        output_prefix=output_prefix,
        plot=plot,
        show_popup=show_popup,
    )
    report["data_label"] = label
    return report


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TSB Casimir falsification test framework")
    p.add_argument("--data", type=str, default=None, help="Path to Casimir CSV")
    p.add_argument("--param_col", type=str, default=None)
    p.add_argument("--value_col", type=str, default=None)
    p.add_argument("--mock", action="store_true", help="Run on synthetic mock data")
    p.add_argument("--prefix", default="tsb_falsification")
    p.add_argument("--no-plot", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    if args.mock or not args.data:
        run_falsification_from_source(
            data_source="mock",
            param_col=args.param_col,
            value_col=args.value_col,
            output_prefix=args.prefix,
            plot=not args.no_plot,
        )
    else:
        df = load_data(args.data)
        run_full_test_suite(
            df,
            param_col=args.param_col,
            value_col=args.value_col,
            output_prefix=args.prefix,
            plot=not args.no_plot,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())