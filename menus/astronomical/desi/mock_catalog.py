#!/usr/bin/env python3
"""
Tau-SB mock catalog: inject using the same model as fit_tau_sb_model, then recover.

Forward model = poly(z) + A_frac·scale·sin(2πs/τ+φ) + hier_frac·scale·tanh-steps(z),
matching :meth:`TauSBScanner.fit_tau_sb_model` exactly.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np

from menus.astronomical.desi.nested_sampling import (
    NESTED_MAX_SAMPLES_DEFAULT,
    NESTED_NLIVE_DEFAULT,
    run_nested_evidence_comparison,
)
from menus.astronomical.desi.prior_bounds import hier_frac_bounds
from tav_shared.artifact_paths import TestSlug, artifact_path, artifact_timestamp, compose_dataset_slug
from menus.astronomical.desi.scanner import (
    DATA_MODE_DH_ONLY,
    GAMMA6_HEX,
    HIGH_POWER_DEDUPE_Z_TOL,
    LATE_UNIVERSE_DELTA_N,
    N_HIER_BINDING,
    R_TAU_MPC,
    STANDARD_DEDUPE_Z_TOL,
    TAU_RESONANCE_PERIOD,
    TauSBScanner,
    _baseline_degree,
    _oscillation_param_count,
    _oscillation_scale,
    load_combined_dr2_data,
    tau_sb_fit_prediction,
)
from menus.astronomical.desi.theory_likelihood import tau_sb_mu

# DESI+legacy high-power DH z grid (n=9)
DEFAULT_Z_TEMPLATE: tuple[float, ...] = (
    0.38,
    0.51,
    0.698,
    0.706,
    0.934,
    1.321,
    1.48,
    1.484,
    2.33,
)

DEFAULT_A_FRACS: tuple[float, ...] = (0.01, 0.02, 0.03, 0.05, 0.08)
DEFAULT_DELTA_N_VALUES: tuple[float, ...] = (0.33, 0.33, 0.33, 0.25, 0.40)
DEFAULT_PHASE_RAD: float = 0.0
# Phase is fixed at injection; fitting φ alongside A causes π degeneracy on sparse z grids.
DEFAULT_FIT_PHASE: bool = False


def load_multi_tracer_grid(
    *,
    local_path: str | None = None,
    gamma: float = 10.0,
    retain_all_z_bins: bool = True,
) -> dict[str, Any] | None:
    """
    Load combined DESI DR2 tracer stack (BGS+LRG+ELG+…) for high-n recovery.

    ``retain_all_z_bins=True`` sets dedupe_z_tol=0 (maximum n for Lomb).

    Returns None if cache is missing — caller falls back to DEFAULT_Z_TEMPLATE.
    """
    dedupe = HIGH_POWER_DEDUPE_Z_TOL if retain_all_z_bins else STANDARD_DEDUPE_Z_TOL
    try:
        data = load_combined_dr2_data(
            DATA_MODE_DH_ONLY,
            local_path=local_path,
            gamma=gamma,
            dedupe_z_tol=dedupe,
        )
        n = int(data.get("n_data", len(data.get("z", []))))
        if n < 12:
            return None
        return data
    except (FileNotFoundError, ValueError, OSError):
        return None


def delta_n_to_hier_frac(delta_n: float, *, reference: float = 0.005) -> float:
    """Map cylinder Δn label to hier_frac used by the fitter (reference at Δn≈0.33)."""
    return float(delta_n) / LATE_UNIVERSE_DELTA_N * reference


def build_fit_aligned_mock_signal(
    z: np.ndarray | Sequence[float],
    *,
    A_osc_frac: float,
    phase_rad: float = DEFAULT_PHASE_RAD,
    hier_frac: float | None = None,
    delta_n: float | None = None,
    gamma: float = 10.0,
    period: float = TAU_RESONANCE_PERIOD,
    quantity: str = "DH_over_rs",
    fit_phase: bool = DEFAULT_FIT_PHASE,
    fit_hier: bool | None = None,
    baseline_coeffs: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Build noiseless mock vector with the same parameterization as fit_tau_sb_model.
    """
    z_arr = np.asarray(z, dtype=float)
    n = len(z_arr)
    if hier_frac is None:
        hier_frac = delta_n_to_hier_frac(delta_n if delta_n is not None else LATE_UNIVERSE_DELTA_N)

    use_hier = (n >= 8) if fit_hier is None else bool(fit_hier)
    extra = _oscillation_param_count(
        fit_A=True, fit_phase=fit_phase, use_hier=use_hier
    )
    deg = _baseline_degree(n, extra)

    anchor = tau_sb_mu(z_arr, quantity=quantity)
    if baseline_coeffs is None:
        baseline_coeffs = np.polyfit(z_arr, anchor, deg)

    baseline_only = np.polyval(baseline_coeffs, z_arr)
    amp_scale = _oscillation_scale(baseline_only)

    y_true = tau_sb_fit_prediction(
        z_arr,
        baseline_coeffs,
        A_osc_frac=A_osc_frac,
        phase_rad=phase_rad,
        hier_frac=hier_frac if use_hier else 0.0,
        gamma=gamma,
        period=period,
        amplitude_scale=amp_scale,
        use_hier=use_hier,
    )

    meta = {
        "A_osc_frac_true": float(A_osc_frac),
        "phase_rad_true": float(phase_rad),
        "hier_frac_true": float(hier_frac if use_hier else 0.0),
        "delta_n_label": delta_n,
        "period": period,
        "gamma": gamma,
        "amplitude_scale": amp_scale,
        "baseline_degree": deg,
        "baseline_coeffs": [float(c) for c in baseline_coeffs],
        "use_hier": use_hier,
        "fit_phase": fit_phase,
        "fit_hier": use_hier,
        "injection_model": "fit_tau_sb_model_aligned",
    }
    return y_true, meta


def build_mock_catalog_entry_multi_tracer(
    *,
    entry_id: int,
    A_frac: float,
    delta_n: float = LATE_UNIVERSE_DELTA_N,
    phase_rad: float = DEFAULT_PHASE_RAD,
    quantity: str = "DH_over_rs",
    seed: int | None = None,
    noise_frac: float = 0.02,
    fit_phase: bool = DEFAULT_FIT_PHASE,
    fit_hier: bool | None = None,
    local_path: str | None = None,
    retain_all_z_bins: bool = True,
) -> dict[str, Any] | None:
    """Inject fit-aligned signal on combined DR2 tracer z-grid (n≫9)."""
    grid = load_multi_tracer_grid(
        local_path=local_path,
        retain_all_z_bins=retain_all_z_bins,
    )
    if grid is None:
        return None
    z_arr = np.asarray(grid["z"], dtype=float)
    base_err = np.asarray(grid["err"], dtype=float)
    cov = np.asarray(grid["cov"], dtype=float)
    entry = build_mock_catalog_entry(
        entry_id=entry_id,
        z=z_arr,
        A_frac=A_frac,
        delta_n=delta_n,
        phase_rad=phase_rad,
        quantity=quantity,
        seed=seed,
        noise_frac=noise_frac,
        fit_phase=fit_phase,
        fit_hier=fit_hier,
    )
    entry["err"] = np.maximum(base_err, entry["err"])
    entry["cov"] = cov
    entry["grid_mode"] = "multi_tracer_stack"
    entry["n_data"] = len(z_arr)
    entry["tracer"] = grid.get("tracer", "COMBINED_DR2")
    return entry


def build_mock_catalog_entry(
    *,
    entry_id: int,
    z: np.ndarray | Sequence[float],
    A_frac: float,
    delta_n: float = LATE_UNIVERSE_DELTA_N,
    phase_rad: float = DEFAULT_PHASE_RAD,
    quantity: str = "DH_over_rs",
    seed: int | None = None,
    noise_frac: float = 0.02,
    fit_phase: bool = DEFAULT_FIT_PHASE,
    fit_hier: bool | None = None,
) -> dict[str, Any]:
    """Single catalog row: fit-aligned signal + diagonal noise."""
    z_arr = np.asarray(z, dtype=float)
    entry_seed = int(seed if seed is not None else 1000 + entry_id)

    y_true, inj_meta = build_fit_aligned_mock_signal(
        z_arr,
        A_osc_frac=A_frac,
        phase_rad=phase_rad,
        delta_n=delta_n,
        quantity=quantity,
        fit_phase=fit_phase,
        fit_hier=fit_hier,
    )
    rng = np.random.default_rng(entry_seed)
    err_frac = noise_frac + 0.005 * np.sqrt(np.maximum(z_arr, 0.0))
    err = err_frac * np.abs(y_true)
    err = np.maximum(err, 1e-6)
    obs = y_true + rng.normal(0.0, err)
    cov = np.diag(err**2)

    return {
        "entry_id": entry_id,
        "seed": entry_seed,
        "z": z_arr,
        "observable": obs,
        "err": err,
        "cov": cov,
        "y_true": y_true,
        "mu_injected": y_true,
        "quantity": quantity,
        "n_data": len(z_arr),
        "injection": inj_meta,
        "label": f"mock_catalog_{entry_id:03d}",
        "data_class": "synthetic",
        "geometric_anchors": {
            "R_tau_mpc": R_TAU_MPC,
            "gamma6": GAMMA6_HEX,
            "n_hier": N_HIER_BINDING,
        },
    }


def generate_full_mock_catalog(
    *,
    z_template: Sequence[float] | None = None,
    A_fracs: Sequence[float] | None = None,
    delta_n_values: Sequence[float] | None = None,
    quantity: str = "DH_over_rs",
    base_seed: int = 42,
    fit_phase: bool = DEFAULT_FIT_PHASE,
    fit_hier: bool | None = None,
    multi_tracer_stack: bool = False,
    local_path: str | None = None,
    retain_all_z_bins: bool = True,
) -> list[dict[str, Any]]:
    """Build catalog crossing A_frac × Δn(hier) grid on the template z vector."""
    z_use = np.asarray(z_template or DEFAULT_Z_TEMPLATE, dtype=float)
    a_list = list(A_fracs or DEFAULT_A_FRACS)
    dn_list = list(delta_n_values or DEFAULT_DELTA_N_VALUES)
    catalog: list[dict[str, Any]] = []
    entry_id = 0
    grid_mode = "single_channel_n9"
    if multi_tracer_stack:
        probe = load_multi_tracer_grid(
            local_path=local_path,
            retain_all_z_bins=retain_all_z_bins,
        )
        if probe is not None:
            grid_mode = "multi_tracer_stack"
            dedupe = probe.get("dedupe_z_tol", "?")
            print(
                f"[Mock catalog] Multi-tracer grid: n={probe['n_data']} "
                f"dedupe_z_tol={dedupe} "
                f"({probe.get('tracer', 'COMBINED_DR2')})"
            )
        else:
            print(
                "[Mock catalog] Multi-tracer stack unavailable — "
                "using n=9 DH template (fetch DESI BAO cache first)"
            )
    for a_frac in a_list:
        for delta_n in dn_list:
            if multi_tracer_stack and grid_mode == "multi_tracer_stack":
                row = build_mock_catalog_entry_multi_tracer(
                    entry_id=entry_id,
                    A_frac=float(a_frac),
                    delta_n=float(delta_n),
                    quantity=quantity,
                    seed=base_seed + entry_id,
                    fit_phase=fit_phase,
                    fit_hier=fit_hier,
                    local_path=local_path,
                    retain_all_z_bins=retain_all_z_bins,
                )
                if row is None:
                    row = build_mock_catalog_entry(
                        entry_id=entry_id,
                        z=z_use,
                        A_frac=float(a_frac),
                        delta_n=float(delta_n),
                        quantity=quantity,
                        seed=base_seed + entry_id,
                        fit_phase=fit_phase,
                        fit_hier=fit_hier,
                    )
            else:
                row = build_mock_catalog_entry(
                    entry_id=entry_id,
                    z=z_use,
                    A_frac=float(a_frac),
                    delta_n=float(delta_n),
                    quantity=quantity,
                    seed=base_seed + entry_id,
                    fit_phase=fit_phase,
                    fit_hier=fit_hier,
                )
            catalog.append(row)
            entry_id += 1
    if catalog:
        catalog[0].setdefault("catalog_grid_mode", grid_mode)
    return catalog


def recover_mock_entry(
    entry: dict[str, Any],
    *,
    scanner: TauSBScanner | None = None,
    run_nested: bool = False,
    nested_nlive: int = 80,
    nested_max_samples: int = 500,
) -> dict[str, Any]:
    """Fit Tau-SB on one mock entry; compare recovered vs injected parameters."""
    sc = scanner or TauSBScanner(auto_calibrate_gamma=False, gamma=10.0)
    z = entry["z"]
    obs = entry["observable"]
    err = entry["err"]
    cov = entry["cov"]
    inj = entry.get("injection") or {}

    a_true = float(inj.get("A_osc_frac_true", inj.get("A_frac_true", 0.0)))
    phase_true = float(inj.get("phase_rad_true", 0.0))
    hier_true = float(inj.get("hier_frac_true", 0.0))

    fit_phase = bool(inj.get("fit_phase", DEFAULT_FIT_PHASE))
    fit_hier_raw = inj.get("fit_hier")
    fit_hier: bool | None
    if fit_hier_raw is None:
        fit_hier = None
    else:
        fit_hier = bool(fit_hier_raw)

    hier_bounds = None
    if hier_true != 0.0:
        hier_bounds = hier_frac_bounds(
            mode="mock_recovery",
            center=hier_true,
        )

    fit = sc.fit_tau_sb_model(
        z,
        obs,
        err,
        cov=cov,
        fit_A=True,
        fit_phase=fit_phase,
        fit_hier=fit_hier,
        hier_bounds=hier_bounds,
    )
    a_rec = float(fit.get("A_osc_frac", fit.get("A_osc", 0.0)))
    params = fit.get("best_params")
    phase_rec = float("nan")
    hier_rec = float("nan")
    if params is not None:
        labels = fit.get("parameter_labels") or []
        for i, lab in enumerate(labels):
            if lab == "phase_rad":
                phase_rec = float(params[i])
            if lab == "hier_frac":
                hier_rec = float(params[i])

    a_bias = (a_rec - a_true) / max(abs(a_true), 1e-8)
    sign_ok = bool(np.sign(a_rec) == np.sign(a_true)) if a_true != 0 else True

    per = sc.scan_periodicity(z, obs, err, cov=cov)

    recovery: dict[str, Any] = {
        "entry_id": entry.get("entry_id"),
        "A_true_frac": a_true,
        "A_recovered_frac": a_rec,
        "A_bias_fraction": float(a_bias),
        "phase_true_rad": phase_true,
        "phase_recovered_rad": phase_rec,
        "hier_frac_true": hier_true,
        "hier_frac_recovered": hier_rec,
        "sign_correct": sign_ok,
        "chi2": float(fit["chi2"]),
        "reduced_chi2": float(fit.get("reduced_chi2", float("nan"))),
        "dof": int(fit.get("dof", 0)),
        "amplitude_scale_fit": float(fit.get("amplitude_scale", float("nan"))),
        "delta_n_label": inj.get("delta_n_label"),
        "lomb_detected": bool((per.get("tav_harmonics") or {}).get("lomb_detected", False)),
        "power_at_1_7": float(per.get("power_at_expected", 0.0)),
    }

    if run_nested and len(z) >= 5:
        try:
            nested = run_nested_evidence_comparison(
                sc,
                z,
                obs,
                err,
                cov,
                nlive=nested_nlive,
                max_samples=nested_max_samples,
                output_prefix=f"mock_{entry.get('entry_id', 0):03d}",
                use_jax_geometric=False,
            )
            recovery["nested_favored"] = nested.get("favored_model")
            recovery["delta_log_evidence"] = nested.get(
                "delta_log_evidence_tau_minus_ade"
            )
            recovery["nested_favored_bic"] = nested.get("favored_model_bic")
            recovery["delta_log_evidence_bic"] = nested.get(
                "delta_log_evidence_bic"
            )
            recovery["nested_favored_occam"] = nested.get("favored_model_occam")
            recovery["nested_favored_sparse"] = nested.get("favored_model_sparse")
            recovery["delta_log_evidence_occam"] = nested.get(
                "delta_log_evidence_occam_corrected"
            )
            recovery["occam_penalty"] = nested.get("occam_penalty_tau_minus_ade")
        except Exception as exc:
            recovery["nested_error"] = str(exc)

    return recovery


def run_full_mock_recovery_pipeline(
    catalog: list[dict[str, Any]] | None = None,
    *,
    run_nested: bool = False,
    nested_nlive: int = NESTED_NLIVE_DEFAULT,
    nested_max_samples: int = NESTED_MAX_SAMPLES_DEFAULT,
    output_prefix: str = "tau_sb_mock_catalog",
) -> dict[str, Any]:
    """Run recovery on every catalog entry; write summary JSON."""
    entries = catalog if catalog is not None else generate_full_mock_catalog()
    scanner = TauSBScanner(auto_calibrate_gamma=False, gamma=10.0)
    recoveries: list[dict[str, Any]] = []

    print(
        f"\n[Mock catalog] {len(entries)} entries — fit-aligned 1/7 + hier injection"
    )
    for entry in entries:
        rec = recover_mock_entry(
            entry,
            scanner=scanner,
            run_nested=run_nested,
            nested_nlive=nested_nlive,
            nested_max_samples=nested_max_samples,
        )
        recoveries.append(rec)
        print(
            f"  id={rec['entry_id']:02d} A_true={rec['A_true_frac']:.3f} "
            f"→ A_rec={rec['A_recovered_frac']:.4f} "
            f"bias={rec['A_bias_fraction']:+.1%} "
            f"χ²_ν={rec['reduced_chi2']:.2f} "
            f"Lomb={rec['lomb_detected']}"
        )

    a_biases = np.array([r["A_bias_fraction"] for r in recoveries], dtype=float)
    sign_acc = float(np.mean([r["sign_correct"] for r in recoveries]))

    grid_mode = entries[0].get("catalog_grid_mode", "single_channel_n9") if entries else "single_channel_n9"
    n_data = int(entries[0].get("n_data", len(DEFAULT_Z_TEMPLATE))) if entries else len(DEFAULT_Z_TEMPLATE)

    summary = {
        "n_entries": len(recoveries),
        "injection_model": "fit_tau_sb_model_aligned",
        "catalog_grid_mode": grid_mode,
        "n_data_per_entry": n_data,
        "recoveries": recoveries,
        "aggregate": {
            "mean_A_bias_fraction": float(np.mean(a_biases)),
            "median_A_bias_fraction": float(np.median(a_biases)),
            "mean_abs_A_bias_fraction": float(np.mean(np.abs(a_biases))),
            "sign_accuracy": sign_acc,
            "mean_reduced_chi2": float(np.mean([r["reduced_chi2"] for r in recoveries])),
            "lomb_detection_rate": float(
                np.mean([r["lomb_detected"] for r in recoveries])
            ),
        },
        "z_template": list(DEFAULT_Z_TEMPLATE),
        "timestamp": artifact_timestamp(),
    }

    out = artifact_path(TestSlug.MOCK_CATALOG, compose_dataset_slug(output_prefix), "recovery", "json")
    from menus.astronomical.desi.json_util import write_json

    write_json(out, summary, indent=2, sort_keys=True)
    summary["summary_path"] = str(out)
    print(f"\n[Mock catalog] Summary: {out}")
    print(
        f"  sign accuracy={sign_acc:.0%} | "
        f"mean |A bias|={summary['aggregate']['mean_abs_A_bias_fraction']:.1%} | "
        f"Lomb detect rate={summary['aggregate']['lomb_detection_rate']:.0%}"
    )
    return summary


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tau-SB mock catalog: fit-aligned injection + recovery"
    )
    parser.add_argument(
        "--nested",
        action="store_true",
        help="Run dynesty evidence comparison on each entry (slow)",
    )
    parser.add_argument("--nlive", type=int, default=NESTED_NLIVE_DEFAULT)
    parser.add_argument("--max-samples", type=int, default=NESTED_MAX_SAMPLES_DEFAULT)
    parser.add_argument("--prefix", default="tau_sb_mock_catalog")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small catalog (3 amplitudes × 2 delta_n)",
    )
    parser.add_argument(
        "--noiseless-check",
        action="store_true",
        help="Run one noiseless entry per A_frac to verify template alignment",
    )
    parser.add_argument(
        "--with-phase",
        action="store_true",
        help="Fit oscillation phase (φ); default fixes φ=0 to avoid π degeneracy on sparse grids",
    )
    parser.add_argument(
        "--multi-tracer",
        action="store_true",
        help="Stack DESI DR2 tracers (n≫9) for Lomb / hier resolution",
    )
    parser.add_argument(
        "--dedupe-z",
        action="store_true",
        help="Dedupe near-duplicate z bins (|Δz|≤0.02); default retains all bins",
    )
    return parser


def run_noiseless_alignment_check(
    A_fracs: Sequence[float] = DEFAULT_A_FRACS[:3],
) -> dict[str, Any]:
    """Verify zero-noise recovery bias is negligible (sanity check)."""
    z = np.asarray(DEFAULT_Z_TEMPLATE, dtype=float)
    scanner = TauSBScanner(auto_calibrate_gamma=False, gamma=10.0)
    rows: list[dict[str, Any]] = []
    print("\n[Mock catalog] Noiseless alignment check")
    for i, a_frac in enumerate(A_fracs):
        y_true, meta = build_fit_aligned_mock_signal(
            z, A_osc_frac=float(a_frac), fit_phase=DEFAULT_FIT_PHASE
        )
        err = np.full(len(z), 1e-9)
        fit = scanner.fit_tau_sb_model(
            z,
            y_true,
            err,
            cov=np.diag(err**2),
            fit_phase=DEFAULT_FIT_PHASE,
            fit_hier=meta.get("fit_hier"),
        )
        a_rec = float(fit.get("A_osc_frac", 0.0))
        bias = (a_rec - a_frac) / max(abs(a_frac), 1e-8)
        rows.append({"A_true": a_frac, "A_rec": a_rec, "bias": bias})
        print(f"  A_true={a_frac:.3f} → A_rec={a_rec:.6f} bias={bias:+.2%}")
    return {"noiseless_rows": rows}


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    if args.noiseless_check:
        run_noiseless_alignment_check()
        return 0
    fit_phase = bool(args.with_phase)
    if args.quick:
        catalog = generate_full_mock_catalog(
            A_fracs=(0.02, 0.05, 0.08),
            delta_n_values=(0.33, 0.25),
            fit_phase=fit_phase,
            multi_tracer_stack=bool(args.multi_tracer),
            retain_all_z_bins=not bool(args.dedupe_z),
        )
    else:
        catalog = generate_full_mock_catalog(
            fit_phase=fit_phase,
            multi_tracer_stack=bool(args.multi_tracer),
            retain_all_z_bins=not bool(args.dedupe_z),
        )
    run_full_mock_recovery_pipeline(
        catalog,
        run_nested=args.nested,
        nested_nlive=args.nlive,
        nested_max_samples=args.max_samples,
        output_prefix=args.prefix,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())