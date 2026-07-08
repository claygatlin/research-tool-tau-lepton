"""
Tav BBN menu engine — improved network with adjustable neutron lifetime.

Artifacts: Public/TauSuperblock/artifacts/prime_past/
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from menus.prime_past.bbn_confrontation import fetch_empirical_data
from menus.prime_past.bbn_enhanced import plot_comparison
from tav_shared.tav_project_paths import TAU_SUPERBLOCK_ROOT

ARTIFACTS_DIR = TAU_SUPERBLOCK_ROOT / "artifacts" / "prime_past"
BEST_CONFIG_PATH = ARTIFACTS_DIR / "best_tav_bbn_config.json"

DEFAULT_PARAMS: dict[str, float] = {
    "A": 0.02,
    "delta_phi_cyl": 0.0,
    "delta_k_wind": 0.469,
    "xi": 8.0,
    "neutron_lifetime": 879.4,
}

G_STAR = 10.75
M_PL_NATURAL = 1.22e19


def ensure_artifacts_dir() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


def get_empirical_data(*, verbose: bool = False) -> dict[str, Any]:
    raw = fetch_empirical_data(verbose=verbose)
    bbn = raw["bbn_abundances"]
    return {
        "bbn_abundances": {
            "D_H": tuple(bbn["D_H"]),
            "Y_p_He4": tuple(bbn["Y_p_He4"]),
            "Li7_H_obs": tuple(bbn["Li7_H_obs"]),
        }
    }


def _tav_interference_delta(T: float, params: dict[str, Any]) -> float:
    s = np.log(10.0 / max(float(T), 0.01))
    phase = float(params.get("delta_phi_cyl", 0.0)) + float(params.get("delta_k_wind", 0.469))
    envelope = np.exp(-s / max(float(params.get("xi", 8.0)), 0.1))
    return float(params.get("A", 0.0)) * np.cos(phase) * envelope


def _modified_hubble(T: float, params: dict[str, Any]) -> float:
    T = max(float(T), 0.01)
    H_std = 1.66 * np.sqrt(G_STAR) * (T**2) / M_PL_NATURAL
    return float(H_std * (1.0 + _tav_interference_delta(T, params)))


def run_improved_bbn(
    params: dict[str, Any] | None = None,
    *,
    t_span: tuple[float, float] = (0.0, 450.0),
) -> dict[str, Any]:
    """Improved BBN with neutron-lifetime-dependent weak rates."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    tau_n = float(params.get("neutron_lifetime", 879.4))

    T0 = 10.0
    Yn0 = 1.0 / (1.0 + np.exp(1.293 / T0))
    Yp0 = 1.0 - Yn0
    y0 = [T0, Yn0, Yp0, 0.0, 0.0]

    def ode(t: float, y: list[float]) -> list[float]:
        T, Yn, Yp, YD, Y3He = y
        H = _modified_hubble(T, params)
        dT_dt = -H * T

        rate_np = 1.0 / (tau_n * (T / 0.8) ** 5 + 1e-10)
        rate_pn = rate_np * np.exp(-1.293 / max(T, 1e-4))
        dYn = -rate_np * Yn + rate_pn * Yp

        YD_eq = 2 * Yn * Yp * np.exp(2.22 / max(T, 0.01)) * 0.08
        destruction_rate = 80.0 * np.exp(-0.4 / max(T, 0.01))
        dYD = (YD_eq - YD) * destruction_rate
        dY3He = 0.25 * dYD - 0.08 * Y3He

        return [dT_dt, dYn, -dYn, dYD, dY3He]

    sol = solve_ivp(ode, t_span, y0, method="RK45", rtol=1e-6, atol=1e-8)
    T_f, Yn_f, Yp_f, YD_f, Y3He_f = (float(v) for v in sol.y[:, -1])

    YHe4 = min(2 * Yn_f, max(0.0, 1.0 - Yp_f - 2 * YD_f - 3 * Y3He_f))
    base = 4.8e-10 * (Yn_f / 0.11)
    interference = 1.0 - 14.0 * float(params.get("A", 0.0)) * np.cos(
        float(params.get("delta_phi_cyl", 0.0))
    )
    YLi7 = max(1e-12, base * max(0.08, interference))

    return {
        "Y_n": Yn_f,
        "Y_p": Yp_f,
        "Y_D": YD_f,
        "Y_3He": Y3He_f,
        "Y_He4": YHe4,
        "Y_Li7": YLi7,
        "Y_p_mass": 4.0 * YHe4,
        "D_H": YD_f / Yp_f if Yp_f > 0 else 0.0,
        "Li7_H": YLi7,
        "success": bool(sol.success),
        "params_used": params,
    }


def calculate_tensions(
    results: dict[str, Any],
    empirical: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    empirical = empirical or get_empirical_data()
    emp = empirical["bbn_abundances"]
    tensions: list[dict[str, Any]] = []
    for name, key, exp_key in (
        ("D_H", "D_H", "D_H"),
        ("Y_p_He4", "Y_p_mass", "Y_p_He4"),
        ("Li7_H", "Li7_H", "Li7_H_obs"),
    ):
        th = float(results.get(key, 0.0))
        ex, unc = emp[exp_key]
        ex, unc = float(ex), float(unc)
        if th > 0 and unc > 0:
            tensions.append(
                {
                    "observable": name,
                    "theory": th,
                    "exp": ex,
                    "tension_sigma": abs(th - ex) / unc,
                }
            )
    return tensions


def run_improved_scan(
    *,
    a_points: int = 9,
    phi_min: float = -0.4,
    phi_max: float = 0.6,
    phi_points: int = 6,
    neutron_lifetime: float = 879.4,
    delta_k_wind: float = 0.469,
) -> dict[str, Any]:
    print("\n=== Running Improved Tav BBN Scan ===")
    empirical = get_empirical_data()
    results: list[dict[str, Any]] = []

    for A in np.linspace(0.0, 0.04, max(int(a_points), 2)):
        for phi in np.linspace(phi_min, phi_max, max(int(phi_points), 2)):
            params = {
                "A": float(A),
                "delta_phi_cyl": float(phi),
                "delta_k_wind": float(delta_k_wind),
                "xi": 8.0,
                "neutron_lifetime": float(neutron_lifetime),
            }
            res = run_improved_bbn(params)
            tensions = calculate_tensions(res, empirical)
            li_tension = next(
                (t["tension_sigma"] for t in tensions if t["observable"] == "Li7_H"),
                999.0,
            )
            results.append(
                {
                    "params": params,
                    "results": res,
                    "tensions": tensions,
                    "li_tension_sigma": float(li_tension),
                }
            )

    best = min(results, key=lambda row: row["li_tension_sigma"])
    print(f"\nBest configuration found: {best['params']}")
    print(f"Li7 tension: {best['li_tension_sigma']:.2f} σ")

    ensure_artifacts_dir()
    BEST_CONFIG_PATH.write_text(
        json.dumps(best, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(f"Best config saved to: {BEST_CONFIG_PATH}")

    scan_path = ARTIFACTS_DIR / f"tav_bbn_menu_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    scan_path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "scan_results": results,
                "best": best,
            },
            indent=2,
            default=float,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Full scan saved to: {scan_path}")
    return {"best": best, "scan_results": results, "scan_path": str(scan_path)}


def analyze_best_config(json_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(json_path) if json_path else BEST_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"No best config at {path}. Run improved scan first.")

    data = json.loads(path.read_text(encoding="utf-8"))
    print("\n=== Best Tav Configuration ===")
    print(json.dumps(data["params"], indent=2))
    print(f"\nLi7 tension: {data['li_tension_sigma']:.2f} σ")
    print(f"Y_p_mass: {data['results']['Y_p_mass']:.4f}")
    print(f"D_H: {data['results']['D_H']:.2e}")
    print(f"Li7_H: {data['results']['Li7_H']:.2e}")
    return data


def compare_standard_vs_tav(
    *,
    best_params: dict[str, Any] | None = None,
    neutron_lifetime: float = 879.4,
    save_plot: bool = True,
    show_plot: bool = False,
) -> dict[str, Any]:
    print("\n=== Standard vs Best Tav Comparison ===")

    if best_params is None and BEST_CONFIG_PATH.is_file():
        best_params = json.loads(BEST_CONFIG_PATH.read_text(encoding="utf-8"))["params"]

    base = {
        "delta_k_wind": 0.469,
        "xi": 8.0,
        "neutron_lifetime": neutron_lifetime,
    }
    std_params = {**base, "A": 0.0, "delta_phi_cyl": 0.0}
    tav_params = best_params or {**base, "A": 0.035, "delta_phi_cyl": 0.0}

    std = run_improved_bbn(std_params)
    best = run_improved_bbn(tav_params)

    if std["Li7_H"] > 0:
        improvement = (std["Li7_H"] - best["Li7_H"]) / std["Li7_H"] * 100.0
    else:
        improvement = float("nan")

    print(f"Standard Li7_H: {std['Li7_H']:.2e}")
    print(f"Best Tav Li7_H:  {best['Li7_H']:.2e}")
    print(f"Improvement: {improvement:.1f}% reduction")

    plot_path = None
    if save_plot:
        plot_path = plot_comparison(std, best, show=show_plot)

    return {
        "standard": std,
        "best": best,
        "improvement_percent": improvement,
        "comparison_plot": plot_path,
    }


def run_high_resolution_around_best(
    *,
    json_path: str | Path | None = None,
    a_span: float = 0.008,
    phi_span: float = 0.15,
    grid: int = 7,
    neutron_lifetime: float | None = None,
) -> dict[str, Any]:
    print("\n=== High-Resolution Scan Around Best Config ===")
    seed = analyze_best_config(json_path)
    center = seed["params"]
    tau_n = float(neutron_lifetime or center.get("neutron_lifetime", 879.4))

    empirical = get_empirical_data()
    results: list[dict[str, Any]] = []
    a0 = float(center["A"])
    p0 = float(center["delta_phi_cyl"])

    for A in np.linspace(a0 - a_span, a0 + a_span, grid):
        for phi in np.linspace(p0 - phi_span, p0 + phi_span, grid):
            params = {
                "A": float(max(A, 0.0)),
                "delta_phi_cyl": float(phi),
                "delta_k_wind": float(center.get("delta_k_wind", 0.469)),
                "xi": float(center.get("xi", 8.0)),
                "neutron_lifetime": tau_n,
            }
            res = run_improved_bbn(params)
            tensions = calculate_tensions(res, empirical)
            li_tension = next(
                (t["tension_sigma"] for t in tensions if t["observable"] == "Li7_H"),
                999.0,
            )
            results.append(
                {
                    "params": params,
                    "results": res,
                    "li_tension_sigma": float(li_tension),
                }
            )
            print(f"A={params['A']:.4f} | phi={params['delta_phi_cyl']:.3f} | Li σ={li_tension:.2f}")

    best = min(results, key=lambda row: row["li_tension_sigma"])
    print(f"\nRefined best: {best['params']} → {best['li_tension_sigma']:.2f} σ")

    ensure_artifacts_dir()
    BEST_CONFIG_PATH.write_text(
        json.dumps(best, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    out = ARTIFACTS_DIR / f"tav_bbn_hires_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(
        json.dumps({"center": center, "scan_results": results, "best": best}, indent=2, default=float)
        + "\n",
        encoding="utf-8",
    )
    print(f"High-res scan saved to: {out}")
    return {"best": best, "scan_path": str(out)}


def settings_menu(*, neutron_lifetime: float = 879.4) -> dict[str, float]:
    print("\n[Tav BBN Settings]")
    print(f"  neutron_lifetime = {neutron_lifetime:.1f} s  (PDG-style default 879.4)")
    print(f"  artifacts_dir    = {ARTIFACTS_DIR}")
    print(f"  best_config      = {BEST_CONFIG_PATH}")
    print("Adjust neutron_lifetime on scan/compare entry forms or options dict.")
    return {"neutron_lifetime": float(neutron_lifetime)}