"""
Enhanced minimal BBN engine with D/^3He channels, lithium heatmaps, and JSON analysis.

Project root: Public/TauSuperblock/
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
from tav_shared.tav_project_paths import TAU_SUPERBLOCK_ROOT

ARTIFACTS_DIR = TAU_SUPERBLOCK_ROOT / "artifacts" / "prime_past"

G_STAR = 10.75
M_PL_NATURAL = 1.22e19
Q_NP_MEV = 1.293
B_D_MEV = 2.22
T_START_MEV = 10.0


def ensure_artifacts_dir() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


def tav_interference_delta(T: float, params: dict[str, Any]) -> float:
    s = np.log(10.0 / max(float(T), 0.01))
    phase = float(params.get("delta_phi_cyl", 0.0)) + float(params.get("delta_k_wind", 0.469))
    envelope = np.exp(-s / max(float(params.get("xi", 8.0)), 0.1))
    return float(params.get("A", 0.0)) * np.cos(phase) * envelope


def modified_hubble(T: float, params: dict[str, Any]) -> float:
    T = max(float(T), 0.01)
    H_std = 1.66 * np.sqrt(G_STAR) * (T**2) / M_PL_NATURAL
    return float(H_std * (1.0 + tav_interference_delta(T, params)))


def run_improved_bbn(
    params: dict[str, Any],
    *,
    t_span: tuple[float, float] = (0.0, 400.0),
) -> dict[str, Any]:
    """Five-state RK45 BBN with improved D bottleneck and lithium interference."""
    T0 = T_START_MEV
    Yn0 = 1.0 / (1.0 + np.exp(Q_NP_MEV / T0))
    Yp0 = 1.0 - Yn0
    y0 = [T0, Yn0, Yp0, 0.0, 0.0]

    def ode(t: float, y: list[float]) -> list[float]:
        T, Yn, Yp, YD, Y3He = y
        H = modified_hubble(T, params)
        dT_dt = -H * T

        rate_np = 1.0 / (880.0 * (T / 0.8) ** 5 + 1e-10)
        rate_pn = rate_np * np.exp(-Q_NP_MEV / max(T, 1e-4))
        dYn = -rate_np * Yn + rate_pn * Yp

        YD_eq = 2 * Yn * Yp * np.exp(B_D_MEV / max(T, 0.01)) * 0.05
        dYD = (YD_eq - YD) * 50.0 * np.exp(-0.5 / max(T, 0.01))
        dY3He = 0.3 * dYD - 0.1 * Y3He

        return [dT_dt, dYn, -dYn, dYD, dY3He]

    sol = solve_ivp(ode, t_span, y0, method="RK45", rtol=1e-6, atol=1e-8)
    T_f, Yn_f, Yp_f, YD_f, Y3He_f = (float(v) for v in sol.y[:, -1])

    YHe4 = min(2 * Yn_f, max(0.0, 1.0 - Yp_f - 2 * YD_f - 3 * Y3He_f))
    base = 4.5e-10 * (Yn_f / 0.12)
    factor = 1.0 - 15.0 * float(params.get("A", 0.0)) * np.cos(
        float(params.get("delta_phi_cyl", 0.0))
    )
    YLi7 = max(1e-12, base * max(0.05, factor))

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
        "Li_H": YLi7,
        "T_final_MeV": T_f,
        "success": bool(sol.success),
    }


def confront(
    results: dict[str, Any],
    empirical: dict[str, Any] | None = None,
    *,
    as_dict: bool = False,
) -> list[Any]:
    empirical = empirical or fetch_empirical_data(verbose=False)
    emp = empirical["bbn_abundances"]
    rows: list[Any] = []
    for name, key, exp_key in (
        ("D_H", "D_H", "D_H"),
        ("Y_p_He4", "Y_p_mass", "Y_p_He4"),
        ("Li7_H", "Li7_H", "Li7_H_obs"),
    ):
        th = float(results.get(key, 0.0))
        ex, unc = emp[exp_key]
        ex, unc = float(ex), float(unc)
        if th <= 0 or unc <= 0:
            continue
        sigma = abs(th - ex) / unc
        if as_dict:
            rows.append(
                {
                    "observable": name,
                    "theory": th,
                    "exp": ex,
                    "tension_sigma": sigma,
                }
            )
        else:
            rows.append((name, th, ex, sigma))
    return rows


def run_enhanced_scan(
    *,
    a_points: int = 8,
    phi_points: int = 5,
    delta_k_wind: float = 0.469,
    t_span: tuple[float, float] = (0.0, 400.0),
    verbose: bool = True,
) -> list[dict[str, Any]]:
    empirical = fetch_empirical_data(verbose=verbose)
    scan_results: list[dict[str, Any]] = []

    if verbose:
        print("\n" + "=" * 70)
        print("ENHANCED TAV BBN — TARGETED SCAN")
        print("=" * 70)

    for A in np.linspace(0.0, 0.035, max(int(a_points), 2)):
        for phi in np.linspace(0.0, np.pi, max(int(phi_points), 2)):
            params = {
                "A": float(A),
                "delta_phi_cyl": float(phi),
                "delta_k_wind": float(delta_k_wind),
                "xi": 8.0,
            }
            res = run_improved_bbn(params, t_span=t_span)
            tens = confront(res, empirical)
            li_tens = next(
                (t[3] if isinstance(t, tuple) else t["tension_sigma"] for t in tens if (
                    t[0] if isinstance(t, tuple) else t["observable"]
                ) == "Li7_H"),
                999.0,
            )
            entry = {
                "params": params,
                "results": res,
                "tensions": confront(res, empirical, as_dict=True),
                "li_tension_sigma": float(li_tens),
            }
            scan_results.append(entry)
            if verbose:
                print(
                    f"A={A:.3f} | phi={phi:.2f} | Li7={res['Li7_H']:.2e} | "
                    f"tension={li_tens:.1f}σ | Y_p={res['Y_p_mass']:.4f}"
                )

    return scan_results


def plot_lithium_tension_heatmap(
    results_list: list[dict[str, Any]],
    *,
    save_path: Path | str | None = None,
    show: bool = False,
) -> str:
    ensure_artifacts_dir()
    As = sorted({r["params"]["A"] for r in results_list})
    phis = sorted({r["params"]["delta_phi_cyl"] for r in results_list})

    tension_grid = np.zeros((len(phis), len(As)))
    for row in results_list:
        i = phis.index(row["params"]["delta_phi_cyl"])
        j = As.index(row["params"]["A"])
        tension_grid[i, j] = row["li_tension_sigma"]

    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.contourf(As, phis, tension_grid, levels=20, cmap="viridis_r")
    fig.colorbar(contour, ax=ax, label="Li7 tension (σ)")
    ax.set_xlabel("Interference amplitude A")
    ax.set_ylabel("delta_phi_cyl (rad)")
    ax.set_title("Lithium Tension vs Tav Parameters")
    plt.tight_layout()

    if save_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        from tav_shared.artifact_paths import TestSlug, artifact_path

        save_path = artifact_path(TestSlug.PRIME_PAST, "li_tension", "heatmap", "png")
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[TAV ENGINE] Heatmap saved to {path}")
    return str(path)


def load_and_analyze_scan(json_path: str | Path, *, top_n: int = 5) -> list[dict[str, Any]]:
    path = Path(json_path)
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    results = data.get("scan_results", data.get("results", []))
    if not results:
        raise ValueError(f"No scan results in {path}")

    best = data.get(
        "best",
        min(results, key=lambda row: row.get("li_tension_sigma", 999.0)),
    )

    print("\n=== BEST CONFIGURATIONS ===")
    sorted_results = sorted(results, key=lambda row: row.get("li_tension_sigma", 999.0))[:top_n]
    for idx, row in enumerate(sorted_results, start=1):
        p = row["params"]
        print(
            f"{idx}. A={p['A']:.3f} | phi={p['delta_phi_cyl']:.2f} | "
            f"kw={p.get('delta_k_wind', 0.469):.3f} "
            f"→ Li tension = {row['li_tension_sigma']:.2f} σ"
        )

    print(f"\nBest overall: {best['params']} → {best['li_tension_sigma']:.2f} σ")
    return sorted_results


def save_enhanced_scan(
    scan_results: list[dict[str, Any]],
    *,
    heatmap_path: str | None = None,
    prefix: str = "enhanced_tav_bbn_scan",
) -> str:
    ensure_artifacts_dir()
    best = min(scan_results, key=lambda row: row["li_tension_sigma"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    from tav_shared.artifact_paths import TestSlug, artifact_path, compose_dataset_slug

    json_path = artifact_path(TestSlug.PRIME_PAST, compose_dataset_slug(prefix), "report", "json")
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scan_results": scan_results,
        "best": best,
        "empirical": fetch_empirical_data(verbose=False),
        "heatmap": heatmap_path,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"[TAV ENGINE] Enhanced scan JSON saved: {json_path}")
    return str(json_path)


BEST_CONFIG_PATH = ARTIFACTS_DIR / "best_tav_bbn_config.json"
COMPARISON_PLOT_PATH = ARTIFACTS_DIR / "tav_vs_standard_comparison.png"


def save_best_config(
    best_result: dict[str, Any],
    *,
    filename: Path | str | None = None,
) -> str:
    """Persist lowest-Li7-tension configuration for reuse in research_tool runs."""
    ensure_artifacts_dir()
    path = Path(filename) if filename else BEST_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "best_params": best_result["params"],
        "results": best_result["results"],
        "li_tension_sigma": best_result["li_tension_sigma"],
        "tensions": best_result.get("tensions"),
        "description": "Best Tav interference configuration for BBN lithium reduction",
    }
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"[TAV ENGINE] Best configuration saved to: {path}")
    return str(path)


def plot_comparison(
    standard_results: dict[str, Any],
    best_results: dict[str, Any],
    *,
    save_path: Path | str | None = None,
    show: bool = False,
) -> str:
    """Bar chart: standard BBN (A=0) vs best Tav configuration."""
    ensure_artifacts_dir()
    labels = ["D/H", "Y_p (He4)", "Li7/H (×10⁹)"]
    std_vals = [
        standard_results["D_H"],
        standard_results["Y_p_mass"],
        standard_results["Li7_H"] * 1e9,
    ]
    tav_vals = [
        best_results["D_H"],
        best_results["Y_p_mass"],
        best_results["Li7_H"] * 1e9,
    ]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, std_vals, width, label="Standard BBN (A=0)", color="#1f77b4")
    ax.bar(x + width / 2, tav_vals, width, label="Best Tav Configuration", color="#ff7f0e")
    ax.set_ylabel("Value")
    ax.set_title("Standard BBN vs Best Tav Interference Run")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2e}" if height < 1 else f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    path = Path(save_path) if save_path else COMPARISON_PLOT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[TAV ENGINE] Comparison plot saved to: {path}")
    return str(path)


def load_and_highlight_best(
    json_path: str | Path,
    *,
    top_n: int = 5,
) -> dict[str, Any] | None:
    """Load prior scan JSON and print top configurations; return best row."""
    path = Path(json_path)
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    results = data.get("scan_results", data.get("results", []))
    if not results:
        print("No results found in JSON.")
        return None

    sorted_results = sorted(results, key=lambda row: row.get("li_tension_sigma", 999.0))[:top_n]
    print(f"\n=== TOP {top_n} BEST TAV CONFIGURATIONS ===")
    for idx, row in enumerate(sorted_results, start=1):
        p = row["params"]
        print(
            f"{idx}. A={p['A']:.3f} | phi={p['delta_phi_cyl']:.2f} | "
            f"kw={p.get('delta_k_wind', 0.469):.3f} "
            f"→ Tension: {row['li_tension_sigma']:.2f} σ"
        )

    best = sorted_results[0]
    print(f"\n>>> BEST: {best['params']} (Li tension: {best['li_tension_sigma']:.2f} σ)")
    return best


def run_final_confrontation_tool(
    *,
    a_points: int = 8,
    phi_points: int = 5,
    delta_k_wind: float = 0.469,
    t_span: tuple[float, float] = (0.0, 400.0),
    verbose: bool = True,
    save_comparison: bool = True,
    show_plots: bool = False,
) -> dict[str, Any]:
    """
    Full pipeline: scan → best config JSON → standard vs best comparison plot.
    """
    if verbose:
        print("Running Tav BBN scan...")

    scan_results = run_enhanced_scan(
        a_points=a_points,
        phi_points=phi_points,
        delta_k_wind=delta_k_wind,
        t_span=t_span,
        verbose=verbose,
    )
    best = min(scan_results, key=lambda row: row["li_tension_sigma"])

    best_json = save_best_config(best)
    outputs: dict[str, Any] = {
        "best": best,
        "best_config_path": best_json,
        "scan_count": len(scan_results),
    }

    if save_comparison:
        std_params = {
            "A": 0.0,
            "delta_phi_cyl": 0.0,
            "delta_k_wind": delta_k_wind,
            "xi": 8.0,
        }
        std_res = run_improved_bbn(std_params, t_span=t_span)
        outputs["comparison_plot"] = plot_comparison(
            std_res,
            best["results"],
            show=show_plots,
        )

    scan_json = save_enhanced_scan(scan_results, prefix="final_tav_bbn_scan")
    outputs["scan_json"] = scan_json

    if verbose:
        print("\nScan complete. Best configuration and comparison plot generated.")

    return outputs