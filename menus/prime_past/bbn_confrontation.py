"""
Complete Tav/Superblock BBN scanner with empirical confrontation.

RK45 minimal network (T, Y_n, Y_p) + lithium interference factor; parameter grid
over (A, delta_phi_cyl, delta_k_wind).

Project root: Public/TauSuperblock/
  - menus/prime_past/bbn_confrontation.py  (this module)
  - tav_bbn_confrontation.py               (CLI entry)
  - artifacts/prime_past/                  (reports + plots)

Used by ``tav_bbn_confrontation.py`` and Prime Past → BBN Confrontation Scan.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from menus.empirical_tests.tests import fetch_empirical_data as _shared_fetch_empirical
from tav_shared.tav_project_paths import TAU_SUPERBLOCK_ROOT

ARTIFACTS_DIR = TAU_SUPERBLOCK_ROOT / "artifacts" / "prime_past"

G_STAR = 10.75
M_PL_NATURAL = 1.22e19
Q_NP_MEV = 1.293
B_D_MEV = 2.22
T_START_MEV = 10.0

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def ensure_artifacts_dir() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


def fetch_empirical_data(*, verbose: bool = False) -> dict[str, Any]:
    """Reliable empirical data for confrontation (PDG + literature via shared pipeline)."""
    raw = _shared_fetch_empirical(verbose=verbose)
    bbn = raw["bbn_abundances"]
    nl = raw["neutron_lifetime"]
    return {
        "neutron_lifetime": {
            "bottle": float(nl["bottle_method"]),
            "beam": float(nl["beam_method"]),
            "pdg": float(nl["pdg_average"]),
            "unc": float(nl["uncertainty"]),
            "tension_sigma": float(nl["tension_sigma"]),
        },
        "bbn_abundances": {
            "D_H": (float(bbn["D_H"][0]), float(bbn["D_H"][1])),
            "Y_p_He4": (float(bbn["Y_p_He4"][0]), float(bbn["Y_p_He4"][1])),
            "Li7_H_obs": (float(bbn["Li7_H"][0]), float(bbn["Li7_H"][1])),
            "Li7_H_std_theory": (
                float(bbn["theory_Li7_H_standard"][0]),
                float(bbn["theory_Li7_H_standard"][1]),
            ),
        },
    }


def tav_interference_delta(T: float, params: dict[str, Any]) -> float:
    """Fractional perturbation to Hubble rate from Tau cylinder."""
    s = np.log(10.0 / max(float(T), 0.01))
    phase = float(params.get("delta_phi_cyl", 0.0)) + float(params.get("delta_k_wind", 0.469))
    envelope = np.exp(-s / max(float(params.get("xi", 8.0)), 0.1))
    return float(params.get("A", 0.0)) * np.cos(phase) * envelope


def modified_hubble(T: float, params: dict[str, Any]) -> float:
    """Standard radiation-dominated Hubble with Tav correction."""
    T = max(float(T), 0.01)
    H_std = 1.66 * np.sqrt(G_STAR) * (T**2) / M_PL_NATURAL
    delta = tav_interference_delta(T, params)
    return float(H_std * (1.0 + delta))


def run_minimal_bbn(
    params: dict[str, Any],
    *,
    t_span: tuple[float, float] = (0.0, 300.0),
    n_steps: int = 600,
) -> dict[str, Any]:
    """
    Improved minimal BBN with corrected lithium direction.

    Positive ``A`` tends to reduce late-time ^7Li. ``n_steps`` is reserved for
    future fixed-step integrators; RK45 uses adaptive stepping.
    """
    del n_steps  # reserved for API compatibility with standalone script
    T_start = T_START_MEV
    Y_n_init = 1.0 / (1.0 + np.exp(Q_NP_MEV / T_start))
    Y_p_init = 1.0 - Y_n_init
    y0 = [T_start, Y_n_init, Y_p_init]

    def ode(t: float, y: list[float]) -> list[float]:
        T, Y_n, Y_p = y
        H = modified_hubble(T, params)
        dT_dt = -H * T
        rate_np = 1.0 / (880.0 * (T / 0.8) ** 5 + 1e-10)
        rate_pn = rate_np * np.exp(-Q_NP_MEV / max(T, 1e-4))
        dY_n = -rate_np * Y_n + rate_pn * Y_p
        dY_p = -dY_n
        return [dT_dt, dY_n, dY_p]

    sol = solve_ivp(
        ode,
        t_span,
        y0,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
        dense_output=False,
    )

    T_f, Y_n_f, Y_p_f = (float(v) for v in sol.y[:, -1])
    Y_D = max(0.0, 2 * Y_n_f * Y_p_f * np.exp(B_D_MEV / max(T_f, 0.01)) * 0.01)
    Y_He4 = min(2 * Y_n_f, 1.0 - Y_p_f)

    base_li = 5.0e-10 * (Y_n_f / 0.15)
    interference_factor = 1.0 - 12.0 * float(params.get("A", 0.0)) * np.cos(
        float(params.get("delta_phi_cyl", 0.0))
    )
    Y_Li7 = max(1e-12, base_li * max(0.1, interference_factor))

    return {
        "Y_n": Y_n_f,
        "Y_p": Y_p_f,
        "Y_D": Y_D,
        "Y_He4": Y_He4,
        "Y_Li7": Y_Li7,
        "Y_p_mass": 4.0 * Y_He4,
        "D_H": Y_D / Y_p_f if Y_p_f > 0 else 0.0,
        "Li7_H": Y_Li7,
        "Li_H": Y_Li7,
        "T_final_MeV": T_f,
        "success": bool(sol.success),
    }


def confront_with_data(
    results: dict[str, Any],
    empirical: dict[str, Any],
) -> list[tuple[str, float, float, float]]:
    """Return (observable, theory, observed, tension_sigma) tuples."""
    emp = empirical["bbn_abundances"]
    tensions: list[tuple[str, float, float, float]] = []

    d_theory = float(results["D_H"])
    d_exp, d_unc = emp["D_H"]
    if d_theory > 0 and d_unc > 0:
        tensions.append(("D_H", d_theory, float(d_exp), abs(d_theory - d_exp) / d_unc))

    y_theory = float(results["Y_p_mass"])
    y_exp, y_unc = emp["Y_p_He4"]
    if y_unc > 0:
        tensions.append(
            ("Y_p_He4", y_theory, float(y_exp), abs(y_theory - float(y_exp)) / float(y_unc))
        )

    li_theory = float(results["Li7_H"])
    li_exp, li_unc = emp["Li7_H_obs"]
    if li_unc > 0:
        tensions.append(
            ("Li7_H", li_theory, float(li_exp), abs(li_theory - float(li_exp)) / float(li_unc))
        )

    return tensions


def print_summary_table(results_list: list[dict[str, Any]], *, top_n: int = 8) -> None:
    """Print ranked scan rows (lowest Li7 tension first)."""
    ranked = sorted(results_list, key=lambda row: row["li_tension_sigma"])
    print("\n" + "-" * 78)
    print(f"{'Rank':<5} {'A':>6} {'phi':>6} {'kw':>6} {'Li7/H':>10} {'Li σ':>7} {'Y_p':>8} {'D/H':>10}")
    print("-" * 78)
    for idx, row in enumerate(ranked[:top_n], start=1):
        p = row["params"]
        res = row["results"]
        print(
            f"{idx:<5} {p['A']:6.3f} {p['delta_phi_cyl']:6.2f} {p['delta_k_wind']:6.3f} "
            f"{res['Li7_H']:10.2e} {row['li_tension_sigma']:7.1f} "
            f"{res['Y_p_mass']:8.4f} {res['D_H']:10.2e}"
        )
    print("-" * 78)


def plot_scan_summary(
    results_list: list[dict[str, Any]],
    *,
    output_path: Path | None = None,
    show: bool = False,
) -> str | None:
    """Scatter Li7 tension vs A coloured by phase (requires matplotlib)."""
    if not HAS_MPL or not results_list:
        return None
    ensure_artifacts_dir()
    A_vals = [row["params"]["A"] for row in results_list]
    li_sigma = [row["li_tension_sigma"] for row in results_list]
    phi_vals = [row["params"]["delta_phi_cyl"] for row in results_list]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    scatter = ax.scatter(A_vals, li_sigma, c=phi_vals, cmap="viridis", alpha=0.75, s=36)
    ax.set_xlabel("Interference amplitude A")
    ax.set_ylabel("^7Li/H tension (σ)")
    ax.set_title("Tav BBN Confrontation — lithium tension vs A")
    ax.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="delta_phi_cyl (rad)")
    plt.tight_layout()

    if output_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = ARTIFACTS_DIR / f"tav_bbn_scan_{stamp}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[TAV ENGINE] Scan plot saved: {output_path}")
    return str(output_path)


def default_scan_ranges(n_points: int = 4) -> dict[str, Any]:
    return {
        "A": np.linspace(0.0, 0.03, max(int(n_points), 2)),
        "delta_phi_cyl": [0.0, np.pi / 2, np.pi],
        "delta_k_wind": [0.0, 0.469, 0.8],
    }


def run_tav_scan(
    scan_ranges: dict[str, Any] | None = None,
    *,
    n_points: int = 4,
    t_span: tuple[float, float] = (0.0, 300.0),
    verbose: bool = True,
    plot: bool = False,
    show_plot: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Grid scan over Tav interference parameters; return all runs + best Li7 match."""
    if scan_ranges is None:
        scan_ranges = default_scan_ranges(n_points)

    empirical = fetch_empirical_data(verbose=verbose)
    results_list: list[dict[str, Any]] = []

    if verbose:
        print("\n" + "=" * 70)
        print("TAV BBN CONFRONTATION — PARAMETER SCAN")
        print("=" * 70)

    for A in scan_ranges["A"]:
        for phi in scan_ranges["delta_phi_cyl"]:
            for kw in scan_ranges["delta_k_wind"]:
                params = {
                    "A": float(A),
                    "delta_phi_cyl": float(phi),
                    "delta_k_wind": float(kw),
                    "xi": 8.0,
                }
                res = run_minimal_bbn(params, t_span=t_span)
                tensions = confront_with_data(res, empirical)
                li_tension = next(t[3] for t in tensions if t[0] == "Li7_H")

                entry = {
                    "params": params,
                    "results": res,
                    "tensions": [
                        {
                            "observable": name,
                            "theory": theory,
                            "exp": exp,
                            "tension_sigma": round(sigma, 2),
                        }
                        for name, theory, exp, sigma in tensions
                    ],
                    "li_tension_sigma": float(li_tension),
                    "Y_p_mass": res["Y_p_mass"],
                    "D_H": res["D_H"],
                }
                results_list.append(entry)

                if verbose:
                    print(
                        f"A={A:.3f} | phi={phi:.2f} | kw={kw:.3f} | "
                        f"Li7_H={res['Li7_H']:.2e} | tension={li_tension:.1f}σ | "
                        f"Y_p={res['Y_p_mass']:.4f}"
                    )

    best = min(results_list, key=lambda row: row["li_tension_sigma"])

    if verbose:
        print_summary_table(results_list)
        print("\n" + "=" * 70)
        print("BEST CONFIGURATION (lowest Li tension)")
        print("=" * 70)
        print(f"Parameters: {best['params']}")
        print(f"Li7_H tension: {best['li_tension_sigma']:.2f} σ")
        print(f"Y_p_mass: {best['Y_p_mass']:.4f} (target ~0.245)")
        print(f"D_H: {best['D_H']:.2e}")

    if plot:
        plot_scan_summary(results_list, show=show_plot)

    return results_list, best


def save_scan_report(
    results_list: list[dict[str, Any]],
    best: dict[str, Any],
    *,
    empirical: dict[str, Any] | None = None,
    prefix: str = "tav_bbn_scan",
    also_cwd: bool = True,
) -> str:
    """Write JSON scan under artifacts/prime_past/ (and optionally cwd)."""
    ensure_artifacts_dir()
    empirical = empirical or fetch_empirical_data(verbose=False)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = ARTIFACTS_DIR / f"{prefix}_{stamp}.json"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scan_results": results_list,
        "best": best,
        "empirical": empirical,
    }
    text = json.dumps(payload, indent=2, default=float) + "\n"
    path.write_text(text, encoding="utf-8")
    print(f"[TAV ENGINE] Results saved to {path}")

    if also_cwd:
        cwd_path = Path("tav_bbn_scan_results.json")
        cwd_path.write_text(text, encoding="utf-8")
        print(f"[TAV ENGINE] Results saved to {cwd_path.resolve()}")

    return str(path)