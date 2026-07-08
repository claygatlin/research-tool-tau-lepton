#!/usr/bin/env python3
"""
Script-facing entry point for minimal Tav-interference BBN.

Re-exports ``menus.prime_past.bbn_interference`` under the names used in
standalone notebooks and confrontation snippets.
"""

from __future__ import annotations

from typing import Any

from menus.empirical_tests.tests import fetch_empirical_data
from menus.prime_past.bbn_interference import (
    DEFAULT_FRAMEWORK_PARAMS,
    FRAMEWORK_PARAMS,
    FrameworkParams,
    confront_empirical_abundances,
    run_bbn,
    run_bbn_comparison,
    save_bbn_report,
)

__all__ = [
    "FRAMEWORK_PARAMS",
    "DEFAULT_FRAMEWORK_PARAMS",
    "FrameworkParams",
    "fetch_empirical_data",
    "run_bbn",
    "run_bbn_comparison",
    "confront_empirical_abundances",
    "confront_lithium_shift",
    "save_bbn_report",
]


def confront_lithium_shift(
    params: dict[str, Any] | FrameworkParams | None = None,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run standard vs Tav BBN and print ^7Li/H shifts against empirical anchors.

    Matches the standalone confrontation snippet:
      std = run_bbn({**FRAMEWORK_PARAMS, 'A': 0.0})
      tav = run_bbn(FRAMEWORK_PARAMS)
    """
    base = FRAMEWORK_PARAMS if params is None else (
        params if isinstance(params, dict) else {
            "A": params.A,
            "delta_phi_cyl": params.delta_phi_cyl,
            "delta_k_wind": params.delta_k_wind,
            "xi": params.xi,
        }
    )

    std = run_bbn({**base, "A": 0.0}, label="Standard", verbose=verbose)
    tav = run_bbn(base, label="Tav Modified", verbose=verbose)

    emp = fetch_empirical_data(verbose=verbose)
    li_obs = float(emp["bbn_abundances"]["Li7_H"][0])
    li_theory_std = float(std["Li_H"])
    li_theory_tav = float(tav["Li_H"])

    std_shift_pct = (li_theory_std - li_obs) / li_obs * 100.0
    tav_residual_pct = (li_theory_tav - li_obs) / li_obs * 100.0
    tav_vs_std_pct = (
        (li_theory_tav - li_theory_std) / li_theory_std * 100.0
        if li_theory_std > 0
        else float("nan")
    )

    if verbose:
        print(f"Observed ^7Li/H: {li_obs:.2e}")
        print(
            f"Standard BBN ^7Li/H: {li_theory_std:.2e}  "
            f"(shift needed: {std_shift_pct:+.1f}%)"
        )
        print(
            f"Tav Modified ^7Li/H: {li_theory_tav:.2e}  "
            f"(residual shift: {tav_residual_pct:+.1f}%)"
        )
        print(f"Tav vs standard ^7Li/H: {tav_vs_std_pct:+.1f}%")

    return {
        "observed_Li7_H": li_obs,
        "standard_Li_H": li_theory_std,
        "tav_Li_H": li_theory_tav,
        "standard_shift_percent": std_shift_pct,
        "tav_residual_shift_percent": tav_residual_pct,
        "tav_vs_standard_percent": tav_vs_std_pct,
        "standard_run": std,
        "tav_run": tav,
        "empirical": emp,
    }


if __name__ == "__main__":
    confront_lithium_shift()