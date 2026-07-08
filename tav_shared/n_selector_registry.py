"""
Central registry: which module actions expose the interactive n selector,
and what n controls for each action.
"""

from __future__ import annotations

from typing import Any

# module extension name → actions where n_points is wired to a real parameter
MEANINGFUL_N_ACTIONS: dict[str, frozenset[str]] = {
    "tau_sb_desi_extension": frozenset(
        {
            "Real DESI DR2 Scan (Cobaya)",
            "Periodicity Scan (1/7 Tracks)",
            "Model Compare (ΛCDM vs aDE vs Tau-SB)",
            "MCMC Posteriors (emcee)",
            "Healpy Dipole Fit",
            "Change-Point Detection",
            "Injection/Recovery Tests",
            "Joint DESI+SN+Planck Fit",
            "Batch Tracer Scan (All DR2)",
            "Full Production Pipeline",
        }
    ),
    "frb_web_extension": frozenset(
        {
            "Classify FRB Paths",
            "Analyze DM Residuals by Path",
            "Scan 1/7 Tav Harmonics",
            "Full FRB Tav Analysis",
        }
    ),
    "tav_resonance_extension": frozenset(
        {
            "FRB Resonance Scan (tav-resonance)",
            "Full Tav-Resonance Demo",
        }
    ),
    "tsb_research_extension": frozenset(
        {
            "Generate Mocks",
            "Generate Full DESI Summary Dashboard",
        }
    ),
}

_N_POINTS_HINTS: dict[str, dict[str, str]] = {
    "tau_sb_desi_extension": {
        "Injection/Recovery Tests": "Monte Carlo injection trials per amplitude (not BAO count)",
        "Periodicity Scan (1/7 Tracks)": "Lomb–Scargle frequency grid size",
        "Joint DESI+SN+Planck Fit": "Max Pantheon+ supernovae in joint fit",
        "Full Production Pipeline": "Periodogram grid + injection trials + SN subsample (BAO n fixed)",
        "Real DESI DR2 Scan (Cobaya)": "Batch tracer limit + Lomb–Scargle grid (BAO n per tracer fixed)",
        "Model Compare (ΛCDM vs aDE vs Tau-SB)": "Batch tracer limit + Lomb–Scargle grid (BAO n fixed)",
        "Batch Tracer Scan (All DR2)": "Max tracers to scan this batch",
        "MCMC Posteriors (emcee)": "Batch tracer prefetch limit before MCMC",
        "Healpy Dipole Fit": "Batch tracer prefetch limit before dipole fit",
        "Change-Point Detection": "Batch tracer prefetch limit before changepoint scan",
    },
    "frb_web_extension": {
        "Classify FRB Paths": "Max FRBs used in path classifier",
        "Analyze DM Residuals by Path": "Max FRBs in residual / periodogram pass",
        "Scan 1/7 Tav Harmonics": "Max FRBs in harmonic scan",
        "Full FRB Tav Analysis": "Max FRBs in end-to-end pipeline",
    },
    "tav_resonance_extension": {
        "FRB Resonance Scan (tav-resonance)": "Max FRBs passed to tav-resonance scan",
        "Full Tav-Resonance Demo": "Max FRBs in demo resonance scan",
    },
    "tsb_research_extension": {
        "Generate Mocks": "Synthetic mock lattice size",
        "Generate Full DESI Summary Dashboard": "Diagnostic plot grid size (resamples real fit)",
    },
}


def parse_n_points(raw: Any) -> int | None:
    if raw is None or str(raw).strip() == "":
        return None
    return max(3, int(float(str(raw).strip())))


def action_uses_n_selector(module_tag: str, action: str) -> bool:
    return action in MEANINGFUL_N_ACTIONS.get(module_tag, frozenset())


def n_points_hint(module_tag: str, action: str) -> str:
    return _N_POINTS_HINTS.get(module_tag, {}).get(
        action,
        "Interactive sample-size control for this action",
    )


def n_points_field(module_tag: str, action: str) -> dict[str, Any]:
    return {
        "key": "n_points",
        "label": "Sample size n (interactive selector will appear)",
        "default": "25",
        "required": False,
        "hint": n_points_hint(module_tag, action),
    }


def desi_n_knobs(action: str, n: int) -> dict[str, int]:
    """Map menu n to DESI scanner / production parameters."""
    n = max(3, int(n))
    if action == "Injection/Recovery Tests":
        return {"n_injection_trials": n}
    if action == "Periodicity Scan (1/7 Tracks)":
        return {"n_freq": n}
    if action == "Joint DESI+SN+Planck Fit":
        return {"max_sne": n}
    if action == "Full Production Pipeline":
        return {"n_freq": n, "n_injection_trials": n, "max_sne": n}
    if action in {"Real DESI DR2 Scan (Cobaya)", "Model Compare (ΛCDM vs aDE vs Tau-SB)"}:
        return {"n_freq": n}
    return {}


def desi_n_usage_message(action: str, n: int) -> str:
    knobs = desi_n_knobs(action, n)
    if action == "Batch Tracer Scan (All DR2)":
        return f"n={n} → batch_limit (max tracers this run; BAO points per tracer unchanged)"
    if not knobs:
        return f"n={n} (no wired parameters for {action})"
    parts = ", ".join(f"{k}={v}" for k, v in knobs.items())
    return f"n={n} → {parts} (DESI BAO vector length unchanged)"


def subsample_rows(frame, n: int):
    """Evenly subsample a pandas DataFrame to at most n rows."""
    import numpy as np

    if len(frame) <= n:
        return frame
    idx = np.linspace(0, len(frame) - 1, int(n), dtype=int)
    return frame.iloc[idx].reset_index(drop=True)