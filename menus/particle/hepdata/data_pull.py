"""
HEPData (Numerical Correlation) — live INSPIRE record fetch + MCMC sweep.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from tav_shared import hepdata_engine, mcmc_engine, require_topology
from tav_research.data_pull_common import COMMON_BATCH_HINT, standard_entry_fields

MENU_LABEL = "HEPData (Numerical Correlation)"


def entry_instructions() -> list[str]:
    return [
        "Query: numeric INSPIRE record ID (e.g. 1803608) or ins1803608.",
        "URLs fetched live: https://www.hepdata.net/record/ins{ID}?format=json",
        "Table JSON: https://www.hepdata.net/download/table/ins{ID}/{table}/1/json",
        "Files: optional local JSON/CSV mirrors; batch lists supported.",
        COMMON_BATCH_HINT,
    ]


def entry_fields() -> list[dict]:
    return standard_entry_fields(
        MENU_LABEL,
        query_hint="INSPIRE / HEPData record ID (e.g. 1803608)",
    )


def fetch_and_graph(query: str, params: dict[str, Any] | None = None) -> None:
    params = params or {}
    record = hepdata_engine.fetch_hepdata_record(query)
    df = hepdata_engine.process_and_filter_table(record)
    x = df["Kinematic_X"]
    y_total = df["Observable_Y"]
    title = f"{MENU_LABEL}: {query}"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y_total, color="red", label="Live Empirical Signal")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Kinematic X")
    ax.set_ylabel("Observable Y")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.show()

    _run_analysis_report(query, df, x, y_total, domain_type="HEPData")


def _run_analysis_report(query, df, x, y_total, *, domain_type: str) -> None:
    dev, is_anomalous = require_topology.calculate_topological_correlation(df, domain=domain_type)
    print(f"\n--- Analytical Report: {query} ---")
    print(f"Mean Flux Intensity: {np.mean(y_total):.6f}")
    print(f"Max Deviation: {np.max(np.abs(dev)):.6f}")
    if is_anomalous:
        print("[CORRELATION FOUND] Anomalous phase-shift detected.")
    else:
        print("[SIGNAL STABLE] No TS dynamic deviations detected.")
    print("\n[TAV ENGINE] Initiating MCMC statistical sweep...")
    comparison = mcmc_engine.run_model_comparison(x, y_total)
    print(mcmc_engine.format_comparison_report(comparison))