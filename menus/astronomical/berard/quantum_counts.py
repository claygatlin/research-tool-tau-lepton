"""
Tiny_Tau / Qiskit Aer quantum measurement histogram analysis.

Ingests sparse bitstring→count JSON from tau-cosmology-berard-framework.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tav_shared.artifact_paths import TestSlug, artifact_path, artifact_timestamp


def load_counts_histogram(path: str | Path) -> dict[str, int]:
    """Load a bitstring counts JSON object."""
    p = Path(path)
    with p.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict counts in {p.name}")
    return {str(k): int(v) for k, v in raw.items()}


def analyze_quantum_counts(
    counts: dict[str, int],
    *,
    label: str = "counts",
) -> dict[str, Any]:
    """
    Summarize sparse Qiskit Aer histograms (Tiny_Tau resonance simulations).
    """
    if not counts:
        return {"label": label, "verdict": "EMPTY", "total_shots": 0}

    bitstrings = list(counts.keys())
    values = np.asarray(list(counts.values()), dtype=np.int64)
    total_shots = int(values.sum())
    n_unique = int(len(bitstrings))
    bit_len = len(bitstrings[0]) if bitstrings else 0
    consistent_len = all(len(s) == bit_len for s in bitstrings)

    probs = values.astype(np.float64) / max(total_shots, 1)
    entropy = float(-np.sum(probs[probs > 0] * np.log2(probs[probs > 0])))

    order = np.argsort(values)[::-1]
    top_k = min(10, n_unique)
    top_peaks = [
        {
            "bitstring": bitstrings[int(i)],
            "count": int(values[int(i)]),
            "fraction": float(values[int(i)] / max(total_shots, 1)),
        }
        for i in order[:top_k]
    ]

    # BC=1.054 encoded in filenames as 1054 — check bit patterns with high '1' density
    one_fractions = [s.count("1") / max(len(s), 1) for s in bitstrings]
    mean_one_frac = float(np.mean(one_fractions)) if one_fractions else 0.0

    bc_hint = None
    match = re.search(r"1054", label)
    if match:
        bc_hint = 1.054

    return {
        "label": label,
        "total_shots": total_shots,
        "unique_bitstrings": n_unique,
        "qubit_register_width": bit_len if consistent_len else None,
        "consistent_bit_width": consistent_len,
        "shannon_entropy_bits": entropy,
        "max_entropy_bits": float(math.log2(n_unique)) if n_unique > 1 else 0.0,
        "mean_bit_one_fraction": mean_one_frac,
        "top_peaks": top_peaks,
        "berard_constant_hint": bc_hint,
        "source": "qiskit_aer_simulator",
        "hardware_note": "Simulator data (noiseless); not hardware counts",
        "verdict": "QUANTUM_COUNTS_ANALYZED",
    }


def compare_shot_convergence(
    analyses: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare multiple shot-count files for peak stability."""
    if len(analyses) < 2:
        return {"available": False, "reason": "need >= 2 count files"}
    shots = [int(a.get("total_shots") or 0) for a in analyses]
    top0 = [
        (a.get("top_peaks") or [{}])[0].get("bitstring") for a in analyses
    ]
    stable = len(set(top0)) == 1
    return {
        "available": True,
        "total_shots_series": shots,
        "dominant_peak_bitstrings": top0,
        "dominant_peak_stable": stable,
        "entropy_series": [a.get("shannon_entropy_bits") for a in analyses],
    }


def run_quantum_counts_analysis(
    paths: list[str | Path],
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """Analyze one or more counts JSON files and optionally check convergence."""
    from menus.astronomical.desi.json_util import write_json

    analyses: list[dict[str, Any]] = []
    for path in paths:
        p = Path(path)
        counts = load_counts_histogram(p)
        analyses.append(analyze_quantum_counts(counts, label=p.name))

    convergence = compare_shot_convergence(analyses)
    report: dict[str, Any] = {
        "action": "Tiny_Tau quantum counts analysis",
        "files": [str(p) for p in paths],
        "analyses": analyses,
        "shot_convergence": convergence,
        "timestamp": artifact_timestamp(datetime.now(timezone.utc)),
        "confidence": "exploratory",
    }
    out = artifact_path(TestSlug.BERARD, "quantum_counts", "report", "json")
    write_json(out, report, indent=2, sort_keys=True)
    report["report_path"] = str(out)

    if verbose:
        print("=" * 60)
        print("BERARD — Quantum counts (Tiny_Tau / Qiskit Aer)")
        for block in analyses:
            print(
                f"  {block['label']}: {block['total_shots']:,} shots, "
                f"{block['unique_bitstrings']:,} unique, "
                f"H={block['shannon_entropy_bits']:.2f} bits"
            )
            if block.get("top_peaks"):
                peak = block["top_peaks"][0]
                print(
                    f"    top peak: {peak['bitstring'][:20]}… "
                    f"({peak['fraction'] * 100:.2f}%)"
                )
        if convergence.get("available"):
            print(
                f"  Peak stable across files: {convergence.get('dominant_peak_stable')}"
            )
        print("=" * 60)
    return report