"""
Berard / tau-cosmology figure styles and NGC 3198 benchmark plots.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from tav_shared.artifact_paths import TestSlug, artifact_path
from tav_shared.tav_project_paths import TAU_SUPERBLOCK_ROOT

BERARD_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#1a1a2e",
    "axes.labelcolor": "#1a1a2e",
    "text.color": "#1a1a2e",
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
}
BERARD_COLORS = {
    "data": "#1a1a2e",
    "baryons": "#6b7280",
    "piso": "#2563eb",
    "berard_hybrid": "#dc2626",
    "residual": "#7c3aed",
    "phase2": "#059669",
    "reference": "#9ca3af",
}


def apply_berard_style() -> None:
    """APS/PRD-inspired serif style from berard-framework figures/."""
    plt.rcParams.update(BERARD_STYLE)


def import_reference_plots(
    *,
    tau_cosmology_root: Path | None = None,
    berard_framework_root: Path | None = None,
) -> list[str]:
    """Copy imported benchmark PNGs/PDFs into datasets/berard_framework/figures/."""
    dest = TAU_SUPERBLOCK_ROOT / "datasets" / "berard_framework" / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []

    tc = tau_cosmology_root or Path("/tmp/tau-cosmology")
    ngc = tc / "NGC_3198_Rotation-Curve_Benchmark"
    if ngc.is_dir():
        for pattern in ("*.png", "*.pdf"):
            for src in ngc.glob(pattern):
                dst = dest / f"ngc3198_{src.name}"
                shutil.copy2(src, dst)
                copied.append(str(dst))

    bf = berard_framework_root or Path("/tmp/tau-cosmology-berard-framework/figures")
    if bf.is_dir():
        for src in bf.glob("*.pdf"):
            dst = dest / src.name
            shutil.copy2(src, dst)
            copied.append(str(dst))

    return copied


def plot_ngc3198_benchmark(
    data: dict[str, np.ndarray],
    models: dict[str, Any],
    best_name: str,
    phase2: dict[str, Any],
) -> dict[str, str]:
    """Generate rotation curve, residual, and model-comparison panels."""
    apply_berard_style()
    r = data["r_kpc"]
    v_obs = data["Vobs"]
    e_vobs = data["eVobs"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: rotation curves
    ax = axes[0]
    ax.errorbar(
        r, v_obs, yerr=e_vobs,
        fmt="o", color=BERARD_COLORS["data"], ms=3, alpha=0.7,
        label="NGC 3198 (SPARC)",
    )
    for name, color_key in (
        ("baryons_only", "baryons"),
        ("plain_piso", "piso"),
        ("berard_hybrid_piso", "berard_hybrid"),
    ):
        if name not in models:
            continue
        v_m = np.asarray(models[name]["v_model"], dtype=float)
        ax.plot(r, v_m, "-", color=BERARD_COLORS[color_key], lw=1.8, label=name)
    ax.set_xlabel("r (kpc)")
    ax.set_ylabel("v (km/s)")
    ax.set_title("(A) Rotation curves")
    ax.legend(fontsize=8)

    # Panel B: residuals (best model)
    ax = axes[1]
    best = models[best_name]
    v_best = np.asarray(best["v_model"], dtype=float)
    dv = v_obs - v_best
    ax.errorbar(r, dv, yerr=e_vobs, fmt="o", color=BERARD_COLORS["residual"], ms=3)
    ax.axhline(0.0, color=BERARD_COLORS["reference"], ls="--", lw=1)
    fixed = phase2.get("fixed_frequency_fit") or {}
    if fixed.get("amplitude_kms", 0) > 0:
        from menus.astronomical.berard.phase2_residual import log_periodic_residual

        amp = float(fixed["amplitude_kms"])
        phi = float(fixed["phase_rad"])
        overlay = log_periodic_residual(r, amp, phi)
        ax.plot(r, overlay, "-", color=BERARD_COLORS["phase2"], lw=1.5,
                label=f"Phase 2 (1/ln7), A={amp:.1f}")
        ax.legend(fontsize=8)
    ax.set_xlabel("r (kpc)")
    ax.set_ylabel("Δv (km/s)")
    ax.set_title(f"(B) Residuals — {best_name}")

    # Panel C: ΔBIC vs baryons
    ax = axes[2]
    names = []
    deltas = []
    colors = []
    base_bic = float(models["baryons_only"]["bic"])
    for name, fit in models.items():
        if name == "baryons_only":
            continue
        names.append(name.replace("_", "\n"))
        deltas.append(float(fit["bic"]) - base_bic)
        colors.append(
            BERARD_COLORS["berard_hybrid"]
            if "berard" in name
            else BERARD_COLORS["piso"]
        )
    ax.bar(range(len(names)), deltas, color=colors, alpha=0.85)
    ax.axhline(6.0, color="#b45309", ls=":", label="ΔBIC=6 (strong)")
    ax.axhline(-6.0, color="#b45309", ls=":")
    ax.axhline(0.0, color=BERARD_COLORS["reference"], lw=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("ΔBIC vs baryons")
    ax.set_title("(C) Model comparison")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = artifact_path(TestSlug.BERARD, "ngc3198_benchmark", "plot", "png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {"ngc3198_benchmark_plot": str(out)}