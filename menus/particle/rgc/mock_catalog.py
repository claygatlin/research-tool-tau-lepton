#!/usr/bin/env python3
"""
RGC mock catalog: inject 1/7 resonance + Δn = 1/14 ladder in normalized x ∈ [0, 1].

Two-component mixture (uniform background + multi-Gaussian signal) for blinded
validation of peak finders, graph builders, and supervised RGC modules.
Distinct from the DESI BAO mock in menus.astronomical.desi.mock_catalog.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tav_shared.run_output import ARTIFACTS_DIR

# Geometric anchors (exact rationals — session 2026-07-09)
RESONANCE_CENTER: float = 1.0 / 7.0
DELTA_N: float = 1.0 / 14.0
MAIN_AMP: float = 1.0
STEP_AMPLITUDE_DECAY: float = 0.35

DEFAULT_SEED: int = 42
DEFAULT_NUM_EVENTS: int = 100_000
DEFAULT_X_MIN: float = 0.0
DEFAULT_X_MAX: float = 1.0
DEFAULT_SIGMA: float = 0.005
DEFAULT_SIG_FRAC: float = 0.10
DEFAULT_NUM_DELTA_STEPS: int = 2

RGC_ARTIFACTS_DIR = ARTIFACTS_DIR / "rgc"


@dataclass(frozen=True)
class MockCatalogConfig:
    seed: int = DEFAULT_SEED
    num_events: int = DEFAULT_NUM_EVENTS
    x_min: float = DEFAULT_X_MIN
    x_max: float = DEFAULT_X_MAX
    resonance_center: float = RESONANCE_CENTER
    sigma: float = DEFAULT_SIGMA
    sig_frac: float = DEFAULT_SIG_FRAC
    main_amp: float = MAIN_AMP
    delta_n: float = DELTA_N
    num_delta_steps: int = DEFAULT_NUM_DELTA_STEPS
    step_amplitude_decay: float = STEP_AMPLITUDE_DECAY
    output_prefix: str = "mock_resonance_catalog_1over7_deltan"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )


def _component_ladder(cfg: MockCatalogConfig) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Return k indices, truth μ_k positions, and normalized weights w_k."""
    ks: list[int] = []
    mus: list[float] = []
    raw_weights: list[float] = []
    r = cfg.step_amplitude_decay
    for k in range(-cfg.num_delta_steps, cfg.num_delta_steps + 1):
        mu_k = cfg.resonance_center + k * cfg.delta_n
        if mu_k < cfg.x_min or mu_k > cfg.x_max:
            continue
        ks.append(k)
        mus.append(mu_k)
        raw_weights.append(cfg.main_amp if k == 0 else r ** abs(k))
    if not ks:
        raise ValueError("No ladder components inside observable window")
    weights = np.asarray(raw_weights, dtype=float)
    weights /= weights.sum()
    return ks, np.asarray(mus, dtype=float), weights


def generate_mock_catalog(
    cfg: MockCatalogConfig | None = None,
) -> tuple[pd.DataFrame, list[int], np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Sample unbinned RGC catalog with truth labels.

    p(x) = (1 - f_sig) p_bkg + f_sig Σ_k w_k N(x | μ_k, σ)
    """
    cfg = cfg or MockCatalogConfig()
    rng = np.random.default_rng(cfg.seed)
    ks, mus, probs = _component_ladder(cfg)

    n_total = int(cfg.num_events)
    n_sig = int(round(n_total * cfg.sig_frac))
    n_bkg = n_total - n_sig

    x_bkg = rng.uniform(cfg.x_min, cfg.x_max, size=n_bkg)
    bkg_rows = [
        {
            "event_id": i,
            "x": float(x_bkg[i]),
            "type": "background",
            "component_k": -999,
            "resonance_center": cfg.resonance_center,
            "weight": 1.0,
        }
        for i in range(n_bkg)
    ]

    n_per_comp = rng.multinomial(n_sig, probs)
    sig_rows: list[dict[str, Any]] = []
    event_id = n_bkg
    for k_idx, (k, mu_k, n_k) in enumerate(zip(ks, mus, n_per_comp)):
        if n_k <= 0:
            continue
        draws = rng.normal(mu_k, cfg.sigma, size=n_k)
        draws = np.clip(draws, cfg.x_min, cfg.x_max)
        for x_val in draws:
            sig_rows.append(
                {
                    "event_id": event_id,
                    "x": float(x_val),
                    "type": "signal",
                    "component_k": int(k),
                    "resonance_center": cfg.resonance_center,
                    "weight": float(probs[k_idx]),
                }
            )
            event_id += 1

    df = pd.DataFrame(bkg_rows + sig_rows)
    df = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    df["event_id"] = np.arange(len(df), dtype=int)

    meta = {
        "n_total": n_total,
        "n_background": n_bkg,
        "n_signal": n_sig,
        "sig_frac": cfg.sig_frac,
        "ks": ks,
        "mu_truth": [float(m) for m in mus],
        "weights_truth": [float(w) for w in probs],
        "n_per_component": [int(n) for n in n_per_comp],
        "resonance_center": cfg.resonance_center,
        "delta_n": cfg.delta_n,
        "sigma": cfg.sigma,
    }
    return df, ks, mus, probs, meta


def plot_catalog(
    df: pd.DataFrame,
    ks: Sequence[int],
    mus: np.ndarray,
    probs: np.ndarray,
    n_per_comp: Sequence[int],
    *,
    cfg: MockCatalogConfig,
    output_path: Path,
    bins: int = 80,
) -> Path:
    """Histogram + analytic expected component overlays."""
    fig, ax = plt.subplots(figsize=(10, 6))
    counts, edges, _ = ax.hist(
        df["x"],
        bins=bins,
        range=(cfg.x_min, cfg.x_max),
        density=False,
        alpha=0.55,
        color="0.55",
        label="Mock events",
    )
    bin_width = edges[1] - edges[0]
    x_fine = np.linspace(cfg.x_min, cfg.x_max, 512)
    n_sig = int((df["type"] == "signal").sum())
    for k, mu_k, w_k, n_k in zip(ks, mus, probs, n_per_comp):
        expected = n_k * _gaussian_pdf(x_fine, float(mu_k), cfg.sigma) * bin_width
        style = "-" if k == 0 else "--"
        color = "crimson" if k == 0 else "darkorange"
        lw = 2.2 if k == 0 else 1.4
        ax.plot(
            x_fine,
            expected,
            style,
            color=color,
            lw=lw,
            label=f"k={k:+d} μ={mu_k:.5f} (N={int(n_k)})",
        )
    ax.axvline(cfg.resonance_center, color="crimson", ls=":", alpha=0.7, lw=1.2)
    ax.axvline(cfg.resonance_center - cfg.delta_n, color="gray", ls=":", alpha=0.5)
    ax.axvline(cfg.resonance_center + cfg.delta_n, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("Normalized observable x")
    ax.set_ylabel("Events per bin")
    ax.set_title(
        f"RGC mock: 1/7 + Δn ladder "
        f"(f_sig={cfg.sig_frac:.0%}, σ={cfg.sigma}, N={cfg.num_events:,})"
    )
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _write_stats_txt(path: Path, cfg: MockCatalogConfig, meta: dict[str, Any]) -> None:
    lines = [
        "RGC mock catalog run",
        f"timestamp_utc: {_utc_stamp()}",
        f"seed: {cfg.seed}",
        f"num_events: {cfg.num_events}",
        f"x_range: [{cfg.x_min}, {cfg.x_max}]",
        f"resonance_center (1/7): {cfg.resonance_center:.12f}",
        f"delta_n (1/14): {cfg.delta_n:.12f}",
        f"sigma: {cfg.sigma}",
        f"sig_frac: {cfg.sig_frac}",
        f"num_delta_steps: {cfg.num_delta_steps}",
        f"step_amplitude_decay: {cfg.step_amplitude_decay}",
        "",
        "Truth ladder:",
    ]
    for k, mu, w, n in zip(
        meta["ks"], meta["mu_truth"], meta["weights_truth"], meta["n_per_component"]
    ):
        lines.append(f"  k={k:+3d}  mu={mu:.8f}  w={w:.5f}  N={n}")
    lines.extend(
        [
            "",
            f"n_background: {meta['n_background']}",
            f"n_signal: {meta['n_signal']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_catalog(df: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    """Console-friendly summary stats."""
    sig = df[df["type"] == "signal"]
    per_k = (
        sig.groupby("component_k")["x"]
        .agg(["count", "mean"])
        .reset_index()
        .sort_values("component_k")
    )
    return {
        "describe_x": df["x"].describe().to_dict(),
        "signal_by_k": per_k.to_dict(orient="records"),
        "meta": meta,
    }


def run_mock_catalog_pipeline(
    cfg: MockCatalogConfig | None = None,
    *,
    plot: bool = True,
    show_popup: bool = False,
) -> dict[str, Any]:
    """Generate catalog, write CSV/PNG/stats, return artifact paths."""
    cfg = cfg or MockCatalogConfig()
    RGC_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _utc_stamp()
    prefix = cfg.output_prefix

    df, ks, mus, probs, meta = generate_mock_catalog(cfg)
    summary = summarize_catalog(df, meta)

    csv_path = RGC_ARTIFACTS_DIR / f"{prefix}.csv"
    png_path = RGC_ARTIFACTS_DIR / f"{prefix}.png"
    from tav_shared.artifact_paths import TestSlug, artifact_path, compose_dataset_slug

    stats_path = artifact_path(TestSlug.RGC, compose_dataset_slug(prefix), "run_stats", "txt")
    json_path = artifact_path(TestSlug.RGC, compose_dataset_slug(prefix), "summary", "json")

    df.to_csv(csv_path, index=False)
    _write_stats_txt(stats_path, cfg, meta)

    if plot:
        plot_catalog(
            df,
            ks,
            mus,
            probs,
            meta["n_per_component"],
            cfg=cfg,
            output_path=png_path,
        )
        if show_popup:
            img = plt.imread(png_path)
            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            plt.axis("off")
            plt.title(png_path.name)
            plt.show()

    from menus.astronomical.desi.json_util import write_json

    report = {
        "module": "rgc_mock_catalog",
        "config": {
            "seed": cfg.seed,
            "num_events": cfg.num_events,
            "sig_frac": cfg.sig_frac,
            "sigma": cfg.sigma,
            "resonance_center": cfg.resonance_center,
            "delta_n": cfg.delta_n,
            "num_delta_steps": cfg.num_delta_steps,
            "step_amplitude_decay": cfg.step_amplitude_decay,
        },
        "summary": summary,
        "artifacts": {
            "csv": str(csv_path),
            "png": str(png_path) if plot else None,
            "stats_txt": str(stats_path),
        },
        "timestamp": stamp,
    }
    write_json(json_path, report, indent=2, sort_keys=True)
    report["summary_path"] = str(json_path)

    print(f"\n[RGC mock] Generated {len(df):,} events "
          f"({meta['n_signal']:,} signal / {meta['n_background']:,} background)")
    print("[RGC mock] Truth ladder (signal allocation):")
    for k, mu, n in zip(meta["ks"], meta["mu_truth"], meta["n_per_component"]):
        mark = " ← primary 1/7" if k == 0 else ""
        print(f"  k={k:+3d}  μ={mu:.5f}  N={n:5d}{mark}")
    print(f"[RGC mock] CSV:   {csv_path}")
    if plot:
        print(f"[RGC mock] Plot:  {png_path}")
    print(f"[RGC mock] Stats: {stats_path}")
    print(f"[RGC mock] JSON:  {json_path}")
    return report


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RGC 1/7 + Δn mock catalog generator")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--num-events", type=int, default=DEFAULT_NUM_EVENTS)
    p.add_argument("--sig-frac", type=float, default=DEFAULT_SIG_FRAC)
    p.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    p.add_argument("--num-delta-steps", type=int, default=DEFAULT_NUM_DELTA_STEPS)
    p.add_argument("--decay", type=float, default=STEP_AMPLITUDE_DECAY)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--quick", action="store_true", help="10k events for smoke test")
    p.add_argument("--prefix", default="mock_resonance_catalog_1over7_deltan")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    cfg = MockCatalogConfig(
        seed=args.seed,
        num_events=10_000 if args.quick else args.num_events,
        sig_frac=args.sig_frac,
        sigma=args.sigma,
        num_delta_steps=args.num_delta_steps,
        step_amplitude_decay=args.decay,
        output_prefix=args.prefix,
    )
    run_mock_catalog_pipeline(cfg, plot=not args.no_plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())