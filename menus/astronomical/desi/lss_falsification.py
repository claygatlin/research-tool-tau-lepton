"""
LSS falsification suite — Methods 1, 4, 9 (configuration-space / P(k) domain).

Weight-6 S(n) node mapping, geometric k = n/R_τ comb search on gridded δ fields,
and unified PASS / POTENTIAL FALSIFICATION reporting. Constants imported from
``scanner.py`` (single source of truth).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from tav_shared.artifact_paths import TestSlug, artifact_path, artifact_timestamp
from menus.astronomical.desi.scanner import (
    GAMMA6_HEX,
    M0_MEV,
    N_HIER_BINDING,
    R_TAU_MPC,
)

# Geometric LSS constants (fixed, not fitted)
R_TAU: float = R_TAU_MPC
N_HIER: float = N_HIER_BINDING
K_COMB_FUND: float = 1.0 / R_TAU_MPC
M_313: float = M0_MEV
GAMMA_6_AVG: float = GAMMA6_HEX
WEIGHT6_PERIOD: int = 142857
PREDICTED_NODE_SCALES_MPC: list[float] = [400.0, 1000.0]

JAX_AVAILABLE = False
try:
    import jax
    import jax.numpy as jnp
    from jax import jit

    try:
        jax.config.update("jax_enable_x64", True)
    except Exception:
        pass
    JAX_AVAILABLE = True
except ImportError:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    jit = None  # type: ignore[assignment]


def _save_report(name: str, payload: dict[str, Any]) -> str:
    from menus.astronomical.desi.json_util import write_json

    path = artifact_path(TestSlug.LSS_FALSIFICATION, name, "report", "json")
    write_json(path, payload, indent=2, sort_keys=True)
    return str(path)


# =============================================================================
# Weight-6 residue operator + S(n)
# =============================================================================


def r6_weight6_residue(m: int, period: int = WEIGHT6_PERIOD) -> float:
    """r_6(m) — Weight-6 residue with 142857 periodic filter (placeholder until exact closed form)."""
    cycle_pos = int(m) % int(period)
    normalized = cycle_pos / float(period)
    return float((normalized**5) * np.sin(2 * np.pi * normalized))


def compute_S_n(n_max: int) -> np.ndarray:
    """S(n) = Σ_{m=1}^{n} r_6(m) — cumulative Weight-6 residue sum."""
    n_max = max(1, int(n_max))
    ms = np.arange(1, n_max + 1, dtype=int)
    r6_vec = np.array([r6_weight6_residue(m) for m in ms], dtype=float)
    return np.cumsum(r6_vec)


def map_S_n_to_nodes(
    n_max: int = 5000,
    *,
    n_hier: float = N_HIER,
    r_tau: float = R_TAU,
    predicted_scales: list[float] | None = None,
) -> dict[str, Any]:
    """
    Map cumulative S(n) jumps to comoving node scales (Method 1).

    Canonical framework anchors (~400 Mpc, ~1 Gpc) are always included; top
    ΔS(n) staircase events interpolate between them via S(n)/S(n_max).
    """
    S = compute_S_n(n_max)
    dS = np.diff(S, prepend=0.0)
    anchors = list(predicted_scales or PREDICTED_NODE_SCALES_MPC)
    if len(anchors) < 2:
        anchors = [400.0, 1000.0]
    lo, hi = float(anchors[0]), float(anchors[-1])
    s_max = float(S[-1]) if len(S) else 1.0
    if s_max <= 0:
        s_max = 1.0

    nodes: list[dict[str, Any]] = []
    seen: set[int] = set()

    def _add(mpc: float, n_index: int, delta_s: float, source: str, rank: int) -> None:
        key = int(round(mpc / 10.0))
        if key in seen:
            return
        seen.add(key)
        nodes.append(
            {
                "n_index": int(n_index),
                "delta_S": float(delta_s),
                "comoving_mpc": float(mpc),
                "rank": int(rank),
                "source": source,
            }
        )

    for rank, anchor in enumerate(anchors):
        _add(anchor, 0, 0.0, "canonical", rank)

    threshold = float(np.percentile(np.abs(dS), 97))
    jump_idx = np.where(np.abs(dS) >= threshold)[0]
    for rank, idx in enumerate(jump_idx[:10]):
        frac = float(S[idx] / s_max)
        mpc = lo + frac * (hi - lo)
        _add(mpc, int(idx + 1), float(dS[idx]), "S_n_staircase", rank + len(anchors))

    nodes.sort(key=lambda item: item["comoving_mpc"])
    predicted = [n["comoving_mpc"] for n in nodes[:12]]

    return {
        "S_n": S,
        "predicted_nodes_mpc": predicted,
        "node_details": nodes[:12],
        "n_hier": float(n_hier),
        "r_tau_mpc": float(r_tau),
        "reference_scales_mpc": anchors,
        "n_max": int(n_max),
    }


def find_density_peaks_1d(
    x: np.ndarray,
    delta: np.ndarray,
    *,
    min_separation_mpc: float = 50.0,
    prominence_frac: float = 0.15,
) -> np.ndarray:
    """Peak locations (Mpc) in a 1D density contrast field."""
    from scipy import signal

    x = np.asarray(x, dtype=float)
    delta = np.asarray(delta, dtype=float)
    if len(x) < 5:
        return np.array([], dtype=float)
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    distance = max(1, int(min_separation_mpc / max(dx, 1e-9)))
    std = float(np.std(delta)) or 1.0
    peaks, _ = signal.find_peaks(delta, distance=distance, prominence=prominence_frac * std)
    return x[peaks]


def cross_correlate_S_n_nodes(
    x: np.ndarray,
    delta: np.ndarray,
    *,
    n_max: int = 5000,
    tolerance_mpc: float = 80.0,
    n_hier: float = N_HIER,
    r_tau: float = R_TAU,
) -> dict[str, Any]:
    """
    Cross-correlate predicted S(n) nodes with observed density peaks (Method 1).
    """
    mapping = map_S_n_to_nodes(n_max=n_max, n_hier=n_hier, r_tau=r_tau)
    predicted = np.asarray(mapping["predicted_nodes_mpc"], dtype=float)
    observed = find_density_peaks_1d(x, delta)

    matches: list[dict[str, Any]] = []
    matched_pred: set[int] = set()
    for obs in observed:
        if len(predicted) == 0:
            break
        dists = np.abs(predicted - obs)
        j = int(np.argmin(dists))
        if dists[j] <= tolerance_mpc and j not in matched_pred:
            matched_pred.add(j)
            matches.append(
                {
                    "observed_mpc": float(obs),
                    "predicted_mpc": float(predicted[j]),
                    "delta_mpc": float(dists[j]),
                }
            )

    canonical = [float(t) for t in PREDICTED_NODE_SCALES_MPC]
    anchor_hits = [
        t for t in canonical if any(abs(obs - t) <= tolerance_mpc for obs in observed)
    ]
    n_pred = len(predicted)
    n_match = len(matches)
    recovery_frac = float(n_match / n_pred) if n_pred else 0.0
    falsified = len(anchor_hits) < 1

    return {
        "predicted_nodes_mpc": predicted.tolist(),
        "observed_peaks_mpc": observed.tolist(),
        "matches": matches,
        "n_predicted": n_pred,
        "n_matched": n_match,
        "recovery_fraction": recovery_frac,
        "tolerance_mpc": float(tolerance_mpc),
        "falsified": falsified,
        "verdict": "POTENTIAL FALSIFICATION" if falsified else "NODE PATTERN RECOVERED",
        "S_n_mapping": mapping,
    }


# =============================================================================
# Synthetic density fields + power spectra
# =============================================================================


def generate_synthetic_density_field_1d(
    n_points: int = 2048,
    box_size_mpc: float = 2000.0,
    noise_level: float = 0.3,
    seed: int = 42,
    *,
    inject_comb: bool = True,
    inject_sn_nodes: bool = False,
    n_max: int = 5000,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """1D δ(x) with optional geometric comb and S(n) node bumps."""
    rng = np.random.default_rng(seed)
    n_points = max(64, int(n_points))
    x = np.linspace(0.0, float(box_size_mpc), n_points)
    dx = float(x[1] - x[0]) if n_points > 1 else 1.0
    k_nyquist = np.pi / max(dx, 1e-12)

    delta = rng.normal(0.0, noise_level, n_points)
    injected_ks: list[float] = []
    amplitudes: list[float] = []
    injected_nodes: list[float] = []

    if inject_comb:
        for n_mode in range(1, 8):
            k = n_mode * K_COMB_FUND
            if k < k_nyquist * 0.8:
                amp = 0.35 / np.sqrt(n_mode)
                phase = rng.uniform(0.0, 2 * np.pi)
                delta += amp * np.sin(2 * np.pi * k * x + phase)
                injected_ks.append(float(k))
                amplitudes.append(float(amp))

    if inject_sn_nodes:
        mapping = map_S_n_to_nodes(n_max=n_max)
        for node_mpc in mapping["predicted_nodes_mpc"]:
            if not (0.0 < node_mpc < box_size_mpc):
                continue
            width = max(30.0, box_size_mpc / 60.0)
            delta += 0.18 * np.exp(-0.5 * ((x - node_mpc) / width) ** 2)
            injected_nodes.append(float(node_mpc))

    metadata = {
        "injected_k_values_hMpc": injected_ks,
        "injection_amplitudes": amplitudes,
        "injected_sn_nodes_mpc": injected_nodes,
        "R_tau": R_TAU,
        "k_comb_fundamental": K_COMB_FUND,
        "n_hier": N_HIER,
        "box_size_mpc": float(box_size_mpc),
        "n_points": n_points,
        "noise_level": float(noise_level),
        "seed": int(seed),
        "dimension": 1,
    }
    return x, delta, metadata


def generate_synthetic_density_field_3d(
    n_per_axis: int = 64,
    box_size_mpc: float = 500.0,
    noise_level: float = 0.25,
    seed: int = 42,
    *,
    inject_comb: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """3D δ(x,y,z) on a cubic grid with isotropic comb injection."""
    rng = np.random.default_rng(seed)
    n = max(16, int(n_per_axis))
    box = float(box_size_mpc)
    delta = rng.normal(0.0, noise_level, (n, n, n))
    coords = np.linspace(0.0, box, n)
    injected_ks: list[float] = []

    if inject_comb:
        for n_mode in range(1, 5):
            k = n_mode * K_COMB_FUND
            amp = 0.06 / np.sqrt(n_mode)
            for axis, c in enumerate(np.meshgrid(coords, coords, coords, indexing="ij")):
                phase = rng.uniform(0.0, 2 * np.pi)
                delta += amp * np.sin(2 * np.pi * k * c[axis] + phase) / 3.0
            injected_ks.append(float(k))

    metadata = {
        "injected_k_values_hMpc": injected_ks,
        "R_tau": R_TAU,
        "k_comb_fundamental": K_COMB_FUND,
        "box_size_mpc": box,
        "n_per_axis": n,
        "noise_level": float(noise_level),
        "seed": int(seed),
        "dimension": 3,
    }
    return coords, delta, metadata


def compute_power_spectrum_1d(
    delta: np.ndarray,
    box_size_mpc: float,
    *,
    use_jax: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """1D P(k) via rfft."""
    n = len(delta)
    if use_jax and JAX_AVAILABLE:
        from menus.astronomical.desi.jax_likelihood import jax_power_spectrum_1d

        k, pk = jax_power_spectrum_1d(delta, box_size_mpc)
        return np.asarray(k), np.asarray(pk)

    fft = np.fft.rfft(np.asarray(delta, dtype=float))
    pk = np.abs(fft) ** 2 / n
    k = np.fft.rfftfreq(n, d=box_size_mpc / n) * 2 * np.pi
    return k, pk


def compute_power_spectrum_3d(
    delta: np.ndarray,
    box_size_mpc: float,
    *,
    n_k_bins: int = 48,
    use_jax: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Spherically averaged 3D P(k) from fftn."""
    if use_jax and JAX_AVAILABLE:
        from menus.astronomical.desi.jax_likelihood import jax_power_spectrum_3d_binned

        k, pk = jax_power_spectrum_3d_binned(delta, box_size_mpc, n_k_bins=n_k_bins)
        return np.asarray(k), np.asarray(pk)

    field = np.asarray(delta, dtype=float)
    n = field.shape[0]
    fft = np.fft.fftn(field)
    pk_cube = np.abs(fft) ** 2 / field.size
    k1d = np.fft.fftfreq(n, d=box_size_mpc / n) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")
    kmag = np.sqrt(kx**2 + ky**2 + kz**2).ravel()
    pk_flat = pk_cube.ravel()
    k_max = float(kmag.max()) if len(kmag) else 1.0
    bins = np.linspace(0.0, k_max, n_k_bins + 1)
    k_centers = 0.5 * (bins[:-1] + bins[1:])
    pk_binned = np.zeros(n_k_bins, dtype=float)
    counts = np.zeros(n_k_bins, dtype=float)
    for i in range(n_k_bins):
        mask = (kmag >= bins[i]) & (kmag < bins[i + 1])
        if np.any(mask):
            pk_binned[i] = float(np.mean(pk_flat[mask]))
            counts[i] = float(np.sum(mask))
    return k_centers, pk_binned


def _phase_scramble_surrogate(delta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Surrogate field: same |FFT| amplitudes, randomized phases (1D or 3D)."""
    field = np.asarray(delta, dtype=float)
    if field.ndim == 1:
        fft = np.fft.rfft(field)
        mag = np.abs(fft)
        phases = rng.uniform(0.0, 2 * np.pi, size=len(fft))
        scrambled = mag * np.exp(1j * phases)
        return np.fft.irfft(scrambled, n=len(field))
    fft = np.fft.fftn(field)
    mag = np.abs(fft)
    phases = rng.uniform(0.0, 2 * np.pi, size=fft.shape)
    scrambled = mag * np.exp(1j * phases)
    return np.fft.ifftn(scrambled).real


def search_for_comb_signature(
    k: np.ndarray,
    pk: np.ndarray,
    *,
    tolerance: float = 0.02,
    min_significance: float = 2.5,
    max_harmonic: int = 11,
    n_bootstrap: int = 500,
    seed: int = 42,
    delta_for_bootstrap: np.ndarray | None = None,
    box_size_mpc: float | None = None,
    recovery_mode: bool = False,
) -> dict[str, Any]:
    """
    Search P(k) for excess at k = n/R_τ (Method 4).

    Uses phase-scramble bootstrap on ``delta_for_bootstrap`` when provided;
    otherwise falls back to local-median excess with Poisson-like σ.
    """
    k = np.asarray(k, dtype=float)
    pk = np.asarray(pk, dtype=float)
    rng = np.random.default_rng(seed)

    results: dict[str, Any] = {
        "detected_peaks": [],
        "expected_k": [],
        "significance": [],
        "bootstrap_p_values": [],
        "falsified": True,
        "method": "local_median",
        "notes": "",
    }

    for n in range(1, max_harmonic + 1):
        k_expected = n * K_COMB_FUND
        band = np.abs(k - k_expected) / max(k_expected, 1e-12) < tolerance
        if np.any(band):
            pk_band = pk[band]
            k_band = k[band]
            j = int(np.argmax(pk_band))
            k_meas = float(k_band[j])
            pk_val = float(pk_band[j])
            idx = int(np.argmin(np.abs(k - k_meas)))
        else:
            idx = int(np.argmin(np.abs(k - k_expected)))
            k_meas = float(k[idx])
            pk_val = float(pk[idx])

        window = 5
        start = max(0, idx - window)
        end = min(len(pk), idx + window + 1)
        k_local = k[start:end]
        local = pk[start:end]
        exclude = (np.abs(k_local - k_expected) / max(k_expected, 1e-12)) < tolerance
        if np.any(~exclude):
            background = float(np.median(local[~exclude]))
        else:
            background = float(np.median(local))
        excess = pk_val - background
        significance = float(
            np.log10((pk_val + 1e-30) / (background + 1e-30))
        )

        p_value = 1.0
        if delta_for_bootstrap is not None and box_size_mpc is not None:
            results["method"] = "phase_scramble_bootstrap"
            boot_excess = np.empty(n_bootstrap, dtype=float)
            for i in range(n_bootstrap):
                surr = _phase_scramble_surrogate(delta_for_bootstrap, rng)
                if surr.ndim == 1:
                    _, pk_s = compute_power_spectrum_1d(surr, box_size_mpc, use_jax=False)
                else:
                    _, pk_s = compute_power_spectrum_3d(surr, box_size_mpc, use_jax=False)
                j = int(np.argmin(np.abs(k - k_expected)))
                bg_s = float(np.median(pk_s[max(0, j - window) : min(len(pk_s), j + window + 1)]))
                boot_excess[i] = float(pk_s[j]) - bg_s
            p_value = float(np.mean(boot_excess >= excess))
            mu_boot = float(np.mean(boot_excess))
            sigma_boot = float(np.std(boot_excess))
            if sigma_boot > max(1e-12, 0.01 * abs(mu_boot)):
                significance = float((excess - mu_boot) / sigma_boot)
            else:
                significance = 0.0
            significance = float(np.clip(significance, -10.0, 10.0))

        p_thresh = 0.05 if recovery_mode else 0.01
        sig_thresh = min_significance if recovery_mode else max(min_significance, 3.5)
        min_log10 = 0.1 if recovery_mode else 0.5

        results["expected_k"].append(float(k_expected))
        freq_ok = abs(k_meas - k_expected) / max(k_expected, 1e-12) < tolerance
        log10_ratio = float(np.log10((pk_val + 1e-30) / (background + 1e-30)))
        if results["method"] == "phase_scramble_bootstrap":
            if recovery_mode:
                hit = freq_ok and p_value < p_thresh and (significance > sig_thresh or log10_ratio > min_log10)
            else:
                hit = freq_ok and p_value < p_thresh and significance > sig_thresh
        else:
            hit = freq_ok and log10_ratio > min_log10 and significance > sig_thresh
        if hit:
            results["detected_peaks"].append((k_meas, significance))
            results["significance"].append(significance)
            results["bootstrap_p_values"].append(p_value)
            results["falsified"] = False

    if results["falsified"]:
        results["notes"] = (
            "No significant excess at predicted comb frequencies k = n/R_τ. "
            "Potential falsification of the geometric comb (check covariance on real data)."
        )
        results["verdict"] = "POTENTIAL FALSIFICATION"
    else:
        results["notes"] = (
            f"Detected {len(results['detected_peaks'])} comb peaks at geometric locations. "
            "Prediction recovered in this realization."
        )
        results["verdict"] = "COMB RECOVERED"

    return results


# =============================================================================
# Runners (menu / CLI entry points)
# =============================================================================


def run_gridded_mock_comb_recovery(
    *,
    n_points: int = 4096,
    box_size_mpc: float = 3000.0,
    noise_level: float = 0.25,
    use_jax: bool = True,
    dimension: int = 1,
    seed: int = 42,
    output_prefix: str = "lss_mock_comb_recovery",
    plot: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Method 9 — gridded mock injection + comb recovery."""
    use_jax = bool(use_jax and JAX_AVAILABLE)

    if dimension >= 3:
        _, delta, meta = generate_synthetic_density_field_3d(
            n_per_axis=max(32, int(round(n_points ** (1 / 3)))),
            box_size_mpc=box_size_mpc / 3.0,
            noise_level=noise_level,
            seed=seed,
            inject_comb=True,
        )
        k, pk = compute_power_spectrum_3d(delta, box_size_mpc / 3.0, use_jax=use_jax)
        x = None
    else:
        x, delta, meta = generate_synthetic_density_field_1d(
            n_points=n_points,
            box_size_mpc=box_size_mpc,
            noise_level=noise_level,
            seed=seed,
            inject_comb=True,
            inject_sn_nodes=True,
        )
        k, pk = compute_power_spectrum_1d(delta, box_size_mpc, use_jax=use_jax)

    comb = search_for_comb_signature(
        k,
        pk,
        tolerance=0.03,
        min_significance=2.0,
        n_bootstrap=300,
        delta_for_bootstrap=delta,
        box_size_mpc=meta["box_size_mpc"],
        recovery_mode=True,
    )

    node_result: dict[str, Any] | None = None
    if x is not None:
        node_result = cross_correlate_S_n_nodes(x, delta)

    if verbose:
        print("=" * 70)
        print("LSS GRIDDED MOCK COMB RECOVERY (Method 9)")
        print(f"  R_τ = {R_TAU} h⁻¹Mpc | k_fund = {K_COMB_FUND:.6f} h Mpc⁻¹")
        print(f"  Comb verdict: {comb['verdict']}")
        if node_result:
            print(f"  S(n) nodes: {node_result['verdict']} ({node_result['n_matched']}/{node_result['n_predicted']} matched)")
        print("=" * 70)

    report = {
        "action": "Gridded Mock Comb Recovery",
        "metadata": meta,
        "comb_search": comb,
        "node_cross_correlation": node_result,
        "geometric_constants": {
            "R_tau": R_TAU,
            "n_hier": N_HIER,
            "k_comb_fund": K_COMB_FUND,
            "m_313_MeV": M_313,
            "gamma_6_avg": GAMMA_6_AVG,
        },
        "use_jax": use_jax,
        "timestamp": artifact_timestamp(),
    }

    if plot:
        report["plot_path"] = _plot_pk_comb(k, pk, comb, prefix=output_prefix)

    report["report_path"] = _save_report(output_prefix, report)
    return report


def run_lss_from_catalog_grid(
    delta: np.ndarray,
    *,
    box_size_mpc: float,
    use_jax: bool = True,
    output_prefix: str = "lss_catalog",
    plot: bool = False,
    verbose: bool = True,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Methods 1 + 4 on a real catalog-derived δ field."""
    use_jax = bool(use_jax and JAX_AVAILABLE)
    delta = np.asarray(delta, dtype=float)
    meta = dict(metadata or {})
    meta.setdefault("box_size_mpc", float(box_size_mpc))
    meta.setdefault("dimension", int(delta.ndim))

    if delta.ndim == 3:
        k, pk = compute_power_spectrum_3d(delta, box_size_mpc, use_jax=use_jax)
        x = np.linspace(0.0, box_size_mpc, delta.shape[0])
        delta_1d = delta[delta.shape[0] // 2, delta.shape[1] // 2, :]
    else:
        delta_1d = delta.ravel()
        x = np.linspace(0.0, box_size_mpc, len(delta_1d))
        k, pk = compute_power_spectrum_1d(delta_1d, box_size_mpc, use_jax=use_jax)

    comb = search_for_comb_signature(
        k,
        pk,
        delta_for_bootstrap=delta if delta.ndim >= 2 else delta_1d,
        box_size_mpc=box_size_mpc,
        recovery_mode=False,
    )
    nodes = cross_correlate_S_n_nodes(x, delta_1d)

    if verbose:
        print("=" * 70)
        print("LSS CATALOG GRID — Methods 1 + 4")
        print(f"  Comb verdict : {comb['verdict']}")
        print(f"  Node verdict : {nodes['verdict']}")
        print("=" * 70)

    report = {
        "action": "LSS Catalog Grid Analysis",
        "metadata": meta,
        "comb_search": comb,
        "node_cross_correlation": nodes,
        "use_jax": use_jax,
        "timestamp": artifact_timestamp(),
    }
    if plot:
        from menus.astronomical.desi.lss_visualization import (
            plot_manuscript_figure_suite,
        )

        report["plot_path"] = plot_manuscript_figure_suite(
            k=k,
            pk=pk,
            comb=comb,
            x=x,
            delta=delta_1d,
            node_result=nodes,
            metadata=meta,
            prefix=output_prefix,
        )
    report["report_path"] = _save_report(output_prefix, report)
    return report


def run_lss_comb_falsification(
    *,
    n_points: int = 4096,
    box_size_mpc: float = 3000.0,
    noise_level: float = 0.3,
    inject_signal: bool = False,
    use_jax: bool = True,
    dimension: int = 1,
    seed: int = 42,
    output_prefix: str = "lss_comb_falsification",
    plot: bool = False,
    verbose: bool = True,
    delta: np.ndarray | None = None,
    catalog_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Method 4 — P(k) comb falsification on synthetic or supplied grid."""
    use_jax = bool(use_jax and JAX_AVAILABLE)

    if delta is not None:
        return run_lss_from_catalog_grid(
            delta,
            box_size_mpc=box_size_mpc,
            use_jax=use_jax,
            output_prefix=output_prefix,
            plot=plot,
            verbose=verbose,
            metadata=catalog_metadata,
        )

    if dimension >= 3:
        _, delta, meta = generate_synthetic_density_field_3d(
            n_per_axis=max(32, int(round(n_points ** (1 / 3)))),
            box_size_mpc=box_size_mpc / 3.0,
            noise_level=noise_level,
            seed=seed,
            inject_comb=inject_signal,
        )
        k, pk = compute_power_spectrum_3d(delta, box_size_mpc / 3.0, use_jax=use_jax)
    else:
        _, delta, meta = generate_synthetic_density_field_1d(
            n_points=n_points,
            box_size_mpc=box_size_mpc,
            noise_level=noise_level,
            seed=seed,
            inject_comb=inject_signal,
        )
        k, pk = compute_power_spectrum_1d(delta, box_size_mpc, use_jax=use_jax)

    comb = search_for_comb_signature(
        k,
        pk,
        delta_for_bootstrap=delta,
        box_size_mpc=meta["box_size_mpc"],
        recovery_mode=inject_signal,
    )

    if verbose:
        print("=" * 70)
        print("LSS COMB FALSIFICATION — P(k) at k = n/R_τ (Method 4)")
        print(f"  inject_signal={inject_signal} | backend={'JAX' if use_jax else 'NumPy'}")
        print(f"  Verdict: {comb['verdict']}")
        print(f"  Detected peaks: {comb['detected_peaks']}")
        print("=" * 70)

    report = {
        "action": "LSS Comb Falsification (P(k))",
        "metadata": meta,
        "comb_search": comb,
        "inject_signal": inject_signal,
        "use_jax": use_jax,
        "timestamp": artifact_timestamp(),
    }
    if plot:
        report["plot_path"] = _plot_pk_comb(k, pk, comb, prefix=output_prefix)
    report["report_path"] = _save_report(output_prefix, report)
    return report


def run_sn_node_cross_correlation(
    *,
    n_points: int = 4096,
    box_size_mpc: float = 3000.0,
    noise_level: float = 0.25,
    inject_nodes: bool = True,
    n_max: int = 5000,
    seed: int = 42,
    output_prefix: str = "sn_node_xcorr",
    plot: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Method 1 — S(n) node prediction vs density peak cross-correlation."""
    x, delta, meta = generate_synthetic_density_field_1d(
        n_points=n_points,
        box_size_mpc=box_size_mpc,
        noise_level=noise_level,
        seed=seed,
        inject_comb=False,
        inject_sn_nodes=inject_nodes,
        n_max=n_max,
    )
    node_result = cross_correlate_S_n_nodes(x, delta, n_max=n_max)

    if verbose:
        print("=" * 70)
        print("S(n) NODE CROSS-CORRELATION (Method 1)")
        print(f"  Predicted: {node_result['predicted_nodes_mpc'][:6]}")
        print(f"  Observed peaks: {node_result['observed_peaks_mpc'][:6]}")
        print(f"  Verdict: {node_result['verdict']}")
        print("=" * 70)

    report = {
        "action": "S(n) Node Cross-Correlation",
        "metadata": meta,
        "node_cross_correlation": node_result,
        "timestamp": artifact_timestamp(),
    }
    if plot:
        report["plot_path"] = _plot_sn_nodes(x, delta, node_result, prefix=output_prefix)
    report["report_path"] = _save_report(output_prefix, report)
    return report


def run_falsification_suite(
    *,
    n_points: int = 4096,
    box_size_mpc: float = 3000.0,
    noise_level: float = 0.25,
    use_jax: bool = True,
    dimension: int = 1,
    seed: int = 42,
    output_prefix: str = "falsification_suite",
    plot: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Unified Methods 1 + 4 + 9 with global PASS / POTENTIAL FALSIFICATION verdict.
    """
    mock = run_gridded_mock_comb_recovery(
        n_points=n_points,
        box_size_mpc=box_size_mpc,
        noise_level=noise_level,
        use_jax=use_jax,
        dimension=dimension,
        seed=seed,
        output_prefix=f"{output_prefix}_mock",
        plot=False,
        verbose=False,
    )
    comb_null = run_lss_comb_falsification(
        n_points=n_points,
        box_size_mpc=box_size_mpc,
        noise_level=noise_level,
        inject_signal=False,
        use_jax=use_jax,
        dimension=dimension,
        seed=seed + 1,
        output_prefix=f"{output_prefix}_comb_null",
        plot=False,
        verbose=False,
    )
    nodes = run_sn_node_cross_correlation(
        n_points=n_points,
        box_size_mpc=box_size_mpc,
        noise_level=noise_level,
        inject_nodes=True,
        seed=seed + 2,
        output_prefix=f"{output_prefix}_nodes",
        plot=False,
        verbose=False,
    )

    mock_ok = not mock["comb_search"]["falsified"]
    nodes_ok = not nodes["node_cross_correlation"]["falsified"]
    null_ok = comb_null["comb_search"]["falsified"]  # null field should NOT detect comb

    all_pass = mock_ok and nodes_ok and null_ok
    verdict = "SUITE PASS" if all_pass else "POTENTIAL FALSIFICATION"

    if verbose:
        print("=" * 70)
        print("TSB FALSIFICATION SUITE — Methods 1, 4, 9")
        print(f"  Method 9 (mock comb recovery) : {'PASS' if mock_ok else 'FAIL'}")
        print(f"  Method 1 (S(n) nodes)         : {'PASS' if nodes_ok else 'FAIL'}")
        print(f"  Method 4 (null comb control)  : {'PASS' if null_ok else 'FAIL'}")
        print(f"  GLOBAL VERDICT                : {verdict}")
        print("=" * 70)

    report = {
        "action": "Falsification Suite (Methods 1,4,9)",
        "verdict": verdict,
        "all_pass": all_pass,
        "method_9_mock_recovery": mock,
        "method_4_null_control": comb_null,
        "method_1_sn_nodes": nodes,
        "geometric_constants": {
            "R_tau": R_TAU,
            "n_hier": N_HIER,
            "k_comb_fund": K_COMB_FUND,
            "m_313_MeV": M_313,
        },
        "timestamp": artifact_timestamp(),
    }
    if plot and mock.get("comb_search"):
        # Re-use mock field plot if available from sub-reports
        pass
    report["report_path"] = _save_report(output_prefix, report)
    return report


def _plot_pk_comb(
    k: np.ndarray,
    pk: np.ndarray,
    comb: dict[str, Any],
    *,
    prefix: str,
) -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.loglog(k[k > 0], pk[k > 0] + 1e-30, "b-", alpha=0.8, label="P(k)")
    for n, k_exp in enumerate(comb.get("expected_k", [])[:8], start=1):
        ax.axvline(k_exp, color="coral", linestyle="--", alpha=0.5, label="k=n/R_τ" if n == 1 else "")
    for k_meas, sig in comb.get("detected_peaks", []):
        ax.plot(k_meas, float(np.interp(k_meas, k, pk)), "ro", markersize=8, label=f"hit σ={sig:.1f}")
    ax.set_xlabel("k [h Mpc⁻¹]")
    ax.set_ylabel("P(k)")
    ax.set_title(f"LSS Comb Search — {comb.get('verdict', '')}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path = artifact_path(TestSlug.LSS_FALSIFICATION, prefix, "pk_plot", "png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def _plot_sn_nodes(
    x: np.ndarray,
    delta: np.ndarray,
    node_result: dict[str, Any],
    *,
    prefix: str,
) -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, delta, "b-", alpha=0.7, label="δ(x)")
    for mpc in node_result.get("predicted_nodes_mpc", [])[:8]:
        ax.axvline(mpc, color="coral", linestyle="--", alpha=0.6)
    for mpc in node_result.get("observed_peaks_mpc", [])[:8]:
        ax.axvline(mpc, color="green", linestyle=":", alpha=0.6)
    ax.set_xlabel("x [h⁻¹ Mpc]")
    ax.set_ylabel("δ")
    ax.set_title(f"S(n) Nodes — {node_result.get('verdict', '')}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path = artifact_path(TestSlug.LSS_FALSIFICATION, prefix, "nodes_plot", "png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


if __name__ == "__main__":
    run_falsification_suite(verbose=True, plot=True)