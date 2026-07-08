"""
Production extensions for the Tau-SB DESI scanner.

1. emcee MCMC posteriors
2. healpy spherical-harmonic dipole fit
3. ruptures change-point detection
4. Injection/recovery tests
5. Pantheon+ / Planck joint fit
6. Batch tracer scanning
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import optimize

from tav_shared.batch_ledger import (
    DESI_BATCH_DONE,
    append_done_entries,
    format_batch_banner,
    select_batch_items,
)
from menus.astronomical.desi.scanner import (
    ARTIFACTS_DIR,
    CHANGPOINTS_MIN_N,
    DATASETS_DIR,
    LATE_UNIVERSE_DELTA_N,
    MCMC_STEPS_DEFAULT,
    MCMC_WALKERS_DEFAULT,
    N_HIER_BINDING,
    TauSBScanner,
    mcmc_burn_in_for_steps,
    gaussian_chi2,
    list_desi_tracers,
    run_desi_scan,
    s_from_z,
    tau_sb_hierarchical_step,
    tau_sb_oscillatory_residual,
    tracer_available_quantities,
)

PANTHEON_DIR = DATASETS_DIR / "pantheon_plus"
PANTHEON_DAT_URL = (
    "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/"
    "Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat"
)
PLANCK_RD_MPC: float = 147.09
PLANCK_RD_ERR_MPC: float = 0.26
# Early-universe friction floor imprint on sound horizon (fractional shift scale)
FRICTION_FLOOR_SHIFT_SCALE: float = 3.131e-3  # tied to 313.1 MeV anchor (×1e-5)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _percentiles(samples: np.ndarray, levels: tuple[float, ...] = (16, 50, 84)) -> dict[str, float]:
    percs = np.percentile(samples, [16, 50, 84])
    return {
        "median": float(percs[1]),
        "low": float(percs[0]),
        "high": float(percs[2]),
    }


# ---------------------------------------------------------------------------
# 1. emcee MCMC
# ---------------------------------------------------------------------------


def run_mcmc_tau_sb(
    scanner: TauSBScanner,
    z: np.ndarray,
    observable: np.ndarray,
    err: np.ndarray | None = None,
    cov: np.ndarray | None = None,
    *,
    n_walkers: int = MCMC_WALKERS_DEFAULT,
    n_steps: int = MCMC_STEPS_DEFAULT,
    burn_in: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """MCMC posteriors for Tau-SB oscillation amplitude, phase, and hierarchy."""
    if burn_in is None:
        burn_in = mcmc_burn_in_for_steps(n_steps)
    burn_in = min(burn_in, max(n_steps - 100, 100))
    try:
        import emcee
    except ImportError as exc:
        raise ImportError("Install emcee: pip install emcee") from exc

    z = np.asarray(z, dtype=float)
    obs = np.asarray(observable, dtype=float)
    from menus.astronomical.desi.scanner import _baseline_degree, _oscillation_scale

    gamma_used = scanner._gamma_for(z)
    s = s_from_z(z, gamma=gamma_used)
    amp_scale = _oscillation_scale(obs)
    deg = _baseline_degree(len(z), extra_params=3)
    n_params = deg + 4  # baseline + A_frac, phase, hier_frac

    def model(params: np.ndarray) -> np.ndarray:
        base = np.polyval(params[: deg + 1], z)
        osc = tau_sb_oscillatory_residual(
            s,
            A=params[deg + 1],
            period=scanner.period,
            phase=params[deg + 2],
            amplitude_scale=amp_scale,
        )
        hier = tau_sb_hierarchical_step(z, amplitude=params[deg + 3] * amp_scale * 0.01)
        return base + osc + hier

    def log_prior(params: np.ndarray) -> float:
        if not np.all(np.isfinite(params)):
            return -np.inf
        if abs(params[deg + 1]) > 0.25:
            return -np.inf
        if abs(params[deg + 3]) > 0.05:
            return -np.inf
        return 0.0

    def log_likelihood(params: np.ndarray) -> float:
        pred = model(params)
        chi2_val = gaussian_chi2(obs, pred, err=err, cov=cov)
        return -0.5 * chi2_val

    def log_probability(params: np.ndarray) -> float:
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(params)

    p0_center = np.concatenate(
        [
            np.polyfit(z, obs, deg),
            np.array([0.01, 0.0, 0.005]),
        ]
    )
    rng = np.random.default_rng(seed)
    initial = p0_center + 1e-3 * rng.standard_normal((n_walkers, n_params))

    sampler = emcee.EnsembleSampler(n_walkers, n_params, log_probability)
    sampler.run_mcmc(initial, n_steps, progress=False)
    chain = sampler.get_chain(discard=burn_in, flat=True)

    labels = [f"c{i}" for i in range(deg + 1)] + ["A_osc", "phase", "hier_amp"]
    posteriors = {label: _percentiles(chain[:, i]) for i, label in enumerate(labels)}

    result = {
        "n_walkers": n_walkers,
        "n_steps": n_steps,
        "burn_in": burn_in,
        "gamma_used": gamma_used,
        "labels": labels,
        "posteriors": posteriors,
        "acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "chain_path": None,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    chain_path = ARTIFACTS_DIR / f"tau_sb_mcmc_chain_{_utc_stamp()}.npz"
    np.savez_compressed(chain_path, chain=chain, labels=np.array(labels))
    result["chain_path"] = str(chain_path)
    scanner.results["mcmc"] = result
    return result


# ---------------------------------------------------------------------------
# 2. healpy dipole
# ---------------------------------------------------------------------------


def _synthetic_patch_catalog(
    n_patches: int = 48,
    dipole_amplitude: float = 0.012,
    noise_sigma: float = 0.004,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate patch sky positions and anisotropic BAO offset measurements."""
    rng = np.random.default_rng(seed)
    npix = n_patches
    # Fibonacci sphere for roughly uniform patches
    indices = np.arange(npix, dtype=float)
    phi = np.arccos(1 - 2 * (indices + 0.5) / npix)
    theta = np.pi * (1 + 5**0.5) * indices
    ra = np.degrees(theta) % 360.0
    dec = 90.0 - np.degrees(phi)
    # Dipole axis (l=1) in galactic-like direction
    axis_ra, axis_dec = 180.0, 30.0
    ax_theta, ax_phi = np.radians(90 - axis_dec), np.radians(axis_ra)
    th, ph = np.radians(90 - dec), np.radians(ra)
    cos_ang = (
        np.sin(th) * np.sin(ax_theta) * np.cos(ph - ax_phi) + np.cos(th) * np.cos(ax_theta)
    )
    signal = dipole_amplitude * cos_ang
    observed = signal + rng.normal(0, noise_sigma, npix)
    return ra, dec, observed


def scan_dipole_healpy(
    scanner: TauSBScanner,
    *,
    ra_deg: np.ndarray | None = None,
    dec_deg: np.ndarray | None = None,
    patch_values: np.ndarray | None = None,
    nside: int = 16,
    dipole_amplitude: float = 0.012,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Fit l=1 dipole on a healpix map built from patch measurements.

    Pass ``ra_deg``, ``dec_deg``, ``patch_values`` for real directional BAO data;
    otherwise a synthetic patch catalog is used for pipeline validation.
    """
    try:
        import healpy as hp
    except ImportError as exc:
        raise ImportError("Install healpy: pip install healpy") from exc

    if ra_deg is None or dec_deg is None or patch_values is None:
        ra_deg, dec_deg, patch_values = _synthetic_patch_catalog(
            n_patches=48, dipole_amplitude=dipole_amplitude, seed=seed
        )
        source = "synthetic_patches"
    else:
        source = "user_patches"

    ra_deg = np.asarray(ra_deg, dtype=float)
    dec_deg = np.asarray(dec_deg, dtype=float)
    patch_values = np.asarray(patch_values, dtype=float)

    npix = hp.nside2npix(nside)
    sky_map = np.zeros(npix, dtype=float)
    weight_map = np.zeros(npix, dtype=float)
    for ra, dec, val in zip(ra_deg, dec_deg, patch_values, strict=True):
        pix = hp.ang2pix(nside, ra, dec, lonlat=True)
        sky_map[pix] += val
        weight_map[pix] += 1.0
    mask = weight_map > 0
    sky_map[mask] /= weight_map[mask]

    alm = hp.map2alm(sky_map)
    # l=1 coefficients: a_{1,m} for m=-1,0,1
    l1_vals = hp.alm2cl(alm, lmax=1)
    # Dipole amplitude from l=1 power
    dipole_rms = float(np.sqrt(max(l1_vals[1], 0.0))) if len(l1_vals) > 1 else 0.0

    # Linear dipole fit in pixel domain for amplitude along reference axis
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    ra_pix = np.degrees(phi)
    dec_pix = 90.0 - np.degrees(theta)

    def dipole_on_pixels(amp: float, ra0: float, dec0: float) -> np.ndarray:
        th0, ph0 = np.radians(90 - dec0), np.radians(ra0)
        thp, php = np.radians(90 - dec_pix), np.radians(ra_pix)
        cos_a = np.sin(thp) * np.sin(th0) * np.cos(php - ph0) + np.cos(thp) * np.cos(th0)
        return amp * cos_a

    def chi2_dipole(x: np.ndarray) -> float:
        amp, ra0, dec0 = x
        model = dipole_on_pixels(amp, ra0, dec0)
        resid = sky_map - model
        return float(np.sum(resid[mask] ** 2) / max(mask.sum(), 1))

    res = optimize.minimize(
        chi2_dipole,
        x0=[dipole_amplitude, 180.0, 30.0],
        method="Nelder-Mead",
    )
    amp_fit, ra_axis, dec_axis = res.x

    result = {
        "method": "healpy_l1",
        "nside": nside,
        "source": source,
        "n_patches": int(len(patch_values)),
        "dipole_rms_l1": dipole_rms,
        "fitted_amplitude": float(amp_fit),
        "dipole_axis_ra_deg": float(ra_axis % 360),
        "dipole_axis_dec_deg": float(dec_axis),
        "chi2": float(res.fun),
        "note": "E8-staggered domain dipole test via healpy map2alm l=1.",
    }
    scanner.results["dipole"] = result
    return result


# ---------------------------------------------------------------------------
# 3. ruptures change-point detection
# ---------------------------------------------------------------------------


def detect_hierarchical_changepoints(
    scanner: TauSBScanner,
    z: np.ndarray,
    observable: np.ndarray,
    *,
    penalty: float = 3.0,
    min_size: int = 2,
    algorithm: str = "pelt",
) -> dict[str, Any]:
    """Detect binding-level transitions via harmonic_binding.detect_binding_transitions."""
    from menus.astronomical.desi.harmonic_binding import detect_binding_transitions

    order = np.argsort(z)
    z_sorted = np.asarray(z, dtype=float)[order]
    obs_sorted = np.asarray(observable, dtype=float)[order]
    n_pts = len(z_sorted)
    if n_pts < CHANGPOINTS_MIN_N:
        result = {
            "algorithm": algorithm,
            "penalty": penalty,
            "skipped": True,
            "skip_reason": (
                f"insufficient data (n={n_pts} < {CHANGPOINTS_MIN_N} required for ruptures)"
            ),
            "break_indices": [],
            "transition_zs": [],
            "detected_transition_zs": [],
            "n_hier_reference_zs": [
                0.3 * (N_HIER_BINDING / 45.8) ** 0.33,
                1.0,
                2.0 * (1 + LATE_UNIVERSE_DELTA_N),
            ],
            "n_breaks": 0,
        }
        scanner.results["changepoints"] = result
        return result

    residuals, _ = scanner.compute_residuals(z_sorted, obs_sorted)
    binding = detect_binding_transitions(
        z_sorted,
        obs_sorted,
        residuals,
        n_hier=scanner.n_hier,
        gamma=float(scanner.gamma),
        penalty=penalty,
        min_size=min_size,
        algorithm=algorithm,
        verbose=True,
    )

    result = {
        "algorithm": binding.get("algorithm", algorithm),
        "penalty": penalty,
        "break_indices": binding.get("break_indices", []),
        "transition_zs": binding.get("detected_transition_zs", []),
        "detected_transition_zs": binding.get("detected_transition_zs", []),
        "predicted_transition_zs": binding.get("predicted_transition_zs", []),
        "n_hier_reference_zs": binding.get("mapping", {}).get(
            "reference_transition_zs",
            [
                0.3 * (N_HIER_BINDING / 45.8) ** 0.33,
                1.0,
                2.0 * (1 + LATE_UNIVERSE_DELTA_N),
            ],
        ),
        "binding_alignment": binding.get("alignment", {}),
        "binding_level_mapping": binding.get("mapping"),
        "n_breaks": binding.get("n_breaks", len(binding.get("detected_transition_zs", []))),
    }
    scanner.results["changepoints"] = result
    return result


# ---------------------------------------------------------------------------
# 4. Injection / recovery
# ---------------------------------------------------------------------------

from menus.astronomical.desi.analysis import (
    DEFAULT_INJECTION_AMPLITUDES,
    DEFAULT_INJECTION_TRIALS,
    run_injection_recovery_suite as run_s_space_injection_recovery,
)

DEFAULT_TRIALS_PER_AMPLITUDE: int = DEFAULT_INJECTION_TRIALS


def run_injection_recovery_suite(
    scanner: TauSBScanner,
    *,
    z_ref: np.ndarray | None = None,
    obs_ref: np.ndarray | None = None,
    err_ref: np.ndarray | None = None,
    cov_ref: np.ndarray | None = None,
    amplitudes: tuple[float, ...] = DEFAULT_INJECTION_AMPLITUDES,
    n_trials_per_amplitude: int = DEFAULT_TRIALS_PER_AMPLITUDE,
    noise_level: float | None = None,
    live_A_osc_frac: float | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Inject known τ oscillations at multiple amplitudes; map recovery bias vs SNR.

    Delegates to :func:`run_injection_recovery_suite` (s-space χ², A_frac+φ fit)
    on the live z/s grid and covariance when provided.
    """
    z_base = np.asarray(
        z_ref if z_ref is not None else [0.3, 0.5, 0.7, 0.9, 1.3, 1.5, 2.3],
        dtype=float,
    )
    obs_arr = np.asarray(obs_ref, dtype=float) if obs_ref is not None else None
    err_arr = np.asarray(err_ref, dtype=float) if err_ref is not None else None
    cov_arr = np.asarray(cov_ref, dtype=float) if cov_ref is not None else None
    baseline = float(np.median(obs_arr)) if obs_arr is not None else 1.0
    y_raw = obs_arr if obs_arr is not None else np.full(len(z_base), baseline)
    deg = min(3, max(1, len(z_base) - 4))
    y_true = np.polyval(np.polyfit(z_base, y_raw, deg=deg), z_base)

    if cov_arr is None:
        if err_arr is not None:
            cov_arr = np.diag(err_arr**2)
        else:
            sigma = float(noise_level if noise_level is not None else 0.01)
            cov_arr = np.diag(np.full(len(z_base), sigma**2))

    s_base = s_from_z(z_base, gamma=scanner.gamma)
    from menus.astronomical.desi.scanner import _oscillation_scale

    amp_scale = _oscillation_scale(y_true)

    def model_func(_s_arr: np.ndarray, a_osc: float, gamma: float) -> np.ndarray:
        ss = s_from_z(z_base, gamma=gamma)
        return y_true + a_osc * np.sin(2 * np.pi * ss / scanner.period)

    sweep = run_s_space_injection_recovery(
        s_base,
        y_true,
        cov_arr,
        model_func,
        z=z_base,
        err=err_arr,
        amplitudes=amplitudes,
        n_injections=n_trials_per_amplitude,
        noise_scale=noise_level,
        period=scanner.period,
        amplitude_scale=amp_scale,
        gamma_guess=float(scanner.gamma),
        seed=seed,
        verbose=True,
    )

    inflations = [
        row["inflation_factor"]
        for row in sweep
        if np.isfinite(row["inflation_factor"]) and row["inflation_factor"] > 0
    ]
    median_inflation = float(np.median(inflations)) if inflations else None
    debiased = None
    bias_note = None
    if live_A_osc_frac is not None and median_inflation and median_inflation > 0:
        debiased = float(live_A_osc_frac / median_inflation)
        bias_note = (
            f"Live A_osc_frac={live_A_osc_frac:.4f} may be inflated ~{median_inflation:.2f}×; "
            f"debiased estimate≈{debiased:.4f} (n={len(z_base)} mock trials, use with caution)."
        )

    # Legacy single-amplitude summary (first sweep point) for downstream readers.
    primary = sweep[0] if sweep else {}
    result = {
        "amplitude_sweep": sweep,
        "amplitudes": list(amplitudes),
        "n_trials_per_amplitude": n_trials_per_amplitude,
        "n_z": len(z_base),
        "used_live_z_grid": z_ref is not None,
        "mock_baseline": baseline,
        "method": "injection_recovery_suite_A_frac_phase_chi2",
        "median_inflation_factor": median_inflation,
        "live_A_osc_frac": live_A_osc_frac,
        "debiased_A_osc_frac": debiased,
        "interpretation_note": bias_note,
        "n_trials": primary.get("n_trials", 0),
        "A_true": primary.get("A_true_frac"),
        "recovery_fraction": primary.get("recovery_fraction"),
        "detection_fraction": primary.get("detection_fraction"),
        "A_recovered_mean": primary.get("A_recovered_mean"),
        "A_recovered_std": primary.get("A_recovered_std"),
        "bias": primary.get("bias"),
        "rmse": primary.get("rmse"),
        "sign_accuracy": primary.get("sign_accuracy"),
        "debiased_median": primary.get("debiased_median"),
        "debiased_median_frac": primary.get("debiased_median_frac"),
    }
    scanner.results["injection_recovery"] = result
    return result


# ---------------------------------------------------------------------------
# 5. Pantheon+ / Planck joint fit
# ---------------------------------------------------------------------------


def ensure_pantheon_plus_dat(force: bool = False) -> Path:
    PANTHEON_DIR.mkdir(parents=True, exist_ok=True)
    target = PANTHEON_DIR / "Pantheon+SH0ES.dat"
    if target.is_file() and not force:
        return target
    urllib.request.urlretrieve(PANTHEON_DAT_URL, target)
    return target


def load_pantheon_plus_subsample(
    path: str | Path | None = None,
    *,
    max_sne: int = 80,
    z_min: float = 0.01,
    z_max: float = 1.5,
) -> dict[str, Any]:
    """Load Pantheon+ distance moduli (diagonal errors) for joint cosmology fit."""
    dat_path = Path(path) if path else ensure_pantheon_plus_dat()
    rows: list[tuple[float, float, float]] = []
    header: list[str] = []
    for line in dat_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if line.startswith("CID") or line.startswith("#"):
            header = line.split()
            continue
        parts = line.split()
        if len(parts) < len(header):
            continue
        row = dict(zip(header, parts, strict=False))
        try:
            z = float(row.get("zCMB", row.get("zHD", "nan")))
            mu = float(row.get("MU_SH0ES", "nan"))
            mu_err = float(row.get("MU_SH0ES_ERR_DIAG", row.get("m_b_corr_err_DIAG", "nan")))
        except ValueError:
            continue
        if not np.isfinite(z) or not np.isfinite(mu) or not np.isfinite(mu_err):
            continue
        if z_min <= z <= z_max and mu_err > 0:
            rows.append((z, mu, mu_err))

    if not rows:
        raise ValueError(f"No Pantheon+ rows parsed from {dat_path}")

    # Subsample for tractable joint fit
    arr = np.array(rows, dtype=float)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    if len(arr) > max_sne:
        idx = np.linspace(0, len(arr) - 1, max_sne, dtype=int)
        arr = arr[idx]

    return {
        "z": arr[:, 0],
        "mu": arr[:, 1],
        "mu_err": arr[:, 2],
        "label": f"Pantheon+_subsample_{len(arr)}",
        "source": str(dat_path),
        "n_sne": len(arr),
    }


def _mu_lcdm_proxy(z: np.ndarray, h0: float = 70.0, om: float = 0.3) -> np.ndarray:
    """Toy distance modulus for flat ΛCDM (Mpc units, approximate integral)."""
    z = np.asarray(z, dtype=float)
    n_int = 64
    z_grid = np.linspace(0, np.max(z) * 1.2 + 0.01, n_int)
    ez = np.sqrt(om * (1 + z_grid) ** 3 + (1 - om))
    integrand = 1.0 / ez
    dz = np.diff(z_grid, prepend=0)
    dc = (299792.458 / h0) * np.cumsum(0.5 * (integrand + np.roll(integrand, 1)) * dz)
    dc[0] = 0
    dc_at_z = np.interp(z, z_grid, dc)
    dl = dc_at_z * (1 + z)
    return 5.0 * np.log10(np.maximum(dl, 1e-6)) + 25.0


def joint_fit_desi_sn_planck(
    scanner: TauSBScanner,
    desi_data: dict[str, Any],
    pantheon_data: dict[str, Any] | None = None,
    *,
    rd_mpc: float = PLANCK_RD_MPC,
    rd_err_mpc: float = PLANCK_RD_ERR_MPC,
    friction_shift: float = 0.0,
    use_covariance: bool = True,
) -> dict[str, Any]:
    """
    Combined χ²: DESI BAO + Pantheon+ μ(z) + Gaussian Planck r_d prior.

    ``friction_shift`` parametrizes early-universe sound-horizon shift from the
    313.1 MeV geometric friction floor (fractional).
    """
    if pantheon_data is None:
        pantheon_data = load_pantheon_plus_subsample()

    z_bao = desi_data["z"]
    obs_bao = desi_data["observable"]
    err_bao = desi_data["err"]
    cov_bao = desi_data.get("cov") if use_covariance else None

    z_sn = pantheon_data["z"]
    mu_sn = pantheon_data["mu"]
    mu_err = pantheon_data["mu_err"]

    gamma_used = scanner._gamma_for(z_bao)
    s_bao = s_from_z(z_bao, gamma=gamma_used)
    deg = min(2, max(1, len(z_bao) - 5))

    # Parameter vector: [poly..., A, phase, hier, mu0_offset, friction_shift]
    x0 = np.array(list(np.polyfit(z_bao, obs_bao, deg)) + [0.01, 0.0, 0.005, 0.0, friction_shift])

    def objective(x: np.ndarray) -> float:
        params = x
        rd_eff = rd_mpc * (1.0 + params[-1])
        base = np.polyval(params[: deg + 1], z_bao)
        osc = tau_sb_oscillatory_residual(
            s_bao, A=params[deg + 1], period=scanner.period, phase=params[deg + 2]
        )
        hier = tau_sb_hierarchical_step(z_bao, amplitude=params[deg + 3])
        bao_pred = (base + osc + hier) * (rd_eff / rd_mpc)
        chi2_bao = gaussian_chi2(obs_bao, bao_pred, err=err_bao, cov=cov_bao)
        chi2_sn = float(np.sum(((mu_sn - (_mu_lcdm_proxy(z_sn) + params[deg + 4])) / mu_err) ** 2))
        chi2_rd = ((rd_eff - rd_mpc) / rd_err_mpc) ** 2
        return chi2_bao + chi2_sn + chi2_rd

    res = optimize.minimize(objective, x0, method="Nelder-Mead")
    best = res.x
    rd_eff = rd_mpc * (1.0 + best[-1])

    bao_pred = (
        np.polyval(best[: deg + 1], z_bao)
        + tau_sb_oscillatory_residual(
            s_bao, A=best[deg + 1], period=scanner.period, phase=best[deg + 2]
        )
        + tau_sb_hierarchical_step(z_bao, amplitude=best[deg + 3])
    ) * (rd_eff / rd_mpc)
    chi2_bao = gaussian_chi2(obs_bao, bao_pred, err=err_bao, cov=cov_bao)
    chi2_sn = float(np.sum(((mu_sn - (_mu_lcdm_proxy(z_sn) + best[deg + 4])) / mu_err) ** 2))
    chi2_rd = ((rd_eff - rd_mpc) / rd_err_mpc) ** 2

    result = {
        "chi2_total": float(res.fun),
        "chi2_components": {"bao": float(chi2_bao), "sne": float(chi2_sn), "planck_rd": float(chi2_rd)},
        "rd_mpc_effective": float(rd_eff),
        "friction_shift_fraction": float(best[-1]),
        "A_osc": float(best[deg + 1]),
        "mu0_offset": float(best[deg + 4]),
        "n_bao": len(z_bao),
        "n_sne": int(pantheon_data["n_sne"]),
        "planck_rd_prior_mpc": rd_mpc,
        "gamma_used": gamma_used,
        "best_params": [float(x) for x in best],
    }
    scanner.results["joint_fit"] = result
    return result


# ---------------------------------------------------------------------------
# 6. Batch tracer scan
# ---------------------------------------------------------------------------


def run_batch_tracer_scan(
    *,
    cobaya_path: str | Path | None = None,
    tracers: list[str] | None = None,
    quantity_filter: str | None = "DM_over_rs",
    auto_calibrate_gamma: bool = True,
    compare_models: bool = True,
    use_covariance: bool = True,
    output_prefix: str = "tau_sb_desi_batch",
    max_tracers: int = 0,
    done_file: Path | str = DESI_BATCH_DONE,
    force_rescan: bool = False,
) -> dict[str, Any]:
    """Run DESI scan across pending DR2 tracers; resumes via datasets/desi/batch_done.txt."""
    available = list_desi_tracers(cobaya_path)
    catalog = tracers or available
    selected, status = select_batch_items(
        catalog,
        done_file,
        key_fn=str,
        limit=max_tracers,
        force_rescan=force_rescan,
    )
    print(format_batch_banner("Tau-SB DESI", status))
    summaries: dict[str, Any] = {}
    completed: list[str] = []

    if not selected:
        if status["done_count"] >= status["catalog_total"] and catalog:
            print(
                "[Tau-SB DESI] All tracers already scanned. "
                "Use force_rescan=yes to re-run from the start."
            )
        batch = {
            "tracers_run": [],
            "quantity_filter": quantity_filter,
            "summaries": summaries,
            "done_count": status["done_count"],
            "pending_count": status["pending_count"],
            "timestamp": _utc_stamp(),
        }
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        out = ARTIFACTS_DIR / f"{output_prefix}_summary_{_utc_stamp()}.json"
        from menus.astronomical.desi.json_util import write_json

        write_json(out, batch, indent=2, sort_keys=True)
        batch["summary_path"] = str(out)
        return batch

    for tracer in selected:
        if tracer not in available:
            summaries[tracer] = {"error": f"tracer not in {available}"}
            continue
        q_filter = quantity_filter
        if q_filter:
            avail_q = tracer_available_quantities(cobaya_path, tracer)
            if q_filter not in avail_q:
                q_filter = avail_q[0] if len(avail_q) == 1 else None
        try:
            result = run_desi_scan(
                cobaya_path=cobaya_path,
                tracer=tracer,
                quantity_filter=q_filter,
                auto_calibrate_gamma=auto_calibrate_gamma,
                compare_models=compare_models,
                use_covariance=use_covariance,
                run_fit=True,
                run_dipole=False,
                plot=False,
                output_prefix=f"{output_prefix}_{tracer}",
                action_name="batch_tracer_scan",
                pipelines_enabled="batch_tracers,tau_sb_fit,model_compare",
                run_residual_diagnostics=True,
            )
            summaries[tracer] = {
                "quantity_used": q_filter or "all",
                "n_data": result.fit.get("n_data") if result.fit else None,
                "dof": result.fit.get("dof") if result.fit else None,
                "gamma_used": result.gamma_used,
                "power_severity": (
                    result.power_assessment.get("overall_severity")
                    if result.power_assessment
                    else None
                ),
                "power_at_expected": result.periodicity["power_at_expected"] if result.periodicity else None,
                "best_model": (
                    result.model_comparison["best_model"] if result.model_comparison else None
                ),
                "report_path": result.report_path,
            }
            completed.append(tracer)
        except Exception as exc:
            summaries[tracer] = {"error": str(exc)}

    if completed:
        append_done_entries(done_file, completed)
        print(f"[Tau-SB DESI] Recorded {len(completed)} tracer(s) in {done_file}")

    batch = {
        "tracers_run": selected,
        "quantity_filter": quantity_filter,
        "summaries": summaries,
        "done_count": status["done_count"] + len(completed),
        "pending_count": max(0, status["pending_count"] - len(completed)),
        "timestamp": _utc_stamp(),
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS_DIR / f"{output_prefix}_summary_{_utc_stamp()}.json"
    from menus.astronomical.desi.json_util import write_json

    write_json(out, batch, indent=2, sort_keys=True)
    batch["summary_path"] = str(out)
    return batch


def run_production_pipeline(
    scanner: TauSBScanner,
    data: dict[str, Any],
    *,
    run_mcmc: bool = False,
    run_healpy_dipole: bool = False,
    run_changepoints: bool = False,
    run_injection: bool = False,
    run_joint: bool = False,
    mcmc_steps: int = MCMC_STEPS_DEFAULT,
    use_covariance: bool = True,
    n_injection_trials: int | None = None,
    max_sne: int | None = None,
) -> dict[str, Any]:
    """Execute selected production modules in order on one dataset."""
    z = data["z"]
    obs = data["observable"]
    err = data["err"]
    cov = data.get("cov") if use_covariance else None
    outputs: dict[str, Any] = {}

    if run_mcmc:
        outputs["mcmc"] = run_mcmc_tau_sb(
            scanner,
            z,
            obs,
            err,
            cov,
            n_steps=mcmc_steps,
            burn_in=mcmc_burn_in_for_steps(mcmc_steps),
        )
    if run_healpy_dipole:
        outputs["dipole"] = scan_dipole_healpy(scanner)
    if run_changepoints:
        outputs["changepoints"] = detect_hierarchical_changepoints(scanner, z, obs)
    if run_injection:
        live_a = None
        fit_block = scanner.results.get("fit")
        if isinstance(fit_block, dict):
            live_a = fit_block.get("A_osc_frac", fit_block.get("A_osc"))
        inj_kw: dict[str, Any] = {
            "live_A_osc_frac": float(live_a) if live_a is not None else None,
        }
        if n_injection_trials is not None:
            inj_kw["n_trials_per_amplitude"] = int(n_injection_trials)
        outputs["injection_recovery"] = run_injection_recovery_suite(
            scanner,
            z_ref=z,
            obs_ref=obs,
            err_ref=err,
            cov_ref=cov,
            **inj_kw,
        )
    if run_joint:
        pantheon = (
            load_pantheon_plus_subsample(max_sne=int(max_sne))
            if max_sne is not None
            else None
        )
        outputs["joint_fit"] = joint_fit_desi_sn_planck(
            scanner,
            data,
            pantheon_data=pantheon,
            use_covariance=use_covariance,
        )

    return outputs