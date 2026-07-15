"""
JAX-accelerated Tau-SB likelihood and dynesty interface (session 2026-07-09).

Implements Eq. 2.1 geometric theory vector (Tav cylinder + 8-phase engine),
augmented covariance, and Bayesian evidence comparison against constrained aDE.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import partial
from typing import Any, Callable

import numpy as np

from tav_shared.artifact_paths import TestSlug, artifact_path, artifact_timestamp, compose_dataset_slug
from menus.astronomical.desi.scanner import (
    GAMMA6_HEX,
    KAPPA_LEAK_DEFAULT,
    N_HIER_BINDING,
    R_TAU_MPC,
)

JAX_AVAILABLE = False
try:
    import jax
    import jax.numpy as jnp
    from jax import jit

    JAX_AVAILABLE = True
except ImportError:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    jit = None  # type: ignore[assignment]

# Geometric prior windows (session doc §5)
R_TAU_PRIOR = (6.5, 7.5)
GAMMA6_PRIOR = (1.35, 1.46)
N_HIER_PRIOR = (44.0, 47.5)
STEP_STRENGTH_PRIOR = (0.0, 0.02)

from menus.astronomical.desi.prior_bounds import (
    ADE_AOSC_MAX_FRAC as ADE_AOSC_MAX,
    ade_omega_bounds,
    ade_z_star_bounds,
    apply_evidence_corrections,
)

ADE_OMEGA_PRIOR = ade_omega_bounds()
ADE_ZSTAR_PRIOR = ade_z_star_bounds()
ADE_AMP_PRIOR = (0.5, 2.5)

NESTED_NLIVE_JAX_DEFAULT = 300
NESTED_MAX_SAMPLES_JAX_DEFAULT = 2000


def _require_jax() -> None:
    if not JAX_AVAILABLE:
        raise ImportError(
            "Install JAX: pip install jax jaxlib (see requirements-desi-production.txt)"
        )


if JAX_AVAILABLE:

    def _cumulative_trapezoid(y, x, *, initial=0.0):
        """JAX cumulative trapezoid (jax.scipy lacks cumulative_trapezoid on some builds)."""
        y = jnp.asarray(y)
        x = jnp.asarray(x)
        dx = jnp.diff(x)
        seg = 0.5 * (y[1:] + y[:-1]) * dx
        return jnp.concatenate([jnp.asarray([initial], dtype=y.dtype), jnp.cumsum(seg)])

    @jit
    def tau_sb_geo_baseline(
        z,
        R_tau=7.0,
        gamma6=1.4050,
        n_hier=45.8,
        kappa_leak=KAPPA_LEAK_DEFAULT,
        rho_aeon=1.0,
        I_norm=1.0,
        observable_scale=15.0,
        z_power=0.25,
    ):
        """Geometric baseline f_geo from 5D EKK Tav cylinder wave dispersion (Eq. 2.1)."""
        z_max = jnp.max(z) + 1.0
        t = jnp.linspace(0.0, z_max, 256)
        integrand = (rho_aeon * R_tau) / (gamma6 * I_norm)
        disp = kappa_leak * _cumulative_trapezoid(integrand * jnp.ones_like(t), t, initial=0.0)
        z_norm = jnp.maximum(z_max, 1e-6)
        base = observable_scale * (1.0 + z) ** z_power
        base = base * jnp.exp(disp[-1] * (z / z_norm))
        stretch = 1.0 + 0.005 * jnp.sin(2 * jnp.pi * n_hier * jnp.log1p(z) / 7.0)
        return base * stretch

    @jit
    def phase_recursion_correction(z, n_hier=45.8, step_strength=0.008):
        """Discrete phase correction delta_phase from 8-phase recursion (Eq. 2.1)."""
        resonance = jnp.sin(2 * jnp.pi * jnp.log1p(z) / 7.0)
        binding_scales = jnp.array([0.3, 1.0, 3.0, 10.0, 30.0])
        steps = 0.0
        for bs in binding_scales:
            log_z = jnp.log1p(z)
            log_bs = jnp.log1p(bs)
            steps = steps + step_strength * jnp.exp(
                -0.5 * ((log_z - log_bs) / 0.15) ** 2
            )
        exclusion = 0.0
        return resonance * 0.003 + steps + exclusion

    @jit
    def tau_sb_mu(
        z,
        R_tau=7.0,
        gamma6=1.4050,
        n_hier=45.8,
        step_strength=0.008,
        observable_scale=15.0,
        z_power=0.25,
    ):
        """Full theory prediction vector mu(theta) — JAX-jittable."""
        base = tau_sb_geo_baseline(
            z,
            R_tau,
            gamma6,
            n_hier,
            observable_scale=observable_scale,
            z_power=z_power,
        )
        phase = phase_recursion_correction(z, n_hier, step_strength)
        return base * (1.0 + phase)

    @jit
    def ade_mu(z, A_osc, omega, z_star, amp, observable_scale=15.0, z_power=0.25):
        """Constrained aDE comparison model (physical priors enforced in ptform)."""
        base = observable_scale * (1.0 + z) ** z_power
        mod = 1.0 + A_osc * jnp.sin(omega * jnp.log1p(z) + z_star)
        return base * mod * amp


class TauSB_Likelihood:
    """
    JAX-accelerated likelihood for Tau-Superblock geometric model.

    Covariance is precomputed once (including optional augmentation) so dynesty
    callbacks stay pure and return Python floats.
    """

    def __init__(
        self,
        z: np.ndarray,
        data: np.ndarray,
        base_cov: np.ndarray,
        *,
        model: str = "tau_sb",
        observable_scale: float | None = None,
        quantity: str = "DH_over_rs",
        augment_cov: bool = False,
    ):
        _require_jax()
        self.z_np = np.asarray(z, dtype=float).ravel()
        self.data_np = np.asarray(data, dtype=float).ravel()
        base = np.asarray(base_cov, dtype=float)
        n = len(self.z_np)
        if base.shape != (n, n):
            raise ValueError(f"base_cov {base.shape} incompatible with n={n}")

        if augment_cov:
            from menus.astronomical.desi.analysis import build_augmented_cov

            self.cov_np = build_augmented_cov(base, self.z_np)
        else:
            self.cov_np = base.copy()

        self.z = jnp.asarray(self.z_np)
        self.data = jnp.asarray(self.data_np)
        self.cov = jnp.asarray(self.cov_np)
        self.model = model
        self.quantity = quantity
        if observable_scale is None:
            observable_scale = float(np.median(np.abs(self.data_np)))
        self.observable_scale = float(observable_scale)
        self.z_power = 0.35 if quantity.startswith("DM") else 0.25

        self.inv_cov = jnp.linalg.inv(self.cov)
        _sign, self.logdet = jnp.linalg.slogdet(2 * jnp.pi * self.cov)

        self._loglike_tau_sb_jit = self._make_loglike_tau_sb()
        self._loglike_ade_jit = self._make_loglike_ade()

    def _make_loglike_tau_sb(self) -> Callable[[np.ndarray], float]:
        z = self.z
        data = self.data
        inv_cov = self.inv_cov
        logdet = self.logdet
        observable_scale = self.observable_scale
        z_power = self.z_power

        @jit
        def _inner(theta):
            R_tau, gamma6, n_hier, step_strength = theta
            mu = tau_sb_mu(
                z,
                R_tau,
                gamma6,
                n_hier,
                step_strength,
                observable_scale=observable_scale,
                z_power=z_power,
            )
            diff = data - mu
            chi2 = diff @ inv_cov @ diff
            return -0.5 * chi2 - 0.5 * logdet

        def loglike(theta: np.ndarray) -> float:
            return float(_inner(jnp.asarray(theta, dtype=float)))

        return loglike

    def _make_loglike_ade(self) -> Callable[[np.ndarray], float]:
        z = self.z
        data = self.data
        inv_cov = self.inv_cov
        logdet = self.logdet
        observable_scale = self.observable_scale
        z_power = self.z_power

        @jit
        def _inner(theta):
            A_osc, omega, z_star, amp = theta
            mu = ade_mu(
                z,
                A_osc,
                omega,
                z_star,
                amp,
                observable_scale=observable_scale,
                z_power=z_power,
            )
            diff = data - mu
            chi2 = diff @ inv_cov @ diff
            return -0.5 * chi2 - 0.5 * logdet

        def loglike(theta: np.ndarray) -> float:
            return float(_inner(jnp.asarray(theta, dtype=float)))

        return loglike

    def loglike_tau_sb(self, theta: np.ndarray) -> float:
        return self._loglike_tau_sb_jit(theta)

    def loglike_ade(self, theta: np.ndarray) -> float:
        return self._loglike_ade_jit(theta)

    def loglike(self, theta: np.ndarray) -> float:
        if self.model == "tau_sb":
            return self.loglike_tau_sb(theta)
        return self.loglike_ade(theta)


def _require_dynesty() -> None:
    try:
        import dynesty  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Install dynesty: pip install dynesty (see requirements-desi-production.txt)"
        ) from exc


def _tau_sb_prior_transform(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    R_tau = R_TAU_PRIOR[0] + (R_TAU_PRIOR[1] - R_TAU_PRIOR[0]) * u[0]
    gamma6 = GAMMA6_PRIOR[0] + (GAMMA6_PRIOR[1] - GAMMA6_PRIOR[0]) * u[1]
    n_hier = N_HIER_PRIOR[0] + (N_HIER_PRIOR[1] - N_HIER_PRIOR[0]) * u[2]
    step_strength = STEP_STRENGTH_PRIOR[0] + (
        STEP_STRENGTH_PRIOR[1] - STEP_STRENGTH_PRIOR[0]
    ) * u[3]
    return np.array([R_tau, gamma6, n_hier, step_strength], dtype=float)


def _ade_prior_transform(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    z_lo, z_hi = ADE_ZSTAR_PRIOR
    o_lo, o_hi = ADE_OMEGA_PRIOR
    A_osc = ADE_AOSC_MAX * u[0]
    omega = o_lo + (o_hi - o_lo) * u[1]
    z_star = z_lo + (z_hi - z_lo) * u[2]
    amp = ADE_AMP_PRIOR[0] + (ADE_AMP_PRIOR[1] - ADE_AMP_PRIOR[0]) * u[3]
    return np.array([A_osc, omega, z_star, amp], dtype=float)


def run_tau_sb_nested_sampling(
    like: TauSB_Likelihood,
    *,
    ndim: int = 4,
    nlive: int = NESTED_NLIVE_JAX_DEFAULT,
    max_samples: int = NESTED_MAX_SAMPLES_JAX_DEFAULT,
    dlogz: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Run dynesty nested sampling on Tau-SB geometric model (4D cylinder params)."""
    _require_dynesty()
    import dynesty

    sampler = dynesty.NestedSampler(
        like.loglike_tau_sb,
        _tau_sb_prior_transform,
        ndim,
        nlive=nlive,
        bound="multi",
        sample="rwalk",
        rstate=np.random.default_rng(int(seed)),
    )
    sampler.run_nested(maxiter=max_samples, dlogz=dlogz, print_progress=False)
    res = sampler.results
    logz = float(res.logz[-1])
    logz_err = float(res.logzerr[-1]) if len(res.logzerr) else float("nan")
    best = res.samples[np.argmax(res.logl)]
    return {
        "model": "Tau-SB (JAX geometric)",
        "log_evidence": logz,
        "log_evidence_err": logz_err,
        "n_live": nlive,
        "ndim": ndim,
        "best_fit_params": {
            "R_tau": float(best[0]),
            "gamma6": float(best[1]),
            "n_hier": float(best[2]),
            "step_strength": float(best[3]),
        },
        "geometric_anchors": {
            "R_tau_mpc": R_TAU_MPC,
            "gamma6_hex": GAMMA6_HEX,
            "n_hier": N_HIER_BINDING,
            "prior_windows": {
                "R_tau": R_TAU_PRIOR,
                "gamma6": GAMMA6_PRIOR,
                "n_hier": N_HIER_PRIOR,
                "step_strength": STEP_STRENGTH_PRIOR,
            },
        },
        "observable_scale": like.observable_scale,
        "method": "dynesty_jax_geometric",
    }


def run_ade_nested_sampling(
    like: TauSB_Likelihood,
    *,
    ndim: int = 4,
    nlive: int = NESTED_NLIVE_JAX_DEFAULT,
    max_samples: int = NESTED_MAX_SAMPLES_JAX_DEFAULT,
    dlogz: float = 0.05,
    seed: int = 43,
) -> dict[str, Any]:
    """Run dynesty on constrained aDE for direct evidence comparison."""
    _require_dynesty()
    import dynesty

    sampler = dynesty.NestedSampler(
        like.loglike_ade,
        _ade_prior_transform,
        ndim,
        nlive=nlive,
        bound="multi",
        sample="rwalk",
        rstate=np.random.default_rng(int(seed)),
    )
    sampler.run_nested(maxiter=max_samples, dlogz=dlogz, print_progress=False)
    res = sampler.results
    logz = float(res.logz[-1])
    logz_err = float(res.logzerr[-1]) if len(res.logzerr) else float("nan")
    best = res.samples[np.argmax(res.logl)]
    return {
        "model": "aDE (JAX physical priors)",
        "log_evidence": logz,
        "log_evidence_err": logz_err,
        "n_live": nlive,
        "ndim": ndim,
        "best_fit_params": {
            "A_osc": float(best[0]),
            "omega": float(best[1]),
            "z_star": float(best[2]),
            "amp": float(best[3]),
        },
        "priors": {
            "A_osc_max": ADE_AOSC_MAX,
            "omega_range": ADE_OMEGA_PRIOR,
            "z_star_range": ADE_ZSTAR_PRIOR,
        },
        "observable_scale": like.observable_scale,
        "method": "dynesty_jax_ade_physical",
    }


def make_likelihood_from_scan_data(
    z: np.ndarray,
    observable: np.ndarray,
    *,
    err: np.ndarray | None = None,
    cov: np.ndarray | None = None,
    quantity: str = "DH_over_rs",
    augment_cov: bool = False,
    observable_scale: float | None = None,
    model: str = "tau_sb",
) -> TauSB_Likelihood:
    """Build TauSB_Likelihood from DESI scan vectors."""
    if cov is None:
        if err is None:
            raise ValueError("JAX likelihood requires cov or err")
        cov = np.diag(np.asarray(err, dtype=float) ** 2)
    return TauSB_Likelihood(
        z,
        observable,
        cov,
        model=model,
        quantity=quantity,
        augment_cov=augment_cov,
        observable_scale=observable_scale,
    )


def run_jax_nested_evidence_comparison(
    z: np.ndarray,
    observable: np.ndarray,
    err: np.ndarray | None = None,
    cov: np.ndarray | None = None,
    *,
    quantity: str = "DH_over_rs",
    cov_already_augmented: bool = False,
    nlive: int = NESTED_NLIVE_JAX_DEFAULT,
    max_samples: int = NESTED_MAX_SAMPLES_JAX_DEFAULT,
    output_prefix: str = "tau_sb_jax_nested",
) -> dict[str, Any]:
    """
    Run Tau-SB (JAX geometric) and aDE (JAX physical) nested sampling; compare ln Z.
    """
    _require_jax()
    observable_scale = float(np.median(np.abs(observable)))
    like_tau = make_likelihood_from_scan_data(
        z,
        observable,
        err=err,
        cov=cov,
        quantity=quantity,
        augment_cov=not cov_already_augmented,
        model="tau_sb",
        observable_scale=observable_scale,
    )
    like_ade = make_likelihood_from_scan_data(
        z,
        observable,
        err=err,
        cov=cov,
        quantity=quantity,
        augment_cov=not cov_already_augmented,
        model="ade",
        observable_scale=observable_scale,
    )

    print("\n[JAX nested sampling] Tau-SB geometric priors …")
    tau = run_tau_sb_nested_sampling(
        like_tau, nlive=nlive, max_samples=max_samples, seed=42
    )
    print(
        f"  ln Z(Tau-SB) = {tau['log_evidence']:.2f} "
        f"± {tau['log_evidence_err']:.2f}"
    )

    print("[JAX nested sampling] aDE physical priors …")
    ade = run_ade_nested_sampling(
        like_ade, nlive=nlive, max_samples=max_samples, seed=43
    )
    print(
        f"  ln Z(aDE)    = {ade['log_evidence']:.2f} "
        f"± {ade['log_evidence_err']:.2f}"
    )

    delta = float(tau["log_evidence"] - ade["log_evidence"])
    favor = "Tau-SB" if delta > 0 else "aDE"
    from menus.astronomical.desi.prior_bounds import SPARSE_N_THRESHOLD

    n_data = len(np.asarray(z, dtype=float))
    ev = apply_evidence_corrections(
        log_evidence_tau=float(tau["log_evidence"]),
        log_evidence_ade=float(ade["log_evidence"]),
        n_data=n_data,
        ndim_tau=int(tau.get("ndim", 4)),
        ndim_ade=int(ade.get("ndim", 4)),
    )
    print(
        f"[JAX nested sampling] Δln Z = {delta:+.2f} → favors {favor} "
        f"(raw evidence)"
    )
    print(
        f"[JAX nested sampling] BIC-corrected Δln Z = "
        f"{ev['delta_log_evidence_bic']:+.2f} "
        f"(k_τ={ev['k_tau']}, k_aDE={ev['k_ade']}, n={n_data}) "
        f"→ favors {ev['favored_model_bic']}"
    )
    if ev["occam_penalty_tau_minus_ade"]:
        print(
            f"[JAX nested sampling] Sparse Occam Δln Z = "
            f"{ev['delta_log_evidence_occam_corrected']:+.2f} "
            f"(penalty={ev['occam_penalty_tau_minus_ade']:.2f}, n<{SPARSE_N_THRESHOLD}) "
            f"→ favors {ev['favored_model_sparse']}"
        )

    report = {
        "tau_sb": tau,
        "ade": ade,
        "delta_log_evidence_tau_minus_ade": delta,
        "favored_model": favor,
        "n_data": n_data,
        **ev,
        "backend": "jax",
        "timestamp": artifact_timestamp(),
    }
    out_path = artifact_path(
        TestSlug.NESTED_SAMPLING,
        compose_dataset_slug(output_prefix, "jax"),
        "evidence",
        "json",
    )
    from menus.astronomical.desi.json_util import write_json

    write_json(out_path, report, indent=2, sort_keys=True)
    report["report_path"] = str(out_path)
    return report


# =============================================================================
# JAX FFT hooks for LSS comb falsification (Methods 4, 9)
# =============================================================================

if JAX_AVAILABLE:

    @jit
    def jax_power_spectrum_1d(delta, box_size_mpc: float):
        """JAX 1D P(k) via rfft."""
        field = jnp.asarray(delta, dtype=jnp.float64)
        n = field.shape[0]
        fft = jnp.fft.rfft(field)
        pk = jnp.abs(fft) ** 2 / n
        k = jnp.fft.rfftfreq(n, d=box_size_mpc / n) * 2 * jnp.pi
        return k, pk

    @partial(jit, static_argnums=(1, 2))
    def jax_power_spectrum_3d_binned(delta, box_size_mpc: float, n_k_bins: int = 48):
        """Full JAX 3D spherically averaged P(k): fftn + digitize + segment_sum."""
        field = jnp.asarray(delta, dtype=jnp.float64)
        n = field.shape[0]
        n_bins = n_k_bins
        fft = jnp.fft.fftn(field)
        pk_flat = (jnp.abs(fft) ** 2 / field.size).ravel()
        k1d = jnp.fft.fftfreq(n, d=box_size_mpc / n) * 2 * jnp.pi
        kx, ky, kz = jnp.meshgrid(k1d, k1d, k1d, indexing="ij")
        kmag = jnp.sqrt(kx**2 + ky**2 + kz**2).ravel()
        k_max = jnp.max(kmag)
        edges = jnp.linspace(0.0, k_max, n_bins + 1)
        bin_idx = jnp.digitize(kmag, edges) - 1
        bin_idx = jnp.clip(bin_idx, 0, n_bins - 1)
        counts = jax.ops.segment_sum(jnp.ones_like(pk_flat), bin_idx, num_segments=n_bins)
        pk_sum = jax.ops.segment_sum(pk_flat, bin_idx, num_segments=n_bins)
        pk_binned = pk_sum / jnp.maximum(counts, 1.0)
        k_centers = 0.5 * (edges[:-1] + edges[1:])
        return k_centers, pk_binned

    def jax_power_spectrum_3d(delta, box_size_mpc: float, n_k_bins: int = 48):
        """Legacy alias — returns binned k, P(k) centers."""
        return jax_power_spectrum_3d_binned(delta, box_size_mpc, n_k_bins)

    @jit
    def jax_comb_excess_at_k(pk, k, k_target: float):
        """Excess power at target comb frequency vs local median window."""
        k = jnp.asarray(k, dtype=jnp.float64)
        pk = jnp.asarray(pk, dtype=jnp.float64)
        idx = jnp.argmin(jnp.abs(k - k_target))
        window = 5
        start = jnp.maximum(0, idx - window)
        end = jnp.minimum(pk.shape[0], idx + window + 1)
        background = jnp.median(pk[start:end])
        return pk[idx] - background, pk[idx], background

else:

    def jax_power_spectrum_1d(delta, box_size_mpc: float):  # type: ignore[misc]
        raise ImportError("JAX not available")

    def jax_power_spectrum_3d(delta, box_size_mpc: float, n_k_bins: int = 48):  # type: ignore[misc]
        raise ImportError("JAX not available")

    def jax_power_spectrum_3d_binned(delta, box_size_mpc: float, n_k_bins: int = 48):  # type: ignore[misc]
        raise ImportError("JAX not available")

    def jax_comb_excess_at_k(pk, k, k_target: float):  # type: ignore[misc]
        raise ImportError("JAX not available")


if __name__ == "__main__":
    _require_jax()
    np.random.seed(42)
    z_np = np.array([0.3, 0.5, 0.8, 1.2, 1.8, 2.5, 3.5])
    true_mu = np.asarray(
        tau_sb_mu(
            jnp.array(z_np),
            R_tau=7.0,
            gamma6=1.405,
            n_hier=45.8,
            step_strength=0.008,
            observable_scale=15.0,
        )
    )
    base_cov = (
        np.eye(len(z_np)) * 0.01
        + 0.005 * np.outer(np.ones(len(z_np)), np.ones(len(z_np)))
    )
    from menus.astronomical.desi.analysis import build_augmented_cov

    aug_cov = build_augmented_cov(base_cov, z_np)
    noise = np.random.multivariate_normal(np.zeros(len(z_np)), aug_cov)
    data = true_mu + noise

    like = TauSB_Likelihood(z_np, data, base_cov, model="tau_sb", augment_cov=True)
    print("Running Tau-SB JAX nested sampling (demo)…")
    sampler_tau = run_tau_sb_nested_sampling(like, nlive=200, max_samples=800)
    print(
        f"Tau-SB lnZ = {sampler_tau['log_evidence']:.2f} "
        f"± {sampler_tau['log_evidence_err']:.2f}"
    )

    like_ade = TauSB_Likelihood(z_np, data, base_cov, model="ade", augment_cov=True)
    sampler_ade = run_ade_nested_sampling(like_ade, nlive=200, max_samples=800)
    print(
        f"aDE lnZ = {sampler_ade['log_evidence']:.2f} "
        f"± {sampler_ade['log_evidence_err']:.2f}"
    )
    delta = sampler_tau["log_evidence"] - sampler_ade["log_evidence"]
    print(f"ΔlnZ (Tau-SB - aDE) = {delta:+.2f}")