import numpy as np
import scipy.optimize as opt

TAV_AIC_BREAKTHROUGH = -29634.0

MODEL_CATALOG = {
    "Tav": {
        "label": "Tav-Superblock",
        "profile": "Softened Kerr-seed soliton",
        "central": "Phase-diffused core (sigma_seed)",
        "outer": "Conformal time-averaging envelope",
        "primary_use": "TS cross-domain correlation with ergosphere damping",
    },
    "NFW": {
        "label": "NFW",
        "profile": "Cusp (r^-1)",
        "central": "Cusp (r^-1)",
        "outer": "Steeper (r^-3)",
        "primary_use": "LCDM standard baseline",
    },
    "Einasto": {
        "label": "Einasto",
        "profile": "Flat / core-like",
        "central": "Flat / core-like",
        "outer": "Varying slope",
        "primary_use": "High-precision fits to simulations",
    },
    "Burkert": {
        "label": "Burkert",
        "profile": "Constant core",
        "central": "Constant core",
        "outer": "Steeper (r^-3)",
        "primary_use": 'Addressing the "cusp-core" problem',
    },
    "Hernquist": {
        "label": "Hernquist",
        "profile": "Cusp (r^-1)",
        "central": "Cusp (r^-1)",
        "outer": "Very steep (r^-4)",
        "primary_use": "Galactic mass distribution modeling",
    },
    "pISO": {
        "label": "pISO",
        "profile": "Constant core",
        "central": "Constant core",
        "outer": "Flattened (r^-2)",
        "primary_use": "Empirical rotation curve fitting",
    },
}

MODEL_ORDER = ["Tav", "NFW", "Einasto", "Burkert", "Hernquist", "pISO"]


def _safe_x(x):
    return np.maximum(np.asarray(x, dtype=float), 1e-9)


def _aic(n_params, n_points, rss):
    rss = max(float(rss), 1e-12)
    return 2 * n_params + n_points * np.log(rss / n_points)


def tav_model(x, amplitude, ratio, sigma_seed):
    """
    Softened Kerr-Seed Tav-Superblock profile.

    amplitude: flux scale
    ratio: domain coupling coefficient
    sigma_seed: ergosphere interface damping (phase-diffusion width)
    """
    r = _safe_x(x)
    sigma = max(sigma_seed, 1e-9)
    xr = r / sigma

    # pISO-like flattened core removes hard central stiffness.
    core = 1.0 / (1.0 + xr ** 2)
    # Phase-diffused Kerr seed becomes a localized soliton, not a singularity.
    seed_soliton = np.exp(-0.5 * xr ** 2)
    # Conformal refresh envelope carries TS outer-domain structure.
    refresh = 1.0 - np.exp(-xr / 2.0)

    return amplitude * (core + ratio * seed_soliton * core + (1.0 + ratio) * refresh * core)


def nfw_model(x, rho_s, r_s):
    xr = _safe_x(x) / max(r_s, 1e-9)
    return rho_s / (xr * (1 + xr) ** 2)


def einasto_model(x, rho_e, r_e, alpha):
    xr = np.maximum(_safe_x(x) / max(r_e, 1e-9), 1e-12)
    alpha = max(alpha, 1e-3)
    return rho_e * np.exp(-2.0 / alpha * (xr ** alpha - 1.0))


def burkert_model(x, rho_0, r_0):
    xr = _safe_x(x) / max(r_0, 1e-9)
    return rho_0 / ((1 + xr) * (1 + xr ** 2))


def hernquist_model(x, rho_0, r_b):
    r = _safe_x(x)
    rb = max(r_b, 1e-9)
    return rho_0 / (r * (r + rb) ** 3)


def piso_model(x, rho_0, r_c):
    xr = _safe_x(x) / max(r_c, 1e-9)
    return rho_0 / (1 + xr ** 2)


def _initial_guess(model_type, x, y_obs):
    x = _safe_x(x)
    y_obs = np.asarray(y_obs, dtype=float)
    amp = max(float(np.median(np.abs(y_obs))), 1e-3)
    scale = max(float(np.median(x)), 1e-3)

    if model_type == "Tav":
        return [amp, 0.2, scale * 0.35]
    if model_type == "NFW":
        return [amp, scale]
    if model_type == "Einasto":
        return [amp, scale, 0.17]
    if model_type == "Burkert":
        return [amp, scale]
    if model_type == "Hernquist":
        return [amp * scale ** 3, scale]
    if model_type == "pISO":
        return [amp, scale]
    raise ValueError(f"Unknown model type: {model_type}")


def _fit_bounds(model_type):
    if model_type == "Tav":
        return ([1e-9, -5.0, 1e-9], [np.inf, 5.0, np.inf])
    if model_type == "Einasto":
        return ([1e-6, 1e-6, 0.01], [np.inf, np.inf, 2.0])
    if model_type in {"NFW", "Burkert", "pISO"}:
        return ([1e-6, 1e-6], [np.inf, np.inf])
    if model_type == "Hernquist":
        return ([1e-12, 1e-6], [np.inf, np.inf])
    return None


def run_mcmc_sweep(x, y_obs, model_type="Tav", return_params=False):
    """
    Fit one model to empirical arrays and return RSS + AIC.
    """
    x = _safe_x(x)
    y_obs = np.asarray(y_obs, dtype=float)
    n_points = len(x)
    if n_points < 2:
        return (np.inf, np.inf, None) if return_params else (np.inf, np.inf)

    model_funcs = {
        "Tav": (tav_model, 3),
        "NFW": (nfw_model, 2),
        "Einasto": (einasto_model, 3),
        "Burkert": (burkert_model, 2),
        "Hernquist": (hernquist_model, 2),
        "pISO": (piso_model, 2),
    }

    if model_type not in model_funcs:
        raise ValueError(f"Unknown model type: {model_type}")

    model_fn, n_params = model_funcs[model_type]
    p0 = _initial_guess(model_type, x, y_obs)
    bounds = _fit_bounds(model_type)
    popt = None

    try:
        if bounds is None:
            popt, _ = opt.curve_fit(model_fn, x, y_obs, p0=p0, maxfev=30000)
        else:
            popt, _ = opt.curve_fit(
                model_fn, x, y_obs, p0=p0, bounds=bounds, maxfev=30000
            )
        y_fit = model_fn(x, *popt)
        rss = float(np.sum((y_obs - y_fit) ** 2))
        aic = _aic(n_params, n_points, rss)
    except Exception:
        rss, aic = np.inf, np.inf

    if return_params:
        return rss, aic, popt
    return rss, aic


def describe_tav_fit(params):
    if params is None:
        return "Softened Kerr-Seed fit unavailable."
    amplitude, ratio, sigma_seed = params
    return (
        "Softened Kerr-Seed fit: "
        f"amplitude={amplitude:.6f}, "
        f"domain_coupling={ratio:.6f}, "
        f"sigma_seed={sigma_seed:.6f}"
    )


def run_model_comparison(x, y_obs):
    """
    Run all cataloged models and return AIC-ranked comparison rows.
    """
    results = []
    for model_type in MODEL_ORDER:
        rss, aic, popt = run_mcmc_sweep(x, y_obs, model_type=model_type, return_params=True)
        meta = MODEL_CATALOG[model_type]
        row = {
            "model_type": model_type,
            "label": meta["label"],
            "profile": meta["profile"],
            "central": meta["central"],
            "outer": meta["outer"],
            "primary_use": meta["primary_use"],
            "rss": rss,
            "aic": aic,
            "params": popt,
        }
        if model_type == "Tav" and popt is not None:
            row["sigma_seed"] = float(popt[2])
            row["domain_coupling"] = float(popt[1])
            row["amplitude"] = float(popt[0])
        results.append(row)

    finite = [r for r in results if np.isfinite(r["aic"])]
    best_aic = min((r["aic"] for r in finite), default=np.inf)
    for row in results:
        row["delta_aic"] = row["aic"] - best_aic if np.isfinite(row["aic"]) else np.inf

    results.sort(key=lambda row: row["aic"])
    return results


def format_comparison_report(results):
    lines = [
        "Model Comparison (lower AIC is better):",
        f"{'Model':<14} {'AIC':>12} {'dAIC':>10}  Profile / Primary Use",
    ]
    for row in results:
        if not np.isfinite(row["aic"]):
            aic_text = "fit failed"
            delta_text = "-"
        else:
            aic_text = f"{row['aic']:.2f}"
            delta_text = f"{row['delta_aic']:.2f}" if row["delta_aic"] > 0 else "best"
        lines.append(
            f"{row['label']:<14} {aic_text:>12} {delta_text:>10}  "
            f"{row['profile']} | {row['primary_use']}"
        )
    return "\n".join(lines)