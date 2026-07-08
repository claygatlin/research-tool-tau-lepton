import numpy as np

import mcmc_engine as me


def _synthetic_burkert(x, rho_0=12.0, r_0=2.5):
    xr = x / r_0
    return rho_0 / ((1 + xr) * (1 + xr ** 2))


def test_all_models_return_finite_aic_on_clean_data():
    x = np.linspace(0.5, 20.0, 80)
    y = _synthetic_burkert(x) + np.random.default_rng(0).normal(0, 0.05, size=x.size)

    for model_type in me.MODEL_ORDER:
        rss, aic = me.run_mcmc_sweep(x, y, model_type=model_type)
        assert np.isfinite(rss), f"{model_type} RSS not finite"
        assert np.isfinite(aic), f"{model_type} AIC not finite"


def test_model_comparison_ranks_burkert_near_top_on_burkert_data():
    x = np.linspace(0.5, 20.0, 100)
    y = _synthetic_burkert(x) + np.random.default_rng(1).normal(0, 0.02, size=x.size)

    results = me.run_model_comparison(x, y)
    top_models = {row["model_type"] for row in results[:2]}
    assert "Burkert" in top_models


def test_catalog_contains_requested_profiles():
    expected = {"NFW", "Einasto", "Burkert", "Hernquist", "pISO", "Tav"}
    assert expected == set(me.MODEL_CATALOG.keys())


def test_format_comparison_report_includes_models():
    x = np.linspace(1.0, 10.0, 40)
    y = me.nfw_model(x, 8.0, 3.0)
    report = me.format_comparison_report(me.run_model_comparison(x, y))
    for label in ("NFW", "Einasto", "Burkert", "Hernquist", "pISO", "Tav-Superblock"):
        assert label in report


def test_softened_tav_returns_sigma_seed_params():
    x = np.linspace(0.5, 20.0, 120)
    y = me.tav_model(x, 4.0, 0.15, 2.5) + np.random.default_rng(3).normal(0, 0.03, size=x.size)
    rss, aic, params = me.run_mcmc_sweep(x, y, model_type="Tav", return_params=True)
    assert np.isfinite(rss)
    assert np.isfinite(aic)
    assert params is not None
    assert len(params) == 3
    assert "sigma_seed" in me.describe_tav_fit(params)


def test_softened_tav_competitive_on_piso_like_core():
    x = np.linspace(0.5, 20.0, 120)
    y = me.piso_model(x, 5.0, 2.0) + np.random.default_rng(4).normal(0, 0.02, size=x.size)
    results = me.run_model_comparison(x, y)
    tav = next(row for row in results if row["model_type"] == "Tav")
    piso = next(row for row in results if row["model_type"] == "pISO")
    assert tav["aic"] <= piso["aic"] + 15.0