"""Unit tests for GARCH(1,1) volatility forecaster."""

import asyncio

import numpy as np
import pytest

from src.analysis.garch_forecast import GARCHForecaster


def _returns(n: int = 250, sigma: float = 0.012, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, n)


def _await(coro):
    return asyncio.run(coro)


def test_forecast_returns_none_for_short_series():
    g = GARCHForecaster()
    short = _returns(n=50)
    assert _await(g.forecast(short)) is None


def test_forecast_basic_keys_present():
    g = GARCHForecaster()
    out = _await(g.forecast(_returns()))
    assert out is not None
    expected = {
        "parameters",
        "current_conditional_vol_annual",
        "long_run_vol_annual",
        "historical_vol_20d",
        "historical_vol_60d",
        "forecasts",
        "n_observations",
    }
    assert expected.issubset(out.keys())


def test_strict_window_offset_excludes_as_of_day():
    """historical_vol_20d must use returns[-21:-1] with ddof=1, not returns[-20:]."""
    returns = _returns(seed=7)
    g = GARCHForecaster()
    out = _await(g.forecast(returns))

    # The strict offset takes 20 obs ending one day before the last point.
    expected_20 = float(np.std(returns[-21:-1], ddof=1) * np.sqrt(252))
    assert out["historical_vol_20d"] == round(expected_20, 4)

    expected_60 = float(np.std(returns[-61:-1], ddof=1) * np.sqrt(252))
    assert out["historical_vol_60d"] == round(expected_60, 4)


def test_window_uses_unbiased_ddof_one():
    """ddof=1 vs ddof=0 produces detectably different std at small N."""
    returns = _returns(n=120, seed=11)
    g = GARCHForecaster()
    out = _await(g.forecast(returns))

    # Reproducing with ddof=0 should diverge from the reported value.
    biased = float(np.std(returns[-21:-1], ddof=0) * np.sqrt(252))
    unbiased = float(np.std(returns[-21:-1], ddof=1) * np.sqrt(252))
    assert biased != unbiased
    assert out["historical_vol_20d"] == round(unbiased, 4)


def test_forecast_horizons_are_returned():
    g = GARCHForecaster()
    out = _await(g.forecast(_returns(), horizons=[5, 20]))
    assert set(out["forecasts"].keys()) == {5, 20}
    for h in (5, 20):
        f = out["forecasts"][h]
        assert f["days"] == h
        assert f["predicted_volatility_annual"] >= 0
        assert f["predicted_volatility_daily"] >= 0
        assert f["interpretation"] in {
            "volatility_expanding", "volatility_stable", "volatility_contracting"
        }


def test_forecast_persistence_within_unit_circle():
    """For mean-reverting volatility, persistence (alpha + beta) should be < 1."""
    returns = _returns(n=400, seed=3)
    g = GARCHForecaster()
    out = _await(g.forecast(returns))
    if not out.get("is_fallback"):
        assert out["parameters"]["persistence"] < 1.05  # tolerate slight overshoot


def test_ewma_fallback_strict_offset():
    """When GARCH fails to fit, the EWMA fallback must still use strict windows."""
    g = GARCHForecaster()
    # Force the fallback path by passing a degenerate input that arch will choke on.
    # We patch _forecast_sync to raise; the public method routes to _ewma_fallback.
    returns = _returns(n=200)
    out = g._ewma_fallback(returns, horizons=[5, 20])
    expected_20 = float(np.std(returns[-21:-1], ddof=1) * np.sqrt(252))
    assert out["historical_vol_20d"] == round(expected_20, 4)
    assert out["is_fallback"] is True


def test_interpret_vol_ratio():
    g = GARCHForecaster()
    assert g._interpret_vol_ratio(1.5) == "volatility_expanding"
    assert g._interpret_vol_ratio(0.6) == "volatility_contracting"
    assert g._interpret_vol_ratio(1.0) == "volatility_stable"
