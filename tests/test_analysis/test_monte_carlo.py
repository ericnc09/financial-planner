"""Unit tests for Monte Carlo GBM simulator."""

import numpy as np
import pytest

from src.analysis.monte_carlo import MonteCarloSimulator


def _gbm_prices(
    n_days: int = 300, mu: float = 0.0005, sigma: float = 0.015, seed: int = 1, s0: float = 100.0
) -> np.ndarray:
    """Generate deterministic GBM price series for testing."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(mu - 0.5 * sigma**2, sigma, n_days)
    return s0 * np.exp(np.cumsum(log_returns))


def test_simulate_returns_none_for_short_series():
    mc = MonteCarloSimulator(n_simulations=200, seed=42)
    short = np.array([100.0] * 10)  # < 30-day minimum
    assert mc.simulate(short) is None


def test_simulate_returns_none_when_volatility_zero():
    mc = MonteCarloSimulator(n_simulations=200, seed=42)
    flat = np.full(60, 100.0)  # zero log-return variance
    assert mc.simulate(flat) is None


def test_simulate_produces_all_horizons_and_keys():
    mc = MonteCarloSimulator(n_simulations=500, seed=42)
    closes = _gbm_prices()
    result = mc.simulate(closes, horizons=[21, 63, 126])

    assert result is not None
    assert set(result["horizons"].keys()) == {21, 63, 126}
    for h, payload in result["horizons"].items():
        assert payload["days"] == h
        assert set(payload["percentiles"].keys()) == {"p10", "p25", "p50", "p75", "p90"}
        assert 0.0 <= payload["probability_of_profit"] <= 1.0


def test_simulate_seed_is_reproducible():
    closes = _gbm_prices(seed=2)
    mc1 = MonteCarloSimulator(n_simulations=500, seed=42)
    mc2 = MonteCarloSimulator(n_simulations=500, seed=42)
    out1 = mc1.simulate(closes)
    out2 = mc2.simulate(closes)
    assert out1["horizons"][21]["expected_return"] == out2["horizons"][21]["expected_return"]
    assert out1["horizons"][126]["value_at_risk_95"] == out2["horizons"][126]["value_at_risk_95"]


def test_simulate_percentiles_are_monotonic():
    mc = MonteCarloSimulator(n_simulations=2000, seed=42)
    closes = _gbm_prices(n_days=400, mu=0.0008, sigma=0.012)
    result = mc.simulate(closes, horizons=[63])
    pcts = result["horizons"][63]["percentiles"]
    assert pcts["p10"] <= pcts["p25"] <= pcts["p50"] <= pcts["p75"] <= pcts["p90"]


def test_simulate_var95_is_negative_for_zero_drift():
    """With zero drift, the 5th-percentile return should be a loss."""
    mc = MonteCarloSimulator(n_simulations=2000, seed=42)
    closes = _gbm_prices(n_days=400, mu=0.0, sigma=0.012)
    result = mc.simulate(closes, horizons=[63])
    assert result["horizons"][63]["value_at_risk_95"] < 0


def test_garch_override_takes_precedence():
    mc = MonteCarloSimulator(n_simulations=500, seed=42)
    closes = _gbm_prices()
    garch = {"current_conditional_vol_annual": 0.30, "is_fallback": False}
    result = mc.simulate(closes, garch_forecast=garch)
    assert result["volatility_source"] == "garch"


def test_regime_override_takes_precedence_over_garch():
    mc = MonteCarloSimulator(n_simulations=500, seed=42)
    closes = _gbm_prices()
    garch = {"current_conditional_vol_annual": 0.30, "is_fallback": False}
    regime = {"mu": 0.001, "sigma": 0.02}
    result = mc.simulate(closes, garch_forecast=garch, regime_params=regime)
    assert result["volatility_source"] == "regime"


def test_p95_max_drawdown_positive_and_bounded():
    mc = MonteCarloSimulator(n_simulations=500, seed=42)
    closes = _gbm_prices(n_days=400)
    result = mc.simulate(closes, horizons=[126])
    md = result["horizons"][126]["max_drawdown_95"]
    assert 0.0 <= md <= 1.0


def test_garch_fallback_does_not_override_volatility_source():
    mc = MonteCarloSimulator(n_simulations=300, seed=42)
    closes = _gbm_prices()
    garch_fb = {"current_conditional_vol_annual": 0.30, "is_fallback": True}
    result = mc.simulate(closes, garch_forecast=garch_fb)
    # When GARCH is a fallback, MC should use historical, not GARCH
    assert result["volatility_source"] == "historical"
