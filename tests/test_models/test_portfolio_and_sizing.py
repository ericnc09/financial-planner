"""Mean-variance optimizer and position sizing."""

import numpy as np
import pytest

from src.analysis.mean_variance import MeanVarianceOptimizer
from src.analysis.position_sizing import MAX_SINGLE_WEIGHT, compute_position_sizes


def _returns(daily_means, vol=0.015, n_days=252, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(daily_means, vol, size=(n_days, len(daily_means)))


def test_shrinkage_keeps_expected_returns_distinct():
    # The old formula always shrank every mean all the way to the grand mean.
    R = _returns([0.003, 0.001, -0.001, 0.002, 0.0])
    opt = MeanVarianceOptimizer()
    from sklearn.covariance import LedoitWolf

    mu = opt._james_stein_shrink(R.mean(axis=0), LedoitWolf().fit(R).covariance_, len(R))
    sample = R.mean(axis=0)
    assert np.ptp(mu) > 0.25 * np.ptp(sample)
    # shrunk toward, never past, the grand mean
    assert np.all(np.abs(mu - sample.mean()) <= np.abs(sample - sample.mean()) + 1e-12)
    assert np.argsort(mu).tolist() == np.argsort(sample).tolist()


def test_optimizer_produces_a_frontier_and_a_diversified_tangency():
    tickers = ["A", "B", "C", "D", "E"]
    result = MeanVarianceOptimizer(risk_free_rate=0.02).optimize(
        tickers, _returns([0.0015, 0.0012, 0.0010, 0.0008, 0.0006])
    )
    assert len(result["efficient_frontier"]) >= 5
    ms, mv = result["max_sharpe"], result["min_variance"]
    assert ms["expected_return"] != mv["expected_return"]
    assert abs(sum(ms["weights"].values()) - 1) < 1e-3
    assert ms["sharpe_ratio"] >= result["equal_weight"]["sharpe_ratio"] - 1e-6
    assert ms["sharpe_ratio"] >= mv["sharpe_ratio"] - 1e-6
    assert result["max_sharpe_note"] is None


def test_no_asset_beats_risk_free_falls_back_to_min_variance():
    # Previously "max Sharpe" maximized volatility here: 100% in one stock.
    tickers = ["A", "B", "C", "D"]
    R = _returns([-0.002, -0.001, -0.0015, -0.003])
    R[:, 3] *= 4  # one very volatile name
    result = MeanVarianceOptimizer(risk_free_rate=0.05).optimize(tickers, R)
    assert result["max_sharpe_note"] is not None
    assert result["max_sharpe"]["weights"] == result["min_variance"]["weights"]
    assert max(result["max_sharpe"]["weights"].values()) < 0.99


def test_large_universe_is_not_rejected_as_singular():
    # det() of a 60-asset daily covariance underflows to 0.0
    n = 60
    R = _returns(list(np.linspace(0.0002, 0.002, n)), n_days=300)
    result = MeanVarianceOptimizer(risk_free_rate=0.02).optimize([f"T{i}" for i in range(n)], R)
    assert result is not None and result["n_assets"] == n


def test_black_litterman_returns_stay_in_total_return_space():
    opt = MeanVarianceOptimizer(risk_free_rate=0.05)
    R = _returns([-0.001, -0.0005, -0.002])  # sample premium is negative
    cov = np.cov(R, rowvar=False) * 252
    mu_bl = opt._black_litterman(["A", "B", "C"], R.mean(axis=0) * 252, cov, {"A": 0.10})
    # risk aversion is clamped positive, so equilibrium excess returns are > 0
    # and the result includes the risk-free rate exactly once
    assert np.all(mu_bl > 0.05)
    assert mu_bl[0] == max(mu_bl)


def _ticker(name, vol):
    return {
        "ticker": name, "verdict": "BUY", "buy_score": 70, "sell_score": 30,
        "models": {"garch": {"current_conditional_vol_annual": vol}},
    }


def test_position_cap_holds_for_any_number_of_positions():
    for vols in ([0.3], [0.3, 0.3], [0.3] * 3, [0.1, 0.4, 0.4, 0.4, 0.5], [0.2] * 8):
        positions = compute_position_sizes([_ticker(f"T{i}", v) for i, v in enumerate(vols)])
        weights = [p["weight_pct"] / 100 for p in positions]
        assert max(weights) <= MAX_SINGLE_WEIGHT + 1e-9, vols
        assert sum(weights) <= 1 + 1e-9


def test_capped_excess_goes_to_uncapped_names_then_cash():
    two = compute_position_sizes([_ticker("A", 0.3), _ticker("B", 0.3)])
    assert sum(p["weight_pct"] for p in two) == 60.0  # 40% stays in cash
    five = compute_position_sizes([_ticker(f"T{i}", v) for i, v in enumerate([0.1, 0.4, 0.4, 0.4, 0.5])])
    assert abs(sum(p["weight_pct"] for p in five) - 100.0) < 0.05


def test_frontier_is_a_smooth_curve_not_a_zigzag():
    # 29 assets, like a real cycle. The unscaled solver used to stop early
    # and return a jagged, non-convex set of points.
    rng = np.random.default_rng(5)
    n = 29
    R = rng.normal(rng.normal(0.0008, 0.0012, n), rng.uniform(0.01, 0.04, n), size=(252, n))
    result = MeanVarianceOptimizer(risk_free_rate=0.03).optimize([f"T{i}" for i in range(n)], R)
    rets = np.array([p["return"] for p in result["efficient_frontier"]])
    vols = np.array([p["volatility"] for p in result["efficient_frontier"]])
    assert len(rets) >= 8
    assert np.all(np.diff(rets) > 0)
    # efficient part only: more return always costs more risk...
    assert np.all(np.diff(vols) >= -1e-4)
    # ...at an increasing rate (convex in return)
    assert np.all(np.diff(vols, 2) >= -2e-3)
    # the frontier starts at the minimum-variance portfolio
    assert vols[0] == pytest.approx(result["min_variance"]["volatility"], abs=2e-3)
    # the tangency portfolio lies on or inside it
    ms = result["max_sharpe"]
    assert ms["volatility"] >= vols[0] - 1e-3
