"""
Monte Carlo Simulation using Geometric Brownian Motion (GBM).

Simulates N price paths to estimate probability distributions of future prices.
Uses historical drift (mu) and volatility (sigma) from log returns.
"""

import numpy as np
import structlog

logger = structlog.get_logger()


class MonteCarloSimulator:
    def __init__(self, n_simulations: int = 10_000, seed: int | None = None):
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(seed)

    def simulate(
        self,
        closes: np.ndarray,
        horizons: list[int] | None = None,
    ) -> dict | None:
        """
        Run GBM Monte Carlo on historical close prices.

        Args:
            closes: Array of historical closing prices (oldest first).
            horizons: Forecast horizons in trading days. Default [21, 63, 126] (~1/3/6 months).

        Returns:
            Dict with per-horizon results: percentiles, probability of profit, expected return.
        """
        if horizons is None:
            horizons = [21, 63, 126]

        if len(closes) < 30:
            logger.warning("monte_carlo.insufficient_data", n_prices=len(closes))
            return None

        # Compute parameters from historical log returns
        log_returns = np.diff(np.log(closes))
        mu = np.mean(log_returns)          # daily drift
        sigma = np.std(log_returns, ddof=1)  # daily volatility
        current_price = float(closes[-1])

        if sigma == 0 or np.isnan(sigma):
            logger.warning("monte_carlo.zero_volatility")
            return None

        results = {
            "current_price": current_price,
            "annual_drift": round(float(mu * 252), 4),
            "annual_volatility": round(float(sigma * np.sqrt(252)), 4),
            "n_simulations": self.n_simulations,
            "horizons": {},
        }

        max_horizon = max(horizons)

        # Generate all random walks at once (vectorized)
        # GBM: S(t) = S(0) * exp((mu - sigma^2/2)*t + sigma*W(t))
        z = self.rng.standard_normal((self.n_simulations, max_horizon))
        daily_returns = (mu - 0.5 * sigma**2) + sigma * z
        cum_returns = np.cumsum(daily_returns, axis=1)
        price_paths = current_price * np.exp(cum_returns)

        for h in horizons:
            terminal_prices = price_paths[:, h - 1]
            returns_pct = (terminal_prices - current_price) / current_price

            percentiles = {
                "p10": round(float(np.percentile(terminal_prices, 10)), 2),
                "p25": round(float(np.percentile(terminal_prices, 25)), 2),
                "p50": round(float(np.percentile(terminal_prices, 50)), 2),
                "p75": round(float(np.percentile(terminal_prices, 75)), 2),
                "p90": round(float(np.percentile(terminal_prices, 90)), 2),
            }

            results["horizons"][h] = {
                "days": h,
                "percentiles": percentiles,
                "probability_of_profit": round(float(np.mean(terminal_prices > current_price)), 4),
                "expected_return": round(float(np.mean(returns_pct)), 4),
                "max_drawdown_median": round(float(self._median_max_drawdown(price_paths[:, :h])), 4),
                "value_at_risk_95": round(float(np.percentile(returns_pct, 5)), 4),
            }

        logger.info(
            "monte_carlo.complete",
            price=current_price,
            vol=results["annual_volatility"],
            horizons=list(results["horizons"].keys()),
        )
        return results

    def _median_max_drawdown(self, paths: np.ndarray) -> float:
        """Compute median max drawdown across all simulation paths."""
        running_max = np.maximum.accumulate(paths, axis=1)
        drawdowns = (running_max - paths) / running_max
        max_dd_per_path = np.max(drawdowns, axis=1)
        return float(np.median(max_dd_per_path))
