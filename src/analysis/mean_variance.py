"""
Mean-Variance Optimization — Markowitz portfolio analysis for signal tickers.

Given a set of tickers from smart money signals, computes:
- Efficient frontier points
- Minimum variance portfolio weights
- Maximum Sharpe ratio (tangency) portfolio weights
- Per-ticker contribution to portfolio risk
"""

import numpy as np
import structlog
from scipy.optimize import minimize
from scipy.linalg import inv
from sklearn.covariance import LedoitWolf

logger = structlog.get_logger()


class MeanVarianceOptimizer:
    def __init__(self, risk_free_rate: float = 0.05):
        """risk_free_rate: annualized (e.g. 0.05 = 5%)."""
        self.rf = risk_free_rate / 252  # daily

    def optimize(
        self,
        tickers: list[str],
        returns_matrix: np.ndarray,
        views: dict[str, float] | None = None,
    ) -> dict | None:
        """
        Run mean-variance optimization with optional Black-Litterman views.

        Args:
            tickers: List of ticker symbols.
            returns_matrix: (n_days, n_tickers) array of daily returns.
            views: Optional dict of ticker -> expected excess return (annualized)
                   derived from conviction scores. Used for Black-Litterman.

        Returns:
            Dict with optimal weights, frontier, and risk decomposition.
        """
        n_days, n_assets = returns_matrix.shape
        if n_assets < 2 or n_days < 30:
            logger.warning("meanvar.insufficient", n_days=n_days, n_assets=n_assets)
            return None

        mu = np.mean(returns_matrix, axis=0)  # daily expected returns
        # Ledoit-Wolf shrinkage: stabilizes covariance estimate for small samples
        lw = LedoitWolf().fit(returns_matrix)
        cov = lw.covariance_

        if np.any(np.isnan(cov)) or np.linalg.det(cov) == 0:
            logger.warning("meanvar.singular_covariance")
            return None

        # Annualize
        mu_ann = mu * 252
        cov_ann = cov * 252

        # --- Black-Litterman adjusted returns (if views provided) ---
        if views:
            mu_bl = self._black_litterman(tickers, mu_ann, cov_ann, views)
            if mu_bl is not None:
                mu_ann = mu_bl
                logger.info("meanvar.black_litterman_applied", n_views=len(views))

        # --- Minimum Variance Portfolio ---
        min_var_w = self._min_variance(cov_ann, n_assets)

        # --- Maximum Sharpe (Tangency) Portfolio ---
        max_sharpe_w = self._max_sharpe(mu_ann, cov_ann, n_assets)

        # --- Equal Weight baseline ---
        eq_w = np.ones(n_assets) / n_assets

        # --- Efficient Frontier (10 points) ---
        frontier = self._efficient_frontier(mu_ann, cov_ann, n_assets, n_points=10)

        # --- Risk contribution for max-sharpe portfolio ---
        risk_contrib = self._risk_contribution(max_sharpe_w, cov_ann)

        # --- Portfolio metrics ---
        def _metrics(w, label):
            ret = float(w @ mu_ann)
            vol = float(np.sqrt(w @ cov_ann @ w))
            sharpe = (ret - self.rf * 252) / vol if vol > 0 else 0
            return {
                "label": label,
                "weights": {t: round(float(wi), 4) for t, wi in zip(tickers, w)},
                "expected_return": round(ret, 4),
                "volatility": round(vol, 4),
                "sharpe_ratio": round(sharpe, 4),
            }

        result = {
            "n_assets": n_assets,
            "n_days": n_days,
            "tickers": tickers,
            "min_variance": _metrics(min_var_w, "min_variance"),
            "max_sharpe": _metrics(max_sharpe_w, "max_sharpe"),
            "equal_weight": _metrics(eq_w, "equal_weight"),
            "efficient_frontier": frontier,
            "risk_contribution": {t: round(float(rc), 4) for t, rc in zip(tickers, risk_contrib)},
            "correlation_matrix": {
                tickers[i]: {tickers[j]: round(float(cov_ann[i, j] / (np.sqrt(cov_ann[i, i] * cov_ann[j, j]))), 4)
                             for j in range(n_assets)}
                for i in range(n_assets)
            },
        }

        logger.info(
            "meanvar.optimized",
            n_assets=n_assets,
            max_sharpe=result["max_sharpe"]["sharpe_ratio"],
            min_var_vol=result["min_variance"]["volatility"],
        )
        return result

    def _min_variance(self, cov: np.ndarray, n: int) -> np.ndarray:
        w0 = np.ones(n) / n
        bounds = [(0, 1)] * n
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        res = minimize(lambda w: w @ cov @ w, w0, method="SLSQP", bounds=bounds, constraints=cons)
        return res.x if res.success else w0

    def _max_sharpe(self, mu: np.ndarray, cov: np.ndarray, n: int) -> np.ndarray:
        rf_ann = self.rf * 252
        w0 = np.ones(n) / n
        bounds = [(0, 1)] * n
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        def neg_sharpe(w):
            ret = w @ mu
            vol = np.sqrt(w @ cov @ w)
            return -(ret - rf_ann) / vol if vol > 1e-10 else 1e10

        res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons)
        return res.x if res.success else w0

    def _efficient_frontier(self, mu: np.ndarray, cov: np.ndarray, n: int, n_points: int = 10) -> list[dict]:
        # Get return range
        min_ret = float(np.min(mu))
        max_ret = float(np.max(mu))
        if min_ret == max_ret:
            return []

        target_returns = np.linspace(min_ret, max_ret, n_points)
        frontier = []

        for target in target_returns:
            w0 = np.ones(n) / n
            bounds = [(0, 1)] * n
            cons = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
            ]
            res = minimize(lambda w: w @ cov @ w, w0, method="SLSQP", bounds=bounds, constraints=cons)
            if res.success:
                vol = float(np.sqrt(res.x @ cov @ res.x))
                frontier.append({"return": round(float(target), 4), "volatility": round(vol, 4)})

        return frontier

    def _risk_contribution(self, w: np.ndarray, cov: np.ndarray) -> np.ndarray:
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol == 0:
            return np.zeros_like(w)
        marginal = cov @ w / port_vol
        rc = w * marginal
        return rc / np.sum(rc)  # normalize to sum to 1

    def _black_litterman(
        self,
        tickers: list[str],
        mu_ann: np.ndarray,
        cov_ann: np.ndarray,
        views: dict[str, float],
    ) -> np.ndarray | None:
        """
        Black-Litterman model: blend equilibrium returns with investor views.

        Uses market-cap-weighted equilibrium as prior (approximated by equal weight
        since we don't have market caps), then tilts toward conviction-based views.

        Args:
            tickers: asset labels
            mu_ann: annualized sample mean returns (N,)
            cov_ann: annualized covariance matrix (N, N)
            views: ticker -> expected excess return (annualized)

        Returns:
            BL-adjusted expected returns (N,) or None if no valid views.
        """
        n = len(tickers)
        ticker_idx = {t: i for i, t in enumerate(tickers)}

        # Build P (pick matrix) and Q (view returns)
        valid_views = [(ticker_idx[t], v) for t, v in views.items() if t in ticker_idx]
        if not valid_views:
            return None

        k = len(valid_views)
        P = np.zeros((k, n))
        Q = np.zeros(k)
        for row, (idx, ret) in enumerate(valid_views):
            P[row, idx] = 1.0
            Q[row] = ret

        # Risk aversion parameter (delta) implied by market
        delta = 2.5  # standard assumption

        # Equilibrium returns: pi = delta * Sigma * w_mkt
        # Approximate market weights as equal weight (no market cap data)
        w_mkt = np.ones(n) / n
        pi = delta * cov_ann @ w_mkt

        # Tau: scaling factor for uncertainty in equilibrium (typical 0.025-0.05)
        tau = 0.05

        # Omega: uncertainty in views (proportional to variance of each view asset)
        omega = np.diag([tau * P[i] @ cov_ann @ P[i].T for i in range(k)])

        # BL posterior: mu_BL = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1
        #                       * [(tau*Sigma)^-1 * pi + P'*Omega^-1*Q]
        tau_cov_inv = inv(tau * cov_ann)
        omega_inv = inv(omega)

        posterior_cov = inv(tau_cov_inv + P.T @ omega_inv @ P)
        mu_bl = posterior_cov @ (tau_cov_inv @ pi + P.T @ omega_inv @ Q)

        return mu_bl
