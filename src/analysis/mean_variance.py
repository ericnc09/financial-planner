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

# Above this the covariance is numerically singular for the optimizer.
MAX_CONDITION_NUMBER = 1e12


class MeanVarianceOptimizer:
    def __init__(self, risk_free_rate: float = 0.05):
        """risk_free_rate: annualized (e.g. 0.05 = 5%)."""
        self.rf = risk_free_rate / 252  # daily

    def optimize(
        self,
        tickers: list[str],
        returns_matrix: np.ndarray,
        views: dict[str, float] | None = None,
        market_caps: dict[str, float] | None = None,
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

        mu_sample = np.mean(returns_matrix, axis=0)
        # Ledoit-Wolf shrinkage: stabilizes covariance estimate for small samples
        lw = LedoitWolf().fit(returns_matrix)
        cov = lw.covariance_

        # Condition number, not determinant: det() of a daily-return covariance
        # underflows to exactly 0.0 for a few dozen assets even when the matrix
        # is perfectly well conditioned.
        if not np.all(np.isfinite(cov)) or np.linalg.cond(cov) > MAX_CONDITION_NUMBER:
            logger.warning("meanvar.singular_covariance", n_assets=n_assets)
            return None

        # Bayes-Stein shrinkage of sample means toward the grand mean
        # (Jorion, 1986). Reduces estimation error in expected returns.
        mu = self._james_stein_shrink(mu_sample, cov, n_days)

        # Annualize
        mu_ann = mu * 252
        cov_ann = cov * 252

        # --- Black-Litterman adjusted returns (if views provided) ---
        if views:
            mu_bl = self._black_litterman(tickers, mu_ann, cov_ann, views, market_caps)
            if mu_bl is not None:
                mu_ann = mu_bl
                logger.info("meanvar.black_litterman_applied", n_views=len(views))

        # --- Minimum Variance Portfolio ---
        min_var_w = self._min_variance(cov_ann, n_assets)

        # --- Maximum Sharpe (Tangency) Portfolio ---
        # A long-only tangency portfolio only exists when at least one asset is
        # expected to beat the risk-free rate. Otherwise "maximizing" a negative
        # Sharpe ratio maximizes volatility and piles into the riskiest stock,
        # so fall back to minimum variance and say so.
        max_sharpe_note = None
        if np.max(mu_ann) <= self.rf * 252:
            max_sharpe_w = min_var_w
            max_sharpe_note = "no_asset_beats_risk_free_rate_min_variance_used"
            logger.warning("meanvar.no_positive_excess_return", n_assets=n_assets)
        else:
            max_sharpe_w = self._max_sharpe(mu_ann, cov_ann, n_assets)

        # --- Equal Weight baseline ---
        eq_w = np.ones(n_assets) / n_assets

        # --- Efficient Frontier (10 points) ---
        frontier = self._efficient_frontier(
            mu_ann, cov_ann, n_assets, n_points=10, min_var_w=min_var_w
        )

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
            "max_sharpe_note": max_sharpe_note,
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

    def _james_stein_shrink(
        self, mu_sample: np.ndarray, cov: np.ndarray, n_days: int
    ) -> np.ndarray:
        """Bayes-Stein shrinkage of sample means toward the grand mean.

        Jorion (1986): mu_bs = (1 - w) * mu + w * mu0, with

            w = (N + 2) / ((N + 2) + T * (mu - mu0)' Sigma^-1 (mu - mu0))

        where N is the number of assets, T the number of observations and
        Sigma the covariance of returns. The quadratic form is dimensionless
        (a squared Mahalanobis distance), so w is always in (0, 1]: means that
        are far apart relative to their noise are shrunk less.
        """
        p = len(mu_sample)
        if p < 3:
            return mu_sample  # Stein shrinkage needs at least 3 assets
        grand_mean = float(np.mean(mu_sample))
        diff = mu_sample - grand_mean
        distance = float(diff @ np.linalg.solve(cov, diff))
        weight = (p + 2) / ((p + 2) + n_days * max(distance, 0.0))
        mu_js = (1.0 - weight) * mu_sample + weight * grand_mean
        logger.debug("meanvar.james_stein", shrinkage_weight=round(weight, 4))
        return mu_js

    def _min_variance(self, cov: np.ndarray, n: int) -> np.ndarray:
        w0 = np.ones(n) / n
        bounds = [(0, 1)] * n
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        # Variance in annualized units is ~1e-2..1e0; scale so SLSQP's default
        # tolerance is meaningful for low-volatility universes.
        scale = 1.0 / max(float(np.mean(np.diag(cov))), 1e-12)
        res = minimize(
            lambda w: scale * (w @ cov @ w), w0, jac=lambda w: 2 * scale * (cov @ w),
            method="SLSQP", bounds=bounds, constraints=cons,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        return self._clean_weights(res.x) if res.success else w0

    def _max_sharpe(self, mu: np.ndarray, cov: np.ndarray, n: int) -> np.ndarray:
        """Long-only tangency portfolio.

        Solved in its convex form: minimize y' Sigma y subject to
        (mu - rf)' y = 1, y >= 0, then w = y / sum(y). Unlike maximizing the
        Sharpe ratio directly this has a single optimum, so the answer does
        not depend on the starting point. Caller guarantees max(mu) > rf.
        """
        excess = mu - self.rf * 252
        best = int(np.argmax(excess))
        y0 = np.zeros(n)
        y0[best] = 1.0 / excess[best]  # feasible start
        scale = 1.0 / max(float(np.mean(np.diag(cov))), 1e-12)
        cons = [{"type": "eq", "fun": lambda y: excess @ y - 1.0, "jac": lambda y: excess}]
        res = minimize(
            lambda y: scale * (y @ cov @ y), y0, jac=lambda y: 2 * scale * (cov @ y),
            method="SLSQP", bounds=[(0, None)] * n, constraints=cons,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if not res.success or np.sum(res.x) <= 0:
            logger.warning("meanvar.max_sharpe_failed", message=str(res.message))
            return self._min_variance(cov, n)
        return self._clean_weights(res.x / np.sum(res.x))

    @staticmethod
    def _clean_weights(w: np.ndarray) -> np.ndarray:
        """Zero out solver noise and renormalize to sum to 1."""
        w = np.where(w < 1e-6, 0.0, w)
        total = float(np.sum(w))
        return w / total if total > 0 else np.ones_like(w) / len(w)

    def _efficient_frontier(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        n: int,
        n_points: int = 10,
        min_var_w: np.ndarray | None = None,
    ) -> list[dict]:
        """Minimum-volatility portfolio for each target return.

        Targets run from the minimum-variance portfolio's return up to the
        best single asset: below the minimum-variance point a portfolio has
        both less return and more risk, so it is not on the efficient part.
        """
        max_ret = float(np.max(mu))
        if min_var_w is not None:
            min_ret = float(min_var_w @ mu)
        else:
            min_ret = float(np.min(mu))
        if max_ret - min_ret < 1e-9:
            return []

        target_returns = np.linspace(min_ret, max_ret, n_points)
        frontier = []
        # Same scaling as _min_variance: annualized variances are small enough
        # that SLSQP's default tolerance would stop it well short of the
        # optimum and leave a jagged "frontier".
        scale = 1.0 / max(float(np.mean(np.diag(cov))), 1e-12)
        w0 = min_var_w if min_var_w is not None else np.ones(n) / n

        for target in target_returns:
            bounds = [(0, 1)] * n
            cons = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1, "jac": lambda w: np.ones(n)},
                {"type": "eq", "fun": lambda w, t=target: w @ mu - t, "jac": lambda w: mu},
            ]
            res = minimize(
                lambda w: scale * (w @ cov @ w), w0, jac=lambda w: 2 * scale * (cov @ w),
                method="SLSQP", bounds=bounds, constraints=cons,
                options={"maxiter": 500, "ftol": 1e-10},
            )
            if res.success:
                vol = float(np.sqrt(res.x @ cov @ res.x))
                frontier.append({"return": round(float(target), 4), "volatility": round(vol, 4)})
                w0 = res.x  # warm-start the next, slightly higher target

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
        market_caps: dict[str, float] | None = None,
    ) -> np.ndarray | None:
        """
        Black-Litterman model: blend equilibrium returns with investor views.

        Args:
            tickers: asset labels
            mu_ann: annualized sample mean returns (N,)
            cov_ann: annualized covariance matrix (N, N)
            views: ticker -> expected excess return (annualized)
            market_caps: ticker -> market capitalization for prior weights.
                         Falls back to equal weight if unavailable.

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

        # Market-cap weights for equilibrium prior (or equal weight fallback)
        if market_caps:
            caps = np.array([market_caps.get(t, 0.0) for t in tickers])
            total_cap = caps.sum()
            w_mkt = caps / total_cap if total_cap > 0 else np.ones(n) / n
        else:
            w_mkt = np.ones(n) / n

        # Risk aversion implied from market: delta = (E[R_m] - R_f) / sigma_m^2.
        # A sample-mean estimate can be negative or absurdly large over a short
        # window; a negative delta would invert every equilibrium return, so
        # keep it in the range used in practice (He & Litterman use 2.5).
        port_var = float(w_mkt @ cov_ann @ w_mkt)
        port_ret = float(w_mkt @ mu_ann)
        rf_ann = self.rf * 252
        delta = (port_ret - rf_ann) / port_var if port_var > 0 else 2.5
        if not np.isfinite(delta):
            delta = 2.5
        delta = float(np.clip(delta, 1.0, 10.0))

        # Equilibrium EXCESS returns: pi = delta * Sigma * w_mkt
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
        mu_bl_excess = posterior_cov @ (tau_cov_inv @ pi + P.T @ omega_inv @ Q)

        # pi and the views are excess returns; everything downstream expects
        # total returns and subtracts the risk-free rate itself. Returning the
        # excess figures would subtract it twice.
        return mu_bl_excess + rf_ann
