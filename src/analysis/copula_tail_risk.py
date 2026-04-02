"""
Copula Tail Risk — Models joint tail dependence between a stock and the market.

Fits a Student-t copula to measure how likely extreme losses co-occur.
Key outputs:
- Lower/upper tail dependence coefficients (lambda)
- VaR and CVaR (Expected Shortfall) at 95%/99%
- Conditional VaR under market stress
- Composite tail risk score (0-100)
"""

import numpy as np
import structlog
from scipy import stats
from scipy.optimize import minimize_scalar

logger = structlog.get_logger()


class CopulaTailRisk:
    def analyze(
        self,
        stock_returns: np.ndarray,
        market_returns: np.ndarray,
    ) -> dict | None:
        n = min(len(stock_returns), len(market_returns))
        if n < 60:
            logger.warning("copula.insufficient_data", n=n)
            return None

        stock = stock_returns[-n:].astype(float)
        market = market_returns[-n:].astype(float)

        mask = np.isfinite(stock) & np.isfinite(market)
        stock, market = stock[mask], market[mask]
        if len(stock) < 60:
            return None

        # Pseudo-observations via empirical CDF
        u_stock = self._rank_cdf(stock)
        u_market = self._rank_cdf(market)

        # Gaussian copula correlation
        z1 = stats.norm.ppf(np.clip(u_stock, 0.001, 0.999))
        z2 = stats.norm.ppf(np.clip(u_market, 0.001, 0.999))
        rho_gauss = float(np.corrcoef(z1, z2)[0, 1])

        # Fit Student-t copula (rho, nu)
        t_rho, t_nu = self._fit_t_copula(u_stock, u_market)

        # Tail dependence: lambda = 2 * t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))
        lambda_tail = self._t_tail_dependence(t_rho, t_nu)

        # Empirical tail concentration at key quantiles
        tail_conc = {}
        for q in [0.05, 0.10]:
            joint = float(np.mean((u_stock <= q) & (u_market <= q)))
            tail_conc[f"lower_{int(q*100)}pct"] = round(joint / q, 4) if q > 0 else 0
        for q in [0.90, 0.95]:
            joint = float(np.mean((u_stock >= q) & (u_market >= q)))
            tail_conc[f"upper_{int(q*100)}pct"] = round(joint / (1 - q), 4)

        # Marginal risk metrics
        var_95 = float(np.percentile(stock, 5))
        var_99 = float(np.percentile(stock, 1))
        cvar_95 = float(np.mean(stock[stock <= var_95])) if np.any(stock <= var_95) else var_95
        cvar_99 = float(np.mean(stock[stock <= var_99])) if np.any(stock <= var_99) else var_99

        # Joint crash probability
        joint_crash = float(np.mean(
            (stock <= np.percentile(stock, 5)) & (market <= np.percentile(market, 5))
        ))
        tail_dep_ratio = round(joint_crash / 0.0025, 2)  # vs independence (0.05*0.05)

        # Conditional VaR under market stress (worst 10%)
        stress_mask = market <= np.percentile(market, 10)
        if np.sum(stress_mask) >= 5:
            cond_var_95 = float(np.percentile(stock[stress_mask], 5))
            stressed = stock[stress_mask]
            cond_cvar_95 = float(np.mean(stressed[stressed <= cond_var_95])) \
                if np.any(stressed <= cond_var_95) else cond_var_95
        else:
            cond_var_95, cond_cvar_95 = var_95, cvar_95

        # Composite tail risk score 0-100
        tail_score = self._score(lambda_tail, tail_dep_ratio, cvar_95, cond_cvar_95, var_95)

        result = {
            "n_observations": len(stock),
            "gaussian_rho": round(rho_gauss, 4),
            "student_t_rho": round(t_rho, 4),
            "student_t_nu": round(t_nu, 2),
            "tail_dep_lower": round(lambda_tail, 4),
            "tail_dep_upper": round(lambda_tail, 4),
            "joint_crash_prob": round(joint_crash, 6),
            "tail_dep_ratio": tail_dep_ratio,
            "tail_concentrations": tail_conc,
            "var_95": round(var_95, 6),
            "var_99": round(var_99, 6),
            "cvar_95": round(cvar_95, 6),
            "cvar_99": round(cvar_99, 6),
            "conditional_var_95": round(cond_var_95, 6),
            "conditional_cvar_95": round(cond_cvar_95, 6),
            "tail_risk_score": round(tail_score, 1),
        }

        logger.info(
            "copula.analyzed",
            n=len(stock), t_rho=round(t_rho, 4), t_nu=round(t_nu, 2),
            lambda_L=round(lambda_tail, 4), score=round(tail_score, 1),
        )
        return result

    def _rank_cdf(self, x: np.ndarray) -> np.ndarray:
        return stats.rankdata(x) / (len(x) + 1)

    def _fit_t_copula(self, u1: np.ndarray, u2: np.ndarray) -> tuple[float, float]:
        def neg_ll(nu):
            nu = max(nu, 2.1)
            t1 = stats.t.ppf(np.clip(u1, 0.001, 0.999), df=nu)
            t2 = stats.t.ppf(np.clip(u2, 0.001, 0.999), df=nu)
            rho = float(np.clip(np.corrcoef(t1, t2)[0, 1], -0.999, 0.999))
            det = 1 - rho ** 2
            if det <= 0:
                return 1e10
            quad = (t1 ** 2 - 2 * rho * t1 * t2 + t2 ** 2) / det
            n = len(t1)
            ll = (
                n * np.log(1 / (2 * np.pi * np.sqrt(det)))
                + (-(nu + 2) / 2) * np.sum(np.log(1 + quad / nu))
                - (-(nu + 1) / 2) * np.sum(np.log(1 + t1 ** 2 / nu))
                - (-(nu + 1) / 2) * np.sum(np.log(1 + t2 ** 2 / nu))
            )
            return -ll

        try:
            res = minimize_scalar(neg_ll, bounds=(2.1, 50), method="bounded")
            best_nu = max(res.x, 2.1)
        except Exception:
            best_nu = 5.0

        t1 = stats.t.ppf(np.clip(u1, 0.001, 0.999), df=best_nu)
        t2 = stats.t.ppf(np.clip(u2, 0.001, 0.999), df=best_nu)
        best_rho = float(np.clip(np.corrcoef(t1, t2)[0, 1], -0.999, 0.999))
        return best_rho, best_nu

    def _t_tail_dependence(self, rho: float, nu: float) -> float:
        if nu <= 0 or abs(rho) >= 1:
            return 0.0
        arg = -np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
        return float(2 * stats.t.cdf(arg, df=nu + 1))

    def _score(self, lam: float, ratio: float, cvar: float, cond_cvar: float, var: float) -> float:
        s1 = min(30, lam * 100)
        s2 = min(25, max(0, (ratio - 1) * 10))
        s3 = min(25, (abs(cvar / var) - 1) * 50) if var != 0 else 0
        s4 = min(20, max(0, (abs(cond_cvar / cvar) - 1) * 40)) if cvar != 0 else 0
        return max(0, min(100, s1 + s2 + s3 + s4))
