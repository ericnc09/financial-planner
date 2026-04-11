"""
Bayesian Signal Decay — Estimates how quickly smart money alpha decays after a signal.

Uses exponential decay model with Gamma prior on the decay rate.
Key outputs:
- Posterior half-life (days until signal alpha halves)
- Signal strength remaining at key horizons
- Optimal entry/exit windows
- Decay quality classification
"""

import numpy as np
import structlog
from scipy import stats

logger = structlog.get_logger()


# Regime-conditional prior half-lives (days).
# Alpha decays faster in bear/recession environments.
REGIME_HALF_LIFE = {
    "expansion": 22.0,
    "transition": 15.0,
    "recession": 8.0,
    "bull": 22.0,
    "mild_bull": 18.0,
    "sideways": 13.0,
    "mild_bear": 9.0,
    "bear": 7.0,
}


class BayesianDecayAnalyzer:
    def __init__(
        self,
        prior_half_life: float = 15.0,
        prior_strength: float = 2.0,
        regime: str | None = None,
    ):
        """
        Args:
            prior_half_life: Default prior on signal half-life in days.
                             Overridden when `regime` is provided.
            prior_strength: Concentration of the Gamma prior (higher = tighter).
            regime: Market regime string ('expansion', 'transition', 'recession',
                    or HMM state 'bull'/'bear'/'sideways').
                    When provided, adjusts the prior half-life accordingly.
        """
        if regime and regime in REGIME_HALF_LIFE:
            effective_hl = REGIME_HALF_LIFE[regime]
            logger.info("bayesian_decay.regime_prior", regime=regime, half_life=effective_hl)
        else:
            effective_hl = prior_half_life
        self.prior_lambda = np.log(2) / effective_hl
        self.prior_strength = prior_strength
        self.regime = regime
        self.rng = np.random.default_rng(42)

    def analyze(
        self,
        abnormal_returns: np.ndarray,
        direction: str = "buy",
    ) -> dict | None:
        """
        Fit Bayesian exponential decay to post-event abnormal returns.

        Args:
            abnormal_returns: Daily abnormal returns (sign-adjusted: positive = alpha).
            direction: 'buy' or 'sell'.
        """
        if len(abnormal_returns) < 5:
            return None

        ar = np.array(abnormal_returns, dtype=float)
        n_days = len(ar)
        days = np.arange(1, n_days + 1)
        car = np.cumsum(ar)
        total_car = float(car[-1])

        # Smoothed absolute AR for decay fitting
        window = min(3, n_days)
        smoothed = np.convolve(np.abs(ar), np.ones(window) / window, mode="valid")
        s_days = days[:len(smoothed)]

        pos_mask = smoothed > 0
        if np.sum(pos_mask) < 3:
            return self._no_signal(direction, n_days)

        # Log-linear regression for MLE initialization
        log_ar = np.log(smoothed[pos_mask])
        t_pos = s_days[pos_mask]
        X = np.column_stack([np.ones(len(t_pos)), t_pos])
        coeffs, _, _, _ = np.linalg.lstsq(X, log_ar, rcond=None)
        mle_A = float(np.exp(coeffs[0]))
        mle_lambda = max(-coeffs[1], 0.001)

        # MCMC posterior sampling for (A, lambda, sigma)
        # Model: |AR(t)| ~ N(A * exp(-lambda * t), sigma^2)
        post_lambda, mle_A, hl_samples, samples = self._mcmc_posterior(
            smoothed[pos_mask], t_pos, mle_A, mle_lambda
        )
        post_half_life = float(np.log(2) / post_lambda)

        ci_lambda = (round(float(np.percentile(samples, 2.5)), 4),
                     round(float(np.percentile(samples, 97.5)), 4))
        ci_hl = (round(float(np.percentile(hl_samples, 2.5)), 1),
                 round(float(np.percentile(hl_samples, 97.5)), 1))

        # Signal strength at horizons
        strength = {}
        for h in [1, 5, 10, 20, 40]:
            remaining = float(np.exp(-post_lambda * h))
            prob_active = float(np.mean(hl_samples > h))
            strength[h] = {"remaining_pct": round(remaining * 100, 1), "prob_active": round(prob_active, 4)}

        # Entry/exit windows
        entry_days = round(float(-np.log(0.70) / post_lambda), 1)
        exit_days = round(float(-np.log(0.30) / post_lambda), 1)

        # Information ratio
        ir = float(np.mean(ar) / np.std(ar, ddof=1)) * np.sqrt(252) if len(ar) > 1 and np.std(ar, ddof=1) > 0 else 0.0

        result = {
            "direction": direction,
            "regime": self.regime,
            "n_days": n_days,
            "total_car": round(total_car, 6),
            "mle_amplitude": round(mle_A, 6),
            "mle_lambda": round(float(mle_lambda), 4),
            "posterior_lambda": round(float(post_lambda), 4),
            "posterior_half_life": round(post_half_life, 1),
            "ci_lambda_95": ci_lambda,
            "ci_half_life_95": ci_hl,
            "signal_strength": strength,
            "entry_window_days": entry_days,
            "exit_window_days": exit_days,
            "annualized_ir": round(ir, 4),
            "decay_quality": self._classify(post_half_life, total_car),
        }

        logger.info(
            "bayesian_decay.analyzed",
            direction=direction, half_life=round(post_half_life, 1),
            total_car=round(total_car, 6), quality=result["decay_quality"],
        )
        return result

    def _mcmc_posterior(
        self,
        obs: np.ndarray,
        t: np.ndarray,
        init_A: float,
        init_lambda: float,
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        """
        MCMC sampling of exponential decay parameters via emcee.

        Model: obs[i] ~ N(A * exp(-lambda * t[i]), sigma^2)
        Priors: A ~ HalfNormal(scale=init_A*3), lambda ~ Gamma(prior_strength, prior_lambda),
                sigma ~ HalfCauchy(scale=0.01)

        Falls back to MLE + Gamma approximation if emcee unavailable.
        """
        try:
            import emcee
        except ImportError:
            # Fallback: use Gamma posterior approximation (original approach, corrected)
            return self._gamma_fallback(obs, t, init_A, init_lambda)

        prior_a = self.prior_strength
        prior_rate = self.prior_strength / self.prior_lambda
        init_sigma = float(np.std(obs - init_A * np.exp(-init_lambda * t)))

        def log_prior(theta):
            A, lam, sig = theta
            if A <= 0 or lam <= 0 or sig <= 0:
                return -np.inf
            # Gamma prior on lambda
            lp = (prior_a - 1) * np.log(lam) - prior_rate * lam
            # HalfNormal on A
            lp += -0.5 * (A / (init_A * 3)) ** 2
            # HalfCauchy on sigma
            lp += -np.log(1 + (sig / 0.01) ** 2)
            return lp

        def log_likelihood(theta):
            A, lam, sig = theta
            model = A * np.exp(-lam * t)
            resid = obs - model
            return -0.5 * np.sum((resid / sig) ** 2) - len(obs) * np.log(sig)

        def log_prob(theta):
            lp = log_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(theta)

        ndim, nwalkers = 3, 16
        p0 = np.array([init_A, init_lambda, max(init_sigma, 0.001)])
        # Initialize walkers in a small ball around MLE
        pos = p0 + 1e-4 * self.rng.standard_normal((nwalkers, ndim))
        pos[:, 0] = np.abs(pos[:, 0])  # A > 0
        pos[:, 1] = np.abs(pos[:, 1])  # lambda > 0
        pos[:, 2] = np.abs(pos[:, 2])  # sigma > 0

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
        sampler.run_mcmc(pos, 1000, progress=False)

        # Discard burn-in, thin
        flat = sampler.get_chain(discard=500, thin=5, flat=True)
        A_samples = flat[:, 0]
        lambda_samples = flat[:, 1]

        post_lambda = float(np.median(lambda_samples))
        post_A = float(np.median(A_samples))
        hl_samples = np.log(2) / np.maximum(lambda_samples, 0.001)

        return post_lambda, post_A, hl_samples, lambda_samples

    def _gamma_fallback(
        self,
        obs: np.ndarray,
        t: np.ndarray,
        init_A: float,
        init_lambda: float,
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        """Corrected Gamma conjugate approximation when emcee not available."""
        a0 = self.prior_strength
        b0 = self.prior_strength / self.prior_lambda
        n = len(obs)
        # Sufficient statistic: sum of squared residuals weighted by time
        model = init_A * np.exp(-init_lambda * t)
        residuals = obs - model
        mse = float(np.mean(residuals ** 2))
        # Posterior shape/rate for lambda
        a_post = a0 + n / 2
        b_post = b0 + n * mse / (2 * max(init_A ** 2, 1e-10))

        post_lambda = float(a_post / b_post)
        samples = self.rng.gamma(a_post, 1 / b_post, 5000)
        samples = np.maximum(samples, 0.001)
        hl_samples = np.log(2) / samples
        return post_lambda, init_A, hl_samples, samples

    def _no_signal(self, direction: str, n_days: int) -> dict:
        return {
            "direction": direction, "n_days": n_days, "total_car": 0.0,
            "mle_amplitude": 0.0, "mle_lambda": 0.0,
            "posterior_lambda": round(float(self.prior_lambda), 4),
            "posterior_half_life": round(float(np.log(2) / self.prior_lambda), 1),
            "ci_lambda_95": (0.0, 0.0), "ci_half_life_95": (0.0, 0.0),
            "signal_strength": {h: {"remaining_pct": 0.0, "prob_active": 0.0} for h in [1, 5, 10, 20, 40]},
            "entry_window_days": 0.0, "exit_window_days": 0.0,
            "annualized_ir": 0.0, "decay_quality": "no_signal",
        }

    def _classify(self, half_life: float, total_car: float) -> str:
        if total_car <= 0:
            return "no_alpha"
        if half_life < 3:
            return "flash"
        if half_life < 10:
            return "fast_decay"
        if half_life < 25:
            return "moderate_decay"
        return "slow_decay"
