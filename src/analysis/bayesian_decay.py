"""
Bayesian Signal Decay — how quickly smart-money alpha fades after a signal
becomes public.

Model, for t = 0, 1, 2, ... trading days after disclosure:

    AR(t) ~ Normal(A * exp(-lambda * t), sigma^2)

fitted to a series of SIGNED abnormal returns (direction-adjusted, so a
positive value is alpha for someone following the signal). A is the day-0
abnormal return and may be negative; lambda is the decay rate.

The series should be an average over many comparable past events (see
src/analysis/pooled_alpha.py). A single event's abnormal returns are almost
entirely noise, and fitting the decay to their absolute value — as an earlier
version did — measures how quickly volatility settles, not alpha.

Key outputs:
- Posterior half-life (days until the daily alpha halves)
- Probability that the alpha is positive at all
- Share of the alpha remaining at key horizons
- Decay quality classification
"""

import numpy as np
import structlog

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

HORIZONS = (1, 5, 10, 20, 40)


class BayesianDecayAnalyzer:
    def __init__(
        self,
        prior_half_life: float = 15.0,
        prior_strength: float = 2.0,
        regime: str | None = None,
        seed: int = 42,
    ):
        """
        Args:
            prior_half_life: Default prior on signal half-life in days.
                             Overridden when `regime` is provided.
            prior_strength: Concentration of the Gamma prior (higher = tighter).
            regime: Market regime string ('expansion', 'transition', 'recession',
                    or HMM state 'bull'/'bear'/'sideways').
                    When provided, adjusts the prior half-life accordingly.
            seed: Seed for the sampler, so the same input gives the same fit.
        """
        if regime and regime in REGIME_HALF_LIFE:
            effective_hl = REGIME_HALF_LIFE[regime]
            logger.info("bayesian_decay.regime_prior", regime=regime, half_life=effective_hl)
        else:
            effective_hl = prior_half_life
        self.prior_lambda = np.log(2) / effective_hl
        self.prior_strength = prior_strength
        self.regime = regime
        self.seed = seed

    def analyze(
        self,
        abnormal_returns: np.ndarray,
        direction: str = "buy",
    ) -> dict | None:
        """
        Fit the exponential decay to post-disclosure abnormal returns.

        Args:
            abnormal_returns: Signed daily abnormal returns for days 0, 1, 2, ...
                after disclosure (direction-adjusted: positive = alpha).
                Normally the mean across many comparable events.
            direction: 'buy' or 'sell' (label only; the input is already signed).

        Returns:
            Fit summary, or None when there is too little data or the sampler
            is unavailable.
        """
        ar = np.asarray(abnormal_returns, dtype=float)
        ar = ar[np.isfinite(ar)]
        if len(ar) < 5:
            return None

        n_days = len(ar)
        t = np.arange(n_days, dtype=float)
        total_car = float(np.sum(ar))
        scale = float(np.std(ar, ddof=1))
        if scale <= 0:
            return self._no_signal(direction, n_days)

        posterior = self._mcmc_posterior(ar, t, scale)
        if posterior is None:
            return None
        lambda_samples, amplitude_samples = posterior

        post_lambda = float(np.median(lambda_samples))
        post_half_life = float(np.log(2) / post_lambda)
        hl_samples = np.log(2) / lambda_samples
        prob_positive = float(np.mean(amplitude_samples > 0))

        ci_lambda = (round(float(np.percentile(lambda_samples, 2.5)), 4),
                     round(float(np.percentile(lambda_samples, 97.5)), 4))
        ci_hl = (round(float(np.percentile(hl_samples, 2.5)), 1),
                 round(float(np.percentile(hl_samples, 97.5)), 1))

        # Share of the alpha still ahead after h days
        strength = {}
        for h in HORIZONS:
            remaining = float(np.exp(-post_lambda * h))
            prob_active = float(np.mean(hl_samples > h))
            strength[h] = {"remaining_pct": round(remaining * 100, 1), "prob_active": round(prob_active, 4)}

        # Entry/exit windows: days until 70% / 30% of the alpha remains
        entry_days = round(float(-np.log(0.70) / post_lambda), 1)
        exit_days = round(float(-np.log(0.30) / post_lambda), 1)

        result = {
            "direction": direction,
            "regime": self.regime,
            "n_days": n_days,
            "total_car": round(total_car, 6),
            "amplitude": round(float(np.median(amplitude_samples)), 6),
            "prob_alpha_positive": round(prob_positive, 4),
            "posterior_lambda": round(post_lambda, 4),
            "posterior_half_life": round(post_half_life, 1),
            "ci_lambda_95": ci_lambda,
            "ci_half_life_95": ci_hl,
            "signal_strength": strength,
            "entry_window_days": entry_days,
            "exit_window_days": exit_days,
            # No information ratio: the input is an average across events, so
            # its mean/std says nothing about the risk of any one position.
            "annualized_ir": None,
            "decay_quality": self._classify(post_half_life, prob_positive),
        }

        logger.info(
            "bayesian_decay.analyzed",
            direction=direction, half_life=round(post_half_life, 1),
            total_car=round(total_car, 6), prob_alpha_positive=round(prob_positive, 3),
            quality=result["decay_quality"],
        )
        return result

    def _mcmc_posterior(
        self,
        obs: np.ndarray,
        t: np.ndarray,
        scale: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Posterior samples of (lambda, A) via emcee.

        Priors: A ~ Normal(0, (3*scale)^2)        (signed, weakly informative)
                lambda ~ Gamma(prior_strength, rate = prior_strength / prior_lambda)
                sigma ~ HalfCauchy(scale)

        There is no closed-form fallback: the Gamma "conjugate" update used
        previously was the posterior of a precision, not of a decay rate.
        """
        try:
            import emcee
        except ImportError:
            logger.warning("bayesian_decay.emcee_missing", hint="pip install emcee")
            return None

        prior_a = self.prior_strength
        prior_rate = self.prior_strength / self.prior_lambda
        amp_scale = 3.0 * scale

        def log_prob(theta):
            A, lam, sig = theta
            if lam <= 0 or sig <= 0:
                return -np.inf
            lp = (prior_a - 1) * np.log(lam) - prior_rate * lam          # Gamma on lambda
            lp += -0.5 * (A / amp_scale) ** 2                             # Normal on A
            lp += -np.log(1 + (sig / scale) ** 2)                         # HalfCauchy on sigma
            resid = obs - A * np.exp(-lam * t)
            return lp - 0.5 * np.sum((resid / sig) ** 2) - len(obs) * np.log(sig)

        # Start at the least-squares amplitude for the prior decay rate
        basis = np.exp(-self.prior_lambda * t)
        init_A = float(obs @ basis / (basis @ basis))
        rng = np.random.default_rng(self.seed)
        ndim, nwalkers = 3, 24
        pos = np.column_stack([
            init_A + 0.1 * scale * rng.standard_normal(nwalkers),
            self.prior_lambda * np.exp(0.2 * rng.standard_normal(nwalkers)),
            scale * np.exp(0.2 * rng.standard_normal(nwalkers)),
        ])

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
        state = emcee.State(pos, random_state=np.random.RandomState(self.seed).get_state())
        sampler.run_mcmc(state, 1500, progress=False)

        flat = sampler.get_chain(discard=750, thin=5, flat=True)
        lambda_samples = np.maximum(flat[:, 1], 1e-4)
        return lambda_samples, flat[:, 0]

    def _no_signal(self, direction: str, n_days: int) -> dict:
        return {
            "direction": direction, "regime": self.regime, "n_days": n_days, "total_car": 0.0,
            "amplitude": 0.0, "prob_alpha_positive": 0.5,
            "posterior_lambda": round(float(self.prior_lambda), 4),
            "posterior_half_life": round(float(np.log(2) / self.prior_lambda), 1),
            "ci_lambda_95": (0.0, 0.0), "ci_half_life_95": (0.0, 0.0),
            "signal_strength": {h: {"remaining_pct": 0.0, "prob_active": 0.0} for h in HORIZONS},
            "entry_window_days": 0.0, "exit_window_days": 0.0,
            "annualized_ir": None, "decay_quality": "no_signal",
        }

    def _classify(self, half_life: float, prob_alpha_positive: float) -> str:
        if prob_alpha_positive < 0.5:
            return "no_alpha"
        if half_life < 3:
            return "flash"
        if half_life < 10:
            return "fast_decay"
        if half_life < 25:
            return "moderate_decay"
        return "slow_decay"


def decay_component(fit: dict | None, age_trading_days: float) -> dict | None:
    """Ensemble input for one signal: how much of the pooled alpha is left.

    Args:
        fit: Output of BayesianDecayAnalyzer.analyze() for the signal's
             source and direction.
        age_trading_days: Trading days since the signal was disclosed.
    """
    if not fit or fit.get("decay_quality") == "no_signal":
        return None
    lam = fit.get("posterior_lambda")
    prob_positive = fit.get("prob_alpha_positive")
    if lam is None or prob_positive is None:
        return None
    age = max(0.0, float(age_trading_days))
    return {
        "remaining_share": round(float(np.exp(-lam * age)), 4),
        "prob_alpha_positive": prob_positive,
        "age_trading_days": round(age, 1),
        "posterior_half_life": fit.get("posterior_half_life"),
        "decay_quality": fit.get("decay_quality"),
    }
