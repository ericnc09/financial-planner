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


class BayesianDecayAnalyzer:
    def __init__(self, prior_half_life: float = 15.0, prior_strength: float = 2.0):
        self.prior_lambda = np.log(2) / prior_half_life
        self.prior_strength = prior_strength
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

        # Log-linear regression: log|AR(t)| = log(A) - lambda*t
        log_ar = np.log(smoothed[pos_mask])
        t_pos = s_days[pos_mask]
        X = np.column_stack([np.ones(len(t_pos)), t_pos])
        coeffs, _, _, _ = np.linalg.lstsq(X, log_ar, rcond=None)
        mle_A = float(np.exp(coeffs[0]))
        mle_lambda = max(-coeffs[1], 0.001)

        # Bayesian update: Gamma prior on lambda
        a0, b0 = self.prior_strength, self.prior_strength / self.prior_lambda
        a_post = a0 + len(t_pos) / 2
        ss = np.sum(t_pos * smoothed[pos_mask] ** 2) / (2 * max(mle_A ** 2, 1e-10))
        b_post = b0 + ss

        post_lambda = a_post / b_post
        post_half_life = float(np.log(2) / post_lambda)

        # Posterior samples for credible intervals
        samples = self.rng.gamma(a_post, 1 / b_post, 5000)
        samples = np.maximum(samples, 0.001)
        hl_samples = np.log(2) / samples

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
