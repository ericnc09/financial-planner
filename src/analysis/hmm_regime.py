"""
Hidden Markov Model (HMM) for market regime detection.

Fits models with n_states=2..5 and selects by BIC. Learns hidden states
(bull/bear/sideways) from observable features: returns, volatility, and
volume changes. Outputs current regime probabilities and transition matrix.

Whatever number of states BIC selects, the published probabilities are always
folded into three buckets (bull / bear / sideways) that sum to 1, so every
consumer sees the same quantities regardless of model order.

A state is published as bull or bear only when its own mean daily return is
distinguishable from zero. The states an HMM finds on daily returns usually
differ in volatility far more than in drift; naming them "bull" and "bear" by
simply ranking their means — as an earlier version did — let the sign of a
difference well inside the noise decide the label, so the same ticker came
out bull or bear depending on the random start.
"""

import asyncio
import logging

import numpy as np
import structlog

logger = structlog.get_logger()

# hmmlearn reports every non-monotone EM step on the stdlib logger; the fit
# quality is checked explicitly in _fit_one instead.
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

# Map state count to labels (sorted bear -> bull)
_STATE_LABELS = {
    2: ["bear", "bull"],
    3: ["bear", "sideways", "bull"],
    4: ["bear", "sideways", "mild_bull", "bull"],
    5: ["bear", "mild_bear", "sideways", "mild_bull", "bull"],
}

# Fine-grained label -> published bucket
_BUCKET = {
    "bull": "bull", "mild_bull": "bull",
    "bear": "bear", "mild_bear": "bear",
    "sideways": "sideways",
}
BUCKETS = ("bull", "bear", "sideways")

# EM restarts per candidate (different seeds); the best log-likelihood wins.
# One start is not enough: on live tickers a single-start fit disagreed with a
# well-optimised one about a third of the time.
N_RESTARTS = 10
# A state counts as directional when its mean daily return is at least this
# many standard errors from zero (and it has enough days to say so).
DIRECTIONAL_T = 1.5
MIN_STATE_DAYS = 10
# A final EM step that LOWERS the log-likelihood by more than this means the
# fit broke down numerically (degenerate covariance), not that it converged.
EM_REGRESSION_TOL = 1.0


def fold_probabilities(
    labelled: dict[str, float], bucket_of: dict[str, str] | None = None
) -> dict[str, float]:
    """Collapse per-state probabilities into bull/bear/sideways.

    Args:
        labelled: state label -> probability.
        bucket_of: state label -> bucket. Defaults to the name-based mapping
                   (mild_bull -> bull, ...).
    """
    bucket_of = bucket_of or _BUCKET
    folded = {b: 0.0 for b in BUCKETS}
    for label, prob in labelled.items():
        folded[bucket_of.get(label, "sideways")] += float(prob)
    return folded


def directional_bucket(state_returns: np.ndarray) -> tuple[str, float]:
    """(bucket, t-statistic) for one state's daily returns."""
    n = len(state_returns)
    if n < MIN_STATE_DAYS:
        return "sideways", 0.0
    spread = float(np.std(state_returns, ddof=1))
    if spread <= 0:
        return "sideways", 0.0
    t_stat = float(np.mean(state_returns) / (spread / np.sqrt(n)))
    if t_stat >= DIRECTIONAL_T:
        return "bull", t_stat
    if t_stat <= -DIRECTIONAL_T:
        return "bear", t_stat
    return "sideways", t_stat


class HMMRegimeDetector:
    def __init__(self, n_states: int | None = None, candidates: list[int] | None = None):
        """
        Args:
            n_states: Fixed number of states (skips BIC selection).
            candidates: States to try for BIC selection. Default [2,3,4,5].
        """
        self.fixed_n_states = n_states
        self.candidates = candidates or [2, 3, 4, 5]

    async def fit_and_predict(
        self,
        returns: np.ndarray,
        volumes: np.ndarray | None = None,
    ) -> dict | None:
        """
        Fit HMM on observable features and predict current regime.

        Args:
            returns: Log returns array.
            volumes: Volume array (optional, same length as returns).

        Returns:
            Dict with current state, probabilities, transition matrix, state history.
        """
        return await asyncio.to_thread(self._fit_predict_sync, returns, volumes)

    @staticmethod
    def _fit_one(GaussianHMM, features: np.ndarray, n: int, seed: int):
        """Fit one model. Returns (model, log_likelihood, em_ok)."""
        model = GaussianHMM(
            n_components=n,
            covariance_type="full",
            n_iter=100,
            random_state=seed,
            verbose=False,
        )
        model.fit(features)
        history = list(model.monitor_.history)
        em_ok = (
            len(history) >= 2
            and model.monitor_.iter < model.n_iter
            and (history[-1] - history[-2]) > -EM_REGRESSION_TOL
        )
        log_ll = float(model.score(features))
        return model, log_ll, bool(em_ok and np.isfinite(log_ll))

    def _best_fit(self, GaussianHMM, features: np.ndarray, n: int):
        """Best of N_RESTARTS fits for one state count, preferring clean EM runs."""
        fits = []
        for restart in range(N_RESTARTS):
            try:
                fits.append(self._fit_one(GaussianHMM, features, n, seed=42 + restart))
            except Exception:
                continue
        if not fits:
            return None
        clean = [f for f in fits if f[2]]
        return max(clean or fits, key=lambda f: f[1])

    def _fit_predict_sync(
        self,
        returns: np.ndarray,
        volumes: np.ndarray | None,
    ) -> dict | None:
        try:
            from hmmlearn.hmm import GaussianHMM

            returns = np.asarray(returns, dtype=float)
            if not np.all(np.isfinite(returns)):
                # Volumes can no longer be aligned once rows are dropped
                returns = returns[np.isfinite(returns)]
                volumes = None

            if len(returns) < 60:
                logger.warning("hmm.insufficient_data", n=len(returns))
                return None

            # Build feature matrix: [returns, rolling_vol, volume_change]
            window = 20
            rolling_vol = np.array([
                np.std(returns[max(0, i - window):i]) if i >= window else np.std(returns[:i + 1])
                for i in range(len(returns))
            ])

            features = np.column_stack([returns, rolling_vol])

            if volumes is not None and len(volumes) == len(returns) + 1:
                vol_changes = np.diff(np.log(np.asarray(volumes, dtype=float) + 1))  # +1 to avoid log(0)
                if np.all(np.isfinite(vol_changes)):
                    features = np.column_stack([features, vol_changes])

            n_features = features.shape[1]
            n_obs = len(features)

            def bic(n: int, log_ll: float) -> float:
                # k = (n^2 - 1) transitions+start, n*d means, n*d(d+1)/2 covariances
                n_params = (n * n - 1) + n * n_features + n * n_features * (n_features + 1) // 2
                return -2 * log_ll + n_params * np.log(n_obs)

            # --- Model selection by BIC (only over fits whose EM ran cleanly) ---
            candidates = [self.fixed_n_states] if self.fixed_n_states else self.candidates
            fitted = {}
            for n in candidates:
                best = self._best_fit(GaussianHMM, features, n)
                if best is not None:
                    fitted[n] = best
            if not fitted:
                logger.warning("hmm.fit_failed", error="no candidate could be fitted")
                return None

            clean = {n: f for n, f in fitted.items() if f[2]}
            pool = clean or fitted
            n_states = min(pool, key=lambda n: bic(n, pool[n][1]))
            model, _log_ll, em_ok = pool[n_states]
            logger.info(
                "hmm.bic_selection",
                best_n=n_states,
                bic_scores={n: round(bic(n, f[1]), 1) for n, f in fitted.items()},
                clean_fits=sorted(clean),
            )

            state_labels = _STATE_LABELS.get(n_states, [f"state_{i}" for i in range(n_states)])

            # Predict states
            hidden_states = model.predict(features)
            state_probs = model.predict_proba(features)

            # Sort states by mean return: lowest → bear, highest → bull
            state_means = []
            for s in range(n_states):
                mask = hidden_states == s
                if mask.sum() > 0:
                    state_means.append(np.mean(returns[mask]))
                else:
                    state_means.append(0.0)

            # Fine-grained labels by rank (unique names for the per-state
            # dictionaries below)...
            sorted_indices = np.argsort(state_means)
            label_map = {int(sorted_indices[i]): state_labels[i] for i in range(n_states)}

            # ...and the published bucket of each state, by whether its mean
            # return is actually distinguishable from zero.
            bucket_of: dict[str, str] = {}
            tstat_of: dict[str, float] = {}
            for s in range(n_states):
                bucket, t_stat = directional_bucket(returns[hidden_states == s])
                bucket_of[label_map[s]] = bucket
                tstat_of[label_map[s]] = t_stat

            # Current state
            current_raw = int(hidden_states[-1])
            current_detail = label_map[current_raw]
            current_probs = {
                label_map[i]: round(float(state_probs[-1, i]), 4)
                for i in range(n_states)
            }
            folded = fold_probabilities(current_probs, bucket_of)
            current_label = bucket_of[current_detail]

            # Transition matrix (relabeled)
            transmat = model.transmat_
            labeled_transmat = {}
            for i in range(n_states):
                from_label = label_map[i]
                labeled_transmat[from_label] = {
                    label_map[j]: round(float(transmat[i, j]), 4)
                    for j in range(n_states)
                }
            transition_from_current = fold_probabilities(
                labeled_transmat[current_detail], bucket_of
            )

            # State means and volatilities
            state_stats = {}
            for s in range(n_states):
                mask = hidden_states == s
                label = label_map[s]
                if mask.sum() > 0:
                    state_stats[label] = {
                        "mean_daily_return": round(float(np.mean(returns[mask])), 6),
                        "mean_annual_return": round(float(np.mean(returns[mask]) * 252), 4),
                        "volatility_annual": round(float(np.std(returns[mask]) * np.sqrt(252)), 4),
                        "days_in_state": int(mask.sum()),
                        "pct_of_time": round(float(mask.sum() / len(returns)), 4),
                        "mean_return_tstat": round(tstat_of[label], 2),
                        "bucket": bucket_of[label],
                    }

            # Recent state history (last 60 days)
            recent_states = [
                label_map[int(s)] for s in hidden_states[-60:]
            ]

            result = {
                # Published regime: always one of bull / bear / sideways
                "current_state": current_label,
                # Rank-based name of the state within the selected model
                "current_state_detail": current_detail,
                "current_state_tstat": round(tstat_of[current_detail], 2),
                "current_probabilities": current_probs,
                "prob_bull": round(folded["bull"], 4),
                "prob_bear": round(folded["bear"], 4),
                "prob_sideways": round(folded["sideways"], 4),
                "transition_matrix": labeled_transmat,
                "transition_from_current": {k: round(v, 4) for k, v in transition_from_current.items()},
                "state_stats": state_stats,
                "recent_states": recent_states,
                "n_observations": len(returns),
                "n_states_selected": n_states,
                "converged": em_ok,
            }

            logger.info(
                "hmm.complete",
                current=current_label,
                detail=current_detail,
                n_states=n_states,
                converged=em_ok,
                probs=folded,
            )
            return result

        except Exception as e:
            logger.warning("hmm.fit_failed", error=str(e))
            return None
