"""
Hidden Markov Model (HMM) for market regime detection.

Fits models with n_states=2..5 and selects by BIC. Learns hidden states
(bull/bear/sideways) from observable features: returns, volatility, and
volume changes. Outputs current regime probabilities and transition matrix.
"""

import asyncio

import numpy as np
import structlog

logger = structlog.get_logger()

# Map state count to labels (sorted bear -> bull)
_STATE_LABELS = {
    2: ["bear", "bull"],
    3: ["bear", "sideways", "bull"],
    4: ["bear", "sideways", "mild_bull", "bull"],
    5: ["bear", "mild_bear", "sideways", "mild_bull", "bull"],
}


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

    def _fit_predict_sync(
        self,
        returns: np.ndarray,
        volumes: np.ndarray | None,
    ) -> dict | None:
        try:
            from hmmlearn.hmm import GaussianHMM

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
                vol_changes = np.diff(np.log(volumes + 1))  # +1 to avoid log(0)
                features = np.column_stack([features, vol_changes])

            # --- Model selection by BIC ---
            if self.fixed_n_states:
                best_n = self.fixed_n_states
            else:
                best_n, best_bic = self.candidates[0], np.inf
                bic_results = {}
                for n in self.candidates:
                    try:
                        m = GaussianHMM(
                            n_components=n,
                            covariance_type="full",
                            n_iter=100,
                            random_state=42,
                            verbose=False,
                        )
                        m.fit(features)
                        log_ll = m.score(features) * len(features)
                        # BIC = -2*LL + k*ln(n_obs)
                        # k = n_states^2 + 2*n_states*n_features - 1 (transitions + means + covs)
                        n_features = features.shape[1]
                        n_params = (n * n - 1) + n * n_features + n * n_features * (n_features + 1) // 2
                        bic = -2 * log_ll + n_params * np.log(len(features))
                        bic_results[n] = round(bic, 1)
                        if bic < best_bic:
                            best_bic = bic
                            best_n = n
                    except Exception:
                        bic_results[n] = None
                        continue
                logger.info("hmm.bic_selection", best_n=best_n, bic_scores=bic_results)

            n_states = best_n
            state_labels = _STATE_LABELS.get(n_states, [f"state_{i}" for i in range(n_states)])

            # Fit final model with selected n_states
            model = GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                n_iter=100,
                random_state=42,
                verbose=False,
            )
            model.fit(features)

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

            sorted_indices = np.argsort(state_means)
            label_map = {int(sorted_indices[i]): state_labels[i] for i in range(n_states)}

            # Current state
            current_raw = int(hidden_states[-1])
            current_label = label_map[current_raw]
            current_probs = {
                label_map[i]: round(float(state_probs[-1, i]), 4)
                for i in range(n_states)
            }

            # Transition matrix (relabeled)
            transmat = model.transmat_
            labeled_transmat = {}
            for i in range(n_states):
                from_label = label_map[i]
                labeled_transmat[from_label] = {
                    label_map[j]: round(float(transmat[i, j]), 4)
                    for j in range(n_states)
                }

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
                    }

            # Recent state history (last 60 days)
            recent_states = [
                label_map[int(s)] for s in hidden_states[-60:]
            ]

            # Ensure standard keys exist for ensemble scorer compatibility
            result = {
                "current_state": current_label,
                "current_probabilities": current_probs,
                "prob_bull": current_probs.get("bull", 0),
                "prob_bear": current_probs.get("bear", 0),
                "prob_sideways": current_probs.get("sideways", 0),
                "transition_matrix": labeled_transmat,
                "state_stats": state_stats,
                "recent_states": recent_states,
                "n_observations": len(returns),
                "n_states_selected": n_states,
            }

            logger.info(
                "hmm.complete",
                current=current_label,
                n_states=n_states,
                probs=current_probs,
            )
            return result

        except Exception as e:
            logger.warning("hmm.fit_failed", error=str(e))
            return None
