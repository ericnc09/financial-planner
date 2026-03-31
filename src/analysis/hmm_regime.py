"""
Hidden Markov Model (HMM) for market regime detection.

Learns 3 hidden states (bull/bear/sideways) from observable features:
returns, volatility, and volume changes. Outputs current regime probabilities
and transition matrix.
"""

import asyncio

import numpy as np
import structlog

logger = structlog.get_logger()


class HMMRegimeDetector:
    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        # Labels assigned post-hoc by sorting states by mean return
        self.state_labels = ["bear", "sideways", "bull"]

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

            # Fit HMM
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=100,
                random_state=42,
                verbose=False,
            )
            model.fit(features)

            # Predict states
            hidden_states = model.predict(features)
            state_probs = model.predict_proba(features)

            # Sort states by mean return: lowest → bear, middle → sideways, highest → bull
            state_means = []
            for s in range(self.n_states):
                mask = hidden_states == s
                if mask.sum() > 0:
                    state_means.append(np.mean(returns[mask]))
                else:
                    state_means.append(0.0)

            sorted_indices = np.argsort(state_means)
            label_map = {int(sorted_indices[i]): self.state_labels[i] for i in range(self.n_states)}

            # Current state
            current_raw = int(hidden_states[-1])
            current_label = label_map[current_raw]
            current_probs = {
                label_map[i]: round(float(state_probs[-1, i]), 4)
                for i in range(self.n_states)
            }

            # Transition matrix (relabeled)
            transmat = model.transmat_
            labeled_transmat = {}
            for i in range(self.n_states):
                from_label = label_map[i]
                labeled_transmat[from_label] = {
                    label_map[j]: round(float(transmat[i, j]), 4)
                    for j in range(self.n_states)
                }

            # State means and volatilities
            state_stats = {}
            for s in range(self.n_states):
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

            result = {
                "current_state": current_label,
                "current_probabilities": current_probs,
                "transition_matrix": labeled_transmat,
                "state_stats": state_stats,
                "recent_states": recent_states,
                "n_observations": len(returns),
            }

            logger.info(
                "hmm.complete",
                current=current_label,
                probs=current_probs,
            )
            return result

        except Exception as e:
            logger.warning("hmm.fit_failed", error=str(e))
            return None
