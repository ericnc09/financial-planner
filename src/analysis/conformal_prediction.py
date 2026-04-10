"""
Conformal Prediction — distribution-free prediction intervals for individual signals.

Unlike bootstrap CIs (which give confidence on aggregate metrics), conformal
prediction provides per-signal coverage guarantees:
    "This signal's 20d return will be in [-3%, +8%] with 90% probability"

Signals where the interval includes zero have no clear directional edge
and should be flagged or downgraded.

Uses split-conformal approach (Vovk et al., 2005):
1. Split data into proper training set and calibration set
2. Fit model on training set
3. Compute nonconformity scores on calibration set
4. For new points, use calibration quantile to build prediction interval
"""

import numpy as np
import structlog

logger = structlog.get_logger()


class ConformalPredictor:
    """
    Split-conformal prediction intervals for signal return forecasts.

    Provides finite-sample, distribution-free coverage guarantees without
    assuming normality of returns or residuals.
    """

    def __init__(self, coverage: float = 0.90, cal_fraction: float = 0.30):
        """
        Args:
            coverage: Desired coverage level (e.g. 0.90 = 90% of true returns
                      will fall within the prediction interval).
            cal_fraction: Fraction of data reserved for calibration.
        """
        self.coverage = coverage
        self.cal_fraction = cal_fraction
        self.calibration_scores = None
        self.quantile_threshold = None
        self.model = None
        self.is_fitted = False

    def fit(
        self,
        records: list[dict],
        score_field: str = "ensemble_score",
        return_field: str = "realized_return",
    ) -> dict:
        """
        Fit conformal predictor on historical signal records.

        Args:
            records: List of dicts sorted chronologically, each with
                     score_field (0-100) and return_field (decimal).

        Returns:
            Dict with calibration stats and coverage diagnostics.
        """
        if len(records) < 30:
            return {"status": "insufficient_data", "n": len(records), "min_required": 30}

        # Temporal split: train on early data, calibrate on later data
        sorted_recs = sorted(records, key=lambda r: r.get("date", ""))
        n = len(sorted_recs)
        cal_start = int(n * (1 - self.cal_fraction))

        train_recs = sorted_recs[:cal_start]
        cal_recs = sorted_recs[cal_start:]

        if len(cal_recs) < 10:
            return {"status": "insufficient_calibration_data", "n_cal": len(cal_recs)}

        # Fit a simple quantile model on training data:
        # Map ensemble_score -> expected_return via linear regression
        train_scores = np.array([r[score_field] for r in train_recs if r.get(return_field) is not None])
        train_returns = np.array([r[return_field] for r in train_recs if r.get(return_field) is not None])

        if len(train_scores) < 10:
            return {"status": "insufficient_training_data", "n_train": len(train_scores)}

        # Simple linear model: E[return] = a * score + b
        coeffs = np.polyfit(train_scores, train_returns, deg=1)
        self.model = coeffs  # [slope, intercept]

        # Compute nonconformity scores on calibration set
        cal_scores = np.array([r[score_field] for r in cal_recs if r.get(return_field) is not None])
        cal_returns = np.array([r[return_field] for r in cal_recs if r.get(return_field) is not None])

        predicted_cal = np.polyval(self.model, cal_scores)
        self.calibration_scores = np.abs(cal_returns - predicted_cal)

        # Quantile for coverage: ceil((n_cal + 1) * coverage) / n_cal
        n_cal = len(self.calibration_scores)
        quantile_idx = int(np.ceil((n_cal + 1) * self.coverage))
        quantile_idx = min(quantile_idx, n_cal) - 1

        sorted_scores = np.sort(self.calibration_scores)
        self.quantile_threshold = float(sorted_scores[quantile_idx])
        self.is_fitted = True

        # Verify coverage on calibration set
        cal_intervals = predicted_cal - self.quantile_threshold, predicted_cal + self.quantile_threshold
        covered = np.sum((cal_returns >= cal_intervals[0]) & (cal_returns <= cal_intervals[1]))
        empirical_coverage = covered / n_cal

        logger.info(
            "conformal.fitted",
            n_train=len(train_scores),
            n_cal=n_cal,
            quantile_threshold=round(self.quantile_threshold, 4),
            empirical_coverage=round(float(empirical_coverage), 4),
            target_coverage=self.coverage,
        )

        return {
            "status": "fitted",
            "n_train": len(train_scores),
            "n_calibration": n_cal,
            "quantile_threshold": round(self.quantile_threshold, 4),
            "empirical_coverage": round(float(empirical_coverage), 4),
            "target_coverage": self.coverage,
            "model_slope": round(float(coeffs[0]), 6),
            "model_intercept": round(float(coeffs[1]), 6),
        }

    def predict_interval(self, ensemble_score: float) -> dict | None:
        """
        Compute prediction interval for a single signal.

        Args:
            ensemble_score: Ensemble score (0-100).

        Returns:
            Dict with predicted return, interval bounds, width, and
            directional clarity flag.
        """
        if not self.is_fitted:
            return None

        predicted = float(np.polyval(self.model, ensemble_score))
        lower = predicted - self.quantile_threshold
        upper = predicted + self.quantile_threshold
        width = upper - lower

        # Directional clarity: does the interval exclude zero?
        clear_direction = (lower > 0) or (upper < 0)
        if lower > 0:
            direction = "bullish"
        elif upper < 0:
            direction = "bearish"
        else:
            direction = "ambiguous"

        return {
            "predicted_return": round(predicted, 4),
            "interval_lower": round(lower, 4),
            "interval_upper": round(upper, 4),
            "interval_width": round(width, 4),
            "coverage": self.coverage,
            "clear_direction": clear_direction,
            "direction": direction,
        }

    def batch_predict(
        self, signals: list[dict], score_field: str = "ensemble_score"
    ) -> list[dict]:
        """
        Add conformal intervals to a batch of signals.

        Signals without clear direction get flagged for potential downgrade.
        """
        if not self.is_fitted:
            return signals

        for sig in signals:
            score = sig.get(score_field)
            if score is not None:
                interval = self.predict_interval(float(score))
                sig["conformal_interval"] = interval
                # Flag ambiguous signals
                if interval and not interval["clear_direction"]:
                    sig["conformal_warning"] = "interval_includes_zero"

        n_ambiguous = sum(1 for s in signals if s.get("conformal_warning"))
        logger.info(
            "conformal.batch_predicted",
            n_signals=len(signals),
            n_ambiguous=n_ambiguous,
        )
        return signals


def format_conformal_report(fit_result: dict) -> str:
    """Human-readable conformal prediction fit report."""
    if fit_result.get("status") != "fitted":
        return f"Conformal Prediction: {fit_result.get('status', 'unknown')}"

    return (
        f"CONFORMAL PREDICTION  ({fit_result['target_coverage']:.0%} coverage)\n"
        f"{'=' * 50}\n"
        f"Training samples: {fit_result['n_train']}\n"
        f"Calibration samples: {fit_result['n_calibration']}\n"
        f"Quantile threshold: ±{fit_result['quantile_threshold']:.4f}\n"
        f"Empirical coverage: {fit_result['empirical_coverage']:.1%} "
        f"(target: {fit_result['target_coverage']:.1%})\n"
        f"Model: return = {fit_result['model_slope']:.6f} × score + {fit_result['model_intercept']:.6f}"
    )
