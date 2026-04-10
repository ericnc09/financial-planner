"""
Adversarial Validation — detects distribution shift between train and test data.

If a classifier can distinguish train samples from test samples (AUC > 0.55),
the two sets come from different distributions and OOS backtest results are
unreliable. Features that drive the shift should be de-trended or dropped.

This catches non-stationarity that temporal CV alone cannot detect:
- Market regime changes between train and test periods
- Seasonal patterns in signal activity
- Data quality changes over time
"""

import numpy as np
import structlog

logger = structlog.get_logger()


# Same features as XGBoost classifier
ADVERSARIAL_FEATURES = [
    "pe_ratio", "market_cap", "revenue_growth_yoy", "eps_latest",
    "eps_growth_yoy", "short_ratio", "momentum_30d", "momentum_90d",
    "rsi_14d", "drawdown_from_52w_high", "avg_volume_30d",
    "hmm_bull_prob", "hmm_bear_prob", "garch_current_vol",
    "ff_alpha", "copula_tail_score",
]


class AdversarialValidator:
    """
    Trains a classifier to distinguish train from test samples.
    High AUC indicates distribution shift — features are non-stationary.
    """

    def __init__(self, auc_threshold: float = 0.55):
        """
        Args:
            auc_threshold: AUC above which we flag distribution shift.
                           0.55 is slightly above random (0.50).
        """
        self.auc_threshold = auc_threshold

    def validate(
        self,
        train_records: list[dict],
        test_records: list[dict],
        features: list[str] | None = None,
    ) -> dict:
        """
        Test whether train and test come from the same distribution.

        Args:
            train_records: Training set signal records.
            test_records: Test set signal records.
            features: Feature names to use. Defaults to ADVERSARIAL_FEATURES.

        Returns:
            Dict with AUC score, feature importance (shift drivers),
            and verdict.
        """
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import cross_val_score
            from sklearn.inspection import permutation_importance
        except ImportError:
            return {"status": "error", "message": "scikit-learn not installed"}

        feat_cols = features or ADVERSARIAL_FEATURES

        # Build feature matrices
        X_train = self._extract_features(train_records, feat_cols)
        X_test = self._extract_features(test_records, feat_cols)

        if X_train.shape[0] < 20 or X_test.shape[0] < 10:
            return {
                "status": "insufficient_data",
                "n_train": X_train.shape[0],
                "n_test": X_test.shape[0],
            }

        # Label: 0 = train, 1 = test
        X = np.vstack([X_train, X_test])
        y = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])

        # Shuffle to avoid temporal ordering artifacts in CV
        rng = np.random.default_rng(42)
        shuffle_idx = rng.permutation(len(y))
        X = X[shuffle_idx]
        y = y[shuffle_idx]

        # Train a lightweight classifier
        clf = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )

        # Cross-validated AUC
        n_splits = min(5, max(2, len(y) // 20))
        cv_auc = cross_val_score(clf, X, y, cv=n_splits, scoring="roc_auc")
        mean_auc = float(np.mean(cv_auc))
        std_auc = float(np.std(cv_auc))

        # Fit on all data for feature importance
        clf.fit(X, y)
        perm = permutation_importance(clf, X, y, n_repeats=10, random_state=42, scoring="roc_auc")

        feature_shift = {}
        for i, feat in enumerate(feat_cols):
            feature_shift[feat] = {
                "importance": round(float(perm.importances_mean[i]), 4),
                "std": round(float(perm.importances_std[i]), 4),
            }

        # Sort by importance (highest shift contribution first)
        feature_shift = dict(sorted(
            feature_shift.items(), key=lambda x: -x[1]["importance"]
        ))

        has_shift = mean_auc > self.auc_threshold
        severity = "none"
        if mean_auc > 0.70:
            severity = "severe"
        elif mean_auc > 0.60:
            severity = "moderate"
        elif mean_auc > 0.55:
            severity = "mild"

        # Identify top shift-driving features (importance > 0.01)
        shift_drivers = [
            f for f, v in feature_shift.items() if v["importance"] > 0.01
        ]

        result = {
            "status": "complete",
            "auc_mean": round(mean_auc, 4),
            "auc_std": round(std_auc, 4),
            "auc_threshold": self.auc_threshold,
            "has_distribution_shift": has_shift,
            "severity": severity,
            "shift_drivers": shift_drivers,
            "feature_shift_importance": feature_shift,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "recommendation": self._recommendation(severity, shift_drivers),
        }

        logger.info(
            "adversarial.validated",
            auc=round(mean_auc, 4),
            has_shift=has_shift,
            severity=severity,
            n_drivers=len(shift_drivers),
        )
        return result

    def _extract_features(self, records: list[dict], feat_cols: list[str]) -> np.ndarray:
        rows = []
        for r in records:
            row = [float(r.get(col) or 0.0) for col in feat_cols]
            rows.append(row)
        return np.array(rows) if rows else np.empty((0, len(feat_cols)))

    def _recommendation(self, severity: str, drivers: list[str]) -> str:
        if severity == "none":
            return "No distribution shift detected. OOS results are reliable."
        if severity == "mild":
            return (
                f"Mild distribution shift detected. Consider de-trending: {', '.join(drivers[:3])}. "
                "OOS results should be interpreted with caution."
            )
        if severity == "moderate":
            return (
                f"Moderate distribution shift. Features driving shift: {', '.join(drivers[:5])}. "
                "Consider z-scoring features on rolling windows or dropping shifted features."
            )
        return (
            f"Severe distribution shift — train and test are from different distributions. "
            f"Top drivers: {', '.join(drivers[:5])}. "
            "OOS backtest results are unreliable. Re-examine data pipeline for temporal artifacts."
        )


def format_adversarial_report(result: dict) -> str:
    """Human-readable adversarial validation report."""
    if result.get("status") != "complete":
        return f"Adversarial Validation: {result.get('status', 'unknown')}"

    lines = [
        "ADVERSARIAL VALIDATION  (Train vs Test Distribution)",
        "=" * 60,
        f"AUC: {result['auc_mean']:.4f} ± {result['auc_std']:.4f}  (threshold: {result['auc_threshold']})",
        f"Distribution shift: {'YES' if result['has_distribution_shift'] else 'NO'}  (severity: {result['severity']})",
        "",
        f"{'Feature':<25} {'Shift Importance':>18}",
        "-" * 45,
    ]
    for feat, vals in list(result["feature_shift_importance"].items())[:10]:
        lines.append(f"{feat:<25} {vals['importance']:>18.4f}")
    lines.append("")
    lines.append(f"Recommendation: {result['recommendation']}")
    return "\n".join(lines)
