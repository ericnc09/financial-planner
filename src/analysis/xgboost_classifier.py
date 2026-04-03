"""
XGBoost signal classifier — learns from historical signal features and
realized 20-day returns to predict signal quality.

Replaces arbitrary hand-tuned scoring with data-driven feature weights.
Requires sufficient historical data (50+ labeled signals) to train.
"""

import numpy as np
import structlog
from datetime import datetime

logger = structlog.get_logger()

# Features extracted from enrichment + conviction + model outputs
FEATURE_COLS = [
    "pe_ratio",
    "market_cap",
    "revenue_growth_yoy",
    "eps_latest",
    "eps_growth_yoy",
    "momentum_30d",
    "momentum_90d",
    "rsi_14d",
    "drawdown_from_52w_high",
    "avg_volume_30d",
    "signal_score",
    "fundamental_score",
    "macro_modifier",
    "conviction",
    "direction_encoded",  # 1 for buy, -1 for sell
]


class XGBoostSignalClassifier:
    """
    Trains XGBoost on historical signals to predict 20-day return direction.

    Usage:
        clf = XGBoostSignalClassifier()
        result = clf.train(records)
        prediction = clf.predict(features_dict)
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 4):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = None
        self.feature_importance = None
        self.train_metrics = None

    def train(self, records: list[dict]) -> dict:
        """
        Train on historical signal records.

        Args:
            records: List of dicts, each with feature columns + "return_20d".

        Returns:
            Dict with training metrics and feature importance.
        """
        try:
            from xgboost import XGBClassifier
            from sklearn.metrics import accuracy_score, roc_auc_score
        except ImportError:
            logger.error("xgboost.import_failed", hint="pip install xgboost")
            return {"status": "error", "message": "xgboost not installed"}

        # Build feature matrix
        X, y, valid_count = self._build_dataset(records)
        if X is None or len(X) < 50:
            return {
                "status": "insufficient_data",
                "n_records": valid_count or 0,
                "min_required": 50,
            }

        # Binary target: 1 if return_20d > 0 (profitable), else 0
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        self.model.fit(X, y)

        # Training metrics
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]
        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0.0

        # Feature importance
        importances = self.model.feature_importances_
        self.feature_importance = {
            FEATURE_COLS[i]: round(float(importances[i]), 4)
            for i in range(len(FEATURE_COLS))
        }
        sorted_features = sorted(
            self.feature_importance.items(), key=lambda x: x[1], reverse=True
        )

        self.train_metrics = {
            "status": "trained",
            "n_samples": len(X),
            "n_positive": int(y.sum()),
            "n_negative": int(len(y) - y.sum()),
            "train_accuracy": round(acc, 4),
            "train_auc": round(auc, 4),
            "feature_importance": dict(sorted_features),
            "trained_at": datetime.utcnow().isoformat(),
        }

        logger.info(
            "xgboost.trained",
            n_samples=len(X),
            accuracy=round(acc, 4),
            auc=round(auc, 4),
            top_features=[f[0] for f in sorted_features[:5]],
        )
        return self.train_metrics

    def predict(self, features: dict) -> dict | None:
        """
        Predict signal quality for a single signal.

        Args:
            features: Dict with feature column values.

        Returns:
            Dict with predicted class, probability, and confidence.
        """
        if self.model is None:
            return None

        row = [features.get(col, 0.0) or 0.0 for col in FEATURE_COLS]
        X = np.array([row])
        prob = float(self.model.predict_proba(X)[0, 1])
        predicted_class = "profitable" if prob >= 0.5 else "unprofitable"

        return {
            "predicted_class": predicted_class,
            "probability_profitable": round(prob, 4),
            "confidence": round(abs(prob - 0.5) * 2, 4),  # 0-1 scale
        }

    def _build_dataset(self, records: list[dict]):
        """Extract features matrix and binary labels from records."""
        rows = []
        labels = []
        for r in records:
            ret = r.get("return_20d")
            if ret is None:
                continue
            row = []
            for col in FEATURE_COLS:
                val = r.get(col)
                row.append(float(val) if val is not None else 0.0)
            rows.append(row)
            labels.append(1 if ret > 0 else 0)

        if not rows:
            return None, None, 0

        return np.array(rows), np.array(labels), len(rows)
