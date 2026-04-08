"""
XGBoost signal classifier — learns from historical signal features and
realized 20-day returns to predict signal quality.

Replaces arbitrary hand-tuned scoring with data-driven feature weights.
Requires sufficient historical data (50+ labeled signals) to train.

New in this version:
- short_ratio feature (C3 integration)
- HMM regime probabilities as features (hmm_bull_prob, hmm_bear_prob)
- Model persistence via pickle (save/load)
"""

import os
import pickle
import numpy as np
import structlog
from datetime import datetime

logger = structlog.get_logger()

# Default save path — sits alongside this file
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgboost_model.pkl")

# Features extracted from enrichment + conviction + model outputs
FEATURE_COLS = [
    # Fundamentals
    "pe_ratio",
    "market_cap",
    "revenue_growth_yoy",
    "eps_latest",
    "eps_growth_yoy",
    "short_ratio",          # C3: short interest days-to-cover
    # Price / momentum
    "momentum_30d",
    "momentum_90d",
    "rsi_14d",
    "drawdown_from_52w_high",
    "avg_volume_30d",
    # Model outputs
    "hmm_bull_prob",        # C1: regime probabilities from HMM
    "hmm_bear_prob",
    "garch_current_vol",    # annualised current conditional vol
    "ff_alpha",             # Fama-French annual alpha
    "copula_tail_score",    # tail risk 0-100
    # Conviction pipeline features (pipeline mode only; 0 when unknown)
    "signal_score",
    "fundamental_score",
    "macro_modifier",
    "conviction",
    "direction_encoded",    # 1 for buy, -1 for sell
]


class XGBoostSignalClassifier:
    """
    Trains XGBoost on historical signals to predict 20-day return direction.

    Usage:
        clf = XGBoostSignalClassifier()
        result = clf.train(records)
        prediction = clf.predict(features_dict)

        # Persist model
        clf.save()
        clf2 = XGBoostSignalClassifier.load()
        prediction = clf2.predict(features_dict)
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 4,
                 model_path: str = DEFAULT_MODEL_PATH):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model_path = model_path
        self.model = None
        self.feature_importance = None
        self.train_metrics = None

    # ── Training ──────────────────────────────────────────────────────────

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

        X, y, valid_count = self._build_dataset(records)
        if X is None or len(X) < 50:
            return {
                "status": "insufficient_data",
                "n_records": valid_count or 0,
                "min_required": 50,
            }

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

        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]
        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0.0

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
            n_samples=len(X), accuracy=round(acc, 4), auc=round(auc, 4),
            top_features=[f[0] for f in sorted_features[:5]],
        )
        return self.train_metrics

    # ── Prediction ────────────────────────────────────────────────────────

    def predict(self, features: dict) -> dict | None:
        """
        Predict signal quality for a single signal.

        Args:
            features: Dict with feature column values (missing → 0.0).

        Returns:
            Dict with predicted class, probability, and confidence, or None
            if model not trained.
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
            "confidence": round(abs(prob - 0.5) * 2, 4),
        }

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | None = None) -> bool:
        """Pickle the trained model to disk."""
        if self.model is None:
            logger.warning("xgboost.save_skipped", reason="no model trained")
            return False
        target = path or self.model_path
        try:
            with open(target, "wb") as f:
                pickle.dump({
                    "model": self.model,
                    "feature_importance": self.feature_importance,
                    "train_metrics": self.train_metrics,
                    "feature_cols": FEATURE_COLS,
                }, f)
            logger.info("xgboost.saved", path=target)
            return True
        except Exception as e:
            logger.warning("xgboost.save_failed", error=str(e))
            return False

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> "XGBoostSignalClassifier":
        """Load a pickled model. Returns an untrained instance if file absent."""
        instance = cls(model_path=path)
        if not os.path.exists(path):
            logger.info("xgboost.no_saved_model", path=path)
            return instance
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            instance.model = data["model"]
            instance.feature_importance = data.get("feature_importance")
            instance.train_metrics = data.get("train_metrics")
            logger.info("xgboost.loaded", path=path,
                        trained_at=instance.train_metrics.get("trained_at") if instance.train_metrics else None)
        except Exception as e:
            logger.warning("xgboost.load_failed", error=str(e))
        return instance

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    # ── Dataset builder ───────────────────────────────────────────────────

    def _build_dataset(self, records: list[dict]):
        rows = []
        labels = []
        for r in records:
            ret = r.get("return_20d")
            if ret is None:
                continue
            row = [float(r.get(col) or 0.0) for col in FEATURE_COLS]
            rows.append(row)
            labels.append(1 if ret > 0 else 0)

        if not rows:
            return None, None, 0
        return np.array(rows), np.array(labels), len(rows)
