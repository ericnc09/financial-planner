"""
XGBoost signal classifier — learns from historical signal features and
realized 20-day returns to predict signal quality.

Replaces arbitrary hand-tuned scoring with data-driven feature weights.

Bias-reduction measures:
- Sample size guards: 200+ for XGBoost, falls back to L1-logistic for <200
- Per-fold feature selection: permutation importance runs inside each CV fold
  (not on full data) to prevent feature selection leakage
- Majority-vote feature consensus: only features selected in >50% of folds survive
- Null model benchmark: compares against random baseline to detect overfitting
"""

import os
import pickle
import numpy as np
import structlog
from datetime import datetime

logger = structlog.get_logger()

# Default save path — sits alongside this file
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgboost_model.pkl")

# Minimum samples for each model type
MIN_SAMPLES_XGBOOST = 200
MIN_SAMPLES_LOGISTIC = 50

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

    Includes sample-size-aware model selection, nested CV, and feature selection.

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
        self.model_type = None  # "xgboost" or "logistic"
        self.selected_features = None  # features that survived selection
        self.feature_importance = None
        self.train_metrics = None

    # ── Training ──────────────────────────────────────────────────────────

    def train(self, records: list[dict]) -> dict:
        """
        Train on historical signal records with sample-size-aware model selection.

        - <50 samples: refuse to train (insufficient data)
        - 50-199 samples: L1-regularized logistic regression (fewer parameters)
        - 200+ samples: XGBoost with nested CV and feature selection

        Args:
            records: List of dicts, each with feature columns + "return_20d".

        Returns:
            Dict with training metrics, feature importance, and CV scores.
        """
        try:
            from xgboost import XGBClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, roc_auc_score
            from sklearn.model_selection import TimeSeriesSplit, cross_val_score
            from sklearn.inspection import permutation_importance
        except ImportError:
            logger.error("xgboost.import_failed", hint="pip install xgboost scikit-learn")
            return {"status": "error", "message": "xgboost or scikit-learn not installed"}

        X, y, valid_count = self._build_dataset(records)
        if X is None or len(X) < MIN_SAMPLES_LOGISTIC:
            return {
                "status": "insufficient_data",
                "n_records": valid_count or 0,
                "min_required": MIN_SAMPLES_LOGISTIC,
                "message": f"Need {MIN_SAMPLES_LOGISTIC}+ labeled signals, got {valid_count or 0}",
            }

        n_samples = len(X)

        # ── Step 1: Select model type based on sample size ──
        if n_samples < MIN_SAMPLES_XGBOOST:
            self.model_type = "logistic"
            self.model = LogisticRegression(
                penalty="l1",
                solver="saga",
                C=1.0,
                max_iter=5000,
                random_state=42,
            )
            logger.info(
                "xgboost.using_logistic_fallback",
                n_samples=n_samples,
                reason=f"<{MIN_SAMPLES_XGBOOST} samples, using L1 logistic regression",
            )
        else:
            self.model_type = "xgboost"
            self.model = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,    # L1 regularization
                reg_lambda=1.0,   # L2 regularization
                random_state=42,
                eval_metric="logloss",
            )

        # ── Step 2: Nested CV with per-fold feature selection ──
        # Feature selection is done INSIDE each CV fold to prevent leakage:
        # fitting on all data first would bias which features survive toward
        # the full sample, inflating CV AUC estimates.
        n_splits = min(5, max(2, n_samples // 40))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        cv_scores = []
        per_fold_features = []  # track which features each fold selects

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]

            # Feature selection on training fold only
            rng = np.random.default_rng(42 + fold_idx)
            noise_col = rng.standard_normal(len(X_train_fold))
            X_train_noise = np.column_stack([X_train_fold, noise_col])

            fold_model = self._make_model()
            fold_model.fit(X_train_noise, y_train_fold)
            perm_result = permutation_importance(
                fold_model, X_train_noise, y_train_fold,
                n_repeats=10, random_state=42, scoring="roc_auc",
            )
            noise_importance = perm_result.importances_mean[-1]

            # Keep features more important than the noise column
            fold_selected = []
            fold_indices = []
            for i, feat in enumerate(FEATURE_COLS):
                if perm_result.importances_mean[i] > noise_importance:
                    fold_selected.append(feat)
                    fold_indices.append(i)

            if len(fold_selected) < 3:
                ranked = np.argsort(perm_result.importances_mean[:-1])[::-1]
                fold_indices = ranked[:8].tolist()
                fold_selected = [FEATURE_COLS[i] for i in fold_indices]

            per_fold_features.append(set(fold_selected))

            # Evaluate on test fold with selected features
            fold_model2 = self._make_model()
            fold_model2.fit(X_train_fold[:, fold_indices], y_train_fold)
            y_prob = fold_model2.predict_proba(X_test_fold[:, fold_indices])[:, 1]
            if len(np.unique(y_test_fold)) > 1:
                fold_auc = roc_auc_score(y_test_fold, y_prob)
            else:
                fold_auc = 0.5
            cv_scores.append(fold_auc)

        cv_auc_mean = float(np.mean(cv_scores))
        cv_auc_std = float(np.std(cv_scores))

        # Final feature selection: keep features selected in majority of folds
        from collections import Counter
        feat_counts = Counter(f for fs in per_fold_features for f in fs)
        majority_threshold = len(per_fold_features) / 2
        self.selected_features = []
        selected_indices = []
        for i, feat in enumerate(FEATURE_COLS):
            if feat_counts.get(feat, 0) >= majority_threshold:
                self.selected_features.append(feat)
                selected_indices.append(i)

        if len(self.selected_features) < 3:
            # Fall back to features selected in at least one fold, ranked by count
            ranked_feats = feat_counts.most_common(8)
            self.selected_features = [f for f, _ in ranked_feats]
            selected_indices = [FEATURE_COLS.index(f) for f in self.selected_features]

        X_selected = X[:, selected_indices]
        logger.info(
            "xgboost.feature_selection",
            n_original=len(FEATURE_COLS),
            n_selected=len(self.selected_features),
            dropped=[f for f in FEATURE_COLS if f not in self.selected_features],
            fold_agreement={f: c for f, c in feat_counts.most_common()},
        )

        # ── Step 4: Null model benchmark ──
        # A null model predicts the majority class — if CV AUC <= 0.55,
        # the model has no real predictive power
        null_auc = 0.5
        is_better_than_null = cv_auc_mean > 0.55

        if not is_better_than_null:
            logger.warning(
                "xgboost.no_predictive_power",
                cv_auc=round(cv_auc_mean, 4),
                message="Model AUC not meaningfully above random (0.55 threshold)",
            )

        # ── Step 4: Final fit on all data with selected features ──
        self.model = self._make_model()
        self.model.fit(X_selected, y)

        y_pred = self.model.predict(X_selected)
        y_prob = self.model.predict_proba(X_selected)[:, 1]
        train_acc = accuracy_score(y, y_pred)
        train_auc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0.0

        # Feature importance (from the final model)
        if self.model_type == "xgboost":
            importances = self.model.feature_importances_
        else:
            importances = np.abs(self.model.coef_[0])

        self.feature_importance = {
            self.selected_features[i]: round(float(importances[i]), 4)
            for i in range(len(self.selected_features))
        }
        sorted_features = sorted(
            self.feature_importance.items(), key=lambda x: x[1], reverse=True
        )

        # Overfit gap: train AUC vs CV AUC
        overfit_gap = train_auc - cv_auc_mean

        self.train_metrics = {
            "status": "trained",
            "model_type": self.model_type,
            "n_samples": n_samples,
            "n_positive": int(y.sum()),
            "n_negative": int(len(y) - y.sum()),
            "n_features_original": len(FEATURE_COLS),
            "n_features_selected": len(self.selected_features),
            "selected_features": self.selected_features,
            "train_accuracy": round(train_acc, 4),
            "train_auc": round(train_auc, 4),
            "cv_auc_mean": round(cv_auc_mean, 4),
            "cv_auc_std": round(cv_auc_std, 4),
            "cv_n_splits": n_splits,
            "overfit_gap": round(overfit_gap, 4),
            "is_overfitting": overfit_gap > 0.15,
            "is_better_than_null": is_better_than_null,
            "feature_importance": dict(sorted_features),
            "trained_at": datetime.utcnow().isoformat(),
        }

        logger.info(
            "xgboost.trained",
            model_type=self.model_type,
            n_samples=n_samples,
            n_features=len(self.selected_features),
            train_auc=round(train_auc, 4),
            cv_auc=round(cv_auc_mean, 4),
            overfit_gap=round(overfit_gap, 4),
            better_than_null=is_better_than_null,
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
            Dict with predicted class, probability, confidence, and model
            reliability flags — or None if model not trained.
        """
        if self.model is None:
            return None

        # Use selected features if available (from feature selection), else all
        cols = self.selected_features if self.selected_features else FEATURE_COLS
        row = [features.get(col, 0.0) or 0.0 for col in cols]
        X = np.array([row])
        prob = float(self.model.predict_proba(X)[0, 1])
        predicted_class = "profitable" if prob >= 0.5 else "unprofitable"

        result = {
            "predicted_class": predicted_class,
            "probability_profitable": round(prob, 4),
            "confidence": round(abs(prob - 0.5) * 2, 4),
            "model_type": self.model_type or "unknown",
        }

        # Attach reliability warnings from training metrics
        if self.train_metrics:
            result["is_better_than_null"] = self.train_metrics.get("is_better_than_null", False)
            result["is_overfitting"] = self.train_metrics.get("is_overfitting", False)
            result["cv_auc"] = self.train_metrics.get("cv_auc_mean")

        return result

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
                    "model_type": self.model_type,
                    "selected_features": self.selected_features,
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
            instance.model_type = data.get("model_type", "xgboost")
            instance.selected_features = data.get("selected_features")
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

    # ── Model factory ────────────────────────────────────────────────────

    def _make_model(self):
        """Create a fresh model instance based on the selected model_type."""
        from xgboost import XGBClassifier
        from sklearn.linear_model import LogisticRegression

        if self.model_type == "logistic":
            return LogisticRegression(
                penalty="l1", solver="saga", C=1.0,
                max_iter=5000, random_state=42,
            )
        return XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
        )

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
