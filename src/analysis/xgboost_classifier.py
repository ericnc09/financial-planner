"""
XGBoost signal classifier — learns from historical signal features and
realized 20-day returns to predict signal quality.

Replaces arbitrary hand-tuned scoring with data-driven feature weights.

Bias-reduction measures:
- Sample size guards: 200+ for XGBoost, falls back to L1-logistic for <200
- Imputation and scaling live INSIDE the model pipeline, so each CV fold
  learns its medians and scales from its own training rows only
- Purged temporal CV: training rows whose 20-day label window reaches into
  the test fold are dropped (their outcome was not known yet at that time)
- Per-fold feature selection: permutation importance runs inside each CV fold
  (not on full data) to prevent feature selection leakage
- Majority-vote feature consensus: only features selected in >50% of folds survive
- Null model benchmark: compares against random baseline to detect overfitting
"""

import os
import pickle
import numpy as np
import structlog
from datetime import datetime, timedelta

logger = structlog.get_logger()

# Default save path — sits alongside this file
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgboost_model.pkl")

# Minimum samples for each model type
MIN_SAMPLES_XGBOOST = 200
MIN_SAMPLES_LOGISTIC = 50

# The label is a 20-trading-day return, known about 30 calendar days after
# the signal. Training rows closer than this to a test fold are purged.
LABEL_HORIZON_DAYS = 30
MIN_FOLD_TRAIN_SAMPLES = 20

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
    # News sentiment (bias-guarded: only articles published <= signal time)
    "news_volume_z",        # attention vs trailing baseline
    "news_sentiment_mean",  # mean headline sentiment [-1, 1]
    "news_sentiment_trend_3d",  # last 3d vs rest of window
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
        self.feature_impute_values = None  # median fallback per feature
        self.feature_importance = None
        self.train_metrics = None
        self._dates: list = []

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
            from sklearn.metrics import accuracy_score, roc_auc_score
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.inspection import permutation_importance
        except Exception as e:
            logger.error("xgboost.import_failed", error=str(e), hint="pip install scikit-learn")
            return {"status": "error", "message": f"model libraries unavailable: {e}"}

        X, y, valid_count = self._build_dataset(records)
        if X is None or len(X) < MIN_SAMPLES_LOGISTIC:
            return {
                "status": "insufficient_data",
                "n_records": valid_count or 0,
                "min_required": MIN_SAMPLES_LOGISTIC,
                "message": f"Need {MIN_SAMPLES_LOGISTIC}+ labeled signals, got {valid_count or 0}",
            }
        if len(np.unique(y)) < 2:
            return {
                "status": "insufficient_data",
                "n_records": valid_count,
                "message": "All labeled signals have the same outcome; nothing to learn.",
            }

        n_samples = len(X)
        dates = self._dates
        # Reported for transparency only; the pipeline does its own imputation
        self.feature_impute_values = self._fit_impute_values(X)

        # ── Step 1: Select model type based on sample size ──
        if n_samples < MIN_SAMPLES_XGBOOST:
            self.model_type = "logistic"
            logger.info(
                "xgboost.using_logistic_fallback",
                n_samples=n_samples,
                reason=f"<{MIN_SAMPLES_XGBOOST} samples, using L1 logistic regression",
            )
        else:
            self.model_type = "xgboost"
            try:
                import xgboost  # noqa: F401
            except Exception as e:
                # More than ImportError: xgboost raises XGBoostError at import
                # time when the OpenMP runtime (libomp) is missing on macOS.
                # The logistic model does not need it, so use that instead of
                # refusing to train.
                self.model_type = "logistic"
                logger.warning(
                    "xgboost.unavailable_using_logistic", error=str(e)[:200],
                    hint="pip install xgboost; on macOS: brew install libomp",
                )

        # ── Step 2: Purged temporal CV with per-fold feature selection ──
        # Feature selection, imputation and scaling all happen INSIDE each
        # fold: fitting any of them on all data first would let the test rows
        # shape the model they are supposed to evaluate.
        n_splits = min(5, max(2, n_samples // 40))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        cv_scores = []
        per_fold_features = []  # track which features each fold selects
        n_purged = 0

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            kept = self._purge(train_idx, test_idx, dates)
            n_purged += len(train_idx) - len(kept)
            train_idx = kept
            if (
                len(train_idx) < MIN_FOLD_TRAIN_SAMPLES
                or len(np.unique(y[train_idx])) < 2
            ):
                continue

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

        if not cv_scores:
            return {
                "status": "insufficient_data",
                "n_records": valid_count,
                "message": "No CV fold had enough purged training data with both outcomes.",
            }

        cv_auc_mean = float(np.mean(cv_scores))
        cv_auc_std = float(np.std(cv_scores))

        # Final feature selection: keep features selected in majority of folds
        from collections import Counter
        feat_counts = Counter(f for fs in per_fold_features for f in fs)
        majority_threshold = len(per_fold_features) / 2
        self.selected_features = []
        selected_indices = []
        for i, feat in enumerate(FEATURE_COLS):
            # strictly more than half the folds
            if feat_counts.get(feat, 0) > majority_threshold:
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
        # ...and with a handful of small folds the mean itself is noisy, so
        # it must also clear 0.5 by more than one standard error.
        null_auc = 0.5
        cv_auc_se = (
            float(np.std(cv_scores, ddof=1) / np.sqrt(len(cv_scores)))
            if len(cv_scores) > 1 else float("inf")
        )
        is_better_than_null = bool(
            cv_auc_mean > 0.55 and (cv_auc_mean - cv_auc_se) > null_auc
        )

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

        # Feature importance (from the final model). The logistic
        # coefficients are on standardized features, so they are comparable.
        estimator = self.model.named_steps["model"]
        if self.model_type == "xgboost":
            importances = estimator.feature_importances_
        else:
            importances = np.abs(estimator.coef_[0])

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
            "cv_auc_se": round(cv_auc_se, 4) if np.isfinite(cv_auc_se) else None,
            "cv_n_splits": len(cv_scores),
            "cv_purged_rows": int(n_purged),
            "label_horizon_days": LABEL_HORIZON_DAYS,
            "overfit_gap": round(overfit_gap, 4),
            "is_overfitting": overfit_gap > 0.15,
            "is_better_than_null": is_better_than_null,
            "feature_importance": dict(sorted_features),
            "missing_imputation": "median (fit per fold, inside the pipeline)",
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
        if self.train_metrics and not self.train_metrics.get("is_better_than_null", False):
            return {
                "status": "unreliable_model",
                "message": "Model CV AUC is not meaningfully above the null threshold.",
                "is_better_than_null": False,
                "cv_auc": self.train_metrics.get("cv_auc_mean"),
            }

        # Use selected features if available (from feature selection), else all
        cols = self.selected_features if self.selected_features else FEATURE_COLS
        row = [self._coerce_float(features.get(col)) for col in cols]
        X = np.array([row])
        if self.feature_impute_values and not hasattr(self.model, "named_steps"):
            # Model saved before imputation moved into the pipeline
            fill = np.array([self.feature_impute_values.get(col, 0.0) for col in cols])
            X = np.where(np.isnan(X), fill.reshape(1, len(cols)), X)
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
                    "feature_impute_values": self.feature_impute_values,
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
            instance.feature_impute_values = data.get("feature_impute_values")
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
        """Fresh impute -> scale -> estimator pipeline for the selected model_type.

        Keeping the imputer and scaler inside the pipeline means they are
        refit on whatever rows the pipeline is trained on (one CV fold, or
        everything for the final model) and applied unchanged at prediction.
        Scaling matters for the L1 logistic model: market cap (~1e10) next to
        RSI (~50) otherwise makes the penalty meaningless and stops the
        solver converging. keep_empty_features keeps column positions stable
        when a feature is missing for every row in a fold.
        """
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        if self.model_type == "logistic":
            estimator = LogisticRegression(
                penalty="l1", solver="saga", C=1.0,
                max_iter=5000, random_state=42,
            )
        else:
            from xgboost import XGBClassifier

            estimator = XGBClassifier(
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
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", estimator),
        ])

    @staticmethod
    def _purge(train_idx: np.ndarray, test_idx: np.ndarray, dates: list) -> np.ndarray:
        """Drop training rows whose label window overlaps the test fold.

        A signal's 20-day outcome is only known ~30 calendar days later. A
        training row dated inside that window before the first test row
        carries an outcome that was still in the future at test time.
        """
        if not dates or any(dates[i] is None for i in train_idx) or any(
            dates[i] is None for i in test_idx
        ):
            return train_idx
        cutoff = min(dates[i] for i in test_idx) - timedelta(days=LABEL_HORIZON_DAYS)
        return np.array([i for i in train_idx if dates[i] <= cutoff], dtype=int)

    # ── Dataset builder ───────────────────────────────────────────────────

    def _build_dataset(self, records: list[dict]):
        # TimeSeriesSplit assumes chronological order — sort by whatever date
        # field is present so the temporal CV folds don't leak future data.
        records = sorted(
            records,
            key=lambda r: str(
                r.get("date") or r.get("trade_date") or r.get("signal_date") or ""
            ),
        )
        rows = []
        labels = []
        dates = []
        for r in records:
            ret = r.get("return_20d")
            if ret is None:
                continue
            row = [self._coerce_float(r.get(col)) for col in FEATURE_COLS]
            rows.append(row)
            labels.append(1 if ret > 0 else 0)
            dates.append(self._parse_date(
                r.get("date") or r.get("trade_date") or r.get("signal_date")
            ))

        # Row-aligned signal dates (None where unknown), used to purge CV folds
        self._dates = dates
        if not rows:
            return None, None, 0
        return np.array(rows), np.array(labels), len(rows)

    @staticmethod
    def _parse_date(value) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.replace(tzinfo=None)
        try:
            return datetime.fromisoformat(str(value)[:19]).replace(tzinfo=None)
        except ValueError:
            return None

    @staticmethod
    def _coerce_float(value) -> float:
        if value is None:
            return np.nan
        try:
            val = float(value)
        except (TypeError, ValueError):
            return np.nan
        return val if np.isfinite(val) else np.nan

    @staticmethod
    def _fit_impute_values(X: np.ndarray) -> dict[str, float]:
        values = {}
        for i, feature in enumerate(FEATURE_COLS):
            col = X[:, i]
            if np.all(np.isnan(col)):
                values[feature] = 0.0
            else:
                values[feature] = float(np.nanmedian(col))
        return values

    def _impute_array(self, X: np.ndarray) -> np.ndarray:
        if self.feature_impute_values is None:
            self.feature_impute_values = self._fit_impute_values(X)
        fill = np.array([self.feature_impute_values.get(f, 0.0) for f in FEATURE_COLS])
        return np.where(np.isnan(X), fill.reshape(1, -1), X)
