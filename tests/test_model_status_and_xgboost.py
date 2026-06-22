import numpy as np
from sqlalchemy import inspect

from scripts.model_status_report import _validation_gates
from src.analysis.xgboost_classifier import FEATURE_COLS, XGBoostSignalClassifier
from src.models.database import init_db


def test_sqlite_schema_upgrade_adds_audit_columns(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'audit.db'}"
    engine = init_db(db_url)
    # Idempotence check: running init_db again should not attempt duplicate ALTERs.
    init_db(db_url)

    inspector = inspect(engine)
    ensemble_cols = {c["name"] for c in inspector.get_columns("ensemble_score_results")}
    enrichment_cols = {c["name"] for c in inspector.get_columns("enrichments")}

    assert {
        "score_options_flow",
        "score_earnings_overlay",
        "score_news_sentiment",
        "score_weather_overlay",
    }.issubset(ensemble_cols)
    assert "short_ratio" in enrichment_cols


def test_xgboost_dataset_uses_explicit_missing_values():
    clf = XGBoostSignalClassifier()
    records = [
        {"date": "2026-01-01", "momentum_30d": None, "return_20d": 0.05},
        {"date": "2026-01-02", "momentum_30d": 0.2, "return_20d": -0.01},
    ]

    X, y, n = clf._build_dataset(records)

    momentum_idx = FEATURE_COLS.index("momentum_30d")
    assert n == 2
    assert y.tolist() == [1, 0]
    assert np.isnan(X[0, momentum_idx])

    clf.feature_impute_values = clf._fit_impute_values(X)
    X_imputed = clf._impute_array(X)

    assert not np.isnan(X_imputed).any()
    assert X_imputed[0, momentum_idx] == 0.2


def test_validation_gates_call_out_known_failures():
    report = {
        "ensemble": {"missing_latest_components": ["event_study"]},
        "performance_20d": {
            "latest_ensemble_correlation": {"n": 100, "spearman": 0.1},
            "conviction_correlation": {"n": 100, "spearman": -0.01},
            "recommendation_buckets": {
                "buy": {"avg_return": 0.05},
                "hold": {"avg_return": 0.0},
                "sell": {"avg_return": -0.02},
            },
            "conviction_buckets": {
                "high_conviction": {"avg_return": 0.01},
                "other": {"avg_return": 0.02},
            },
        },
    }

    result = _validation_gates(report)

    assert result["gates"]["minimum_20d_samples"] is True
    assert result["gates"]["buy_bucket_beats_hold"] is True
    assert result["gates"]["conviction_rank_ic_positive"] is False
    assert result["gates"]["sell_bucket_positive"] is False
    assert result["gates"]["high_conviction_beats_other"] is False
    assert result["gates"]["all_claimed_components_present"] is False
    assert result["overall_status"] == "fail"
