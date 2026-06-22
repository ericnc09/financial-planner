"""
Model status and performance audit.

Produces a no-network report that answers:
- Which model tables currently have data?
- Which ensemble components are actually persisted?
- Do conviction/ensemble scores rank realized 20-day outcomes?
- How do recommendation buckets perform?

Output: reports/model_status_report.json
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from scipy import stats
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import Settings
from src.analysis.xgboost_classifier import DEFAULT_MODEL_PATH, XGBoostSignalClassifier
from src.backtesting.validator import (
    ScoreCalibrator,
    bootstrap_metrics,
    calibration_analysis,
    deflated_sharpe_ratio,
)
from src.models.database import init_db


OUT = ROOT / "reports" / "model_status_report.json"

MODEL_TABLES = {
    "events": "smart_money_events",
    "enrichments": "enrichments",
    "conviction_scores": "conviction_scores",
    "ensemble_scores": "ensemble_score_results",
    "performance": "signal_performance",
    "monte_carlo": "monte_carlo_results",
    "hmm": "hmm_regime_states",
    "garch": "garch_forecasts",
    "fama_french": "fama_french_exposures",
    "copula_tail": "copula_tail_risk_results",
    "event_study": "event_study_results",
    "bayesian_decay": "bayesian_decay_results",
    "options_flow": "options_flow",
    "news_sentiment": "news_sentiment_results",
    "mean_variance": "mean_variance_results",
}

ENSEMBLE_COMPONENT_COLUMNS = {
    "monte_carlo": "score_monte_carlo",
    "hmm_regime": "score_hmm",
    "garch": "score_garch",
    "fama_french": "score_fama_french",
    "copula_tail": "score_copula",
    "bayesian_decay": "score_bayesian_decay",
    "event_study": "score_event_study",
    "options_flow": "score_options_flow",
    "earnings_overlay": "score_earnings_overlay",
    "news_sentiment": "score_news_sentiment",
    "weather_overlay": "score_weather_overlay",
}


def _sqlite_path(database_url: str) -> Path:
    if database_url.startswith("sqlite:///"):
        raw = database_url.removeprefix("sqlite:///")
        p = Path(unquote(raw))
        return p if p.is_absolute() else ROOT / p
    if database_url.startswith("sqlite://"):
        raw = database_url.removeprefix("sqlite://")
        return Path(unquote(raw))
    raise SystemExit("model_status_report currently supports SQLite DATABASE_URL only")


def _query(conn: sqlite3.Connection, sql: str, params=()) -> list[sqlite3.Row]:
    return conn.execute(sql, params).fetchall()


def _count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _corr(rows: list[sqlite3.Row], x_key: str, y_key: str) -> dict:
    pairs = [
        (float(r[x_key]), float(r[y_key]))
        for r in rows
        if r[x_key] is not None and r[y_key] is not None
    ]
    if len(pairs) < 3:
        return {"n": len(pairs), "pearson": None, "spearman": None}
    x = np.array([p[0] for p in pairs], dtype=float)
    y = np.array([p[1] for p in pairs], dtype=float)
    pearson = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None
    spear = spearmanr(x, y).correlation
    return {
        "n": len(pairs),
        "pearson": round(pearson, 4) if pearson is not None and np.isfinite(pearson) else None,
        "spearman": round(float(spear), 4) if spear is not None and np.isfinite(spear) else None,
    }


def _bucket_summary(rows: list[sqlite3.Row], bucket_key: str, return_key: str) -> dict:
    grouped: dict[str, list[float]] = {}
    for r in rows:
        bucket = r[bucket_key]
        ret = r[return_key]
        if bucket is None or ret is None:
            continue
        grouped.setdefault(str(bucket), []).append(float(ret))

    out = {}
    for bucket, vals in sorted(grouped.items()):
        arr = np.array(vals, dtype=float)
        out[bucket] = {
            "n": len(vals),
            "avg_return": round(float(np.mean(arr)), 4),
            "win_rate": round(float(np.mean(arr > 0)), 4),
        }
    return out


def _annualized_sharpe(returns: list[float], hold_days: int = 20) -> float | None:
    if len(returns) < 3:
        return None
    arr = np.array(returns, dtype=float)
    std = float(np.std(arr, ddof=1))
    if std <= 0:
        return None
    return float(np.mean(arr) / std * np.sqrt(252 / hold_days))


def _validation_gates(report: dict) -> dict:
    perf = report["performance_20d"]
    rec = perf["recommendation_buckets"]
    conv = perf["conviction_buckets"]
    ens_corr = perf["latest_ensemble_correlation"]
    conv_corr = perf["conviction_correlation"]

    def avg(bucket_map: dict, name: str):
        return (bucket_map.get(name) or {}).get("avg_return")

    gates = {
        "minimum_20d_samples": ens_corr["n"] >= 50,
        "ensemble_rank_ic_positive": (ens_corr["spearman"] or 0) > 0,
        "conviction_rank_ic_positive": (conv_corr["spearman"] or 0) > 0,
        "buy_bucket_beats_hold": (
            avg(rec, "buy") is not None
            and avg(rec, "hold") is not None
            and avg(rec, "buy") > avg(rec, "hold")
        ),
        "sell_bucket_positive": (
            avg(rec, "sell") is not None
            and avg(rec, "sell") > 0
        ),
        "high_conviction_beats_other": (
            avg(conv, "high_conviction") is not None
            and avg(conv, "other") is not None
            and avg(conv, "high_conviction") > avg(conv, "other")
        ),
        "all_claimed_components_present": not report["ensemble"]["missing_latest_components"],
    }
    return {
        "gates": gates,
        "passed": sum(1 for v in gates.values() if v),
        "total": len(gates),
        "overall_status": "pass" if all(gates.values()) else "fail",
    }


def _xgboost_status() -> dict:
    path = Path(DEFAULT_MODEL_PATH)
    clf = XGBoostSignalClassifier.load()
    if not clf.is_trained:
        return {
            "status": "not_trained",
            "model_path": str(path),
            "model_file_exists": path.exists(),
        }
    metrics = clf.train_metrics or {}
    return {
        "status": "trained",
        "model_path": str(path),
        "model_file_exists": path.exists(),
        "model_type": clf.model_type,
        "n_samples": metrics.get("n_samples"),
        "cv_auc_mean": metrics.get("cv_auc_mean"),
        "is_better_than_null": metrics.get("is_better_than_null"),
        "is_overfitting": metrics.get("is_overfitting"),
        "selected_features": clf.selected_features,
        "missing_imputation": metrics.get("missing_imputation"),
        "trained_at": metrics.get("trained_at"),
    }


def _json_default(value):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def main() -> int:
    settings = Settings()
    init_db(settings.database_url)
    db_path = _sqlite_path(settings.database_url)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        table_counts = {name: _count(conn, table) for name, table in MODEL_TABLES.items()}

        latest_ensemble_sql = """
            SELECT es.*
            FROM ensemble_score_results es
            JOIN (
                SELECT event_id, MAX(run_date) AS max_date
                FROM ensemble_score_results
                GROUP BY event_id
            ) latest
              ON latest.event_id = es.event_id
             AND latest.max_date = es.run_date
        """

        latest_rows = _query(conn, latest_ensemble_sql)
        n_models_distribution = dict(sorted(Counter(r["n_models"] for r in latest_rows).items()))

        component_presence = {}
        for component, col in ENSEMBLE_COMPONENT_COLUMNS.items():
            latest_present = sum(1 for r in latest_rows if r[col] is not None)
            all_present = int(conn.execute(
                f"SELECT COUNT(*) FROM ensemble_score_results WHERE {col} IS NOT NULL"
            ).fetchone()[0])
            component_presence[component] = {
                "latest_rows_present": latest_present,
                "all_rows_present": all_present,
                "latest_coverage": round(latest_present / len(latest_rows), 4) if latest_rows else 0.0,
            }

        conviction_rows = _query(conn, """
            SELECT cs.conviction AS score, sp.return_20d AS realized_return
            FROM conviction_scores cs
            JOIN signal_performance sp ON sp.event_id = cs.event_id
            WHERE sp.return_20d IS NOT NULL
        """)

        ensemble_perf_rows = _query(conn, f"""
            SELECT es.total_score AS score,
                   es.run_date AS date,
                   es.recommendation AS recommendation,
                   sp.return_20d AS realized_return,
                   {", ".join(f"es.{col} AS {component}" for component, col in ENSEMBLE_COMPONENT_COLUMNS.items())}
            FROM ({latest_ensemble_sql}) es
            JOIN signal_performance sp ON sp.event_id = es.event_id
            WHERE sp.return_20d IS NOT NULL
        """)

        component_correlations = {}
        for component in ENSEMBLE_COMPONENT_COLUMNS:
            component_correlations[component] = _corr(
                ensemble_perf_rows,
                component,
                "realized_return",
            )

        conviction_bucket_rows = _query(conn, """
            SELECT CASE WHEN cs.passes_threshold THEN 'high_conviction' ELSE 'other' END AS bucket,
                   sp.return_20d AS realized_return
            FROM signal_performance sp
            JOIN conviction_scores cs ON cs.event_id = sp.event_id
            WHERE sp.return_20d IS NOT NULL
        """)

        source_rows = _query(conn, """
            SELECT e.source_type AS bucket, sp.return_20d AS realized_return
            FROM smart_money_events e
            JOIN signal_performance sp ON sp.event_id = e.id
            WHERE sp.return_20d IS NOT NULL
        """)

        direction_rows = _query(conn, """
            SELECT e.direction AS bucket, sp.return_20d AS realized_return
            FROM smart_money_events e
            JOIN signal_performance sp ON sp.event_id = e.id
            WHERE sp.return_20d IS NOT NULL
        """)

        missing_claimed_components = [
            name for name, stats in component_presence.items()
            if stats["latest_rows_present"] == 0
        ]

        ensemble_records = [
            {
                "date": r["date"],
                "ensemble_score": r["score"],
                "realized_return": r["realized_return"],
            }
            for r in ensemble_perf_rows
            if r["score"] is not None and r["realized_return"] is not None
        ]
        ensemble_records = sorted(ensemble_records, key=lambda r: str(r["date"]))
        all_20d_returns = [float(r["realized_return"]) for r in ensemble_perf_rows if r["realized_return"] is not None]
        buy_20d_returns = [
            float(r["realized_return"])
            for r in ensemble_perf_rows
            if r["recommendation"] == "buy" and r["realized_return"] is not None
        ]

        calibration = calibration_analysis(ensemble_records)
        calibrator = ScoreCalibrator()
        isotonic = calibrator.fit(ensemble_records)
        all_sharpe = _annualized_sharpe(all_20d_returns)
        buy_sharpe = _annualized_sharpe(buy_20d_returns)
        n_strategies = len(ENSEMBLE_COMPONENT_COLUMNS) + 4
        dsr_all = (
            deflated_sharpe_ratio(
                observed_sharpe=all_sharpe,
                n_observations=len(all_20d_returns),
                n_strategies_tried=n_strategies,
                skewness=float(stats.skew(all_20d_returns)) if len(all_20d_returns) >= 3 else 0.0,
                kurtosis=float(stats.kurtosis(all_20d_returns, fisher=False)) if len(all_20d_returns) >= 4 else 3.0,
            )
            if all_sharpe is not None else {"status": "insufficient_data"}
        )
        dsr_buy = (
            deflated_sharpe_ratio(
                observed_sharpe=buy_sharpe,
                n_observations=len(buy_20d_returns),
                n_strategies_tried=n_strategies,
                skewness=float(stats.skew(buy_20d_returns)) if len(buy_20d_returns) >= 3 else 0.0,
                kurtosis=float(stats.kurtosis(buy_20d_returns, fisher=False)) if len(buy_20d_returns) >= 4 else 3.0,
            )
            if buy_sharpe is not None else {"status": "insufficient_data"}
        )

        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "database": str(db_path),
            "table_counts": table_counts,
            "ensemble": {
                "latest_rows": len(latest_rows),
                "n_models_distribution": n_models_distribution,
                "component_presence": component_presence,
                "missing_latest_components": missing_claimed_components,
            },
            "performance_20d": {
                "conviction_correlation": _corr(conviction_rows, "score", "realized_return"),
                "latest_ensemble_correlation": _corr(ensemble_perf_rows, "score", "realized_return"),
                "component_correlations": component_correlations,
                "recommendation_buckets": _bucket_summary(
                    ensemble_perf_rows, "recommendation", "realized_return"
                ),
                "conviction_buckets": _bucket_summary(
                    conviction_bucket_rows, "bucket", "realized_return"
                ),
                "source_buckets": _bucket_summary(source_rows, "bucket", "realized_return"),
                "direction_buckets": _bucket_summary(direction_rows, "bucket", "realized_return"),
            },
            "validation_20d": {
                "calibration": calibration,
                "isotonic_calibrator": isotonic,
                "bootstrap_all": bootstrap_metrics(all_20d_returns),
                "bootstrap_buy_recommendations": bootstrap_metrics(buy_20d_returns),
                "deflated_sharpe_all": dsr_all,
                "deflated_sharpe_buy_recommendations": dsr_buy,
                "n_strategies_tried_assumption": n_strategies,
            },
            "xgboost": _xgboost_status(),
        }
        report["validation_20d"]["acceptance_gates"] = _validation_gates(report)

        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2, default=_json_default))

        print(f"saved {OUT}")
        print(json.dumps({
            "ensemble_latest_rows": report["ensemble"]["latest_rows"],
            "missing_latest_components": report["ensemble"]["missing_latest_components"],
            "conviction_spearman_20d": report["performance_20d"]["conviction_correlation"]["spearman"],
            "ensemble_spearman_20d": report["performance_20d"]["latest_ensemble_correlation"]["spearman"],
            "validation_gates": report["validation_20d"]["acceptance_gates"],
            "xgboost_status": report["xgboost"]["status"],
        }, indent=2, default=_json_default))
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
