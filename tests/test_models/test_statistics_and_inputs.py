"""Performance statistics, symbol handling, scorers and the classifier."""

import asyncio
import math
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.analysis.sector_momentum import compute_sector_momentum
from src.analysis.xgb_features import build_feature_record
from src.analysis.xgboost_classifier import FEATURE_COLS, LABEL_HORIZON_DAYS, XGBoostSignalClassifier
from src.backtesting.backtester import Backtester
from src.backtesting.validator import (
    ScoreCalibrator,
    annualized_sharpe,
    bootstrap_metrics,
    deflated_sharpe_ratio,
    evaluate_split,
    model_contribution_analysis,
)
from src.clients.symbols import classify_security, normalize_symbol, yahoo_candidates
from src.models.schemas import SourceType
from src.scoring.signal_scorer import SignalScorer


# ── Sharpe ────────────────────────────────────────────────────────────────

def test_sharpe_annualizes_by_holding_period_not_by_days():
    returns = list(np.random.default_rng(1).normal(0.02, 0.10, 400))
    per_period = np.mean(returns) / np.std(returns, ddof=1)
    assert annualized_sharpe(returns, hold_days=20) == pytest.approx(per_period * math.sqrt(252 / 20))
    assert annualized_sharpe(returns, hold_days=1) == pytest.approx(per_period * math.sqrt(252))
    records = [{"realized_return": r} for r in returns]
    assert evaluate_split(records)["sharpe_ratio"] == pytest.approx(annualized_sharpe(returns, 20), abs=1e-4)


def test_every_sharpe_in_the_validator_agrees():
    returns = list(np.random.default_rng(2).normal(0.03, 0.08, 300))
    point = annualized_sharpe(returns, 20)
    boot = bootstrap_metrics(returns, n_bootstrap=400)["sharpe_ratio"]
    assert boot["lower"] < point < boot["upper"]
    assert boot["point"] == pytest.approx(point, rel=0.15)


def test_deflated_sharpe_uses_per_period_units():
    periods = 252 / 20
    # annualized 1.0 on 150 independent periods, 15 strategies tried
    out = deflated_sharpe_ratio(1.0, n_observations=150, n_strategies_tried=15, periods_per_year=periods)
    sr = 1.0 / math.sqrt(periods)
    assert out["observed_sharpe_per_period"] == pytest.approx(sr, abs=1e-4)
    expected_max = math.sqrt(1 / 149) * 1.7707  # sqrt(V) * E[max of 15 standard normals]
    assert out["expected_max_sharpe_per_period"] == pytest.approx(expected_max, abs=1e-3)
    stat = (sr - expected_max) / math.sqrt((1 + 0.5 * sr**2) / 149)
    assert out["dsr_statistic"] == pytest.approx(stat, abs=1e-2)
    assert 0 <= out["deflated_sharpe_ratio"] <= 1


def test_deflated_sharpe_gets_harder_with_more_trials_and_fewer_observations():
    kw = dict(periods_per_year=252 / 20)
    base = deflated_sharpe_ratio(2.0, 100, 5, **kw)["p_value"]
    assert deflated_sharpe_ratio(2.0, 100, 500, **kw)["p_value"] > base
    assert deflated_sharpe_ratio(2.0, 20, 5, **kw)["p_value"] > base
    assert deflated_sharpe_ratio(2.0, 5, 5, **kw)["status"] == "insufficient_data"


def test_leave_one_out_baseline_uses_the_same_filter():
    rng = np.random.default_rng(0)
    records = [{"a": float(rng.uniform(0, 100)), "b": float(rng.uniform(0, 100)),
                "realized_return": float(rng.normal(0, 0.05))} for _ in range(200)]

    def score_fn(r, excluded):
        used = [m for m in ("a", "b") if m not in excluded]
        return sum(r[m] for m in used) / len(used)

    out = model_contribution_analysis(records, score_fn, ["a", "b"])
    n_pass = sum(1 for r in records if score_fn(r, []) >= 50)
    assert out["baseline"]["n"] == n_pass < len(records)


def test_calibrator_reports_brier_against_the_base_rate():
    rng = np.random.default_rng(0)
    records = []
    for i in range(300):
        score = float(rng.uniform(0, 100))
        records.append({"date": f"2026-{1 + i // 30:02d}-{1 + i % 28:02d}", "ensemble_score": score,
                        "realized_return": 0.05 if rng.random() < score / 100 else -0.05})
    stats = ScoreCalibrator().fit(list(reversed(records)))  # unsorted on purpose
    assert stats["oos_brier"] < stats["oos_brier_base_rate"] and stats["beats_base_rate"]


def test_backtest_metrics_are_order_stable_and_sortino_uses_all_trades():
    bt = Backtester.__new__(Backtester)
    returns = [0.10, -0.05, 0.02, -0.01, 0.04]
    m = bt._compute_metrics(returns, hold_days=30)
    downside = math.sqrt((0.05**2 + 0.01**2) / 5)
    assert m.sortino_ratio == pytest.approx(np.mean(returns) / downside * math.sqrt(365 / 30), abs=1e-3)
    assert m.sharpe_ratio == pytest.approx(np.mean(returns) / np.std(returns, ddof=1) * math.sqrt(365 / 30), abs=1e-3)


# ── symbols ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,stored", [
    ("NYSE: VTEX", "VTEX"), ("NYSE:FLG", "FLG"), ("nasdaq: aapl", "AAPL"),
    ("BRK/B", "BRK.B"), (" msft ", "MSFT"), ("", None), (None, None), ("WAY TOO LONG SYM", None),
])
def test_normalize_symbol(raw, stored):
    assert normalize_symbol(raw) == stored


@pytest.mark.parametrize("raw,first", [
    ("BRK/B", "BRK-B"), ("BRK.B", "BRK-B"), ("ARES.PRB", "ARES-PB"),
    ("NYSE: KRC", "KRC"), ("AAPL", "AAPL"), ("^GSPC", "^GSPC"),
])
def test_yahoo_candidates_lead_with_the_price_source_spelling(raw, first):
    assert yahoo_candidates(raw)[0] == first


def test_ambiguous_symbols_keep_the_literal_form_first():
    assert yahoo_candidates("AMH-H") == ["AMH-H", "AMH-PH"]
    assert yahoo_candidates("UHALB") == ["UHALB", "UHAL-B"]
    assert yahoo_candidates("GOOGL") == ["GOOGL"]


def test_notes_and_preferreds_are_not_common_stock():
    equity = {"quoteType": "EQUITY"}
    assert classify_security("AAPL", {**equity, "longName": "Apple Inc."}) == "common"
    assert classify_security("CGABL", {**equity, "longName": "The Carlyle Group Inc. 4.625% Subordinated Notes due 2061"}) == "preferred_or_debt"
    assert classify_security("ARES-PB", {**equity, "longName": "Ares Management Corp"}) == "preferred_or_debt"
    assert classify_security("SPY", {"quoteType": "ETF"}) == "fund_or_other"


# ── scorers ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("actor,boosted", [
    ("Cook Taylor", False), ("Cooper Jane", False), ("Coons Jordan", False),
    ("Lin Morgan (VP)", False), ("Doe Jane (Senior Vice President)", False),
    ("Yeo Sam (COO)", True), ("Mills Pat (President)", True),
    ("Lomax Terry (Chief Financial Officer)", True), ("Tully Robin K (President and CEO)", True),
])
def test_title_boost_reads_whole_words_in_the_title_only(actor, boosted):
    assert SignalScorer()._actor_reputation(actor, SourceType.INSIDER) == (0.7 if boosted else 0.5)


def test_chairman_is_boosted_once_and_names_never_match():
    s = SignalScorer()
    assert s._actor_reputation("LEFKOWITZ ERIN P (CEO and Chairman)", SourceType.INSIDER) == 0.7
    assert s._actor_reputation("Rep. Tom Cooper (R-OH)", SourceType.CONGRESSIONAL) == 0.6


def _sector_row(ticker, verdict, buy, sell, sector):
    return {"ticker": ticker, "verdict": verdict, "buy_score": buy, "sell_score": sell, "sector": sector, "models": {}}


def test_sector_momentum_is_signed():
    out = compute_sector_momentum([
        _sector_row("A", "SELL", 30, 70, "Energy"), _sector_row("B", "SELL", 35, 68, "Energy"),
        _sector_row("C", "HOLD", 50, 50, "Energy"),
        _sector_row("D", "BUY", 70, 30, "Technology"), _sector_row("E", "BUY", 66, 35, "Technology"),
        _sector_row("F", "HOLD", 50, 50, "Technology"),
        _sector_row("G", "HOLD", 50, 50, "Utilities"),
    ])
    assert out["Energy"]["recommendation"] == "underweight" and out["Energy"]["momentum_score"] < 40
    assert out["Technology"]["recommendation"] == "overweight" and out["Technology"]["momentum_score"] > 60
    assert out["Utilities"]["momentum_score"] == 50.0


# ── classifier ────────────────────────────────────────────────────────────

def _records(n, signal=True, seed=0):
    g = np.random.default_rng(seed)
    out = []
    for i in range(n):
        mom = g.normal(0, 0.1)
        p = 1 / (1 + np.exp(-(8 * mom if signal else 0)))
        r = build_feature_record(
            direction="buy",
            enrichment={"momentum_30d": mom, "market_cap": float(np.exp(g.normal(22, 2))),
                        "rsi_14d": g.uniform(20, 80), "avg_volume_30d": float(np.exp(g.normal(13, 1))),
                        "pe_ratio": None if g.random() < 0.3 else g.uniform(5, 40)},
            date=(datetime(2025, 1, 1) + timedelta(days=2 * i)).isoformat(),
        )
        r["return_20d"] = 0.05 if g.random() < p else -0.05
        out.append(r)
    return out


def test_feature_record_never_substitutes_numbers_for_unknowns():
    record = build_feature_record(direction="sell", enrichment={"momentum_30d": 0.1})
    assert set(FEATURE_COLS) <= set(record)
    assert record["avg_volume_30d"] is None and record["conviction"] is None
    assert record["momentum_30d"] == 0.1 and record["direction_encoded"] == -1.0


def test_purge_drops_training_rows_whose_outcome_was_unknown_at_test_time():
    start = datetime(2026, 1, 1)
    dates = [start + timedelta(days=i) for i in range(100)]
    kept = XGBoostSignalClassifier._purge(np.arange(0, 80), np.arange(80, 100), dates)
    assert max(dates[i] for i in kept) <= dates[80] - timedelta(days=LABEL_HORIZON_DAYS)
    assert len(kept) == 80 - LABEL_HORIZON_DAYS + 1
    # unknown dates: nothing to purge on
    assert len(XGBoostSignalClassifier._purge(np.arange(0, 80), np.arange(80, 100), [None] * 100)) == 80


def test_logistic_path_trains_on_unscaled_features_and_finds_the_signal():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = XGBoostSignalClassifier()
        metrics = clf.train(_records(150, signal=True, seed=150))
    assert metrics["status"] == "trained" and metrics["model_type"] == "logistic"
    assert metrics["cv_purged_rows"] > 0
    assert metrics["cv_auc_mean"] > 0.6 and metrics["is_better_than_null"]
    assert next(iter(metrics["feature_importance"])) == "momentum_30d"
    # market cap is ~1e10 next to momentum ~0.1: without scaling it dominates
    pred = clf.predict({"momentum_30d": 0.25, "direction_encoded": 1.0})
    assert pred["probability_profitable"] > 0.6


def test_missing_features_are_imputed_inside_the_pipeline():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = XGBoostSignalClassifier()
        clf.train(_records(120, seed=3))
        assert hasattr(clf.model, "named_steps")
        assert clf.predict({}) is not None
