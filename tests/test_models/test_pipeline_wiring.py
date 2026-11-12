"""The repaired models reach each signal the way the pipeline wires them."""

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from config.settings import Settings
from src.analysis.event_study import EventStudyAnalyzer
from src.analysis.weather_overlay import WeatherOverlay
from src.api import api
from src.clients.options import METHOD_VERSION
from src.models.database import (
    BayesianDecayResult,
    EventStudyResult,
    FamaFrenchExposure,
    HMMRegimeState,
    OptionsFlow,
    SignalPerformance,
    SmartMoneyEvent,
    WeatherSnapshot,
    get_session_factory,
    init_db,
)
from src.models.schemas import Direction, SourceType
from src.pipeline.orchestrator import Orchestrator

NOW = datetime.utcnow()


def _prices(n=500, drift=0.0, jump_at=None, seed=0, vol=0.01):
    """Business-day price history ending today."""
    g = np.random.default_rng(seed)
    dates = list(pd.bdate_range(end=NOW.date(), periods=n))
    rets = g.normal(drift, vol, n - 1)
    if jump_at is not None:
        rets[jump_at] += 0.08
    closes = 50 * np.exp(np.concatenate([[0.0], np.cumsum(rets)]))
    return {"dates": dates, "closes": closes, "returns": np.diff(np.log(closes)),
            "volumes": np.full(n, 1e6)}


class FakeYahoo:
    def __init__(self, data):
        self.data = data

    async def get_price_history(self, ticker, days=504):
        return self.data.get(ticker)


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'p.db'}")
    settings = Settings(_env_file=None)
    p = Orchestrator(settings)
    p.engine = init_db(settings.database_url)
    p.session_factory = get_session_factory(p.engine)
    p.event_study = EventStudyAnalyzer()
    yield p
    p.engine.dispose()


def _event(session, ticker, days_ago, direction=Direction.BUY, source=SourceType.INSIDER, trade_lag=3):
    disclosed = NOW - timedelta(days=days_ago)
    e = SmartMoneyEvent(
        ticker=ticker, actor=f"Insider {ticker} {days_ago}", direction=direction,
        trade_date=disclosed - timedelta(days=trade_lag), disclosure_date=disclosed,
        source_type=source,
    )
    session.add(e)
    session.flush()
    return e.id


# ── event studies ─────────────────────────────────────────────────────────

def test_event_study_job_measures_matured_signals_from_disclosure(pipeline):
    dates = list(pd.bdate_range(end=NOW.date(), periods=500))
    disclosed_idx = len(dates) - 60
    days_ago = (NOW.date() - dates[disclosed_idx].date()).days
    # +8% on the first trading day AFTER disclosure; also +8% two days BEFORE
    # it (between trade and disclosure), which a follower could not capture.
    stock = _prices(jump_at=disclosed_idx, seed=1, vol=0.002)   # return index i -> date i+1
    stock["returns"][disclosed_idx - 3] += 0.08
    pipeline.yahoo = FakeYahoo({"SPY": _prices(seed=2, vol=0.002), "ACME": stock})

    with pipeline.session_factory() as s:
        matured = _event(s, "ACME", days_ago)
        fresh = _event(s, "ACME", 5)
        s.commit()

    asyncio.run(pipeline._run_event_studies())

    with pipeline.session_factory() as s:
        rows = s.query(EventStudyResult).all()
    assert [r.event_id for r in rows] == [matured]          # the fresh one must wait
    assert fresh not in [r.event_id for r in rows]
    row = rows[0]
    assert row.anchor == "disclosure"
    assert row.car_1d == pytest.approx(0.08, abs=0.02)       # the post-disclosure jump
    assert row.car_20d == pytest.approx(0.08, abs=0.04)      # pre-disclosure +8% excluded

    asyncio.run(pipeline._run_event_studies())               # idempotent
    with pipeline.session_factory() as s:
        assert s.query(EventStudyResult).count() == 1


def test_unmeasurable_events_are_marked_once_they_are_old(pipeline):
    pipeline.yahoo = FakeYahoo({"SPY": _prices(seed=2)})     # no data for the ticker
    with pipeline.session_factory() as s:
        _event(s, "GONE", 90)
        _event(s, "GONE", 40)                                # may still get prices: keep trying
        s.commit()
    asyncio.run(pipeline._run_event_studies())
    with pipeline.session_factory() as s:
        rows = s.query(EventStudyResult).all()
    assert len(rows) == 1 and rows[0].car_20d is None and rows[0].anchor == "disclosure"


def _seed_event_studies(session, n_days, per_day, car, source="insider", direction="buy"):
    import json
    daily = list(np.cumsum(np.concatenate([np.zeros(5), np.full(21, car / 21)])))
    for d in range(n_days):
        for k in range(per_day):
            eid = _event(session, f"T{d}_{k}", 40 + d)
            session.add(EventStudyResult(
                ticker=f"T{d}_{k}", event_id=eid, run_date=NOW, direction=direction, source_type=source,
                car_1d=car / 21, car_5d=car * 6 / 21, car_10d=car * 11 / 21, car_20d=car,
                daily_cars=json.dumps(daily), anchor="disclosure",
                anchor_date=NOW - timedelta(days=40 + d),
            ))


def test_pooled_alpha_appears_only_with_enough_disclosure_dates(pipeline):
    with pipeline.session_factory() as s:
        _seed_event_studies(s, n_days=5, per_day=30, car=0.04)
        s.commit()
    assert pipeline._pooled_signal_alpha() == {}             # 150 events, 5 dates

    with pipeline.session_factory() as s:
        for d in range(5, 25):
            eid = _event(s, f"U{d}", 40 + d)
            s.add(EventStudyResult(
                ticker=f"U{d}", event_id=eid, run_date=NOW, direction="buy", source_type="insider",
                car_5d=0.012, car_20d=0.04 + 0.001 * (d % 3),
                daily_cars=None, anchor="disclosure", anchor_date=NOW - timedelta(days=40 + d),
            ))
        # legacy trade-date-anchored and unmeasurable rows must be ignored
        eid = _event(s, "OLD", 200)
        s.add(EventStudyResult(ticker="OLD", event_id=eid, run_date=NOW, direction="buy",
                               source_type="insider", car_5d=-0.9, car_20d=-0.9, anchor=None))
        s.commit()

    pooled = pipeline._pooled_signal_alpha()
    stats = pooled[("insider", "buy")]["event_study"]
    assert stats["n_clusters"] == 25
    assert stats["mean_car_20d"] == pytest.approx(0.04, abs=0.002)
    assert stats["is_significant"]
    assert ("insider", "sell") not in pooled


def test_decay_fit_is_persisted_for_the_dashboard(pipeline):
    fit = {"n_days": 21, "total_car": 0.03, "posterior_half_life": 9.5, "posterior_lambda": 0.073,
           "ci_half_life_95": (5.0, 20.0), "entry_window_days": 4.9, "exit_window_days": 16.5,
           "annualized_ir": 1.2, "decay_quality": "fast_decay",
           "signal_strength": {5: {"remaining_pct": 69.4}, 20: {"remaining_pct": 23.2}}}
    with pipeline.session_factory() as s:
        eid = _event(s, "ACME", 2)
        s.commit()
    pipeline._persist_bayesian_decay("ACME", eid, NOW, "buy", fit)
    with pipeline.session_factory() as s:
        row = s.query(BayesianDecayResult).one()
    assert row.posterior_half_life == 9.5 and row.signal_strength_5d == 69.4
    assert (row.ci_half_life_lower, row.ci_half_life_upper) == (5.0, 20.0)


# ── options, weather, HMM, portfolio ──────────────────────────────────────

def _options_row(ticker, pcr, skew, version=METHOD_VERSION, days_ago=1):
    return OptionsFlow(ticker=ticker, analysis_date=NOW - timedelta(days=days_ago), pcr=pcr,
                       iv_skew=skew, unusual_volume_score=0.2, method_version=version)


def test_options_are_ranked_against_current_method_rows_only(pipeline):
    with pipeline.session_factory() as s:
        s.add_all([_options_row(f"R{i}", pcr=0.2 + 0.05 * i, skew=0.02 * i) for i in range(25)])
        s.add_all([_options_row(f"L{i}", pcr=40.0, skew=5000.0, version=None) for i in range(50)])
        s.add(_options_row("LEGACY", pcr=0.1, skew=0.9, version=None))
        s.commit()

    reference = pipeline._options_reference()
    assert len(reference["pcr"]) == 25 and max(reference["pcr"]) < 2

    with pipeline.session_factory() as s:
        low = pipeline._options_dict(s, s.query(OptionsFlow).filter_by(ticker="R1").one(), reference)
        high = pipeline._options_dict(s, s.query(OptionsFlow).filter_by(ticker="R23").one(), reference)
        legacy = pipeline._options_dict(s, s.query(OptionsFlow).filter_by(ticker="LEGACY").one(), reference)
    assert low["pcr_percentile"] < 0.15 < 0.85 < high["pcr_percentile"]
    assert legacy is None


def test_small_cycles_give_no_options_percentile(pipeline):
    with pipeline.session_factory() as s:
        s.add_all([_options_row(f"R{i}", 0.5 + 0.1 * i, 0.05) for i in range(5)])
        s.commit()
        row = s.query(OptionsFlow).filter_by(ticker="R1").one()
        out = pipeline._options_dict(s, row, pipeline._options_reference())
    assert out["pcr_percentile"] is None and out["iv_skew_percentile"] is None


def test_weather_context_builds_its_own_alert_baseline(pipeline):
    class FakeWeather:
        async def get_context(self):
            return {"temp_anomaly_c": 0.2, "degree_day_delta": 0.1, "n_severe_alerts": 100, "n_cities": 6}

    pipeline.weather = FakeWeather()
    first = asyncio.run(pipeline._weather_context())
    assert first["alert_z"] is None                           # no history yet
    assert WeatherOverlay.score("Financial Services", "buy", first) is None

    with pipeline.session_factory() as s:
        assert s.query(WeatherSnapshot).count() == 1
        s.add_all([WeatherSnapshot(snapshot_date=NOW - timedelta(days=d), n_severe_alerts=95 + d % 10)
                   for d in range(1, 15)])
        s.commit()
    later = asyncio.run(pipeline._weather_context())
    assert later["alert_baseline_days"] == 14 and abs(later["alert_z"]) < 1


def test_hmm_rows_store_folded_probabilities(pipeline):
    pipeline._persist_hmm("ACME", NOW, {
        "current_state": "bull", "current_state_detail": "mild_bull",
        "current_probabilities": {"mild_bull": 0.9, "bull": 0.05, "sideways": 0.03, "bear": 0.02},
        "prob_bull": 0.95, "prob_bear": 0.02, "prob_sideways": 0.03,
        "transition_matrix": {}, "transition_from_current": {"bull": 0.9, "bear": 0.04, "sideways": 0.06},
        "n_observations": 300,
    })
    with pipeline.session_factory() as s:
        row = s.query(HMMRegimeState).one()
    assert row.prob_bull + row.prob_bear + row.prob_sideways == pytest.approx(1.0)
    assert row.trans_to_bull == 0.9 and row.current_state == "bull"


def test_portfolio_candidates_are_top_conviction_common_stock_buys(pipeline):
    def scored(ticker, direction, conviction):
        return (SimpleNamespace(ticker=ticker, direction=SimpleNamespace(value=direction)), None,
                SimpleNamespace(conviction=conviction))

    pipeline._security_types = {"NOTE": "preferred_or_debt", "AAA": "common"}
    rows = [scored("AAA", "buy", 0.5), scored("BBB", "buy", 0.9), scored("SHORT", "sell", 0.99),
            scored("NOTE", "buy", 0.95), scored("BBB", "buy", 0.3)]
    rows += [scored(f"X{i}", "buy", 0.1) for i in range(60)]
    picks = pipeline._portfolio_candidates(rows)
    assert picks[:2] == ["BBB", "AAA"]
    assert "SHORT" not in picks and "NOTE" not in picks
    assert len(picks) == pipeline.MEANVAR_MAX_ASSETS


# ── API ───────────────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'api.db'}"
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("DATABASE_URL", database_url)
        patch.setenv("ADMIN_API_TOKEN", "")
        patch.setenv("SENTRY_DSN", "")
        patch.setattr(api, "settings", Settings(_env_file=None))
        with TestClient(api.app) as c:
            yield c
        api.app.state.session_factory.kw["bind"].dispose()


def _perf(session, ticker, source, direction, ret, entry=10.0):
    eid = _event(session, ticker, 60, direction=Direction(direction), source=SourceType(source))
    session.add(SignalPerformance(
        event_id=eid, ticker=ticker, direction=direction, source_type=source, entry_price=entry,
        entry_date=NOW - timedelta(days=59), return_5d=ret, return_20d=ret, return_60d=ret,
        is_winner_5d=(ret > 0) if ret is not None else None,
        is_winner_20d=(ret > 0) if ret is not None else None,
    ))


def test_performance_summary_excludes_unpriced_and_separates_13f(client):
    with api.app.state.session_factory() as s:
        _perf(s, "AAA", "insider", "buy", 0.10)
        _perf(s, "BBB", "congressional", "buy", 0.02)
        for i in range(40):
            _perf(s, f"F{i}", "institutional", "buy", -0.05)
        _perf(s, "BRK/B", "insider", "sell", 1.0, entry=0.0)   # legacy phantom +100%
        _perf(s, "XSP", "insider", "sell", None, entry=0.0)    # unpriced marker
        s.commit()

    body = client.get("/api/performance/summary").json()
    assert body["unpriced_signals"] == 2 and body["tracked_signals"] == 42
    assert body["headline_basis"] == "insider + congressional"
    assert body["avg_return_20d"] == pytest.approx(0.06)       # not dragged to -0.045 by 13F
    assert body["win_rate_20d"] == 1.0
    assert body["by_source"]["institutional"]["n"] == 40
    assert body["by_source"]["insider"]["n"] == 1
    assert "sell" not in body["by_direction"]
    assert all(w["return_20d"] < 1.0 for w in body["top_winners"])


def test_training_features_are_taken_as_of_the_signal(client):
    with api.app.state.session_factory() as s:
        eid = _event(s, "ACME", 90)
        evt = s.get(SmartMoneyEvent, eid)
        evt.created_at = NOW - timedelta(days=90)
        for days_ago, alpha in [(120, 0.01), (90, 0.02), (3, 0.99)]:
            s.add(FamaFrenchExposure(ticker="ACME", run_date=NOW - timedelta(days=days_ago),
                                     alpha_annual=alpha))
        s.commit()
        history = api._xgb_model_history(s, {"ACME"})
        record = api._build_xgb_record(evt, None, None, history)
    # 0.99 was estimated over the outcome window itself
    assert record["ff_alpha"] == 0.02
    assert record["avg_volume_30d"] is None and record["hmm_bull_prob"] is None


def test_ticker_analysis_reads_the_stored_decay_fit_for_recent_signals(pipeline):
    from src.analysis.pooled_alpha import recent_decay_components

    url = str(pipeline.engine.url)
    fit = {"posterior_lambda": 0.0693, "posterior_half_life": 10.0, "prob_alpha_positive": 0.9,
           "decay_quality": "moderate_decay", "signal_strength": {}}
    with pipeline.session_factory() as s:
        recent_buy = _event(s, "ACME", 7)
        old_sell = _event(s, "ACME", 80, direction=Direction.SELL)
        s.commit()
    pipeline._persist_bayesian_decay("ACME", recent_buy, NOW, "buy", fit)
    pipeline._persist_bayesian_decay("ACME", old_sell, NOW, "sell", fit)

    buy, sell = recent_decay_components("ACME", database_url=url)
    assert sell is None                                       # signal too old to matter
    assert buy["prob_alpha_positive"] == 0.9
    assert buy["remaining_share"] == pytest.approx(np.exp(-0.0693 * 5), abs=0.02)  # 7 days ~ 5 trading days
    assert recent_decay_components("NOPE", database_url=url) == (None, None)
    assert recent_decay_components("ACME", database_url="sqlite:////nonexistent/dir/x.db") == (None, None)


def test_event_study_summary_tests_significance_across_disclosure_dates(client):
    # 300 signals disclosed on just 3 days, all sharing that day's +4% move
    with api.app.state.session_factory() as s:
        for d in range(3):
            for k in range(100):
                eid = _event(s, f"E{d}_{k}", 40 + d)
                s.add(EventStudyResult(
                    ticker=f"E{d}_{k}", event_id=eid, run_date=NOW, direction="buy", source_type="insider",
                    car_1d=0.01, car_5d=0.02, car_10d=0.03, car_20d=0.04 + 0.001 * d,
                    is_significant=False, anchor="disclosure", anchor_date=NOW - timedelta(days=40 + d),
                ))
        # a legacy trade-date-anchored row and an unmeasurable marker: both ignored
        eid = _event(s, "LEGACY", 200)
        s.add(EventStudyResult(ticker="LEGACY", event_id=eid, run_date=NOW, direction="buy",
                               source_type="insider", car_5d=-0.5, car_20d=-0.5))
        eid = _event(s, "GONE", 90)
        s.add(EventStudyResult(ticker="GONE", event_id=eid, run_date=NOW, direction="buy",
                               source_type="insider", anchor="disclosure"))
        s.commit()

    body = client.get("/api/analysis/event-study/summary").json()
    assert body["n_events"] == 300 and body["n_clusters"] == 3 and body["anchor"] == "disclosure"
    assert body["mean_car_20d"] == pytest.approx(0.041, abs=1e-4)
    # df = 2, not 299: three days cannot give a five-sigma result
    assert body["by_source"]["insider"]["car_20d"]["n_clusters"] == 3
    assert body["p_value_20d"] > 1e-4
    assert body["win_rate_20d"] == 1.0


def test_api_does_not_serve_results_from_before_the_fixes(client):
    from src.models.database import MeanVarianceResult

    with api.app.state.session_factory() as s:
        # newest row is the broken optimizer's signature: identical returns, no frontier, 100% one stock
        s.add(MeanVarianceResult(
            run_date=NOW, n_assets=2, tickers='["WDC", "AAA"]', ms_weights='{"WDC": 1.0, "AAA": 0.0}',
            mv_weights='{"WDC": 0.1, "AAA": 0.9}', ms_return=-0.155, mv_return=-0.155, ew_return=-0.155,
            ms_volatility=0.84, ms_sharpe=-0.24, efficient_frontier="[]", risk_contribution="{}",
        ))
        s.add(MeanVarianceResult(
            run_date=NOW - timedelta(days=1), n_assets=2, tickers='["AAA", "BBB"]',
            ms_weights='{"AAA": 0.6, "BBB": 0.4}', mv_weights='{"AAA": 0.5, "BBB": 0.5}',
            ms_return=0.12, mv_return=0.09, ew_return=0.10, ms_volatility=0.2, ms_sharpe=0.4,
            efficient_frontier='[{"return": 0.09, "volatility": 0.18}]', risk_contribution="{}",
        ))
        s.add(HMMRegimeState(ticker="LEG", run_date=NOW, current_state="mild_bull",
                             prob_bull=0.01, prob_bear=0.01, prob_sideways=0.0, trans_to_bull=0.0))
        s.add(OptionsFlow(ticker="LEG", analysis_date=NOW, pcr=0.8, iv_skew=47927.9, method_version=None))
        s.commit()

    mv = client.get("/api/analysis/mean-variance").json()
    assert mv["tickers"] == ["AAA", "BBB"] and mv["max_sharpe"]["weights"]["AAA"] == 0.6

    detail = client.get("/api/analysis/LEG").json()
    assert detail["hmm"]["current_state"] == "bull" and detail["hmm"]["prob_bull"] is None
    assert detail["options_flow"]["pcr"] == 0.8 and detail["options_flow"]["iv_skew"] is None
