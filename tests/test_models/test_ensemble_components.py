"""Ensemble scorer and the components repaired in the 2026-11 audit."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.earnings_overlay import EarningsOverlay
from src.analysis.ensemble_scoring import EnsembleScorer
from src.analysis.pooled_alpha import (
    pooled_event_stats,
    post_event_abnormal_returns,
    trading_days_since,
)
from src.analysis.weather_overlay import WeatherOverlay
from src.clients.fama_french import FamaFrenchClient
from src.clients.options import OptionsClient, percentile_rank

scorer = EnsembleScorer()


# ── direction symmetry ────────────────────────────────────────────────────

@pytest.mark.parametrize("fn,payload", [
    ("_score_hmm", {"prob_bull": 0.6, "prob_bear": 0.1, "prob_sideways": 0.3}),
    ("_score_fama_french", {"alpha_tstat": 1.7}),
    ("_score_options_flow", {"pcr_percentile": 0.2, "iv_skew_percentile": 0.3, "unusual_volume_score": 0.8}),
    ("_score_monte_carlo", {"horizons": {30: {"probability_of_profit": 0.62, "expected_return": 0.03}}}),
])
def test_directional_components_mirror_between_buy_and_sell(fn, payload):
    buy = getattr(scorer, fn)(payload, "buy")
    sell = getattr(scorer, fn)(payload, "sell")
    assert buy > 50 > sell
    assert buy + sell == pytest.approx(100.0)


def test_positive_alpha_no_longer_supports_a_sell():
    assert scorer._score_fama_french({"alpha_tstat": 2.5}, "sell") < 50
    # no standard error -> no score, rather than one built on a noisy point estimate
    assert scorer._score_fama_french({"alpha_annual": 0.4}, "buy") is None


def test_neutral_inputs_score_exactly_fifty():
    assert scorer._score_garch({"forecast_5d_ratio": 1.0, "forecast_20d_ratio": 1.0}) == 50.0
    assert scorer._score_hmm({"prob_bull": 0, "prob_bear": 0, "prob_sideways": 1.0}, "buy") == 50.0
    assert scorer._score_event_study({"mean_car_5d": 0.0, "mean_car_20d": 0.0}, "buy") == 50.0
    assert scorer._score_options_flow({"pcr_percentile": 0.5, "iv_skew_percentile": 0.5}, "buy") == 50.0


def test_components_without_usable_input_are_absent():
    result = scorer.score(
        direction="buy",
        fama_french={"alpha_annual": 0.2},          # no t-stat
        options_flow={"pcr": 0.4},                  # no percentile reference
        bayesian_decay={"decay_quality": "slow_decay"},
        hmm={"prob_bull": 0.8, "prob_bear": 0.1, "prob_sideways": 0.1},
    )
    assert set(result["components"]) == {"hmm_regime"}


# ── agreement gate ────────────────────────────────────────────────────────

BULLISH = dict(
    monte_carlo={"horizons": {30: {"probability_of_profit": 0.75, "expected_return": 0.08}}},
    hmm={"prob_bull": 0.9, "prob_bear": 0.05, "prob_sideways": 0.05},
    fama_french={"alpha_tstat": 2.5},
    options_flow={"pcr_percentile": 0.1, "iv_skew_percentile": 0.1},
)


def test_risk_components_do_not_vote():
    calm = {"garch": {"forecast_5d_ratio": 0.6, "forecast_20d_ratio": 0.6}, "copula": {"tail_risk_score": 5}}
    only_risk = scorer.score(direction="buy", **calm)
    assert only_risk["n_directional_votes"] == 0
    assert only_risk["total_score"] >= 55 and only_risk["recommendation"] == "hold"

    with_views = scorer.score(direction="buy", **calm, **BULLISH)
    assert with_views["n_directional_votes"] == 4
    assert with_views["agreement_met"] and with_views["recommendation"] in ("buy", "strong_buy")


def test_neutral_components_abstain_from_the_vote():
    result = scorer.score(
        direction="buy", **BULLISH,
        earnings_overlay={"score": 50},
        news_sentiment={"n_articles": 1, "news_sentiment_mean": 0, "news_sentiment_trend_3d": 0, "news_volume_z": 0},
    )
    assert result["n_models"] == 6
    assert result["n_directional_votes"] == 4 and result["agreement_ratio"] == 1.0


def test_split_models_are_held_even_with_a_high_total():
    result = scorer.score(
        direction="buy",
        monte_carlo={"horizons": {30: {"probability_of_profit": 0.95, "expected_return": 0.2}}},
        hmm={"prob_bull": 1.0, "prob_bear": 0.0, "prob_sideways": 0.0},
        fama_french={"alpha_tstat": -0.3},
        options_flow={"pcr_percentile": 0.6, "iv_skew_percentile": 0.6},
        earnings_overlay={"score": 48},
    )
    assert result["total_score"] >= 55
    assert result["agreement_ratio"] < 0.6 and result["recommendation"] == "hold"


# ── earnings ──────────────────────────────────────────────────────────────

def _earnings(reported, upcoming=1):
    n = len(reported)
    idx = pd.date_range("2026-12-01", periods=upcoming, freq="91D").tolist()[::-1] + \
        pd.date_range(end="2026-09-01", periods=n, freq="91D").tolist()[::-1]
    return pd.DataFrame({
        "EPS Estimate": [1.0] * (upcoming + n),
        "Reported EPS": [np.nan] * upcoming + list(reported),
    }, index=pd.DatetimeIndex(idx))


def test_unreported_quarters_are_not_counted_as_misses():
    history = EarningsOverlay()._get_surprise_history(_earnings([1.1, 1.1, 1.1, 1.1], upcoming=3))
    assert history["n_quarters"] == 4
    assert history["beat_rate"] == 1.0
    assert history["last_surprise_pct"] == pytest.approx(0.1)
    assert np.isfinite(history["avg_surprise_pct"])


def test_quarters_reported_after_the_signal_are_ignored():
    df = _earnings([0.5, 1.2, 1.2, 1.2, 1.2], upcoming=0)  # latest quarter was a big miss
    as_of = df.index.sort_values()[-2].to_pydatetime()
    history = EarningsOverlay()._get_surprise_history(df, as_of=as_of)
    assert history["n_quarters"] == 4 and history["beat_rate"] == 1.0


def test_earnings_proximity_alone_moves_nothing():
    eo = EarningsOverlay()
    for direction in ("buy", "sell"):
        assert eo._compute_score(direction, True, 3, {"beat_rate": None}) == 50.0
        assert eo._compute_score(direction, False, None, {"beat_rate": 1.0, "n_quarters": 8}) == 50.0


def test_earnings_score_is_symmetric_around_fifty():
    eo = EarningsOverlay()
    history = {"beat_rate": 1.0, "n_quarters": 8, "last_surprise_pct": 0.1}
    buy, sell = eo._compute_score("buy", True, 5, history), eo._compute_score("sell", True, 5, history)
    assert buy > 50 > sell and buy + sell == pytest.approx(100.0)
    typical = {"beat_rate": 0.70, "n_quarters": 8, "last_surprise_pct": 0.0}
    assert eo._compute_score("buy", True, 5, typical) == pytest.approx(50.0)


# ── options ───────────────────────────────────────────────────────────────

def _chain(put_slope=0.004, call_slope=0.001):
    strikes = np.arange(80, 121, 2.5)
    base = dict(strike=strikes, openInterest=500, volume=50)
    calls = pd.DataFrame({**base, "impliedVolatility": 0.30 + call_slope * np.abs(strikes - 100)})
    puts = pd.DataFrame({**base, "impliedVolatility": 0.30 + np.where(strikes < 100, put_slope, call_slope) * np.abs(strikes - 100)})
    return calls, puts


def test_junk_quotes_do_not_move_the_skew():
    calls, puts = _chain()
    clean = OptionsClient._compute_iv_skew(calls, puts, 100.0)
    junk = pd.DataFrame({"strike": [92.5, 95.0, 105.0], "impliedVolatility": [1e-5, 480.0, 300.0],
                         "openInterest": [900, 900, 900], "volume": [1, 1, 1]})
    illiquid = pd.DataFrame({"strike": [95.0], "impliedVolatility": [2.5], "openInterest": [0], "volume": [0]})
    dirty = OptionsClient._compute_iv_skew(pd.concat([calls, junk]), pd.concat([puts, junk, illiquid]), 100.0)
    assert dirty == pytest.approx(clean)
    assert 0 < clean < 0.5


def test_symmetric_smile_has_no_skew():
    calls, puts = _chain(put_slope=0.002, call_slope=0.002)
    assert OptionsClient._compute_iv_skew(calls, puts, 100.0) == pytest.approx(0.0, abs=1e-6)


def test_put_call_ratio_needs_real_volume_and_is_capped():
    assert OptionsClient._compute_pcr(3, 40) is None
    assert OptionsClient._compute_pcr(500, 300) == 0.6
    assert OptionsClient._compute_pcr(10, 5000) == 10.0


def test_percentile_rank():
    assert percentile_rank(0.3, [0.1, 0.2, 0.3, 0.5, 0.9]) == 0.5
    assert percentile_rank(5, [1, 2, 3]) == 1.0
    assert percentile_rank(None, [1, 2]) is None and percentile_rank(1, []) is None


# ── weather ───────────────────────────────────────────────────────────────

CTX = {"temp_anomaly_c": 0.4, "degree_day_delta": 0.3, "n_severe_alerts": 100}


def test_alert_channels_wait_for_a_baseline():
    thin = WeatherOverlay.with_alert_baseline(CTX, [100] * 5)
    assert thin["alert_z"] is None
    assert WeatherOverlay.score("Financial Services", "buy", thin) is None
    assert WeatherOverlay.score("Industrials", "buy", CTX) is None


def test_a_normal_alert_count_is_neutral_and_a_spike_is_not():
    usual = [90, 110, 100, 95, 105, 100, 120, 80, 100, 100, 110, 90]
    normal = WeatherOverlay.with_alert_baseline(CTX, usual)
    assert WeatherOverlay.score("Financial Services", "buy", normal)["score"] == pytest.approx(50.0, abs=0.5)

    spike = WeatherOverlay.with_alert_baseline({**CTX, "n_severe_alerts": 220}, usual)
    buy = WeatherOverlay.score("Financial Services", "buy", spike)["score"]
    sell = WeatherOverlay.score("Financial Services", "sell", spike)["score"]
    assert buy < 45 and buy + sell == pytest.approx(100.0)


def test_readings_inside_the_normal_band_say_nothing():
    assert WeatherOverlay.score("Utilities", "buy", CTX)["score"] == 50.0
    assert WeatherOverlay.score("Consumer Defensive", "buy", CTX)["score"] == 50.0
    hot = {**CTX, "temp_anomaly_c": 5.0, "degree_day_delta": 4.0}
    assert WeatherOverlay.score("Utilities", "buy", hot)["score"] > 50
    assert WeatherOverlay.score("Consumer Defensive", "buy", hot)["score"] < 50


def test_temperature_anomaly_is_measured_against_the_same_week_in_prior_years():
    forecast = {f"2026-11-{d:02d}": 6.0 for d in range(11, 18)}
    # every prior year this week averaged 6C: a cold week in November is normal
    archive = {f"{y}-11-{d:02d}": 6.0 for y in range(2021, 2026) for d in range(11, 18)}
    anomaly, dd_delta = WeatherOverlay._city_anomaly(forecast, archive)
    assert anomaly == pytest.approx(0.0) and dd_delta == pytest.approx(0.0)
    warm_years = {k: 10.0 for k in archive}
    assert WeatherOverlay._city_anomaly(forecast, warm_years)[0] == pytest.approx(-4.0)
    assert WeatherOverlay._city_anomaly(forecast, {}) is None


# ── Fama-French ───────────────────────────────────────────────────────────

def _factor_frame(n=250, seed=0):
    g = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-01", periods=n)
    frame = pd.DataFrame(g.normal(0, 0.01, (n, 5)), index=idx, columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA"])
    frame["RF"] = 0.0001
    return frame


def test_alpha_tstat_separates_real_alpha_from_noise():
    factors = _factor_frame()
    g = np.random.default_rng(1)
    noise = g.normal(0, 0.02, len(factors))
    base = factors["RF"].values + 1.1 * factors["Mkt-RF"].values + noise
    client = FamaFrenchClient()

    flat = client.compute_factor_exposure(base, factors, stock_dates=list(factors.index))
    strong = client.compute_factor_exposure(base + 0.006, factors, stock_dates=list(factors.index))
    assert abs(flat["alpha_tstat"]) < 2.5
    assert strong["alpha_tstat"] > 3
    assert strong["beta_market"] == pytest.approx(1.1, abs=0.3)
    # the old score saturated on a point estimate this noisy
    assert flat["alpha_se_annual"] > 0.15


# ── pooled event study ────────────────────────────────────────────────────

def test_post_event_returns_start_at_day_zero():
    cars = np.cumsum(np.arange(26, dtype=float))  # days -5..20, AR_i = i
    ars = post_event_abnormal_returns(cars, evt_start=-5)
    assert len(ars) == 21 and ars[0] == 5.0 and ars[-1] == 25.0


def test_pooling_needs_enough_distinct_disclosure_dates():
    one_day = [{"anchor_date": "2026-08-14", "car_5d": 0.02, "car_20d": 0.05} for _ in range(500)]
    assert pooled_event_stats(one_day) is None  # 500 signals, one observation


def test_same_day_signals_count_once():
    g = np.random.default_rng(0)
    rows = []
    for day in range(30):
        shared = g.normal(0, 0.05)  # that day's market move, common to all
        for _ in range(40):
            rows.append({"anchor_date": day, "car_5d": shared, "car_20d": shared,
                         "daily_ars": np.full(21, shared / 21)})
    stats = pooled_event_stats(rows)
    assert stats["n_events"] == 1200 and stats["n_clusters"] == 30
    # 1,200 "observations" of pure noise would look wildly significant if
    # treated as independent; across 30 dates they are not
    naive_t = np.mean([r["car_20d"] for r in rows]) / (np.std([r["car_20d"] for r in rows], ddof=1) / np.sqrt(1200))
    assert abs(stats["t_statistic"]) < abs(naive_t) / 3
    assert len(stats["mean_ar_by_day"]) == 21


def test_pooled_event_study_score():
    assert scorer._score_event_study({"mean_car_5d": 0.02, "mean_car_20d": 0.04, "is_significant": True}, "buy") > 70
    assert scorer._score_event_study({"mean_car_5d": -0.02, "mean_car_20d": -0.04, "is_significant": True}, "sell") < 30
    assert scorer._score_event_study({}, "buy") is None


def test_trading_days_since():
    from datetime import datetime, timedelta
    now = datetime(2026, 11, 11)
    assert trading_days_since(now - timedelta(days=14), now) == pytest.approx(10.0)
    assert trading_days_since(now + timedelta(days=3), now) == 0.0
