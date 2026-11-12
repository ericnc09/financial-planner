"""
Pooled post-disclosure alpha.

A fresh signal has no post-event returns of its own, so its event-study and
decay components cannot come from the signal itself. They come from what
happened after comparable past signals (same source type and direction)
became public: event studies anchored on the first trading day after
disclosure, pooled.

Signals disclosed on the same day share one market move and are not
independent observations. Everything here is therefore averaged within each
disclosure date first, and significance is computed across dates.
"""

from collections import defaultdict

import numpy as np
from scipy import stats

# Fewer distinct disclosure dates than this and the pooled figures are too
# thin to score anything; the components are simply absent until then.
MIN_CLUSTERS = 20

# Rough trading days per calendar day, for ageing a signal
TRADING_DAYS_PER_CALENDAR_DAY = 5.0 / 7.0


def post_event_abnormal_returns(daily_cars, evt_start: int = -5) -> np.ndarray | None:
    """Daily abnormal returns for day 0 onward from a stored cumulative series.

    `daily_cars` is the cumulative abnormal return for days evt_start..end
    as written by EventStudyAnalyzer (so index -evt_start is day 0).
    """
    car = np.asarray(daily_cars, dtype=float)
    offset = -evt_start
    if len(car) <= offset or not np.all(np.isfinite(car)):
        return None
    ar = np.diff(np.concatenate([[0.0], car]))
    return ar[offset:]


def pooled_event_stats(rows: list[dict], min_clusters: int = MIN_CLUSTERS) -> dict | None:
    """Pool matured, disclosure-anchored event studies for one source x direction.

    Args:
        rows: One dict per event with "anchor_date" (anything hashable that
              identifies the disclosure day), "car_5d", "car_20d" and
              optionally "daily_ars" (array, day 0 onward).

    Returns:
        Dict with cluster-level mean CARs, a t-test across disclosure dates
        and the mean abnormal-return curve — or None below `min_clusters`.
    """
    by_day: dict = defaultdict(list)
    for r in rows:
        if r.get("car_20d") is None or r.get("anchor_date") is None:
            continue
        by_day[r["anchor_date"]].append(r)

    n_clusters = len(by_day)
    if n_clusters < min_clusters:
        return None

    day_car5, day_car20, day_curves = [], [], []
    for day_rows in by_day.values():
        car5 = [r["car_5d"] for r in day_rows if r.get("car_5d") is not None]
        if car5:
            day_car5.append(float(np.mean(car5)))
        day_car20.append(float(np.mean([r["car_20d"] for r in day_rows])))
        curves = [np.asarray(r["daily_ars"], dtype=float) for r in day_rows if r.get("daily_ars") is not None]
        if curves:
            length = min(len(c) for c in curves)
            day_curves.append(np.mean([c[:length] for c in curves], axis=0))

    car20 = np.asarray(day_car20)
    mean20 = float(np.mean(car20))
    se20 = float(np.std(car20, ddof=1) / np.sqrt(n_clusters))
    t_stat = mean20 / se20 if se20 > 0 else 0.0
    p_value = float(2 * stats.t.sf(abs(t_stat), df=n_clusters - 1))

    mean_curve = None
    if len(day_curves) >= min_clusters:
        length = min(len(c) for c in day_curves)
        mean_curve = np.mean([c[:length] for c in day_curves], axis=0)

    return {
        "pooled": True,
        "n_events": sum(len(v) for v in by_day.values()),
        "n_clusters": n_clusters,
        "mean_car_5d": round(float(np.mean(day_car5)), 6) if day_car5 else None,
        "mean_car_20d": round(mean20, 6),
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(p_value, 4),
        "is_significant": p_value < 0.05,
        "win_rate_20d": round(float(np.mean(car20 > 0)), 4),
        "mean_ar_by_day": mean_curve,
    }


def trading_days_since(disclosed_at, now) -> float:
    """Approximate trading days between disclosure and now (never negative)."""
    if disclosed_at is None:
        return 0.0
    days = (now - disclosed_at).total_seconds() / 86400.0
    return max(0.0, days * TRADING_DAYS_PER_CALENDAR_DAY)


_engines: dict = {}


def recent_decay_components(
    ticker: str,
    database_url: str | None = None,
    max_age_days: int = 30,
    now=None,
) -> tuple[dict | None, dict | None]:
    """(buy, sell) decay components for a ticker's most recent signals.

    Reads the pooled decay fit the pipeline stored for the ticker's latest
    signal in each direction and ages it to today. Used by the ticker-analysis
    path (Slack/WhatsApp verdicts), which has no signal context of its own.
    Returns (None, None) whenever there is no recent signal, no stored fit,
    or no database — never raises.
    """
    try:
        from datetime import datetime, timedelta

        from sqlalchemy.orm import Session

        from src.analysis.bayesian_decay import decay_component
        from src.models.database import BayesianDecayResult, SmartMoneyEvent, get_engine

        if database_url is None:
            from config.settings import Settings
            database_url = Settings().database_url
        engine = _engines.get(database_url)
        if engine is None:
            engine = _engines[database_url] = get_engine(database_url)

        now = now or datetime.utcnow()
        out: dict[str, dict | None] = {"buy": None, "sell": None}
        with Session(engine) as session:
            rows = (
                session.query(BayesianDecayResult, SmartMoneyEvent)
                .join(SmartMoneyEvent, SmartMoneyEvent.id == BayesianDecayResult.event_id)
                .filter(BayesianDecayResult.ticker == ticker)
                .order_by(BayesianDecayResult.run_date.desc())
                .all()
            )
            for fit, event in rows:
                disclosed = event.disclosure_date or event.trade_date
                if out.get(fit.direction) is not None or disclosed is None:
                    continue
                if disclosed < now - timedelta(days=max_age_days):
                    continue
                out[fit.direction] = decay_component(
                    {
                        "posterior_lambda": fit.posterior_lambda,
                        "posterior_half_life": fit.posterior_half_life,
                        "prob_alpha_positive": fit.prob_alpha_positive,
                        "decay_quality": fit.decay_quality,
                    },
                    trading_days_since(disclosed, now),
                )
        return out["buy"], out["sell"]
    except Exception:
        return None, None
