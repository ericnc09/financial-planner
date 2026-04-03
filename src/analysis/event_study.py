"""
Event Study Analysis — Measures abnormal returns around smart money signals.

For each insider/congressional trade event:
1. Fit a market model on pre-event estimation window [-120, -10] days
2. Compute abnormal returns (actual - expected) in event window [-5, +20] days
3. Accumulate into Cumulative Abnormal Return (CAR)
4. Test statistical significance via cross-sectional t-test
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd
import structlog
from scipy import stats

logger = structlog.get_logger()


class EventStudyAnalyzer:
    def __init__(
        self,
        estimation_window: tuple[int, int] = (-120, -10),
        event_window: tuple[int, int] = (-5, 20),
    ):
        self.est_start, self.est_end = estimation_window
        self.evt_start, self.evt_end = event_window

    def analyze_event(
        self,
        ticker: str,
        event_date: datetime,
        direction: str,
        price_dates: list,
        price_returns: np.ndarray,
        market_returns: np.ndarray,
    ) -> dict | None:
        """
        Run event study for a single event.

        Args:
            ticker: Stock ticker.
            event_date: Date of the insider/congressional trade.
            direction: 'buy' or 'sell'.
            price_dates: List of trading dates for the stock.
            price_returns: Log returns array aligned with price_dates[1:].
            market_returns: Market excess returns (Mkt-RF) aligned with price_dates[1:].

        Returns:
            Dict with daily abnormal returns, CAR at key horizons, or None if insufficient data.
        """
        # Find event date index in the dates array
        event_idx = self._find_event_index(price_dates, event_date)
        if event_idx is None:
            logger.debug("event_study.date_not_found", ticker=ticker, event_date=event_date)
            return None

        # Returns arrays are 1 shorter than dates (diff-based)
        # Return index = date index - 1
        ret_event_idx = event_idx - 1

        # Check we have enough data for estimation window
        est_start_idx = ret_event_idx + self.est_start
        est_end_idx = ret_event_idx + self.est_end
        evt_start_idx = ret_event_idx + self.evt_start
        evt_end_idx = ret_event_idx + self.evt_end

        if est_start_idx < 0:
            logger.debug("event_study.insufficient_pre_data", ticker=ticker)
            return None
        if evt_end_idx >= len(price_returns):
            logger.debug("event_study.insufficient_post_data", ticker=ticker)
            return None

        n_market = len(market_returns)
        if est_end_idx >= n_market or evt_end_idx >= n_market:
            logger.debug("event_study.market_data_mismatch", ticker=ticker)
            return None

        # Estimation window: fit market model
        est_stock = price_returns[est_start_idx:est_end_idx]
        est_market = market_returns[est_start_idx:est_end_idx]

        if len(est_stock) < 30:
            logger.debug("event_study.short_estimation", ticker=ticker, n=len(est_stock))
            return None

        # OLS: R_stock = alpha + beta * R_market
        X = np.column_stack([np.ones(len(est_market)), est_market])
        beta, residuals, rank, sv = np.linalg.lstsq(X, est_stock, rcond=None)
        alpha, mkt_beta = float(beta[0]), float(beta[1])

        # Estimation window residual std for significance testing
        est_predicted = alpha + mkt_beta * est_market
        est_residuals = est_stock - est_predicted
        sigma_est = float(np.std(est_residuals, ddof=2))

        if sigma_est == 0:
            return None

        # Event window: compute abnormal returns
        evt_stock = price_returns[evt_start_idx:evt_end_idx + 1]
        evt_market = market_returns[evt_start_idx:evt_end_idx + 1]

        expected_returns = alpha + mkt_beta * evt_market
        abnormal_returns = evt_stock - expected_returns

        # For SELL signals, invert AR (negative price move = positive alpha for seller)
        if direction == "sell":
            abnormal_returns = -abnormal_returns

        # Cumulative abnormal returns
        car = np.cumsum(abnormal_returns)
        daily_cars = [round(float(c), 6) for c in car]

        # Day indices relative to event (evt_start to evt_end)
        days = list(range(self.evt_start, self.evt_end + 1))
        event_day_0_offset = -self.evt_start  # index of day 0 in the arrays

        # Extract CAR at key horizons
        def car_at_day(d: int) -> float | None:
            idx = d - self.evt_start
            if 0 <= idx < len(car):
                return round(float(car[idx]), 6)
            return None

        car_1d = car_at_day(1)
        car_5d = car_at_day(5)
        car_10d = car_at_day(10)
        car_20d = car_at_day(min(20, self.evt_end))

        # T-statistic for the post-event CAR
        # Using standardized CAR: t = CAR / (sigma_est * sqrt(n_actual_post_days))
        # n_post = actual number of event-window observations (not hardcoded horizon)
        n_post = len(evt_stock)
        t_stat = float(car[-1]) / (sigma_est * np.sqrt(n_post)) if sigma_est > 0 and n_post > 0 else 0
        p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=len(est_stock) - 2)))

        result = {
            "ticker": ticker,
            "event_date": event_date.isoformat(),
            "direction": direction,
            "alpha": round(alpha, 6),
            "beta": round(mkt_beta, 4),
            "sigma_est": round(sigma_est, 6),
            "car_1d": car_1d,
            "car_5d": car_5d,
            "car_10d": car_10d,
            "car_20d": car_20d,
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 4),
            "is_significant": p_value < 0.05,
            "daily_cars": daily_cars,
            "days": days,
            "n_estimation": len(est_stock),
        }

        logger.info(
            "event_study.analyzed",
            ticker=ticker,
            direction=direction,
            car_5d=car_5d,
            car_20d=car_20d,
            t_stat=round(t_stat, 4),
            p_value=round(p_value, 4),
            significant=p_value < 0.05,
        )
        return result

    def aggregate_results(self, results: list[dict]) -> dict:
        """
        Compute cross-event aggregate statistics.

        Args:
            results: List of per-event result dicts from analyze_event().

        Returns:
            Dict with mean CAR, t-stats, p-values, win rates, breakdowns by source/direction.
        """
        if not results:
            return {"n_events": 0}

        def _aggregate_group(group: list[dict], label: str) -> dict:
            if not group:
                return {"n_events": 0, "label": label}

            cars_1d = [r["car_1d"] for r in group if r["car_1d"] is not None]
            cars_5d = [r["car_5d"] for r in group if r["car_5d"] is not None]
            cars_10d = [r["car_10d"] for r in group if r["car_10d"] is not None]
            cars_20d = [r["car_20d"] for r in group if r["car_20d"] is not None]

            def _stats(values: list[float], horizon: str) -> dict:
                if len(values) < 2:
                    return {"mean": values[0] if values else None, "t_stat": None, "p_value": None, "win_rate": None}
                arr = np.array(values)
                mean = float(np.mean(arr))
                se = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
                t = mean / se if se > 0 else 0
                p = float(2 * (1 - stats.t.cdf(abs(t), df=len(arr) - 1)))
                win = float(np.mean(arr > 0))
                return {
                    "mean": round(mean, 6),
                    "std": round(float(np.std(arr, ddof=1)), 6),
                    "t_stat": round(t, 4),
                    "p_value": round(p, 4),
                    "is_significant": p < 0.05,
                    "win_rate": round(win, 4),
                    "n": len(values),
                }

            significant_count = sum(1 for r in group if r.get("is_significant"))

            return {
                "label": label,
                "n_events": len(group),
                "n_significant": significant_count,
                "pct_significant": round(significant_count / len(group), 4),
                "car_1d": _stats(cars_1d, "1d"),
                "car_5d": _stats(cars_5d, "5d"),
                "car_10d": _stats(cars_10d, "10d"),
                "car_20d": _stats(cars_20d, "20d"),
            }

        # Overall
        overall = _aggregate_group(results, "all")

        # By direction
        buys = [r for r in results if r["direction"] == "buy"]
        sells = [r for r in results if r["direction"] == "sell"]

        # By source (need to be set by caller via extra field)
        insiders = [r for r in results if r.get("source_type") == "insider"]
        congressional = [r for r in results if r.get("source_type") == "congressional"]

        aggregate = {
            "overall": overall,
            "by_direction": {
                "buy": _aggregate_group(buys, "buy"),
                "sell": _aggregate_group(sells, "sell"),
            },
            "by_source": {
                "insider": _aggregate_group(insiders, "insider"),
                "congressional": _aggregate_group(congressional, "congressional"),
            },
        }

        logger.info(
            "event_study.aggregated",
            n_events=overall["n_events"],
            n_significant=overall["n_significant"],
            mean_car_5d=overall["car_5d"].get("mean"),
            mean_car_20d=overall["car_20d"].get("mean"),
        )
        return aggregate

    def _find_event_index(self, dates: list, event_date: datetime) -> int | None:
        """Find the closest trading date index on or after event_date."""
        event_d = event_date.date() if hasattr(event_date, 'date') else event_date
        for i, d in enumerate(dates):
            d_date = d.date() if hasattr(d, 'date') else d
            if d_date >= event_d:
                return i
        return None
