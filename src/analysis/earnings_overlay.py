"""
Earnings Surprise Overlay — flags insider signals near earnings dates.

Insider buys within 30 days before a positive earnings surprise are among
the strongest signals. This module checks proximity to earnings and
historical surprise direction using yfinance earnings data.
"""

import asyncio
from datetime import datetime, timedelta

import numpy as np
import structlog

logger = structlog.get_logger()


class EarningsOverlay:
    """
    For a given ticker and signal date, determines:
    - Whether the signal is within 30 days of an upcoming earnings date
    - Historical earnings surprise direction (beat/miss ratio)
    - A composite earnings alignment score (0-100)
    """

    def __init__(self, lookback_days: int = 30, lookforward_days: int = 30):
        self.lookback_days = lookback_days
        self.lookforward_days = lookforward_days

    async def analyze(
        self,
        ticker: str,
        signal_date: datetime,
        direction: str = "buy",
    ) -> dict | None:
        """
        Check earnings proximity and surprise history for a signal.

        Returns:
            Dict with earnings_nearby, days_to_earnings, surprise metrics,
            and alignment score. None if data unavailable.
        """
        return await asyncio.to_thread(
            self._analyze_sync, ticker, signal_date, direction
        )

    def _analyze_sync(
        self, ticker: str, signal_date: datetime, direction: str
    ) -> dict | None:
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)

            # Get earnings dates
            try:
                earnings_dates = stock.earnings_dates
            except Exception:
                earnings_dates = None

            if earnings_dates is None or earnings_dates.empty:
                logger.debug("earnings.no_data", ticker=ticker)
                return {
                    "ticker": ticker,
                    "earnings_nearby": False,
                    "days_to_earnings": None,
                    "earnings_data_available": False,
                    "score": 50,  # neutral when no data
                }

            # Find nearest upcoming earnings relative to signal_date
            signal_dt = signal_date if hasattr(signal_date, 'date') else datetime.strptime(str(signal_date), "%Y-%m-%d")

            upcoming = None
            days_to_earnings = None
            for dt in earnings_dates.index:
                dt_naive = dt.to_pydatetime().replace(tzinfo=None)
                delta = (dt_naive - signal_dt).days
                if -self.lookback_days <= delta <= self.lookforward_days:
                    if upcoming is None or abs(delta) < abs(days_to_earnings):
                        upcoming = dt_naive
                        days_to_earnings = delta

            earnings_nearby = days_to_earnings is not None

            # Historical surprise data (quarters reported on or before the signal)
            surprise_data = self._get_surprise_history(earnings_dates, as_of=signal_dt)

            # Compute alignment score
            score = self._compute_score(
                direction, earnings_nearby, days_to_earnings, surprise_data
            )

            result = {
                "ticker": ticker,
                "earnings_nearby": earnings_nearby,
                "days_to_earnings": days_to_earnings,
                "next_earnings_date": upcoming.strftime("%Y-%m-%d") if upcoming else None,
                "earnings_data_available": True,
                "surprise_history": surprise_data,
                "score": round(score, 1),
            }

            logger.info(
                "earnings.analyzed",
                ticker=ticker,
                nearby=earnings_nearby,
                days=days_to_earnings,
                score=round(score, 1),
            )
            return result

        except Exception as e:
            logger.warning("earnings.failed", ticker=ticker, error=str(e))
            return None

    def _get_surprise_history(self, earnings_dates, as_of: datetime | None = None) -> dict:
        """Extract beat/miss history from the earnings_dates DataFrame.

        Only quarters that have actually been reported count. yfinance lists
        up to a year of *upcoming* dates with NaN "Reported EPS"; treating
        those as misses dragged every beat rate down and left the latest
        surprise undefined. With `as_of`, quarters reported after that moment
        are ignored too, so a historical signal never sees later results.
        """
        import pandas as pd

        surprises = []  # most recent first
        for dt, row in earnings_dates.sort_index(ascending=False).iterrows():
            if as_of is not None:
                try:
                    if dt.to_pydatetime().replace(tzinfo=None) > as_of:
                        continue
                except (AttributeError, TypeError):
                    pass
            estimate = row.get("EPS Estimate")
            actual = row.get("Reported EPS")
            if pd.isna(estimate) or pd.isna(actual):
                continue
            try:
                est = float(estimate)
                act = float(actual)
            except (ValueError, TypeError):
                continue
            if est == 0:
                continue
            surprises.append((act - est) / abs(est))

        if not surprises:
            return {"n_quarters": 0, "beat_rate": None, "avg_surprise_pct": None, "last_surprise_pct": None}

        beats = sum(1 for s in surprises if s > 0)
        return {
            "n_quarters": len(surprises),
            "beat_rate": round(beats / len(surprises), 4),
            "avg_surprise_pct": round(float(np.mean(surprises)), 4),
            "last_surprise_pct": round(surprises[0], 4),
        }

    # Roughly 70% of US companies beat consensus EPS in a typical quarter, so
    # that — not 50% — is the rate that carries no information.
    NEUTRAL_BEAT_RATE = 0.70
    MIN_QUARTERS = 4

    def _compute_score(
        self,
        direction: str,
        earnings_nearby: bool,
        days_to_earnings: int | None,
        surprise_data: dict,
    ) -> float:
        """
        Score 0-100: 50 + proximity x directional evidence.

        Directional evidence is the company's surprise record relative to the
        norm. Being close to earnings only *scales* that evidence; with no
        evidence the score stays at 50 for buys and sells alike. (Adding a
        proximity bonus on its own lifted both directions at once.)
        """
        if not earnings_nearby or days_to_earnings is None:
            return 50.0

        evidence = 0.0  # > 0 favours buyers, < 0 favours sellers
        beat_rate = surprise_data.get("beat_rate")
        if beat_rate is not None and (surprise_data.get("n_quarters") or 0) >= self.MIN_QUARTERS:
            evidence += (beat_rate - self.NEUTRAL_BEAT_RATE) / (1.0 - self.NEUTRAL_BEAT_RATE)

        last_surprise = surprise_data.get("last_surprise_pct")
        if last_surprise is not None:
            if last_surprise > 0.05:
                evidence += 0.25
            elif last_surprise < -0.05:
                evidence -= 0.25

        evidence = max(-1.0, min(1.0, evidence))
        if direction == "sell":
            evidence = -evidence

        abs_days = abs(days_to_earnings)
        if abs_days <= 7:
            proximity = 1.0
        elif abs_days <= 14:
            proximity = 0.75
        else:
            proximity = 0.5
        # A trade placed before the report says more than one placed after it
        if days_to_earnings > 0:
            proximity = min(1.2, proximity * 1.2)

        return max(0.0, min(100.0, 50.0 + 25.0 * evidence * proximity))
