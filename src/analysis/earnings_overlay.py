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

            # Historical surprise data
            surprise_data = self._get_surprise_history(earnings_dates)

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

    def _get_surprise_history(self, earnings_dates) -> dict:
        """Extract beat/miss history from earnings_dates DataFrame."""
        surprises = []
        for _, row in earnings_dates.iterrows():
            estimate = row.get("EPS Estimate")
            actual = row.get("Reported EPS")
            if estimate is not None and actual is not None:
                try:
                    est = float(estimate)
                    act = float(actual)
                    if est != 0:
                        surprise_pct = (act - est) / abs(est)
                        surprises.append(surprise_pct)
                except (ValueError, TypeError):
                    continue

        if not surprises:
            return {"n_quarters": 0, "beat_rate": None, "avg_surprise_pct": None}

        beats = sum(1 for s in surprises if s > 0)
        return {
            "n_quarters": len(surprises),
            "beat_rate": round(beats / len(surprises), 4),
            "avg_surprise_pct": round(float(np.mean(surprises)), 4),
            "last_surprise_pct": round(surprises[0], 4) if surprises else None,
        }

    def _compute_score(
        self,
        direction: str,
        earnings_nearby: bool,
        days_to_earnings: int | None,
        surprise_data: dict,
    ) -> float:
        """
        Score 0-100 based on earnings alignment with signal direction.

        Key insight: insider buy + upcoming earnings + history of beats = strongest signal.
        """
        score = 50.0  # neutral baseline

        if not earnings_nearby:
            return score  # no earnings proximity, stay neutral

        # Proximity boost: closer to earnings = more informative
        if days_to_earnings is not None:
            abs_days = abs(days_to_earnings)
            if abs_days <= 7:
                proximity_boost = 20
            elif abs_days <= 14:
                proximity_boost = 15
            else:
                proximity_boost = 8

            # Before earnings (insider knows something) is stronger than after
            if days_to_earnings > 0:  # earnings upcoming
                proximity_boost *= 1.2

            score += proximity_boost

        # Historical surprise alignment
        beat_rate = surprise_data.get("beat_rate")
        if beat_rate is not None:
            if direction == "buy":
                # Buying before a company that usually beats = bullish
                if beat_rate >= 0.75:
                    score += 15
                elif beat_rate >= 0.5:
                    score += 5
                else:
                    score -= 10  # buying into a serial misser
            else:  # sell
                if beat_rate <= 0.25:
                    score += 15  # selling a serial misser
                elif beat_rate <= 0.5:
                    score += 5
                else:
                    score -= 10  # selling against a serial beater

        # Last surprise direction
        last_surprise = surprise_data.get("last_surprise_pct")
        if last_surprise is not None:
            if direction == "buy" and last_surprise > 0.05:
                score += 5  # momentum from last beat
            elif direction == "sell" and last_surprise < -0.05:
                score += 5

        return max(0, min(100, score))
