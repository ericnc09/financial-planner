import asyncio
from datetime import datetime, timedelta

import numpy as np
import structlog
import yfinance as yf

logger = structlog.get_logger()


class OptionsClient:
    """Fetch and analyze options chain data via yfinance (free, no rate limits)."""

    async def get_options_analysis(self, ticker: str) -> dict | None:
        """Fetch options chain and compute sentiment metrics."""
        try:
            return await asyncio.to_thread(self._analyze, ticker)
        except Exception as e:
            logger.warning("options.analysis_failed", ticker=ticker, error=str(e))
            return None

    def _analyze(self, ticker: str) -> dict | None:
        t = yf.Ticker(ticker)

        # Get available expirations
        try:
            expirations = t.options
        except Exception:
            return None

        if not expirations:
            return None

        # Pick nearest monthly expiration (14-60 days out)
        today = datetime.now().date()
        target_exp = None
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            days_out = (exp_date - today).days
            if 14 <= days_out <= 60:
                target_exp = exp_str
                break

        # Fallback: take the first available expiration
        if not target_exp:
            target_exp = expirations[0]

        # Fetch option chain
        try:
            chain = t.option_chain(target_exp)
        except Exception:
            return None

        calls = chain.calls
        puts = chain.puts

        if calls.empty and puts.empty:
            return None

        # Current price
        try:
            info = t.info or {}
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        except Exception:
            current_price = None

        # Compute metrics
        total_call_volume = int(calls["volume"].sum()) if "volume" in calls.columns else 0
        total_put_volume = int(puts["volume"].sum()) if "volume" in puts.columns else 0
        total_call_oi = int(calls["openInterest"].sum()) if "openInterest" in calls.columns else 0
        total_put_oi = int(puts["openInterest"].sum()) if "openInterest" in puts.columns else 0

        pcr = self._compute_pcr(total_call_volume, total_put_volume)
        iv_skew = self._compute_iv_skew(calls, puts, current_price)
        max_pain = self._compute_max_pain(calls, puts)
        unusual_volume_score = self._compute_unusual_volume(
            total_call_volume, total_put_volume, total_call_oi, total_put_oi
        )

        return {
            "ticker": ticker,
            "analysis_date": datetime.utcnow(),
            "pcr": pcr,
            "unusual_volume_score": unusual_volume_score,
            "iv_skew": iv_skew,
            "max_pain": max_pain,
            "nearest_expiry": target_exp,
            "total_call_volume": total_call_volume,
            "total_put_volume": total_put_volume,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
        }

    @staticmethod
    def _compute_pcr(call_volume: int, put_volume: int) -> float | None:
        """Put/call volume ratio. < 0.7 = bullish, > 1.0 = bearish."""
        if call_volume <= 0:
            return None
        return round(put_volume / call_volume, 4)

    @staticmethod
    def _compute_unusual_volume(
        call_vol: int, put_vol: int, call_oi: int, put_oi: int
    ) -> float:
        """Score 0-1 based on volume-to-open-interest ratio (higher = more unusual activity)."""
        total_vol = call_vol + put_vol
        total_oi = call_oi + put_oi
        if total_oi <= 0:
            return 0.0
        # Volume/OI ratio: typically 0.1-0.3 is normal, >0.5 is unusual
        ratio = total_vol / total_oi
        # Normalize to 0-1 with sigmoid-like curve
        score = min(1.0, ratio / 0.8)
        return round(score, 4)

    @staticmethod
    def _compute_iv_skew(calls, puts, current_price: float | None) -> float | None:
        """IV skew: avg IV of OTM puts vs ATM calls. Positive = bearish skew."""
        if current_price is None or current_price <= 0:
            return None

        iv_col = "impliedVolatility"
        if iv_col not in calls.columns or iv_col not in puts.columns:
            return None

        # ATM calls: strikes within 5% of current price
        atm_calls = calls[
            (calls["strike"] >= current_price * 0.95) &
            (calls["strike"] <= current_price * 1.05)
        ]
        atm_call_iv = atm_calls[iv_col].dropna().mean() if not atm_calls.empty else None

        # OTM puts: strikes below 95% of current price
        otm_puts = puts[puts["strike"] < current_price * 0.95]
        otm_put_iv = otm_puts[iv_col].dropna().mean() if not otm_puts.empty else None

        if atm_call_iv is None or otm_put_iv is None or atm_call_iv <= 0:
            return None

        # Positive skew = puts more expensive = bearish sentiment
        skew = (otm_put_iv - atm_call_iv) / atm_call_iv
        return round(float(skew), 4)

    @staticmethod
    def _compute_max_pain(calls, puts) -> float | None:
        """Strike where total option holder losses are maximized (max pain theory)."""
        if calls.empty and puts.empty:
            return None

        oi_col = "openInterest"
        if oi_col not in calls.columns or oi_col not in puts.columns:
            return None

        # Get all unique strikes
        all_strikes = sorted(
            set(calls["strike"].tolist() + puts["strike"].tolist())
        )
        if not all_strikes:
            return None

        min_pain = float("inf")
        max_pain_strike = all_strikes[0]

        for strike in all_strikes:
            # Call holders lose money when price < strike
            call_pain = 0
            for _, row in calls.iterrows():
                oi = row.get(oi_col, 0) or 0
                if strike > row["strike"]:
                    call_pain += (strike - row["strike"]) * oi

            # Put holders lose money when price > strike
            put_pain = 0
            for _, row in puts.iterrows():
                oi = row.get(oi_col, 0) or 0
                if strike < row["strike"]:
                    put_pain += (row["strike"] - strike) * oi

            total_pain = call_pain + put_pain
            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = strike

        return float(max_pain_strike)

    async def close(self):
        pass
