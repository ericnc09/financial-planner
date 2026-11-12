import asyncio
from datetime import datetime, timedelta

import numpy as np
import structlog
import yfinance as yf

from src.clients.symbols import yahoo_candidates

logger = structlog.get_logger()


# Quotes outside these bounds are stale or broken (Yahoo reports implied
# volatilities like 0.00001 or 500 for strikes that never trade).
IV_RANGE = (0.01, 5.0)
MIN_OPEN_INTEREST = 10
# Below this many contracts across calls and puts a put/call ratio is noise.
MIN_TOTAL_VOLUME = 100
MAX_PCR = 10.0
# Bumped whenever the definition of pcr / iv_skew changes, so percentile
# references never mix rows computed differently.
METHOD_VERSION = 2


def percentile_rank(value: float | None, reference) -> float | None:
    """Share of `reference` at or below `value` (mid-rank for ties), 0..1."""
    if value is None:
        return None
    ref = np.asarray([r for r in reference if r is not None], dtype=float)
    ref = ref[np.isfinite(ref)]
    if len(ref) == 0:
        return None
    below = float(np.sum(ref < value))
    equal = float(np.sum(ref == value))
    return (below + 0.5 * equal) / len(ref)


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
        import pandas as pd

        # Resolve the price source's spelling of the symbol (BRK.B -> BRK-B)
        t, expirations = None, None
        for symbol in yahoo_candidates(ticker) or [ticker]:
            try:
                candidate = yf.Ticker(symbol)
                exps = candidate.options
            except Exception:
                continue
            if exps:
                t, expirations = candidate, exps
                break
        if not expirations:
            return None

        # Nearest expirations 14-60 days out (up to two, pooled for volume).
        today = datetime.now().date()
        in_window = []
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            if 14 <= (exp_date - today).days <= 60:
                in_window.append(exp_str)
        targets = in_window[:2] or [expirations[0]]

        chains = []
        for exp_str in targets:
            try:
                chains.append(t.option_chain(exp_str))
            except Exception:
                continue
        if not chains:
            return None

        target_exp = targets[0]
        calls = pd.concat([c.calls for c in chains], ignore_index=True)
        puts = pd.concat([c.puts for c in chains], ignore_index=True)
        near_calls, near_puts = chains[0].calls, chains[0].puts

        if calls.empty and puts.empty:
            return None

        current_price = self._current_price(t)

        # Compute metrics
        total_call_volume = int(calls["volume"].sum()) if "volume" in calls.columns else 0
        total_put_volume = int(puts["volume"].sum()) if "volume" in puts.columns else 0
        total_call_oi = int(calls["openInterest"].sum()) if "openInterest" in calls.columns else 0
        total_put_oi = int(puts["openInterest"].sum()) if "openInterest" in puts.columns else 0

        pcr = self._compute_pcr(total_call_volume, total_put_volume)
        iv_skew = self._compute_iv_skew(near_calls, near_puts, current_price)
        max_pain = self._compute_max_pain(near_calls, near_puts)
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
            "method_version": METHOD_VERSION,
        }

    @staticmethod
    def _current_price(t) -> float | None:
        try:
            price = t.fast_info.get("last_price") or t.fast_info.get("lastPrice")
            if price and price > 0:
                return float(price)
        except Exception:
            pass
        try:
            info = t.info or {}
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            return float(price) if price and price > 0 else None
        except Exception:
            return None

    @staticmethod
    def _compute_pcr(call_volume: int, put_volume: int) -> float | None:
        """Put/call volume ratio, or None when volume is too thin to mean anything."""
        if call_volume <= 0 or call_volume + put_volume < MIN_TOTAL_VOLUME:
            return None
        return round(min(put_volume / call_volume, MAX_PCR), 4)

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
        """Wing skew: (OTM-put IV - OTM-call IV) / ATM IV.

        Put wing = strikes at 90-97.5% of spot, call wing = 102.5-110%, ATM =
        within 2.5% of spot, all at the nearest expiry and only on quotes
        with a plausible IV and real open interest.

        Comparing OTM puts with ATM calls (the previous definition) measures
        the ordinary volatility smile, which is positive for nearly every
        stock; comparing the two wings at matching distance isolates how much
        more the market pays for downside than for upside. Even so the level
        is usually above zero, so consumers should read it relative to other
        tickers, not against a fixed threshold.
        """
        if current_price is None or current_price <= 0:
            return None

        iv_col = "impliedVolatility"
        if iv_col not in calls.columns or iv_col not in puts.columns:
            return None

        def clean(df):
            df = df[(df[iv_col] > IV_RANGE[0]) & (df[iv_col] < IV_RANGE[1])]
            if "openInterest" in df.columns:
                df = df[df["openInterest"].fillna(0) >= MIN_OPEN_INTEREST]
            return df

        calls, puts = clean(calls), clean(puts)
        call_m = calls["strike"] / current_price
        put_m = puts["strike"] / current_price

        atm = np.concatenate([
            calls[(call_m - 1).abs() <= 0.025][iv_col].values,
            puts[(put_m - 1).abs() <= 0.025][iv_col].values,
        ])
        put_wing = puts[(put_m >= 0.90) & (put_m <= 0.975)][iv_col].values
        call_wing = calls[(call_m >= 1.025) & (call_m <= 1.10)][iv_col].values

        if len(atm) < 1 or len(put_wing) < 2 or len(call_wing) < 2:
            return None
        atm_iv = float(np.mean(atm))
        if atm_iv <= 0:
            return None

        skew = (float(np.mean(put_wing)) - float(np.mean(call_wing))) / atm_iv
        return round(float(skew), 4)

    @staticmethod
    def _compute_max_pain(calls, puts) -> float | None:
        """Strike at which option holders in aggregate are paid the least."""
        if calls.empty and puts.empty:
            return None

        oi_col = "openInterest"
        if oi_col not in calls.columns or oi_col not in puts.columns:
            return None

        call_k = calls["strike"].to_numpy(dtype=float)
        put_k = puts["strike"].to_numpy(dtype=float)
        call_oi = np.nan_to_num(calls[oi_col].to_numpy(dtype=float))
        put_oi = np.nan_to_num(puts[oi_col].to_numpy(dtype=float))

        strikes = np.unique(np.concatenate([call_k, put_k]))
        if len(strikes) == 0:
            return None

        # Payout to holders if the stock settles at each candidate strike
        call_payout = (np.maximum(strikes[:, None] - call_k[None, :], 0) * call_oi[None, :]).sum(axis=1)
        put_payout = (np.maximum(put_k[None, :] - strikes[:, None], 0) * put_oi[None, :]).sum(axis=1)
        return float(strikes[int(np.argmin(call_payout + put_payout))])

    async def close(self):
        pass
