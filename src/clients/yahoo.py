import asyncio
from datetime import datetime, timedelta

import numpy as np
import structlog
import yfinance as yf

from src.clients.symbols import classify_security, yahoo_candidates

logger = structlog.get_logger()


class YahooClient:
    """Free unlimited price data via yfinance. All calls wrapped in to_thread."""

    async def get_price_history(
        self, ticker: str, days: int = 504
    ) -> dict | None:
        """Returns dict with 'closes', 'returns', 'volumes', 'highs', 'lows' arrays."""
        try:
            df = await asyncio.to_thread(self._fetch_history, ticker, days)
            if df is None or df.empty:
                return None
            # A NaN or zero close turns into a non-finite log return that
            # poisons every model downstream. Drop those bars once, here.
            close = df["Close"].astype(float)
            df = df[np.isfinite(close) & (close > 0)]
            if len(df) < 2:
                return None
            closes = df["Close"].values.astype(float)
            volumes = np.nan_to_num(df["Volume"].values.astype(float), nan=0.0)
            highs = df["High"].values.astype(float)
            lows = df["Low"].values.astype(float)
            returns = np.diff(np.log(closes))  # log returns
            return {
                "closes": closes,
                "returns": returns,
                "volumes": volumes,
                "highs": highs,
                "lows": lows,
                "dates": df.index.tolist(),
            }
        except Exception as e:
            logger.warning("yahoo.price_history_failed", ticker=ticker, error=str(e))
            return None

    def _fetch_history(self, ticker: str, days: int):
        # yfinance's `end` is exclusive, so today's bar is left out on
        # purpose: during market hours it is a partial bar (part of a day's
        # move, a fraction of a day's volume) and feeding it to the models as
        # if it were a full session flips regimes and distorts returns.
        end = datetime.now()
        start = end - timedelta(days=days)
        for symbol in yahoo_candidates(ticker):
            df = yf.Ticker(symbol).history(
                start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d")
            )
            if df is not None and not df.empty:
                if symbol != ticker:
                    logger.info("yahoo.symbol_resolved", ticker=ticker, yahoo_symbol=symbol)
                return df
        return None

    async def enrich_ticker(self, ticker: str):
        """Enrich a ticker using yfinance — free, no rate limits."""
        from src.models.schemas import EnrichmentSchema
        try:
            info, hist, symbol = await asyncio.to_thread(self._fetch_enrichment, ticker)
            if hist is None or hist.empty:
                return None

            close = hist["Close"].astype(float)
            hist = hist[np.isfinite(close) & (close > 0)]
            if hist.empty:
                return None
            closes = hist["Close"].values.astype(float)
            volumes = np.nan_to_num(hist["Volume"].values.astype(float), nan=0.0)

            current_price = float(closes[-1]) if len(closes) > 0 else None

            # Momentum
            momentum_30d = None
            if len(closes) >= 22:
                momentum_30d = (closes[-1] - closes[-22]) / closes[-22]
            momentum_90d = None
            if len(closes) >= 63:
                momentum_90d = (closes[-1] - closes[-63]) / closes[-63]

            # RSI-14
            rsi = self._compute_rsi(closes)

            # Drawdown from 52-week high (252 trading days, same window the
            # ticker-analysis path uses)
            drawdown = None
            if len(closes) > 0:
                high_52w = float(np.max(closes[-252:]))
                if high_52w > 0 and current_price:
                    drawdown = (high_52w - current_price) / high_52w

            # Average volume 30d
            avg_vol = float(np.mean(volumes[-30:])) if len(volumes) >= 30 else (
                float(np.mean(volumes)) if len(volumes) > 0 else None
            )

            # Fundamentals from yfinance .info
            pe_ratio = info.get("trailingPE") or info.get("forwardPE")
            market_cap = info.get("marketCap")
            eps_latest = info.get("trailingEps")
            sector = info.get("sector")
            short_ratio = info.get("shortRatio")

            # Revenue/EPS growth
            revenue_growth = info.get("revenueGrowth")
            earnings_growth = info.get("earningsGrowth")

            return EnrichmentSchema(
                pe_ratio=pe_ratio,
                market_cap=market_cap,
                revenue_growth_yoy=revenue_growth,
                eps_latest=eps_latest,
                eps_growth_yoy=earnings_growth,
                price_at_signal=current_price,
                momentum_30d=momentum_30d,
                momentum_90d=momentum_90d,
                rsi_14d=rsi,
                drawdown_from_52w_high=drawdown,
                avg_volume_30d=avg_vol,
                sector=sector,
                short_ratio=float(short_ratio) if short_ratio is not None else None,
                security_type=classify_security(symbol, info),
            )
        except Exception as e:
            logger.warning("yahoo.enrich_failed", ticker=ticker, error=str(e))
            return None

    def _fetch_enrichment(self, ticker: str, days: int = 400):
        """Synchronous fetch of info + history for enrichment."""
        end = datetime.now()  # exclusive: complete sessions only
        start = end - timedelta(days=days)
        for symbol in yahoo_candidates(ticker):
            t = yf.Ticker(symbol)
            hist = t.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
            if hist is None or hist.empty:
                continue
            try:
                info = t.info or {}
            except Exception:
                info = {}
            return info, hist, symbol
        return {}, None, None

    @staticmethod
    def _compute_rsi(closes: np.ndarray, period: int = 14) -> float | None:
        if len(closes) < period + 1:
            return None
        deltas = np.diff(closes)
        recent = deltas[-period:]
        gains = np.where(recent > 0, recent, 0)
        losses = np.where(recent < 0, -recent, 0)
        avg_gain = float(np.mean(gains))
        avg_loss = float(np.mean(losses))
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return round(100.0 - (100.0 / (1.0 + rs)), 2)

    async def get_price_range(
        self, ticker: str, start_date: str, end_date: str
    ) -> list[dict]:
        """Fetch daily prices between start_date and end_date.
        Returns list of dicts with 'adjClose' key (matches Tiingo format for backtester).
        """
        try:
            df = await asyncio.to_thread(self._fetch_range, ticker, start_date, end_date)
            if df is None or df.empty:
                return []
            result = []
            for date, row in df.iterrows():
                result.append({
                    "date": date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
                    "adjClose": float(row["Close"]),
                })
            return result
        except Exception as e:
            logger.warning("yahoo.price_range_failed", ticker=ticker, error=str(e))
            return []

    def _fetch_range(self, ticker: str, start_date: str, end_date: str):
        for symbol in yahoo_candidates(ticker):
            df = yf.Ticker(symbol).history(start=start_date, end=end_date)
            if df is not None and not df.empty:
                close = df["Close"].astype(float)
                df = df[np.isfinite(close) & (close > 0)]
                if not df.empty:
                    return df
        return None

    async def close(self):
        pass
