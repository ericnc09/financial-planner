import asyncio
from datetime import datetime, timedelta

import numpy as np
import structlog
import yfinance as yf

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
            closes = df["Close"].values.astype(float)
            volumes = df["Volume"].values.astype(float)
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
        end = datetime.now()
        start = end - timedelta(days=days)
        t = yf.Ticker(ticker)
        df = t.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        return df if not df.empty else None

    async def enrich_ticker(self, ticker: str):
        """Enrich a ticker using yfinance — free, no rate limits."""
        from src.models.schemas import EnrichmentSchema
        try:
            info, hist = await asyncio.to_thread(self._fetch_enrichment, ticker)
            if hist is None or hist.empty:
                return None

            closes = hist["Close"].values.astype(float)
            volumes = hist["Volume"].values.astype(float)

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

            # Drawdown from 52-week high
            drawdown = None
            if len(closes) > 0:
                high_52w = float(np.max(closes))
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
            )
        except Exception as e:
            logger.warning("yahoo.enrich_failed", ticker=ticker, error=str(e))
            return None

    def _fetch_enrichment(self, ticker: str, days: int = 400):
        """Synchronous fetch of info + history for enrichment."""
        t = yf.Ticker(ticker)
        try:
            info = t.info or {}
        except Exception:
            info = {}
        end = datetime.now()
        start = end - timedelta(days=days)
        hist = t.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        return info, hist if not hist.empty else (info, None)

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
        t = yf.Ticker(ticker)
        df = t.history(start=start_date, end=end_date)
        return df if not df.empty else None

    async def close(self):
        pass
