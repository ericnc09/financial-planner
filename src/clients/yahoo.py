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

    async def get_sector_tickers(self, sector: str) -> list[str]:
        """Placeholder — returns empty list. Could be expanded with yfinance screener."""
        return []

    async def close(self):
        pass
