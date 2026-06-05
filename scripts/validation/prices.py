"""Shared price-data layer for the validation scripts.

Daily adjusted closes + volumes from yfinance, cached to disk so the
backtest is deterministic and reruns are instant. auto_adjust=True folds
splits and dividends into Close, giving an (approximate) total-return series.
"""

import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

CACHE = Path("reports/cache/prices")
CACHE.mkdir(parents=True, exist_ok=True)

_START = "2021-06-01"  # buffer before 2023 for 52w-high / momentum lookbacks


class PriceStore:
    def __init__(self, end: str | None = None):
        self.end = end or datetime.utcnow().strftime("%Y-%m-%d")
        self._mem: dict[str, dict] = {}

    def get(self, ticker: str) -> dict | None:
        """Return {'dates': list[date], 'closes': np.ndarray, 'vols': np.ndarray} or None."""
        ticker = ticker.upper()
        if ticker in self._mem:
            return self._mem[ticker] or None

        # path-safe cache name (some tickers contain "/" or ":")
        safe = ticker.replace("/", "_").replace("\\", "_").replace(":", "_")
        cache = CACHE / f"{safe}.csv"
        df = None
        if cache.exists():
            try:
                df = pd.read_csv(cache, parse_dates=["date"])
            except Exception:
                df = None
        if df is None:
            df = self._download(ticker)
            if df is not None:
                df.to_csv(cache, index=False)
            else:
                # write a sentinel empty file so we don't refetch dead tickers
                pd.DataFrame(columns=["date", "close", "vol"]).to_csv(cache, index=False)
                df = pd.DataFrame(columns=["date", "close", "vol"])

        if df is None or df.empty:
            self._mem[ticker] = {}
            return None

        rec = {
            "dates": [d.date() if hasattr(d, "date") else d for d in pd.to_datetime(df["date"]).tolist()],
            "closes": df["close"].to_numpy(dtype=float),
            "vols": df["vol"].to_numpy(dtype=float),
        }
        self._mem[ticker] = rec
        return rec

    def _download(self, ticker: str):
        for attempt in range(3):
            try:
                df = yf.Ticker(ticker).history(
                    start=_START, end=self.end, auto_adjust=True
                )
                if df is None or df.empty:
                    return None
                out = pd.DataFrame({
                    "date": df.index.tz_localize(None) if df.index.tz is not None else df.index,
                    "close": df["Close"].to_numpy(dtype=float),
                    "vol": df["Volume"].to_numpy(dtype=float),
                })
                out["date"] = pd.to_datetime(out["date"]).dt.date
                return out
            except Exception:
                time.sleep(1.0 * (attempt + 1))
        return None


def entry_index(dates: list[date], after: date) -> int | None:
    """First trading-day index strictly after `after` (next-day execution)."""
    for i, d in enumerate(dates):
        if d > after:
            return i
    return None


def asof_index(dates: list[date], on_or_before: date) -> int | None:
    """Last trading-day index with date <= on_or_before (point-in-time)."""
    idx = None
    for i, d in enumerate(dates):
        if d <= on_or_before:
            idx = i
        else:
            break
    return idx
