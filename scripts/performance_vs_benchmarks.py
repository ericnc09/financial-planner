"""
Live track record: system buy signals vs S&P 500 and Nasdaq Composite.

Builds equal-weight, buy-and-hold equity curves from actual signals in
smart_money.db (entry = first close on/after the signal's created_at date —
the day the live system ingested it and could actually have traded; using
disclosure_date would leak backfilled pre-launch signals into the record). Two portfolio lines:

  - all buys:        every distinct ticker with a buy signal
  - high conviction: tickers with at least one threshold-passing buy

Benchmarks (^GSPC, ^IXIC) are rebased to 100 at the same start date.
Before a pick's entry date its capital sits in cash (ratio 1.0), so the
curve is money-weighted the same way a real account funding each pick
equally would be. Tickers with no price data at all are carried at 0
(delisted = total loss, no survivorship bias); tickers whose history ends
early are carried flat at their last price.

Outputs: reports/performance_vs_benchmarks.json and .png
"""

import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "smart_money.db"
OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(exist_ok=True)

COST_PER_TRADE = 0.0030  # same 30bps round-trip assumption as PerformanceTracker


def load_picks() -> tuple[dict[str, str], dict[str, str]]:
    """Return {ticker: first_signal_date} for all buys and the high-conviction subset."""
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        """
        SELECT e.ticker,
               MIN(date(e.created_at)) AS first_date,
               MAX(COALESCE(cs.passes_threshold, 0)) AS ever_passed
        FROM smart_money_events e
        LEFT JOIN conviction_scores cs ON cs.event_id = e.id
        WHERE e.direction = 'buy'
        GROUP BY e.ticker
        """
    ).fetchall()
    conn.close()
    all_buys = {t: d for t, d, _ in rows}
    high_conv = {t: d for t, d, p in rows if p}
    return all_buys, high_conv


def fetch_prices(tickers: list[str], start: str) -> pd.DataFrame:
    """Daily closes, one column per ticker. Missing tickers come back all-NaN."""
    data = yf.download(
        tickers, start=start, auto_adjust=True, progress=False, group_by="column"
    )
    closes = data["Close"]
    if isinstance(closes, pd.Series):  # single ticker edge case
        closes = closes.to_frame(tickers[0])
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    return closes


def equity_curve(closes: pd.DataFrame, picks: dict[str, str], calendar: pd.DatetimeIndex):
    """Equal-weight curve over `calendar`; returns (series, per_pick_stats, dead)."""
    ratios = pd.DataFrame(index=calendar, dtype=float)
    stats, dead = [], []
    for ticker, signal_date in picks.items():
        col = closes[ticker].dropna() if ticker in closes else pd.Series(dtype=float)
        col = col[col.index >= pd.Timestamp(signal_date)]
        if col.empty:
            # never traded after signal -> assume delisted, total loss from signal date
            r = pd.Series(1.0, index=calendar)
            r[calendar >= pd.Timestamp(signal_date)] = 0.0
            ratios[ticker] = r
            dead.append(ticker)
            stats.append({"ticker": ticker, "entry_date": signal_date,
                          "entry": None, "last": None, "return_pct": -100.0})
            continue
        entry_date, entry_px = col.index[0], float(col.iloc[0])
        r = (col / entry_px).reindex(calendar)
        r[calendar < entry_date] = 1.0          # cash before entry
        r = r.ffill()                            # carry flat if history ends early
        # one-time entry cost
        r[calendar >= entry_date] *= (1 - COST_PER_TRADE)
        ratios[ticker] = r
        stats.append({
            "ticker": ticker,
            "entry_date": entry_date.strftime("%Y-%m-%d"),
            "entry": round(entry_px, 2),
            "last": round(float(col.iloc[-1]), 2),
            "return_pct": round((float(col.iloc[-1]) / entry_px - 1 - COST_PER_TRADE) * 100, 2),
        })
    curve = 100.0 * ratios.mean(axis=1)
    return curve, sorted(stats, key=lambda s: s["return_pct"], reverse=True), dead


def main():
    all_buys, high_conv = load_picks()
    start = min(all_buys.values())
    print(f"{len(all_buys)} buy tickers ({len(high_conv)} high-conviction), since {start}")

    bench = fetch_prices(["^GSPC", "^IXIC"], start)
    bench = bench.dropna(how="all")
    calendar = bench.index

    closes = fetch_prices(sorted(all_buys), start)

    curve_all, stats_all, dead = equity_curve(closes, all_buys, calendar)
    curve_hc, stats_hc, _ = equity_curve(closes, high_conv, calendar)
    sp = 100.0 * bench["^GSPC"] / bench["^GSPC"].iloc[0]
    nq = 100.0 * bench["^IXIC"] / bench["^IXIC"].iloc[0]

    payload = {
        "as_of": date.today().isoformat(),
        "start": start,
        "n_all": len(all_buys),
        "n_high_conviction": len(high_conv),
        "dead_tickers": dead,
        "dates": [d.strftime("%Y-%m-%d") for d in calendar],
        "series": {
            "all_buys": [round(v, 3) for v in curve_all],
            "high_conviction": [round(v, 3) for v in curve_hc],
            "sp500": [round(v, 3) for v in sp],
            "nasdaq": [round(v, 3) for v in nq],
        },
        "totals_pct": {
            "all_buys": round(curve_all.iloc[-1] - 100, 2),
            "high_conviction": round(curve_hc.iloc[-1] - 100, 2),
            "sp500": round(sp.iloc[-1] - 100, 2),
            "nasdaq": round(nq.iloc[-1] - 100, 2),
        },
        "picks_all": stats_all,
        "picks_high_conviction": stats_hc,
    }
    (OUT_DIR / "performance_vs_benchmarks.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["totals_pct"], indent=2))
    print("dead:", dead)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(calendar, curve_hc, label=f"High-conviction picks ({len(high_conv)})", lw=2.2, color="#7c3aed")
    ax.plot(calendar, curve_all, label=f"All buy signals ({len(all_buys)})", lw=2.2, color="#0ea5e9")
    ax.plot(calendar, sp, label="S&P 500", lw=1.6, color="#64748b", ls="--")
    ax.plot(calendar, nq, label="Nasdaq Composite", lw=1.6, color="#f59e0b", ls="--")
    ax.axhline(100, color="#cbd5e1", lw=0.8)
    ax.set_title(f"Smart Money Follows — live signals vs benchmarks ({start} → {date.today()})")
    ax.set_ylabel("Growth of 100 (equal-weight, 30bps cost, incl. delistings)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "performance_vs_benchmarks.png", dpi=150)
    print(f"saved {OUT_DIR / 'performance_vs_benchmarks.png'}")


if __name__ == "__main__":
    sys.exit(main())
