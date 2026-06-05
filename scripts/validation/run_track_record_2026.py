"""
Real since-launch track record (no backtest, no reconstruction).

These are the actual signals the live system scored and stored in
smart_money.db between March and May 2026. We take the buy signals that
cleared the live conviction threshold — i.e. the ones the system would have
acted on — and measure their realized return against SPY/QQQ over the
identical holding window.

This record is short (the system is only a few months old) and the window is
still open, so it is shown as honest, unvarnished evidence alongside the
longer 2023 backtest — not as a finished result.

Outputs reports/track_record_2026_results.json.
"""

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path

import numpy as np

from src.models.schemas import Direction
from src.backtesting.backtester import Backtester
from scripts.validation.prices import PriceStore
from scripts.validation.run_backtest_2023 import forward_returns, bench_return, equity_curves, HORIZONS, BENCHMARKS

DB = "smart_money.db"
OUT_FILE = Path("reports/track_record_2026_results.json")
THRESHOLD = 0.6  # live actionable conviction threshold


def load_actionable_buys():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT e.ticker, e.actor, e.direction,
               e.trade_date, e.disclosure_date,
               c.conviction, c.passes_threshold,
               en.avg_volume_30d
        FROM smart_money_events e
        JOIN conviction_scores c ON e.id = c.event_id
        LEFT JOIN enrichments en ON e.id = en.event_id
        WHERE e.direction = 'buy' AND c.conviction >= ?
        ORDER BY c.conviction DESC
        """,
        (THRESHOLD,),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def _d(s: str) -> date:
    return datetime.fromisoformat(str(s)).date() if s else None


def main():
    rows = load_actionable_buys()
    # de-dup to one (best-conviction) entry per ticker, like the backtest
    best: dict[str, dict] = {}
    for r in rows:
        b = best.get(r["ticker"])
        if b is None or r["conviction"] > b["conviction"]:
            best[r["ticker"]] = r
    picks_meta = sorted(best.values(), key=lambda x: x["conviction"], reverse=True)
    print(f"actionable buy signals (conviction>={THRESHOLD}): {len(rows)} rows, "
          f"{len(picks_meta)} distinct tickers")

    store = PriceStore()
    store.get("SPY"); store.get("QQQ")

    picks = []
    skipped_bad = 0
    for s in picks_meta:
        disc = _d(s["disclosure_date"]) or _d(s["trade_date"])
        trade = _d(s["trade_date"])
        if disc is None:
            continue
        # data hygiene: drop records disclosed before the trade (impossible) and
        # anything outside the system's real operating window (launched 2026).
        if (trade and disc < trade) or disc < date(2026, 1, 1):
            skipped_bad += 1
            continue
        cost = Backtester._estimate_transaction_cost(s.get("avg_volume_30d"), None)
        rr = forward_returns(store, s["ticker"], disc, Direction.BUY, cost)
        bench = {}
        if rr["entry_date"]:
            ed = date.fromisoformat(rr["entry_date"])
            xd = date.fromisoformat(rr.get("exit_date_todate") or rr["entry_date"])
            for bname, bsym in BENCHMARKS.items():
                bench[bname] = bench_return(store, bsym, ed, xd)
        picks.append({
            "ticker": s["ticker"], "actor": s["actor"],
            "trade_date": str(_d(s["trade_date"])), "disclosure_date": str(disc),
            "conviction": s["conviction"], "txn_cost": round(cost, 4),
            "returns": rr, "benchmark_to_date": bench,
        })

    valid = [p for p in picks if p["returns"]["entry_date"]]

    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return round(float(np.mean(xs)), 6) if xs else None

    td = [p["returns"]["to_date"] for p in valid]
    spy = [p["benchmark_to_date"].get("sp500") for p in valid]
    qqq = [p["benchmark_to_date"].get("nasdaq") for p in valid]
    beat = sum(1 for p in valid
               if p["returns"]["to_date"] is not None
               and p["benchmark_to_date"].get("sp500") is not None
               and p["returns"]["to_date"] > p["benchmark_to_date"]["sp500"])

    curves = equity_curves(store, valid)
    summary = {
        "n_signals_rows": len(rows),
        "n_distinct_tickers": len(valid),
        "data_hygiene_skipped": skipped_bad,
        "avg_to_date": _mean(td),
        "median_to_date": round(float(np.median([x for x in td if x is not None])), 6) if td else None,
        "avg_sp500_to_date": _mean(spy),
        "avg_nasdaq_to_date": _mean(qqq),
        "beating_sp500": f"{beat}/{len(valid)}",
        "portfolio_to_date": (curves["portfolio"][-1] / 100 - 1) if curves else None,
        "sp500_to_date_matched": (curves["sp500"][-1] / 100 - 1) if curves else None,
        "nasdaq_to_date_matched": (curves["nasdaq"][-1] / 100 - 1) if curves else None,
        "note": "Window opened Mar-May 2026 and is still running; holding periods are weeks, not years.",
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({
        "generated_at": datetime.utcnow().isoformat(),
        "kind": "real_live_track_record",
        "threshold": THRESHOLD,
        "summary": summary,
        "picks": picks,
        "equity_curve": curves,
    }, indent=2, default=str))
    print("\n==== 2026 TRACK RECORD ====")
    print(json.dumps(summary, indent=2, default=str))
    print(f"wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
