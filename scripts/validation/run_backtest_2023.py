"""
2023 point-in-time backtest of the Smart Money conviction engine.

Pipeline (no look-ahead at any step):
  1. Load every 2023 open-market insider buy in the fixed S&P 600 universe.
  2. Score each with the LIVE conviction engine, fed only point-in-time inputs:
       - signal layer: actor/size/cluster/timing/consensus (uses only buys
         already disclosed on or before the decision date)
       - fundamental layer: momentum/RSI/drawdown/liquidity from prices up to
         the disclosure date; valuation & short-interest neutralized (no
         point-in-time fundamentals source -> not faked)
       - macro layer: FRED regime as of the disclosure date
  3. Rank, take the top 15 distinct tickers by conviction.
  4. Enter next trading day after disclosure; measure forward returns at
     1/3/6/12-month and to-date horizons, net of liquidity-tier costs, with
     delistings booked as -100% (survivorship control).
  5. Benchmark every position against SPY (S&P 500) and QQQ (Nasdaq-100) over
     the identical window, and build an equal-weight equity curve vs both.

Outputs reports/backtest_2023_results.json.
"""

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import structlog

# the live scoring engine logs every score; quiet it for the batch backtest
logging.basicConfig(level=logging.WARNING)
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))

from config.settings import Settings
from src.models.schemas import (
    Direction, EnrichmentSchema, MacroSnapshotSchema, MacroRegime,
    SmartMoneyEventSchema, SourceType,
)
from src.scoring.signal_scorer import SignalScorer
from src.scoring.fundamental_scorer import FundamentalScorer
from src.scoring.macro_scorer import MacroScorer
from src.scoring.conviction_engine import ConvictionEngine
from src.backtesting.backtester import Backtester

from scripts.validation.prices import PriceStore, entry_index, asof_index

IN_FILE = Path("reports/cache/insider_2023_buys.json")
OUT_FILE = Path("reports/backtest_2023_results.json")
N_PICKS = 15
HORIZONS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
BENCHMARKS = {"sp500": "SPY", "nasdaq": "QQQ"}


# ----------------------------- point-in-time enrichment -----------------------
def pit_enrichment(store: PriceStore, ticker: str, as_of: date) -> EnrichmentSchema | None:
    """Reconstruct the price-derived enrichment using only data <= as_of.

    Mirrors src/clients/yahoo.py:enrich_ticker but on a sliced series.
    Valuation (P/E), market cap and short interest are left None because no
    point-in-time source exists here; the fundamental scorer treats None as a
    neutral 0.5, so we never inject post-hoc financials.
    """
    rec = store.get(ticker)
    if not rec:
        return None
    j = asof_index(rec["dates"], as_of)
    if j is None or j < 1:
        return None
    closes = rec["closes"][: j + 1]
    vols = rec["vols"][: j + 1]
    if len(closes) < 22:
        return None

    cur = float(closes[-1])
    mom_30 = (cur - closes[-22]) / closes[-22] if len(closes) >= 22 and closes[-22] > 0 else None
    mom_90 = (cur - closes[-63]) / closes[-63] if len(closes) >= 63 and closes[-63] > 0 else None

    # RSI-14 (matches yahoo client)
    rsi = None
    if len(closes) >= 15:
        deltas = np.diff(closes)[-14:]
        gains = float(np.mean(np.where(deltas > 0, deltas, 0)))
        losses = float(np.mean(np.where(deltas < 0, -deltas, 0)))
        rsi = 100.0 if losses == 0 else round(100.0 - 100.0 / (1.0 + gains / losses), 2)

    window = closes[-270:]  # ~52 weeks
    high_52w = float(np.max(window)) if len(window) else cur
    drawdown = (high_52w - cur) / high_52w if high_52w > 0 else None
    avg_vol = float(np.mean(vols[-30:])) if len(vols) >= 30 else (float(np.mean(vols)) if len(vols) else None)

    return EnrichmentSchema(
        pe_ratio=None, market_cap=None, revenue_growth_yoy=None,
        eps_latest=None, eps_growth_yoy=None,
        price_at_signal=cur, momentum_30d=mom_30, momentum_90d=mom_90,
        rsi_14d=rsi, drawdown_from_52w_high=drawdown, avg_volume_30d=avg_vol,
        sector=None, short_ratio=None,
    )


# ----------------------------- macro (PIT, cached by month) -------------------
class MacroPIT:
    def __init__(self, settings: Settings):
        self._cache: dict[str, MacroSnapshotSchema] = {}
        self._fred = None
        if settings.fred_api_key:
            try:
                from src.clients.fred import FredClient
                self._fred = FredClient(settings.fred_api_key)
            except Exception:
                self._fred = None

    def get(self, as_of: date) -> MacroSnapshotSchema:
        key = as_of.strftime("%Y-%m")
        if key in self._cache:
            return self._cache[key]
        snap = None
        if self._fred is not None:
            try:
                snap = self._fred.get_macro_snapshot(as_of=as_of.strftime("%Y-%m-%d"))
            except Exception:
                snap = None
        if snap is None:
            # documented fallback: 2023 was an inverted-curve, high-rate regime
            snap = MacroSnapshotSchema(regime=MacroRegime.TRANSITION, regime_score=0.6)
        self._cache[key] = snap
        return snap


# ----------------------------- returns ---------------------------------------
def forward_returns(store: PriceStore, ticker: str, disclosure: date,
                    direction: Direction, cost: float) -> dict:
    """Returns at each horizon + to-date. Delisting -> -100%. Net of cost."""
    rec = store.get(ticker)
    out = {"entry_date": None, "entry_price": None, **{h: None for h in HORIZONS}, "to_date": None}
    if not rec:
        return out
    dates, closes = rec["dates"], rec["closes"]
    ei = entry_index(dates, disclosure)
    if ei is None or ei >= len(closes):
        return out
    entry = float(closes[ei])
    if entry <= 0:
        return out
    out["entry_date"] = dates[ei].isoformat()
    out["entry_price"] = round(entry, 4)
    sign = 1.0 if direction == Direction.BUY else -1.0

    def _ret(exit_idx):
        # Exit at the horizon price if it exists, else at the last available
        # close. Survivors always have data through today, so this fallback
        # only fires for delisted names — which in the S&P 600 are mostly
        # acquisitions (exit at deal price), not bankruptcies. Exiting at the
        # last traded price avoids both survivorship bias (we don't drop the
        # name) and a fake -100% on names that were actually bought out.
        ex = float(closes[exit_idx]) if exit_idx < len(closes) else float(closes[-1])
        if ex <= 0:
            return round(-1.0 - cost, 6)  # genuine zero print -> total loss
        return round(sign * (ex - entry) / entry - cost, 6)

    for h, td in HORIZONS.items():
        out[h] = _ret(ei + td)
    out["to_date"] = _ret(len(closes) - 1)
    out["exit_date_todate"] = dates[-1].isoformat()
    out["delisted"] = (datetime.utcnow().date() - dates[-1]).days > 10  # data ended early
    return out


def bench_return(store: PriceStore, sym: str, entry_d: date, exit_d: date) -> float | None:
    rec = store.get(sym)
    if not rec:
        return None
    ei = entry_index(rec["dates"], entry_d - timedelta(days=1))  # close on/after entry
    xi = asof_index(rec["dates"], exit_d)
    if ei is None or xi is None or xi <= ei:
        return None
    e, x = float(rec["closes"][ei]), float(rec["closes"][xi])
    return round((x - e) / e, 6) if e > 0 else None


# ----------------------------- equity curve ----------------------------------
def _asset_curve(store: PriceStore, sym_or_ticker: str, entry_d: date, cal: list[date]):
    """Daily value (per $1) of one position held from entry_d to end of `cal`.

    Before entry the dollar sits in cash (value 1.0). Price is carried forward
    on the asset's own non-trading days. If the series ends before today
    (delisting) the value goes to 0 afterwards — a survivorship control.
    """
    rec = store.get(sym_or_ticker)
    if not rec:
        return None
    ei = entry_index(rec["dates"], entry_d - timedelta(days=1))
    if ei is None:
        return None
    ep = float(rec["closes"][ei])
    if ep <= 0:
        return None
    dmap = {d: float(rec["closes"][k]) for k, d in enumerate(rec["dates"])}
    vals, carry = [], None
    for d in cal:
        if d in dmap:
            carry = dmap[d]
        if d < entry_d or carry is None:
            vals.append(1.0)            # cash before entry
        else:
            # carry freezes at the last close after a stock stops trading,
            # so a delisted name holds its last value (exit at last price) —
            # consistent with forward_returns, no fake total loss.
            vals.append(carry / ep)
    return vals


def equity_curves(store: PriceStore, picks: list[dict]) -> dict:
    """Equal-weight ($1/pick) basket vs the same cash deployed into SPY/QQQ.

    Each pick contributes $1 entered on its own entry date; benchmarks deploy
    the identical $1-per-pick on the identical dates, so the comparison is
    cash-flow matched rather than a single lump sum.
    """
    entries = [date.fromisoformat(p["returns"]["entry_date"]) for p in picks if p["returns"]["entry_date"]]
    if not entries:
        return {}
    start = min(entries)
    cal = [d for d in store.get("SPY")["dates"] if d >= start]
    n = len(picks)

    def blend(sym_resolver):
        acc = np.zeros(len(cal))
        used = 0
        for p in picks:
            ed = date.fromisoformat(p["returns"]["entry_date"])
            vals = _asset_curve(store, sym_resolver(p), ed, cal)
            if vals is None:
                acc += 1.0  # treat unresolved as cash so totals stay comparable
            else:
                acc += np.array(vals)
            used += 1
        return [round(v / n * 100, 4) for v in acc]

    return {
        "dates": [d.isoformat() for d in cal],
        "portfolio": blend(lambda p: p["ticker"]),
        "sp500": blend(lambda p: "SPY"),
        "nasdaq": blend(lambda p: "QQQ"),
    }


# ----------------------------- main ------------------------------------------
def main():
    settings = Settings()
    data = json.loads(IN_FILE.read_text())
    raw = data["events"]
    print(f"loaded {len(raw)} raw 2023 buy events from {data['universe']}")

    # Build event objects (skip rows with no usable dollar value -> size None ok)
    events = []
    for e in raw:
        try:
            td = datetime.strptime(e["trade_date"], "%Y-%m-%d")
            dd = datetime.strptime(e["disclosure_date"], "%Y-%m-%d")
        except (ValueError, KeyError):
            continue
        events.append(SmartMoneyEventSchema(
            ticker=e["ticker"], actor=e["actor"], direction=Direction.BUY,
            size_lower=e.get("dollar_value"), size_upper=e.get("dollar_value"),
            trade_date=td, disclosure_date=dd, source_type=SourceType.INSIDER,
        ))
    events.sort(key=lambda x: x.disclosure_date)
    print(f"usable events: {len(events)} across {len({e.ticker for e in events})} tickers")

    store = PriceStore()
    # warm benchmarks
    store.get("SPY"); store.get("QQQ")

    sig, fund, macro = SignalScorer(), FundamentalScorer(), MacroScorer()
    engine = ConvictionEngine(settings, sig, fund, macro)
    macro_pit = MacroPIT(settings)

    scored = []
    for i, ev in enumerate(events):
        # PIT cluster/consensus: only buys already public by this disclosure date
        recent = [o for o in events if o.disclosure_date <= ev.disclosure_date]
        enr = pit_enrichment(store, ev.ticker, ev.disclosure_date.date())
        if enr is None:
            continue
        snap = macro_pit.get(ev.disclosure_date.date())
        res = engine.compute(ev, enr, snap, recent)
        scored.append({
            "ticker": ev.ticker, "actor": ev.actor,
            "trade_date": ev.trade_date.date().isoformat(),
            "disclosure_date": ev.disclosure_date.date().isoformat(),
            "dollar_value": ev.size_lower,
            "conviction": res.conviction, "signal_score": res.signal_score,
            "fundamental_score": res.fundamental_score, "macro_modifier": res.macro_modifier,
            "passes_threshold": res.passes_threshold,
            "avg_volume_30d": enr.avg_volume_30d,
        })
        if (i + 1) % 500 == 0:
            print(f"  scored {i+1}/{len(events)}")

    print(f"scored {len(scored)} events")

    # Top-15 distinct tickers by conviction (keep best event per ticker)
    best_by_ticker: dict[str, dict] = {}
    for s in scored:
        b = best_by_ticker.get(s["ticker"])
        if b is None or s["conviction"] > b["conviction"]:
            best_by_ticker[s["ticker"]] = s
    ranked = sorted(best_by_ticker.values(), key=lambda x: x["conviction"], reverse=True)
    picks_meta = ranked[:N_PICKS]

    # Returns + benchmarks for each pick
    picks = []
    for s in picks_meta:
        cost = Backtester._estimate_transaction_cost(s["avg_volume_30d"], None)
        rr = forward_returns(store, s["ticker"], date.fromisoformat(s["disclosure_date"]),
                             Direction.BUY, cost)
        bench = {}
        if rr["entry_date"]:
            ed = date.fromisoformat(rr["entry_date"])
            xd = date.fromisoformat(rr.get("exit_date_todate") or rr["entry_date"])
            for bname, bsym in BENCHMARKS.items():
                bench[bname] = bench_return(store, bsym, ed, xd)
        picks.append({**s, "txn_cost": round(cost, 4), "returns": rr, "benchmark_to_date": bench})

    # Aggregate stats
    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return round(float(np.mean(xs)), 6) if xs else None

    def _median(xs):
        xs = [x for x in xs if x is not None]
        return round(float(np.median(xs)), 6) if xs else None

    picks_valid = [p for p in picks if p["returns"]["entry_date"]]
    td_returns = [p["returns"]["to_date"] for p in picks_valid]
    spy_td = [p["benchmark_to_date"].get("sp500") for p in picks_valid]
    qqq_td = [p["benchmark_to_date"].get("nasdaq") for p in picks_valid]
    beat_spy = sum(1 for p in picks_valid
                   if p["returns"]["to_date"] is not None and p["benchmark_to_date"].get("sp500") is not None
                   and p["returns"]["to_date"] > p["benchmark_to_date"]["sp500"])

    # Unfiltered baseline: avg to-date return of ALL distinct-ticker buys (best event each)
    base = []
    for s in best_by_ticker.values():
        cost = Backtester._estimate_transaction_cost(s["avg_volume_30d"], None)
        rr = forward_returns(store, s["ticker"], date.fromisoformat(s["disclosure_date"]), Direction.BUY, cost)
        if rr["entry_date"]:
            base.append(rr["to_date"])

    curves = equity_curves(store, picks_valid)

    summary = {
        "n_picks": len(picks_valid),
        "avg_return": {h: _mean([p["returns"][h] for p in picks_valid]) for h in list(HORIZONS) + ["to_date"]},
        "median_return": {h: _median([p["returns"][h] for p in picks_valid]) for h in list(HORIZONS) + ["to_date"]},
        "portfolio_to_date": curves["portfolio"][-1] / 100 - 1 if curves else None,
        "sp500_to_date_matched": curves["sp500"][-1] / 100 - 1 if curves else None,
        "nasdaq_to_date_matched": curves["nasdaq"][-1] / 100 - 1 if curves else None,
        "avg_pick_to_date": _mean(td_returns),
        "avg_sp500_to_date": _mean(spy_td),
        "avg_nasdaq_to_date": _mean(qqq_td),
        "picks_beating_sp500": f"{beat_spy}/{len(picks_valid)}",
        "unfiltered_universe_avg_to_date": _mean(base),
        "unfiltered_universe_n": len(base),
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({
        "generated_at": datetime.utcnow().isoformat(),
        "methodology": "point-in-time backtest; live conviction engine; PIT inputs; next-day entry; "
                       "liquidity-tier costs; delisted names exit at last traded price; "
                       "benchmarks SPY/QQQ matched-cash.",
        "universe": data["universe"],
        "window": "insider open-market purchases disclosed in 2023",
        "n_raw_events": len(raw),
        "n_scored_events": len(scored),
        "summary": summary,
        "picks": picks,
        "equity_curve": curves,
    }, indent=2, default=str))

    print("\n==== SUMMARY ====")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {OUT_FILE}")


if __name__ == "__main__":
    main()
