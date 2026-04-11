"""
Fetch today's insider + congressional trades, run full model analysis
on every unique ticker that appeared, and post results to Slack.

Usage:
    python scripts/insider_signal_run.py [--days N]

    --days N   Look-back window in calendar days (default: 1 = today only)
"""

import asyncio
import sys
import os
import argparse
from datetime import datetime
from collections import defaultdict

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clients.edgar import EdgarClient
from src.clients.congress import CongressClient
from src.clients.yahoo import YahooClient
from src.clients.fama_french import FamaFrenchClient
from src.analysis.monte_carlo import MonteCarloSimulator
from src.analysis.hmm_regime import HMMRegimeDetector
from src.analysis.garch_forecast import GARCHForecaster
from src.analysis.copula_tail_risk import CopulaTailRisk
from src.analysis.ensemble_scoring import EnsembleScorer
from src.analysis.ticker_analysis import analyze_ticker
from src.slack_bot.formatter import format_results
from config.settings import Settings


async def fetch_signals(since_days: int) -> tuple[list, dict]:
    """
    Fetch insider + congressional trades for the past `since_days` days.

    Returns:
        (all_events, ticker_signals) — flat list of SmartMoneyEventSchema objects
        and a dict mapping ticker -> list of signal summaries.
    """
    edgar = EdgarClient()
    congress = CongressClient()

    print(f"Fetching insider trades (EDGAR Form 4, last {since_days}d)...")
    insider_events = await edgar.get_bulk_insider_trades(since_days=since_days)
    print(f"  Found {len(insider_events)} insider trades")

    print(f"Fetching congressional trades (Capitol Trades, last {since_days}d)...")
    congressional_events = await congress.get_all_congressional_trades(since_days=since_days)
    print(f"  Found {len(congressional_events)} congressional trades")

    await edgar.close()
    await congress.close()

    all_events = insider_events + congressional_events

    # Group by ticker: store who traded what, so we can annotate Slack
    ticker_signals: dict[str, list[dict]] = defaultdict(list)
    for e in all_events:
        ticker_signals[e.ticker].append({
            "actor": e.actor,
            "direction": e.direction.value,
            "source": e.source_type.value,
            "date": e.trade_date.strftime("%Y-%m-%d"),
            "value": e.size_lower,
        })

    return all_events, dict(ticker_signals)


def build_signal_context_block(ticker: str, signals: list[dict]) -> dict:
    """
    Build a Slack Block Kit section showing who traded this ticker.
    """
    lines = []
    for s in sorted(signals, key=lambda x: x["date"], reverse=True)[:5]:
        direction_emoji = ":large_green_circle:" if s["direction"] == "buy" else ":red_circle:"
        source_tag = "insider" if s["source"] == "insider" else "congress"
        value_str = f"  ${s['value']:,.0f}" if s.get("value") else ""
        lines.append(
            f"{direction_emoji} `{source_tag}` *{s['actor']}* — {s['direction'].upper()} {s['date']}{value_str}"
        )
    if len(signals) > 5:
        lines.append(f"_...and {len(signals) - 5} more trades_")
    return {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"*Signals for {ticker}:*\n" + "\n".join(lines),
        },
    }


async def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=1,
                        help="Look-back window in days (default: 1 = today)")
    args = parser.parse_args()

    settings = Settings()
    webhook_url = settings.slack_webhook_url
    if not webhook_url:
        print("ERROR: SLACK_WEBHOOK_URL not set in config/.env")
        sys.exit(1)

    print("=" * 70)
    print(f"INSIDER + CONGRESSIONAL SIGNAL RUN  ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print(f"Look-back: {args.days} day(s)")
    print("=" * 70)
    print()

    # ── Step 1: Fetch signals ──────────────────────────────────────────────
    all_events, ticker_signals = await fetch_signals(args.days)

    if not ticker_signals:
        print("No trades found for today. Exiting.")
        return

    unique_tickers = sorted(ticker_signals.keys())
    print(f"\n{len(unique_tickers)} unique tickers with trades today: {', '.join(unique_tickers)}\n")

    # ── Step 2: Run model analysis ─────────────────────────────────────────
    print("Running analysis models...")
    yahoo = YahooClient()
    ff_client = FamaFrenchClient()
    mc = MonteCarloSimulator(n_simulations=10_000, seed=42)
    hmm = HMMRegimeDetector()
    garch = GARCHForecaster()
    copula = CopulaTailRisk()
    ensemble = EnsembleScorer()

    ff_factors = await ff_client.get_factors(days=504)
    print(f"  Fama-French factors: {'OK' if ff_factors is not None else 'FAILED'}\n")

    results = []
    failed = []

    for ticker in unique_tickers:
        signals = ticker_signals[ticker]
        n_buys = sum(1 for s in signals if s["direction"] == "buy")
        n_sells = sum(1 for s in signals if s["direction"] == "sell")
        print(f"  {ticker}  [{n_buys}B/{n_sells}S]...", end=" ", flush=True)
        try:
            result = await asyncio.wait_for(
                analyze_ticker(ticker, yahoo, ff_client, ff_factors,
                               mc, hmm, garch, copula, ensemble),
                timeout=120,
            )
            if result:
                result["_trade_signals"] = signals  # attach signal context
                results.append(result)
                regime = result.get("hmm_regime") or "?"
                print(
                    f"{result['verdict']:4}  "
                    f"buy={result['buy_score']:.0f} sell={result['sell_score']:.0f}  "
                    f"regime={regime}"
                )
            else:
                failed.append(ticker)
                print("NO DATA")
        except asyncio.TimeoutError:
            failed.append(ticker)
            print("TIMEOUT")
        except Exception as e:
            failed.append(ticker)
            print(f"ERROR: {e}")

    if not results:
        print("\nNo results to send.")
        return

    # Print local summary
    print()
    buys  = [r for r in results if r["verdict"] == "BUY"]
    sells = [r for r in results if r["verdict"] == "SELL"]
    holds = [r for r in results if r["verdict"] == "HOLD"]
    print(f"VERDICT SUMMARY: {len(buys)} BUY  |  {len(holds)} HOLD  |  {len(sells)} SELL  |  {len(failed)} failed")

    # ── Step 3: Build Slack payload ────────────────────────────────────────
    payload = format_results(results, failed if failed else None)

    # Header block
    run_header = {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": (
                f":mag: *Insider + Congressional Signal Run* — "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M')} ET\n"
                f"_{len(all_events)} trades across {len(unique_tickers)} tickers "
                f"(past {args.days}d)  •  "
                f"7-model ensemble: MC · HMM · GARCH · FF · Copula · Earnings · Decay_"
            ),
        },
    }
    payload["blocks"] = [run_header, {"type": "divider"}] + payload.get("blocks", [])

    # Inject trade-signal context into each ticker's attachment
    ticker_result_map = {r["ticker"]: r for r in results}
    augmented_attachments = []
    for attachment in payload.get("attachments", []):
        # Determine which ticker this attachment is for from the header block
        header_text = ""
        for blk in attachment.get("blocks", []):
            if blk.get("type") == "header":
                header_text = blk.get("text", {}).get("text", "")
                break
        # Extract ticker from header "● TICK — VERDICT ..."
        ticker_in_header = None
        for t in unique_tickers:
            if f" {t} " in header_text or header_text.startswith(t) or f":{t}" in header_text:
                ticker_in_header = t
                break
        if not ticker_in_header:
            # Try matching against all result tickers
            for r in results:
                if r["ticker"] in header_text:
                    ticker_in_header = r["ticker"]
                    break

        if ticker_in_header and ticker_in_header in ticker_signals:
            context_block = build_signal_context_block(
                ticker_in_header, ticker_signals[ticker_in_header]
            )
            attachment["blocks"].append(context_block)
        augmented_attachments.append(attachment)

    payload["attachments"] = augmented_attachments

    # ── Step 4: POST to Slack ──────────────────────────────────────────────
    print("\nSending to Slack...", end=" ", flush=True)
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(webhook_url, json=payload)
        if resp.status_code == 200:
            print("sent.")
        else:
            print(f"ERROR {resp.status_code}: {resp.text}")


if __name__ == "__main__":
    asyncio.run(run())
