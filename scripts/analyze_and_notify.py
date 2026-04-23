"""
Run analysis on all tickers and post results to Slack via webhook.

Usage:
    python scripts/analyze_and_notify.py [TICKER1 TICKER2 ...]

If no tickers provided, runs the default Canadian banks + oil & gas watchlist.
"""

import asyncio
import sys
import os
from datetime import datetime

import httpx

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


DEFAULT_TICKERS = ["RY", "CM", "TD", "BNS", "E", "VLO", "EPD", "ET", "SUN", "SLB", "BKR", "FLS"]


async def run_and_notify(tickers: list[str]):
    settings = Settings()
    webhook_url = settings.slack_webhook_url
    if not webhook_url:
        print("ERROR: SLACK_WEBHOOK_URL not set in config/.env")
        sys.exit(1)

    print(f"Running analysis on {len(tickers)} tickers: {', '.join(tickers)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Init models
    yahoo = YahooClient()
    ff_client = FamaFrenchClient()
    mc = MonteCarloSimulator(n_simulations=10_000, seed=settings.random_seed)
    hmm = HMMRegimeDetector()
    garch = GARCHForecaster()
    copula = CopulaTailRisk()
    ensemble = EnsembleScorer()

    print("Loading Fama-French factors...")
    ff_factors = await ff_client.get_factors(days=504)
    print(f"  {'OK' if ff_factors is not None else 'FAILED — FF/Copula models will be skipped'}\n")

    results = []
    failed = []

    for ticker in tickers:
        print(f"  {ticker}...", end=" ", flush=True)
        try:
            result = await asyncio.wait_for(
                analyze_ticker(ticker, yahoo, ff_client, ff_factors, mc, hmm, garch, copula, ensemble),
                timeout=120,
            )
            if result:
                results.append(result)
                print(f"{result['verdict']}  (buy={result['buy_score']:.0f} sell={result['sell_score']:.0f}  {result['n_models']}/5 models)")
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

    # Build Slack payload
    payload = format_results(results, failed if failed else None)

    # Prepend a run-header block
    run_header = {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": (
                f":robot_face: *Smart Money Analysis* — "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M')} ET\n"
                f"_5-model ensemble: Monte Carlo · HMM Regime · GARCH · Fama-French · Copula Tail Risk_"
            ),
        },
    }
    payload["blocks"] = [run_header, {"type": "divider"}] + payload.get("blocks", [])

    # POST to webhook
    print(f"\nSending to Slack...")
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(webhook_url, json=payload)
        if resp.status_code == 200:
            print("Slack message sent successfully.")
        else:
            print(f"Slack error {resp.status_code}: {resp.text}")


if __name__ == "__main__":
    tickers = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TICKERS
    asyncio.run(run_and_notify([t.upper() for t in tickers]))
