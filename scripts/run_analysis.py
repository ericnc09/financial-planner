"""
Run multi-model quantitative analysis on a list of tickers.
Produces buy/hold/sell verdicts, sector momentum, and risk-parity position sizing.

Usage:
    python scripts/run_analysis.py [TICKER1 TICKER2 ...]
"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Settings
from src.clients.yahoo import YahooClient
from src.clients.fama_french import FamaFrenchClient
from src.analysis.monte_carlo import MonteCarloSimulator
from src.analysis.hmm_regime import HMMRegimeDetector
from src.analysis.garch_forecast import GARCHForecaster
from src.analysis.copula_tail_risk import CopulaTailRisk
from src.analysis.ensemble_scoring import EnsembleScorer
from src.analysis.ticker_analysis import analyze_ticker
from src.analysis.sector_momentum import compute_sector_momentum, format_sector_report
from src.analysis.position_sizing import compute_position_sizes, format_sizing_report


CANADIAN_BANKS = ["RY", "CM", "TD", "BNS"]
OIL_GAS = ["E", "VLO", "EPD", "ET", "SUN", "SLB", "BKR", "FLS"]
ALL_TICKERS = CANADIAN_BANKS + OIL_GAS

TICKER_NAMES = {
    "RY": "Royal Bank of Canada",
    "CM": "CIBC",
    "TD": "TD Bank",
    "BNS": "Bank of Nova Scotia",
    "E": "Eni S.p.A.",
    "VLO": "Valero Energy",
    "EPD": "Enterprise Products",
    "ET": "Energy Transfer",
    "SUN": "Sunoco LP",
    "SLB": "Schlumberger",
    "BKR": "Baker Hughes",
    "FLS": "Flowserve",
}

SECTOR_MAP = {
    "RY": "Financials", "CM": "Financials", "TD": "Financials", "BNS": "Financials",
    "E": "Energy", "VLO": "Energy", "EPD": "Energy", "ET": "Energy",
    "SUN": "Energy", "SLB": "Energy", "BKR": "Energy", "FLS": "Industrials",
}


async def main():
    tickers = [t.upper() for t in sys.argv[1:]] if len(sys.argv) > 1 else ALL_TICKERS

    print("=" * 80)
    print("MULTI-MODEL STOCK ANALYSIS  (C1-C6 + D1-D4)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    print()

    settings = Settings()
    yahoo = YahooClient()
    ff_client = FamaFrenchClient()
    mc = MonteCarloSimulator(n_simulations=10_000, seed=settings.random_seed)
    hmm = HMMRegimeDetector()
    garch_model = GARCHForecaster()
    copula = CopulaTailRisk()
    ensemble = EnsembleScorer()

    print("Loading Fama-French factors...")
    ff_factors = await ff_client.get_factors(days=504)
    print(f"  {'OK' if ff_factors is not None else 'FAILED'}\n")

    results = []
    for ticker in tickers:
        print(f"  {ticker} ({TICKER_NAMES.get(ticker, '')})...", end=" ", flush=True)
        try:
            result = await asyncio.wait_for(
                analyze_ticker(ticker, yahoo, ff_client, ff_factors,
                               mc, hmm, garch_model, copula, ensemble),
                timeout=120,
            )
            if result:
                # Inject sector for sector momentum
                result["sector"] = SECTOR_MAP.get(ticker, "Unknown")
                results.append(result)
                regime = result.get("hmm_regime") or "?"
                earnings_score = (result["models"].get("earnings") or {}).get("score", "N/A")
                bd_hl = None
                bd = result["models"].get("bayesian_decay")
                if bd:
                    bd_hl = bd.get("posterior_half_life")
                print(
                    f"{result['verdict']:4}  "
                    f"buy={result['buy_score']:.0f} sell={result['sell_score']:.0f}  "
                    f"regime={regime}  "
                    f"earnings={earnings_score}  "
                    f"decay_hl={bd_hl or 'N/A'}"
                )
            else:
                print("NO DATA")
        except asyncio.TimeoutError:
            print("TIMEOUT")
        except Exception as e:
            print(f"ERROR: {e}")

    if not results:
        print("\nNo results.")
        return

    # ── Summary Table ──────────────────────────────────────────────────────
    print()
    print("=" * 120)
    print("RESULTS SUMMARY")
    print("=" * 120)
    _print_table(results, CANADIAN_BANKS, "CANADIAN BANKS")
    print()
    _print_table(results, OIL_GAS, "OIL & GAS")

    # ── Sector Momentum (C4) ───────────────────────────────────────────────
    print()
    print("=" * 120)
    sector_scores = compute_sector_momentum(results)
    print(format_sector_report(sector_scores))

    # ── Position Sizing (C5) ───────────────────────────────────────────────
    print()
    print("=" * 120)
    positions = compute_position_sizes(results, portfolio_value=100_000)
    print(format_sizing_report(positions, portfolio_value=100_000))

    # ── Save JSON ──────────────────────────────────────────────────────────
    output_file = os.path.join(os.path.dirname(__file__), "analysis_results.json")
    serializable = []
    for r in results:
        entry = {k: v for k, v in r.items() if k != "models"}
        serializable.append(entry)
    with open(output_file, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")


def _print_table(results, group, label):
    print(f"-- {label} " + "-" * (116 - len(label)))
    print(f"{'Ticker':<8} {'Name':<24} {'Price':>9} {'Regime':<12} {'Earnings':>10} "
          f"{'30d':>8} {'90d':>8} {'Buy':>7} {'Sell':>7} {'N':>4}  {'VERDICT':>8}")
    print("-" * 120)
    for r in results:
        if r["ticker"] not in group:
            continue
        mom30 = f"{r['momentum_30d']:+.1f}%" if r["momentum_30d"] is not None else "N/A"
        mom90 = f"{r['momentum_90d']:+.1f}%" if r["momentum_90d"] is not None else "N/A"
        regime = r.get("hmm_regime") or "?"
        e_score = (r["models"].get("earnings") or {}).get("score", "--") if isinstance(r.get("models"), dict) else "--"
        print(
            f"{r['ticker']:<8} {TICKER_NAMES.get(r['ticker'], r['ticker']):<24} "
            f"{r['price']:>9.2f} {str(regime):<12} {str(e_score):>10} "
            f"{mom30:>8} {mom90:>8} "
            f"{r['buy_score']:>7.1f} {r['sell_score']:>7.1f} "
            f"{r['n_models']:>3}/7  {r['verdict']:>8}"
        )


if __name__ == "__main__":
    asyncio.run(main())
