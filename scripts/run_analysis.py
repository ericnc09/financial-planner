"""
Run multi-model quantitative analysis on a list of tickers.
Produces buy/hold/sell verdicts using the ensemble scoring engine.

Usage:
    python scripts/run_analysis.py
"""

import asyncio
import sys
import os
import json
from datetime import datetime

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


# ── Tickers ──────────────────────────────────────────────────────────────
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


async def main():
    print("=" * 80)
    print("MULTI-MODEL STOCK ANALYSIS")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    print()

    yahoo = YahooClient()
    ff_client = FamaFrenchClient()
    mc = MonteCarloSimulator(n_simulations=10_000, seed=42)
    hmm = HMMRegimeDetector()
    garch_model = GARCHForecaster()
    copula = CopulaTailRisk()
    ensemble = EnsembleScorer()

    # Pre-load Fama-French factors
    print("Loading Fama-French factors...")
    ff_factors = await ff_client.get_factors(days=504)
    if ff_factors is not None:
        print(f"  Loaded {len(ff_factors)} days of factor data")
    else:
        print("  WARNING: Could not load FF factors — FF and Copula models will be skipped")
    print()

    # Run analysis on all tickers
    results = []
    for ticker in ALL_TICKERS:
        print(f"Analyzing {ticker} ({TICKER_NAMES.get(ticker, '')})...", end=" ", flush=True)
        try:
            result = await analyze_ticker(
                ticker, yahoo, ff_client, ff_factors,
                mc, hmm, garch_model, copula, ensemble,
            )
            if result:
                results.append(result)
                print(f"Score: Buy={result['buy_score']:.0f} Sell={result['sell_score']:.0f} -> {result['verdict']}")
            else:
                print("NO DATA")
        except Exception as e:
            print(f"ERROR: {e}")

    # ── Summary Table ────────────────────────────────────────────────────
    print()
    print("=" * 120)
    print("RESULTS SUMMARY")
    print("=" * 120)
    print()

    # Canadian Banks
    print("-- CANADIAN BANKS " + "-" * 102)
    print(f"{'Ticker':<8} {'Name':<24} {'Price':>10} {'30d Mom':>10} {'90d Mom':>10} {'52w DD':>10} {'Buy':>8} {'Sell':>8} {'Models':>8} {'VERDICT':>10}")
    print("-" * 120)
    for r in results:
        if r["ticker"] in CANADIAN_BANKS:
            mom30 = f"{r['momentum_30d']:+.1f}%" if r['momentum_30d'] is not None else "N/A"
            mom90 = f"{r['momentum_90d']:+.1f}%" if r['momentum_90d'] is not None else "N/A"
            dd = f"-{r['drawdown_52w']:.1f}%" if r['drawdown_52w'] is not None else "N/A"
            print(f"{r['ticker']:<8} {r.get('name', r['ticker']):<24} {r['price']:>10.2f} {mom30:>10} {mom90:>10} {dd:>10} {r['buy_score']:>7.1f} {r['sell_score']:>7.1f} {r['n_models']:>6}/5  {r['verdict']:>10}")

    print()

    # Oil & Gas
    print("-- OIL & GAS " + "-" * 107)
    print(f"{'Ticker':<8} {'Name':<24} {'Price':>10} {'30d Mom':>10} {'90d Mom':>10} {'52w DD':>10} {'Buy':>8} {'Sell':>8} {'Models':>8} {'VERDICT':>10}")
    print("-" * 120)
    for r in results:
        if r["ticker"] in OIL_GAS:
            mom30 = f"{r['momentum_30d']:+.1f}%" if r['momentum_30d'] is not None else "N/A"
            mom90 = f"{r['momentum_90d']:+.1f}%" if r['momentum_90d'] is not None else "N/A"
            dd = f"-{r['drawdown_52w']:.1f}%" if r['drawdown_52w'] is not None else "N/A"
            print(f"{r['ticker']:<8} {r.get('name', r['ticker']):<24} {r['price']:>10.2f} {mom30:>10} {mom90:>10} {dd:>10} {r['buy_score']:>7.1f} {r['sell_score']:>7.1f} {r['n_models']:>6}/5  {r['verdict']:>10}")

    # ── Detailed Model Breakdown ─────────────────────────────────────────
    print()
    print("=" * 120)
    print("DETAILED MODEL BREAKDOWN")
    print("=" * 120)

    for r in results:
        print(f"\n{'-' * 80}")
        print(f"  {r['ticker']} -- {r.get('name', r['ticker'])}  |  Price: ${r['price']}  |  VERDICT: {r['verdict']}")
        print(f"{'-' * 80}")

        # Monte Carlo
        mc_data = r["models"].get("monte_carlo")
        if mc_data:
            print(f"  Monte Carlo (10k sims):")
            print(f"    Annual Drift: {mc_data['annual_drift']*100:.1f}%  |  Annual Vol: {mc_data['annual_volatility']*100:.1f}%")
            for h, hdata in mc_data["horizons"].items():
                pcts = hdata["percentiles"]
                print(f"    {h}d horizon: P(profit)={hdata['probability_of_profit']*100:.0f}%  "
                      f"E[ret]={hdata['expected_return']*100:+.1f}%  "
                      f"VaR95={hdata['value_at_risk_95']*100:.1f}%  "
                      f"Range=[${pcts['p10']:.0f} - ${pcts['p90']:.0f}]")

        # HMM
        hmm_data = r["models"].get("hmm")
        if hmm_data:
            print(f"  HMM Regime ({hmm_data['n_states_selected']} states):")
            print(f"    Current: {hmm_data['current_state'].upper()}  "
                  f"|  P(bull)={hmm_data.get('prob_bull',0)*100:.0f}%  "
                  f"P(bear)={hmm_data.get('prob_bear',0)*100:.0f}%  "
                  f"P(sideways)={hmm_data.get('prob_sideways',0)*100:.0f}%")

        # GARCH
        garch_data = r["models"].get("garch")
        if garch_data:
            print(f"  GARCH(1,1):")
            print(f"    Current Vol: {garch_data['current_conditional_vol_annual']*100:.1f}%  "
                  f"|  Long-Run: {garch_data['long_run_vol_annual']*100:.1f}%  "
                  f"|  Persistence: {garch_data['parameters']['persistence']:.3f}")
            for h, fdata in garch_data["forecasts"].items():
                print(f"    {h}d forecast: {fdata['predicted_volatility_annual']*100:.1f}% ({fdata['interpretation']})")

        # Fama-French
        ff_data = r["models"].get("fama_french")
        if ff_data:
            print(f"  Fama-French 5-Factor:")
            print(f"    Alpha: {ff_data['alpha_annual']*100:.2f}%/yr  "
                  f"|  Beta: {ff_data['beta_market']:.2f}  "
                  f"|  R2: {ff_data['r_squared']:.2f}")
            print(f"    SMB={ff_data['beta_smb']:+.2f}  HML={ff_data['beta_hml']:+.2f}  "
                  f"RMW={ff_data['beta_rmw']:+.2f}  CMA={ff_data['beta_cma']:+.2f}")

        # Copula
        cop_data = r["models"].get("copula")
        if cop_data:
            print(f"  Copula Tail Risk:")
            print(f"    Tail Dep: {cop_data['tail_dep_lower']:.3f}  "
                  f"|  Joint Crash: {cop_data['joint_crash_prob']*100:.2f}%  "
                  f"|  VaR95: {cop_data['var_95']*100:.2f}%  "
                  f"|  CVaR95: {cop_data['cvar_95']*100:.2f}%  "
                  f"|  Score: {cop_data['tail_risk_score']:.0f}/100")

        # Component scores
        print(f"  Ensemble Components (Buy): {r['buy_components']}")
        print(f"  Ensemble Components (Sell): {r['sell_components']}")

    # ── Save JSON ────────────────────────────────────────────────────────
    output_file = os.path.join(os.path.dirname(__file__), "analysis_results.json")
    serializable = []
    for r in results:
        entry = {k: v for k, v in r.items() if k != "models"}
        serializable.append(entry)
    with open(output_file, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
