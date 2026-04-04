"""
Analysis runner — wraps the shared analyze_ticker function with client setup.
"""

import asyncio

import structlog

from src.clients.yahoo import YahooClient
from src.clients.fama_french import FamaFrenchClient
from src.analysis.monte_carlo import MonteCarloSimulator
from src.analysis.hmm_regime import HMMRegimeDetector
from src.analysis.garch_forecast import GARCHForecaster
from src.analysis.copula_tail_risk import CopulaTailRisk
from src.analysis.ensemble_scoring import EnsembleScorer
from src.analysis.ticker_analysis import analyze_ticker

logger = structlog.get_logger()


async def run_analysis(tickers: list[str]) -> tuple[list[dict], list[str]]:
    """
    Run all 5 models on a list of tickers.

    Returns:
        (results, failed) — list of successful result dicts and list of failed ticker strings.
    """
    yahoo = YahooClient()
    ff_client = FamaFrenchClient()
    mc = MonteCarloSimulator(n_simulations=10_000, seed=42)
    hmm = HMMRegimeDetector()
    garch = GARCHForecaster()
    copula = CopulaTailRisk()
    ensemble = EnsembleScorer()

    # Pre-load Fama-French factors once
    ff_factors = await ff_client.get_factors(days=504)

    results = []
    failed = []

    for ticker in tickers:
        try:
            result = await asyncio.wait_for(
                analyze_ticker(
                    ticker, yahoo, ff_client, ff_factors,
                    mc, hmm, garch, copula, ensemble,
                ),
                timeout=120,
            )
            if result:
                results.append(result)
            else:
                failed.append(ticker)
                logger.warning("runner.no_data", ticker=ticker)
        except asyncio.TimeoutError:
            failed.append(ticker)
            logger.warning("runner.timeout", ticker=ticker)
        except Exception as e:
            failed.append(ticker)
            logger.warning("runner.error", ticker=ticker, error=str(e))

    return results, failed
