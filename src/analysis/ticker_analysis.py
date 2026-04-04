"""
Shared ticker analysis — runs all quantitative models on a single ticker.

Used by both scripts/run_analysis.py and the Slack bot.
"""

import asyncio

import numpy as np
import structlog

from src.clients.yahoo import YahooClient
from src.clients.fama_french import FamaFrenchClient
from src.analysis.monte_carlo import MonteCarloSimulator
from src.analysis.hmm_regime import HMMRegimeDetector
from src.analysis.garch_forecast import GARCHForecaster
from src.analysis.copula_tail_risk import CopulaTailRisk
from src.analysis.ensemble_scoring import EnsembleScorer

logger = structlog.get_logger()


async def analyze_ticker(
    ticker: str,
    yahoo: YahooClient,
    ff_client: FamaFrenchClient,
    ff_factors,
    mc: MonteCarloSimulator,
    hmm: HMMRegimeDetector,
    garch: GARCHForecaster,
    copula: CopulaTailRisk,
    ensemble: EnsembleScorer,
) -> dict | None:
    """Run all models on a single ticker and return results with verdict."""
    price_data = await yahoo.get_price_history(ticker, days=504)
    if price_data is None:
        logger.warning("ticker_analysis.no_data", ticker=ticker)
        return None

    closes = price_data["closes"]
    returns = price_data["returns"]
    volumes = price_data["volumes"]
    current_price = float(closes[-1])

    # Run models concurrently
    mc_result = mc.simulate(closes, horizons=[21, 63, 126])
    hmm_task = hmm.fit_and_predict(returns, volumes)
    garch_task = garch.forecast(returns, horizons=[5, 20])
    hmm_result, garch_result = await asyncio.gather(hmm_task, garch_task)

    # Fama-French
    ff_result = None
    if ff_factors is not None:
        ff_result = ff_client.compute_factor_exposure(returns, ff_factors)

    # Copula (needs market returns from FF factors)
    copula_result = None
    if ff_factors is not None and "Mkt-RF" in ff_factors.columns:
        mkt_returns = ff_factors["Mkt-RF"].values
        n = min(len(returns), len(mkt_returns))
        if n >= 60:
            copula_result = copula.analyze(returns[-n:], mkt_returns[-n:])

    # Flatten GARCH ratios for ensemble scorer
    garch_for_ensemble = None
    if garch_result:
        garch_for_ensemble = dict(garch_result)
        forecasts = garch_result.get("forecasts", {})
        garch_for_ensemble["forecast_5d_ratio"] = forecasts.get(5, {}).get("volatility_ratio", 1.0)
        garch_for_ensemble["forecast_20d_ratio"] = forecasts.get(20, {}).get("volatility_ratio", 1.0)

    # Score for both directions
    buy_score = ensemble.score(
        direction="buy",
        monte_carlo=mc_result,
        hmm=hmm_result,
        garch=garch_for_ensemble,
        fama_french=ff_result,
        copula=copula_result,
    )
    sell_score = ensemble.score(
        direction="sell",
        monte_carlo=mc_result,
        hmm=hmm_result,
        garch=garch_for_ensemble,
        fama_french=ff_result,
        copula=copula_result,
    )

    # Determine verdict
    buy_total = buy_score["total_score"]
    sell_total = sell_score["total_score"]

    if buy_total >= 60 and buy_total > sell_total + 10:
        verdict = "BUY"
    elif sell_total >= 60 and sell_total > buy_total + 10:
        verdict = "SELL"
    else:
        verdict = "HOLD"

    # Key metrics
    momentum_30d = None
    if len(closes) >= 22:
        momentum_30d = (closes[-1] - closes[-22]) / closes[-22]
    momentum_90d = None
    if len(closes) >= 63:
        momentum_90d = (closes[-1] - closes[-63]) / closes[-63]

    drawdown_52w = None
    if len(closes) > 0:
        high_52w = float(np.max(closes[-252:] if len(closes) >= 252 else closes))
        drawdown_52w = (high_52w - current_price) / high_52w

    return {
        "ticker": ticker,
        "price": round(current_price, 2),
        "momentum_30d": round(momentum_30d * 100, 1) if momentum_30d is not None else None,
        "momentum_90d": round(momentum_90d * 100, 1) if momentum_90d is not None else None,
        "drawdown_52w": round(drawdown_52w * 100, 1) if drawdown_52w is not None else None,
        "buy_score": buy_total,
        "sell_score": sell_total,
        "buy_rec": buy_score["recommendation"],
        "sell_rec": sell_score["recommendation"],
        "verdict": verdict,
        "n_models": buy_score["n_models"],
        "confidence": buy_score["confidence"],
        "models": {
            "monte_carlo": mc_result,
            "hmm": hmm_result,
            "garch": garch_result,
            "fama_french": ff_result,
            "copula": copula_result,
        },
        "buy_components": buy_score.get("components", {}),
        "sell_components": sell_score.get("components", {}),
    }
