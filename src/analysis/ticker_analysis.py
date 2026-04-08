"""
Shared ticker analysis — runs all quantitative models on a single ticker.

Used by both scripts/run_analysis.py and the Slack bot.

Models run per ticker:
  1. Monte Carlo (GBM, 10k sims, fat tails)
  2. HMM Regime Detection (BIC-selected, 2-5 states)
  3. GARCH(1,1) volatility forecast
  4. Fama-French 5-factor regression
  5. Copula tail risk (Student-t)
  6. Earnings Surprise Overlay (C2)
  7. XGBoost signal classifier predict-only (C1, if model trained)
  8. Bayesian Decay, regime-conditional (C6, uses HMM state as prior)
"""

import asyncio
from datetime import datetime

import numpy as np
import structlog

from src.clients.yahoo import YahooClient
from src.clients.fama_french import FamaFrenchClient
from src.analysis.monte_carlo import MonteCarloSimulator
from src.analysis.hmm_regime import HMMRegimeDetector
from src.analysis.garch_forecast import GARCHForecaster
from src.analysis.copula_tail_risk import CopulaTailRisk
from src.analysis.ensemble_scoring import EnsembleScorer
from src.analysis.earnings_overlay import EarningsOverlay
from src.analysis.bayesian_decay import BayesianDecayAnalyzer
from src.analysis.xgboost_classifier import XGBoostSignalClassifier

logger = structlog.get_logger()

# Load XGBoost model once at module import (returns untrained instance if no file)
_xgb_classifier = XGBoostSignalClassifier.load()


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

    # Run compute-heavy models concurrently
    hmm_task = hmm.fit_and_predict(returns, volumes)
    garch_task = garch.forecast(returns, horizons=[5, 20])
    earnings_task = _earnings_overlay().analyze(ticker, datetime.now(), direction="buy")

    mc_result = mc.simulate(closes, horizons=[21, 63, 126])
    hmm_result, garch_result, earnings_result = await asyncio.gather(
        hmm_task, garch_task, earnings_task
    )

    # Fama-French
    ff_result = None
    if ff_factors is not None:
        ff_result = ff_client.compute_factor_exposure(returns, ff_factors)

    # Copula (needs market returns from FF Mkt-RF)
    copula_result = None
    if ff_factors is not None and "Mkt-RF" in ff_factors.columns:
        mkt_returns = ff_factors["Mkt-RF"].values
        n = min(len(returns), len(mkt_returns))
        if n >= 60:
            copula_result = copula.analyze(returns[-n:], mkt_returns[-n:])

    # C6: Bayesian decay with regime-conditional prior from HMM state
    bayesian_result = None
    hmm_regime_state = None
    if hmm_result:
        hmm_regime_state = hmm_result.get("current_state")
    # Use recent 40-day abnormal returns (proxy: excess over mean)
    if len(returns) >= 20:
        mean_ret = float(np.mean(returns[-252:] if len(returns) >= 252 else returns))
        abnormal = returns[-40:] - mean_ret
        bd = BayesianDecayAnalyzer(regime=hmm_regime_state)
        bayesian_result = bd.analyze(abnormal, direction="buy")

    # Flatten GARCH ratios for ensemble scorer
    garch_for_ensemble = None
    if garch_result:
        garch_for_ensemble = dict(garch_result)
        forecasts = garch_result.get("forecasts", {})
        garch_for_ensemble["forecast_5d_ratio"] = forecasts.get(5, {}).get("volatility_ratio", 1.0)
        garch_for_ensemble["forecast_20d_ratio"] = forecasts.get(20, {}).get("volatility_ratio", 1.0)

    # C1: XGBoost predict-only (uses model outputs as features)
    xgb_result = None
    if _xgb_classifier.is_trained:
        xgb_features = _build_xgb_features(
            hmm_result, garch_result, ff_result, copula_result, closes, returns,
            direction="buy",
        )
        xgb_raw = _xgb_classifier.predict(xgb_features)
        if xgb_raw:
            # Convert probability_profitable to a 0-100 score
            prob = xgb_raw["probability_profitable"]
            xgb_result = {
                "score": round(prob * 100, 1),
                "predicted_class": xgb_raw["predicted_class"],
                "confidence": xgb_raw["confidence"],
            }

    # Score for both directions
    buy_score = ensemble.score(
        direction="buy",
        monte_carlo=mc_result,
        hmm=hmm_result,
        garch=garch_for_ensemble,
        fama_french=ff_result,
        copula=copula_result,
        earnings_overlay=earnings_result,
        bayesian_decay=bayesian_result,
    )
    sell_score = ensemble.score(
        direction="sell",
        monte_carlo=mc_result,
        hmm=hmm_result,
        garch=garch_for_ensemble,
        fama_french=ff_result,
        copula=copula_result,
        earnings_overlay=earnings_result,
        bayesian_decay=bayesian_result,
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
        "hmm_regime": hmm_regime_state,
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
            "earnings": earnings_result,
            "bayesian_decay": bayesian_result,
            "xgboost": xgb_result,
        },
        "buy_components": buy_score.get("components", {}),
        "sell_components": sell_score.get("components", {}),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _earnings_overlay() -> EarningsOverlay:
    """Lazy singleton for EarningsOverlay (no init cost)."""
    return EarningsOverlay(lookback_days=30, lookforward_days=30)


def _build_xgb_features(
    hmm_result, garch_result, ff_result, copula_result,
    closes, returns, direction: str,
) -> dict:
    """Build XGBoost feature dict from model outputs + price data."""
    features: dict = {}

    # HMM features
    if hmm_result:
        features["hmm_bull_prob"] = hmm_result.get("prob_bull", 0.0) or 0.0
        features["hmm_bear_prob"] = hmm_result.get("prob_bear", 0.0) or 0.0

    # GARCH
    if garch_result:
        features["garch_current_vol"] = garch_result.get("current_conditional_vol_annual", 0.25)

    # Fama-French alpha
    if ff_result:
        features["ff_alpha"] = ff_result.get("alpha_annual", 0.0) or 0.0

    # Copula tail score
    if copula_result:
        features["copula_tail_score"] = copula_result.get("tail_risk_score", 50.0)

    # Price-derived
    if len(closes) >= 22:
        features["momentum_30d"] = float((closes[-1] - closes[-22]) / closes[-22])
    if len(closes) >= 63:
        features["momentum_90d"] = float((closes[-1] - closes[-63]) / closes[-63])

    if len(closes) >= 252:
        high_52w = float(np.max(closes[-252:]))
        features["drawdown_from_52w_high"] = (high_52w - float(closes[-1])) / high_52w

    if len(closes) >= 15:
        features["rsi_14d"] = float(_compute_rsi(closes))

    if len(returns) >= 20:
        features["avg_volume_30d"] = 0.0  # volume not in returns; left neutral

    features["direction_encoded"] = 1.0 if direction == "buy" else -1.0

    return features


def _compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    recent = deltas[-period:]
    gains = np.where(recent > 0, recent, 0.0)
    losses = np.where(recent < 0, -recent, 0.0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 2)
