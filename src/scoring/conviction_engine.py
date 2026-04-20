import json
from pathlib import Path

import structlog

from config.settings import Settings
from src.models.schemas import (
    ConvictionResult,
    Direction,
    EnrichmentSchema,
    MacroSnapshotSchema,
    SmartMoneyEventSchema,
)
from src.scoring.fundamental_scorer import FundamentalScorer
from src.scoring.macro_scorer import MacroScorer
from src.scoring.signal_scorer import SignalScorer

logger = structlog.get_logger()

_RISK_PROFILES_PATH = Path(__file__).resolve().parents[2] / "config" / "risk_profiles.json"


def _load_risk_profile(name: str) -> dict | None:
    """Load a risk profile from config/risk_profiles.json. None on miss."""
    try:
        with open(_RISK_PROFILES_PATH) as f:
            profiles = json.load(f)
        return profiles.get(name)
    except Exception:
        return None


class ConvictionEngine:
    """Combines signal, fundamental, and macro scores into a single conviction.

    Supports adaptive thresholding: the conviction threshold adjusts based
    on the current macro regime and volatility environment.
    """

    # Adaptive threshold adjustments by regime
    REGIME_THRESHOLD_ADJUSTMENTS = {
        "recession": 0.10,    # raise threshold in recession (fewer buys)
        "contraction": 0.08,
        "transition": 0.03,
        "expansion": -0.05,   # relax threshold in expansion
    }

    def __init__(
        self,
        settings: Settings,
        signal_scorer: SignalScorer,
        fundamental_scorer: FundamentalScorer,
        macro_scorer: MacroScorer,
    ):
        self.settings = settings
        self.signal_scorer = signal_scorer
        self.fundamental_scorer = fundamental_scorer
        self.macro_scorer = macro_scorer

    def compute(
        self,
        event: SmartMoneyEventSchema,
        enrichment: EnrichmentSchema,
        macro_snapshot: MacroSnapshotSchema,
        recent_events: list[SmartMoneyEventSchema],
        risk_profile: str | None = None,
    ) -> ConvictionResult:
        # Score each layer
        signal_score = self.signal_scorer.score(event, recent_events)
        fundamental_score = self.fundamental_scorer.score(
            enrichment, event.direction
        )
        regime, macro_modifier = self.macro_scorer.score(
            macro_snapshot, event.direction
        )

        # Direction boost
        direction_boost = self._direction_boost(enrichment, event.direction)

        # Weighted base (before macro)
        w_sig = self.settings.signal_weight
        w_fund = self.settings.fundamental_weight
        base = (signal_score * w_sig + fundamental_score * w_fund) / (
            w_sig + w_fund
        )

        # Apply macro modifier and direction boost
        conviction = base * macro_modifier * direction_boost
        conviction = round(max(0.0, min(1.0, conviction)), 4)

        # Adaptive threshold: adjust based on regime and volatility
        threshold = self._adaptive_threshold(
            regime, enrichment, event.direction, risk_profile
        )
        passes = conviction >= threshold

        logger.info(
            "conviction_engine.scored",
            ticker=event.ticker,
            actor=event.actor,
            direction=event.direction.value,
            signal=signal_score,
            fundamental=fundamental_score,
            macro_mod=macro_modifier,
            dir_boost=direction_boost,
            conviction=conviction,
            passes=passes,
        )

        return ConvictionResult(
            signal_score=signal_score,
            fundamental_score=fundamental_score,
            macro_modifier=macro_modifier,
            direction_boost=direction_boost,
            conviction=conviction,
            passes_threshold=passes,
        )

    def _adaptive_threshold(
        self,
        regime: str,
        enrichment: EnrichmentSchema,
        direction: Direction,
        risk_profile: str | None = None,
    ) -> float:
        """
        Compute regime- and volatility-adjusted conviction threshold.

        In bear/high-vol regimes, raise the bar for buy signals (require
        stronger conviction). In calm expansions, relax slightly.
        For sell signals in bear markets, we relax (easier to confirm).

        When risk_profile is provided (conservative / moderate / aggressive),
        its min_conviction overrides settings.conviction_threshold as the base.
        """
        base = self.settings.conviction_threshold
        if risk_profile:
            profile = _load_risk_profile(risk_profile)
            if profile and "min_conviction" in profile:
                base = float(profile["min_conviction"])

        # Regime adjustment
        regime_adj = self.REGIME_THRESHOLD_ADJUSTMENTS.get(
            regime.lower() if regime else "transition", 0.0
        )

        # For sell signals, invert the regime adjustment
        # (it's easier to sell in recessions, harder in expansions)
        if direction == Direction.SELL:
            regime_adj = -regime_adj

        # Volatility adjustment: high RSI extremes → raise threshold
        rsi = enrichment.rsi_14d
        vol_adj = 0.0
        if rsi is not None:
            if rsi < 20 or rsi > 80:
                vol_adj = 0.05  # extreme territory, be cautious
            elif rsi < 30 or rsi > 70:
                vol_adj = 0.02  # moderate caution

        threshold = base + regime_adj + vol_adj
        # Clamp to reasonable range
        threshold = max(0.45, min(0.80, threshold))

        if threshold != base:
            logger.debug(
                "conviction_engine.adaptive_threshold",
                base=base,
                regime=regime,
                regime_adj=regime_adj,
                vol_adj=vol_adj,
                final=round(threshold, 4),
            )

        return round(threshold, 4)

    def _direction_boost(
        self, enrichment: EnrichmentSchema, direction: Direction
    ) -> float:
        """
        BUY + significant drawdown → 1.1 (dip-buy confirmation)
        SELL + overbought RSI → 1.1 (overbought sell confirmation)
        Otherwise → 1.0
        """
        if direction == Direction.BUY:
            dd = enrichment.drawdown_from_52w_high
            if dd is not None and dd >= 0.15:
                return 1.1
        else:
            rsi = enrichment.rsi_14d
            if rsi is not None and rsi >= 70:
                return 1.1
        return 1.0
