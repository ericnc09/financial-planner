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


class ConvictionEngine:
    """Combines signal, fundamental, and macro scores into a single conviction."""

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
    ) -> ConvictionResult:
        # Score each layer
        signal_score = self.signal_scorer.score(event, recent_events)
        fundamental_score = self.fundamental_scorer.score(
            enrichment, event.direction
        )
        _regime, macro_modifier = self.macro_scorer.score(
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

        passes = conviction >= self.settings.conviction_threshold

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
