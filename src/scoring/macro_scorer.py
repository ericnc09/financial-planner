import structlog

from src.models.schemas import Direction, MacroRegime, MacroSnapshotSchema

logger = structlog.get_logger()


class MacroScorer:
    """Produces a regime classification and conviction modifier (0.5-1.5)."""

    def score(
        self, snapshot: MacroSnapshotSchema, direction: Direction
    ) -> tuple[MacroRegime, float]:
        regime = snapshot.regime or MacroRegime.TRANSITION

        if snapshot.regime_score is not None:
            # Continuous scaling when regime_score is available
            # regime_score 0=expansion, 1=recession
            regime_score = snapshot.regime_score
            if direction == Direction.BUY:
                # Scale from 1.2 (expansion) to 0.7 (recession)
                modifier = 1.2 - (regime_score * 0.5)
            else:
                # Scale from 0.8 (expansion) to 1.3 (recession)
                modifier = 0.8 + (regime_score * 0.5)
        elif regime == MacroRegime.EXPANSION:
            modifier = 1.2 if direction == Direction.BUY else 0.8
        elif regime == MacroRegime.RECESSION:
            modifier = 0.7 if direction == Direction.BUY else 1.3
        else:  # TRANSITION
            modifier = 1.0

        modifier = round(max(0.5, min(1.5, modifier)), 3)

        logger.debug(
            "macro_scorer.result",
            regime=regime.value,
            direction=direction.value,
            modifier=modifier,
        )
        return regime, modifier
