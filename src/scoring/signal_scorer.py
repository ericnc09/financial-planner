from datetime import datetime, timedelta

import structlog

from src.models.schemas import Direction, SmartMoneyEventSchema, SourceType

logger = structlog.get_logger()


class SignalScorer:
    """Scores the smart money signal itself (0-1). Five weighted sub-factors."""

    WEIGHTS = {
        "actor_reputation": 0.25,
        "trade_size": 0.20,
        "cluster_signal": 0.25,
        "disclosure_timing": 0.15,
        "consensus": 0.15,
    }

    def score(
        self,
        event: SmartMoneyEventSchema,
        recent_events: list[SmartMoneyEventSchema],
    ) -> float:
        scores = {
            "actor_reputation": self._actor_reputation(event.actor, event.source_type),
            "trade_size": self._trade_size(
                event.size_lower, event.size_upper, event.source_type
            ),
            "cluster_signal": self._cluster_signal(
                event.ticker, event.trade_date, recent_events
            ),
            "disclosure_timing": self._disclosure_timing(
                event.trade_date, event.disclosure_date
            ),
            "consensus": self._consensus(
                event.ticker, event.direction, recent_events
            ),
        }
        composite = sum(
            scores[k] * self.WEIGHTS[k] for k in self.WEIGHTS
        )
        return round(min(1.0, max(0.0, composite)), 4)

    def _actor_reputation(self, actor: str, source_type: SourceType) -> float:
        """
        Base score by source type. Congressional gets a slight edge
        (informational advantage is the thesis). C-suite insiders
        (CEO, CFO, COO) get a boost. Default 0.5 for unknowns.
        """
        base = 0.6 if source_type == SourceType.CONGRESSIONAL else 0.5
        actor_lower = actor.lower()
        # Boost for high-ranking insiders
        if any(
            title in actor_lower
            for title in ["ceo", "cfo", "coo", "chairman", "director", "president"]
        ):
            base = min(1.0, base + 0.2)
        # Boost for committee chairs / leadership
        if any(
            kw in actor_lower
            for kw in ["speaker", "leader", "chair", "whip"]
        ):
            base = min(1.0, base + 0.15)
        return base

    def _trade_size(
        self,
        size_lower: float | None,
        size_upper: float | None,
        source_type: SourceType,
    ) -> float:
        if size_lower is None:
            return 0.3  # unknown size = low score
        midpoint = (
            (size_lower + size_upper) / 2 if size_upper else size_lower
        )
        if source_type == SourceType.CONGRESSIONAL:
            if midpoint >= 500_000:
                return 1.0
            elif midpoint >= 100_000:
                return 0.8
            elif midpoint >= 50_000:
                return 0.6
            else:
                return 0.3
        else:  # insider
            if midpoint >= 1_000_000:
                return 1.0
            elif midpoint >= 500_000:
                return 0.8
            elif midpoint >= 100_000:
                return 0.6
            elif midpoint >= 50_000:
                return 0.4
            else:
                return 0.2

    def _cluster_signal(
        self,
        ticker: str,
        trade_date: datetime,
        recent_events: list[SmartMoneyEventSchema],
    ) -> float:
        """Count other actors trading same ticker within +/- 7 days."""
        window = timedelta(days=7)
        actors = set()
        for e in recent_events:
            if (
                e.ticker == ticker
                and abs((e.trade_date - trade_date).days) <= window.days
            ):
                actors.add(e.actor)
        # Remove the current actor
        unique_others = len(actors) - 1
        if unique_others >= 3:
            return 1.0
        elif unique_others == 2:
            return 0.7
        elif unique_others == 1:
            return 0.4
        return 0.2

    def _disclosure_timing(
        self, trade_date: datetime, disclosure_date: datetime | None
    ) -> float:
        if disclosure_date is None:
            return 0.5  # unknown = neutral
        lag_days = (disclosure_date - trade_date).days
        if lag_days < 0:
            lag_days = 0
        if lag_days <= 7:
            return 1.0
        elif lag_days <= 15:
            return 0.8
        elif lag_days <= 30:
            return 0.5
        elif lag_days <= 45:
            return 0.3
        return 0.1

    def _consensus(
        self,
        ticker: str,
        direction: Direction,
        recent_events: list[SmartMoneyEventSchema],
    ) -> float:
        """Fraction of recent events for this ticker that agree on direction."""
        ticker_events = [e for e in recent_events if e.ticker == ticker]
        if not ticker_events:
            return 0.5
        same_dir = sum(1 for e in ticker_events if e.direction == direction)
        ratio = same_dir / len(ticker_events)
        if ratio >= 0.8:
            return 1.0
        elif ratio >= 0.6:
            return 0.7
        return 0.4
