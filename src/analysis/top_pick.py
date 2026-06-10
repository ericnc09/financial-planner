"""
Morning Top Pick — selects the single stock most probable for high
performance over the following days.

Ranks all BUY signals from the recent lookback window by a blend of:
  - conviction score        (signal x fundamental x macro pipeline)
  - ensemble score          (multi-model 0-100, direction=buy)
  - Monte Carlo P(profit)   (30-day horizon probability of profit)
  - cluster breadth bonus   (multiple distinct actors buying = stronger)

Tickers whose latest ensemble recommendation is "avoid" are excluded.
Designed to run after the morning pipeline cycle so it can always produce
a pick from the existing database, even when no new signals arrived today.
"""

from datetime import datetime, timedelta

import structlog
from sqlalchemy import desc

from src.models.database import (
    ConvictionScore,
    Direction,
    Enrichment,
    EnsembleScoreResult,
    MonteCarloResult,
    SmartMoneyEvent,
)

logger = structlog.get_logger()

# Blend weights for the composite rank score (must sum to 1.0)
W_CONVICTION = 0.40
W_ENSEMBLE = 0.40
W_MC_PROB = 0.20

# Bonus per additional distinct actor buying the same ticker (capped)
CLUSTER_BONUS_PER_ACTOR = 0.02
CLUSTER_BONUS_CAP = 0.06


def select_top_pick(session_factory, lookback_days: int = 7) -> dict | None:
    """
    Select today's single highest-probability BUY from recent signals.

    Args:
        session_factory: SQLAlchemy session factory.
        lookback_days: How many days of signals to consider.

    Returns:
        Dict describing the pick (ticker, score, components, context),
        or None when no qualifying candidate exists.
    """
    session = session_factory()
    try:
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        rows = (
            session.query(SmartMoneyEvent, ConvictionScore, Enrichment)
            .outerjoin(ConvictionScore, SmartMoneyEvent.id == ConvictionScore.event_id)
            .outerjoin(Enrichment, SmartMoneyEvent.id == Enrichment.event_id)
            .filter(SmartMoneyEvent.trade_date >= cutoff)
            .filter(SmartMoneyEvent.direction == Direction.BUY)
            .all()
        )
        if not rows:
            logger.info("top_pick.no_recent_buys", lookback_days=lookback_days)
            return None

        # Group events per ticker
        by_ticker: dict[str, dict] = {}
        for event, cs, enr in rows:
            t = by_ticker.setdefault(event.ticker, {
                "events": [], "actors": set(),
                "best_conviction": None, "sector": None, "price": None,
                "latest_event": None,
            })
            t["events"].append(event)
            t["actors"].add(event.actor)
            if cs and cs.conviction is not None:
                if t["best_conviction"] is None or cs.conviction > t["best_conviction"]:
                    t["best_conviction"] = cs.conviction
            if enr:
                t["sector"] = t["sector"] or enr.sector
                t["price"] = t["price"] or enr.price_at_signal
            if t["latest_event"] is None or event.trade_date > t["latest_event"].trade_date:
                t["latest_event"] = event

        candidates = []
        for ticker, info in by_ticker.items():
            # Latest BUY-direction ensemble score for this ticker
            ens = (
                session.query(EnsembleScoreResult)
                .filter(EnsembleScoreResult.ticker == ticker)
                .filter(EnsembleScoreResult.direction == "buy")
                .order_by(desc(EnsembleScoreResult.run_date))
                .first()
            )
            if ens and ens.recommendation == "avoid":
                continue

            # Latest Monte Carlo run for the 30d profit probability
            mc = (
                session.query(MonteCarloResult)
                .filter(MonteCarloResult.ticker == ticker)
                .order_by(desc(MonteCarloResult.run_date))
                .first()
            )

            conviction = info["best_conviction"]
            ensemble_score = ens.total_score if ens else None
            mc_prob = mc.h30_prob_profit if mc else None

            # Require at least one model-based component — never pick on
            # raw signal presence alone.
            if conviction is None and ensemble_score is None:
                continue

            # Blend available components, renormalizing weights for the
            # missing ones (same approach as the ensemble scorer).
            parts = []
            if conviction is not None:
                parts.append((W_CONVICTION, conviction))
            if ensemble_score is not None:
                parts.append((W_ENSEMBLE, ensemble_score / 100.0))
            if mc_prob is not None:
                parts.append((W_MC_PROB, mc_prob))
            total_w = sum(w for w, _ in parts)
            blended = sum(w * v for w, v in parts) / total_w

            cluster_bonus = min(
                CLUSTER_BONUS_CAP,
                CLUSTER_BONUS_PER_ACTOR * (len(info["actors"]) - 1),
            )
            final_score = blended + cluster_bonus

            latest = info["latest_event"]
            candidates.append({
                "ticker": ticker,
                "score": round(final_score, 4),
                "conviction": round(conviction, 4) if conviction is not None else None,
                "ensemble_score": round(ensemble_score, 1) if ensemble_score is not None else None,
                "ensemble_recommendation": ens.recommendation if ens else None,
                "mc_prob_profit_30d": round(mc_prob, 4) if mc_prob is not None else None,
                "n_signals": len(info["events"]),
                "n_distinct_actors": len(info["actors"]),
                "cluster_bonus": round(cluster_bonus, 4),
                "sector": info["sector"],
                "price_at_signal": info["price"],
                "latest_actor": latest.actor if latest else None,
                "latest_trade_date": latest.trade_date.isoformat() if latest else None,
                "source_type": latest.source_type.value if latest else None,
            })

        if not candidates:
            logger.info("top_pick.no_qualifying_candidates")
            return None

        candidates.sort(key=lambda c: -c["score"])
        pick = candidates[0]
        pick["runner_ups"] = [
            {"ticker": c["ticker"], "score": c["score"]} for c in candidates[1:4]
        ]
        pick["lookback_days"] = lookback_days
        pick["selected_at"] = datetime.utcnow().isoformat()

        logger.info(
            "top_pick.selected",
            ticker=pick["ticker"], score=pick["score"],
            n_candidates=len(candidates),
        )
        return pick
    finally:
        session.close()
