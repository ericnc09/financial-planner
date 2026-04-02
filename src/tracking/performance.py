"""
Signal Performance Tracker — Measures actual returns after each signal.

Fetches current/historical prices for past signals and computes realized
returns at 5, 10, 20, and 60 day horizons. Determines win/loss for each.
"""

import asyncio
from datetime import datetime, timedelta

import numpy as np
import structlog

from src.clients.yahoo import YahooClient
from src.models.database import (
    ConvictionScore,
    Enrichment,
    SignalPerformance,
    SmartMoneyEvent,
    get_session_factory,
    init_db,
)
from sqlalchemy.exc import IntegrityError

logger = structlog.get_logger()


class PerformanceTracker:
    def __init__(self, session_factory):
        self.session_factory = session_factory
        self.yahoo = YahooClient()

    async def update_performance(self):
        """Scan all events and update performance for those with enough history."""
        session = self.session_factory()
        try:
            # Get all events that don't have complete performance records
            events = session.query(SmartMoneyEvent).all()
            existing = {
                r.event_id: r
                for r in session.query(SignalPerformance).all()
            }
            logger.info("performance.scanning", total_events=len(events), existing=len(existing))
        finally:
            session.close()

        # Group by ticker for efficient price fetching
        ticker_events: dict[str, list] = {}
        for evt in events:
            ticker_events.setdefault(evt.ticker, []).append(evt)

        updated = 0
        for ticker, evts in ticker_events.items():
            try:
                price_data = await self.yahoo.get_price_history(ticker, days=504)
                if price_data is None:
                    continue

                dates = price_data["dates"]
                closes = price_data["closes"]

                for evt in evts:
                    perf = existing.get(evt.id)
                    needs_update = (
                        perf is None or
                        perf.return_60d is None
                    )
                    if not needs_update:
                        continue

                    result = self._compute_returns(
                        evt, dates, closes
                    )
                    if result:
                        self._persist(evt, result, existing.get(evt.id))
                        updated += 1

            except Exception as e:
                logger.warning("performance.ticker_failed", ticker=ticker, error=str(e))

        logger.info("performance.done", updated=updated)
        await self.yahoo.close()
        return updated

    def _compute_returns(self, event, dates, closes) -> dict | None:
        """Compute returns at various horizons from trade date."""
        trade_date = event.trade_date.date() if hasattr(event.trade_date, 'date') else event.trade_date

        # Find entry index (first trading day on or after trade date)
        entry_idx = None
        for i, d in enumerate(dates):
            d_date = d.date() if hasattr(d, 'date') else d
            if d_date >= trade_date:
                entry_idx = i
                break

        if entry_idx is None:
            return None

        entry_price = float(closes[entry_idx])
        if entry_price <= 0:
            return None

        result = {
            "entry_price": entry_price,
            "entry_date": dates[entry_idx],
        }

        # Direction multiplier: BUY profits from price going up, SELL from down
        direction = event.direction.value
        mult = 1.0 if direction == "buy" else -1.0

        for horizon, label in [(5, "5d"), (10, "10d"), (20, "20d"), (60, "60d")]:
            target_idx = entry_idx + horizon
            if target_idx < len(closes):
                future_price = float(closes[target_idx])
                raw_return = (future_price - entry_price) / entry_price
                adjusted_return = raw_return * mult
                result[f"price_{label}"] = future_price
                result[f"return_{label}"] = round(adjusted_return, 6)
            else:
                result[f"price_{label}"] = None
                result[f"return_{label}"] = None

        # Win/loss flags
        r5 = result.get("return_5d")
        r20 = result.get("return_20d")
        result["is_winner_5d"] = r5 > 0 if r5 is not None else None
        result["is_winner_20d"] = r20 > 0 if r20 is not None else None

        return result

    def _persist(self, event, result: dict, existing_row):
        """Create or update performance record."""
        session = self.session_factory()
        try:
            # Get conviction score
            cs = session.query(ConvictionScore).filter(
                ConvictionScore.event_id == event.id
            ).first()
            conviction = cs.conviction if cs else None

            if existing_row:
                # Update
                row = session.query(SignalPerformance).filter(
                    SignalPerformance.event_id == event.id
                ).first()
                if row:
                    for key, val in result.items():
                        if val is not None and key not in ("entry_price", "entry_date"):
                            setattr(row, key, val)
                    row.updated_at = datetime.utcnow()
                    session.commit()
            else:
                row = SignalPerformance(
                    event_id=event.id,
                    ticker=event.ticker,
                    direction=event.direction.value,
                    source_type=event.source_type.value,
                    conviction=conviction,
                    entry_price=result["entry_price"],
                    entry_date=result["entry_date"],
                    price_5d=result.get("price_5d"),
                    price_10d=result.get("price_10d"),
                    price_20d=result.get("price_20d"),
                    price_60d=result.get("price_60d"),
                    return_5d=result.get("return_5d"),
                    return_10d=result.get("return_10d"),
                    return_20d=result.get("return_20d"),
                    return_60d=result.get("return_60d"),
                    is_winner_5d=result.get("is_winner_5d"),
                    is_winner_20d=result.get("is_winner_20d"),
                )
                session.add(row)
                try:
                    session.commit()
                except IntegrityError:
                    session.rollback()
        finally:
            session.close()


async def run_performance_update():
    """CLI entry point."""
    from config.settings import Settings
    settings = Settings()
    engine = init_db(settings.database_url)
    sf = get_session_factory(engine)
    tracker = PerformanceTracker(sf)
    updated = await tracker.update_performance()
    print(f"Updated {updated} performance records")


if __name__ == "__main__":
    asyncio.run(run_performance_update())
