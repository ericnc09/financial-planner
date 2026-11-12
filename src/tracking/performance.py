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
    # Transaction costs: 10bps slippage + 5bps commission per side = 30bps round-trip
    COST_PER_TRADE = 0.0030
    # Yahoo returns None for delisted tickers, for symbols it spells
    # differently, and for transient failures (rate limits, network).
    PRICE_FETCH_ATTEMPTS = 3
    PRICE_RETRY_DELAY_S = 2.0
    # A price history whose last bar is older than this has stopped trading
    # (acquired, delisted): remaining horizons settle at the last traded price.
    STALE_HISTORY_DAYS = 10
    # If the first available bar is this long after the signal, the signal
    # predates the price window and no honest entry price exists.
    MAX_ENTRY_GAP_DAYS = 7

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
            # Delisted placeholders (entry_price 0) are re-checked every run so
            # a mistaken -100% heals once prices are available again.
            pending = [
                evt for evt in evts
                if (perf := existing.get(evt.id)) is None
                or perf.return_60d is None
                or perf.entry_price == 0
            ]
            if not pending:
                continue
            try:
                price_data = await self._fetch_prices(ticker)

                for evt in pending:
                    perf = existing.get(evt.id)

                    if price_data is None:
                        # No price data after retries. That can mean delisted,
                        # but in practice it is usually a symbol the price
                        # source spells differently or an outage, so a return
                        # is never invented: the event is marked unpriced
                        # (entry_price 0, no returns), excluded from every
                        # statistic, and re-checked on each run.
                        if perf is not None:
                            if perf.entry_price != 0:
                                logger.warning("performance.price_unavailable", ticker=ticker, event_id=evt.id)
                            elif self._clear_placeholder_returns(evt.id):
                                updated += 1
                                logger.info("performance.phantom_return_cleared", ticker=ticker, event_id=evt.id)
                            continue
                        days_since = (datetime.utcnow() - evt.trade_date).days
                        if days_since > 10:
                            self._persist(evt, self._unpriced(evt), None)
                            updated += 1
                            logger.info("performance.unpriced", ticker=ticker, event_id=evt.id)
                        continue

                    dates = price_data["dates"]
                    closes = price_data["closes"]

                    # A row that already has a real entry keeps that entry, so
                    # its horizons stay on one basis while they mature.
                    fixed_entry = (
                        perf.entry_date if perf is not None and perf.entry_price else None
                    )
                    result = self._compute_returns(
                        evt, dates, closes, entry_date=fixed_entry
                    )
                    if result:
                        self._persist(evt, result, existing.get(evt.id))
                        updated += 1

            except Exception as e:
                logger.warning("performance.ticker_failed", ticker=ticker, error=str(e))

        logger.info("performance.done", updated=updated)
        await self.yahoo.close()
        return updated

    async def _fetch_prices(self, ticker: str) -> dict | None:
        for attempt in range(self.PRICE_FETCH_ATTEMPTS):
            price_data = await self.yahoo.get_price_history(ticker, days=504)
            if price_data is not None:
                return price_data
            if attempt < self.PRICE_FETCH_ATTEMPTS - 1:
                await asyncio.sleep(self.PRICE_RETRY_DELAY_S * (attempt + 1))
        logger.info("performance.no_price_data", ticker=ticker, attempts=self.PRICE_FETCH_ATTEMPTS)
        return None

    def _unpriced(self, event) -> dict:
        """Marker row for an event with no price data: entry_price 0, no returns."""
        return {
            "entry_price": 0.0,
            "entry_date": getattr(event, "disclosure_date", None) or event.trade_date,
            "price_5d": None, "return_5d": None,
            "price_10d": None, "return_10d": None,
            "price_20d": None, "return_20d": None,
            "price_60d": None, "return_60d": None,
            "is_winner_5d": None,
            "is_winner_20d": None,
        }

    def _clear_placeholder_returns(self, event_id: int) -> bool:
        """Remove the +/-100% returns older versions booked on unpriced rows."""
        session = self.session_factory()
        try:
            row = session.query(SignalPerformance).filter(
                SignalPerformance.event_id == event_id
            ).first()
            if row is None or row.entry_price != 0 or row.return_5d is None:
                return False
            for key, val in self._unpriced_fields().items():
                setattr(row, key, val)
            row.updated_at = datetime.utcnow()
            session.commit()
            return True
        finally:
            session.close()

    @staticmethod
    def _unpriced_fields() -> dict:
        fields = {f"{kind}_{h}": None for kind in ("price", "return") for h in ("5d", "10d", "20d", "60d")}
        fields.update({"is_winner_5d": None, "is_winner_20d": None})
        return fields

    @staticmethod
    def _as_date(value):
        return value.date() if hasattr(value, "date") else value

    def _compute_returns(self, event, dates, closes, entry_date=None) -> dict | None:
        """Compute returns at each horizon from the first tradable close.

        The signal becomes actionable on its disclosure_date (trade_date if
        that is missing). Form 4s and 13Fs are routinely filed after the
        close, so entry is the close of the NEXT trading day, never the
        disclosure day itself.

        Args:
            entry_date: Keep an existing row on its recorded entry instead of
                        re-deriving it.
        """
        if len(dates) == 0:
            return None

        if entry_date is not None:
            target = self._as_date(entry_date)
            entry_idx = next(
                (i for i, d in enumerate(dates) if self._as_date(d) >= target), None
            )
        else:
            signal_date = self._as_date(
                getattr(event, "disclosure_date", None) or event.trade_date
            )
            entry_idx = next(
                (i for i, d in enumerate(dates) if self._as_date(d) > signal_date), None
            )
            # First bar far after the signal: the signal predates the price
            # window, so that bar is not the entry.
            if entry_idx == 0 and (
                self._as_date(dates[0]) - signal_date
            ).days > self.MAX_ENTRY_GAP_DAYS:
                return None

        if entry_idx is None:
            return None

        entry_price = float(closes[entry_idx])
        if not np.isfinite(entry_price) or entry_price <= 0:
            return None

        # History that ended a while ago means the security stopped trading.
        stopped_trading = (
            datetime.utcnow().date() - self._as_date(dates[-1])
        ).days > self.STALE_HISTORY_DAYS

        result = {
            "entry_price": entry_price,
            "entry_date": dates[entry_idx],
        }

        # Direction multiplier: BUY profits from price going up, SELL from down
        direction = event.direction.value
        mult = 1.0 if direction == "buy" else -1.0

        for horizon, label in [(5, "5d"), (10, "10d"), (20, "20d"), (60, "60d")]:
            target_idx = entry_idx + horizon
            if target_idx >= len(closes) and stopped_trading:
                # Acquired or delisted mid-hold: settle at the last traded
                # price (direction applied below), not at an assumed zero.
                target_idx = len(closes) - 1
            if target_idx < len(closes):
                future_price = float(closes[target_idx])
                raw_return = (future_price - entry_price) / entry_price
                adjusted_return = raw_return * mult
                # Deduct round-trip transaction costs (slippage + commission)
                adjusted_return -= self.COST_PER_TRADE
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
                    # A delisted placeholder is replaced wholesale once real
                    # prices exist, including entry and still-maturing horizons.
                    replacing_placeholder = row.entry_price == 0
                    for key, val in result.items():
                        if replacing_placeholder or (
                            val is not None and key not in ("entry_price", "entry_date")
                        ):
                            setattr(row, key, val)
                    if row.conviction is None:
                        row.conviction = conviction
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
