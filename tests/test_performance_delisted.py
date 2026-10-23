"""A failed price fetch must not book a false -100% delisted loss."""

import asyncio
from datetime import datetime, timedelta

import numpy as np
import pytest

from src.models.database import (
    SignalPerformance,
    SmartMoneyEvent,
    get_session_factory,
    init_db,
)
from src.models.schemas import Direction, SourceType
from src.tracking.performance import PerformanceTracker

TRADE_DATE = datetime.utcnow() - timedelta(days=200)


def prices():
    dates = [TRADE_DATE + timedelta(days=i) for i in range(100)]
    closes = np.linspace(10.0, 20.0, 100)
    return {"dates": dates, "closes": closes}


class FakeYahoo:
    """Returns None for the first `failures` calls per ticker, then `data`."""

    def __init__(self, data, failures):
        self.data = data
        self.failures = failures
        self.calls = 0

    async def get_price_history(self, ticker, days=504):
        self.calls += 1
        return None if self.calls <= self.failures else self.data

    async def close(self):
        pass


@pytest.fixture
def session_factory(tmp_path):
    engine = init_db(f"sqlite:///{tmp_path / 'perf.db'}")
    sf = get_session_factory(engine)
    with sf() as session:
        session.add(SmartMoneyEvent(
            ticker="BB", actor="Example insider", direction=Direction.BUY,
            trade_date=TRADE_DATE, disclosure_date=TRADE_DATE,
            source_type=SourceType.INSIDER,
        ))
        session.commit()
    yield sf
    engine.dispose()


def run(sf, yahoo):
    tracker = PerformanceTracker(sf)
    tracker.yahoo = yahoo
    tracker.PRICE_RETRY_DELAY_S = 0
    asyncio.run(tracker.update_performance())
    with sf() as session:
        return session.query(SignalPerformance).one_or_none()


def test_transient_failure_is_retried(session_factory):
    yahoo = FakeYahoo(prices(), failures=2)
    row = run(session_factory, yahoo)
    assert yahoo.calls == 3
    assert row.entry_price == pytest.approx(10.0)
    assert row.return_60d > 0


def test_persistent_failure_books_delisted_loss(session_factory):
    row = run(session_factory, FakeYahoo(None, failures=99))
    assert row.entry_price == 0
    assert row.return_60d == -1.0


def test_delisted_placeholder_heals_when_prices_return(session_factory):
    run(session_factory, FakeYahoo(None, failures=99))
    row = run(session_factory, FakeYahoo(prices(), failures=0))
    assert row.entry_price == pytest.approx(10.0)
    assert row.return_5d > 0 and row.return_60d > 0


def test_realized_returns_are_never_overwritten(session_factory):
    data = prices()
    # Only 30 days of history: 5d/20d measured, 60d still maturing.
    data = {"dates": data["dates"][:30], "closes": data["closes"][:30]}
    first = run(session_factory, FakeYahoo(data, failures=0))
    assert first.return_20d > 0 and first.return_60d is None

    row = run(session_factory, FakeYahoo(None, failures=99))
    assert row.entry_price == pytest.approx(first.entry_price)
    assert row.return_20d == pytest.approx(first.return_20d)
    assert row.return_60d is None
