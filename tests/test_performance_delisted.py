"""A ticker without prices must never be booked as a +/-100% return."""

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
COST = PerformanceTracker.COST_PER_TRADE


def prices(start=TRADE_DATE, n=100):
    dates = [start + timedelta(days=i) for i in range(n)]
    closes = np.linspace(10.0, 20.0, 100)[:n]
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


def make_db(tmp_path, trade_date=TRADE_DATE, direction=Direction.BUY):
    engine = init_db(f"sqlite:///{tmp_path / 'perf.db'}")
    sf = get_session_factory(engine)
    with sf() as session:
        session.add(SmartMoneyEvent(
            ticker="BB", actor="Example insider", direction=direction,
            trade_date=trade_date, disclosure_date=trade_date,
            source_type=SourceType.INSIDER,
        ))
        session.commit()
    return engine, sf


@pytest.fixture
def session_factory(tmp_path):
    engine, sf = make_db(tmp_path)
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
    assert row.return_60d > 0


def test_entry_is_the_close_after_the_disclosure_day(session_factory):
    data = prices()
    row = run(session_factory, FakeYahoo(data, failures=0))
    # Bar 0 is the disclosure day itself; filings land after the close.
    assert row.entry_price == pytest.approx(data["closes"][1])
    assert row.return_5d == pytest.approx(data["closes"][6] / data["closes"][1] - 1 - COST)


def test_persistent_failure_marks_unpriced_without_a_return(session_factory):
    row = run(session_factory, FakeYahoo(None, failures=99))
    assert row.entry_price == 0
    assert row.return_5d is None and row.return_20d is None and row.return_60d is None
    assert row.is_winner_20d is None


def test_unpriced_marker_heals_when_prices_return(session_factory):
    run(session_factory, FakeYahoo(None, failures=99))
    data = prices()
    row = run(session_factory, FakeYahoo(data, failures=0))
    assert row.entry_price == pytest.approx(data["closes"][1])
    assert row.return_5d > 0 and row.return_60d > 0


def test_legacy_phantom_return_is_cleared(session_factory):
    with session_factory() as session:
        session.add(SignalPerformance(
            event_id=1, ticker="BB", direction="sell", source_type="insider",
            entry_price=0.0, entry_date=TRADE_DATE,
            return_5d=1.0, return_10d=1.0, return_20d=1.0, return_60d=1.0,
            is_winner_5d=True, is_winner_20d=True,
        ))
        session.commit()
    row = run(session_factory, FakeYahoo(None, failures=99))
    assert row.entry_price == 0
    assert row.return_20d is None and row.return_60d is None
    assert row.is_winner_20d is None


def test_realized_returns_are_never_overwritten(tmp_path):
    recent = datetime.utcnow() - timedelta(days=35)
    engine, sf = make_db(tmp_path, trade_date=recent)
    # History still live but only 30 bars deep: 5d/20d measured, 60d maturing.
    first = run(sf, FakeYahoo(prices(start=recent, n=30), failures=0))
    assert first.return_20d > 0 and first.return_60d is None

    row = run(sf, FakeYahoo(None, failures=99))
    assert row.entry_price == pytest.approx(first.entry_price)
    assert row.return_20d == pytest.approx(first.return_20d)
    assert row.return_60d is None
    engine.dispose()


@pytest.mark.parametrize("direction,sign", [(Direction.BUY, 1.0), (Direction.SELL, -1.0)])
def test_stopped_trading_settles_at_last_price(tmp_path, direction, sign):
    engine, sf = make_db(tmp_path, direction=direction)
    data = prices(n=30)  # history ended ~170 days ago: acquired or delisted
    row = run(sf, FakeYahoo(data, failures=0))
    move = data["closes"][-1] / data["closes"][1] - 1
    assert row.return_60d == pytest.approx(sign * move - COST)
    assert abs(row.return_60d) < 1.0
    engine.dispose()


def test_signal_older_than_price_window_gets_no_entry(session_factory):
    late = prices(start=TRADE_DATE + timedelta(days=30))
    assert run(session_factory, FakeYahoo(late, failures=0)) is None
