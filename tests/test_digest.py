import asyncio
import json
import subprocess
from datetime import datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.alerts.digest import build_digest, publish_digest, select_highlights
from src.alerts.digest_publisher import DigestPublisher
from src.alerts.notifier import SlackNotifier
from src.models.database import Base, Direction, Enrichment, SmartMoneyEvent, SourceType

MONDAY_MORNING = datetime(2026, 11, 9, 7, 5)
MONDAY_EVENING = datetime(2026, 11, 9, 17, 35)
SATURDAY_MORNING = datetime(2026, 11, 7, 7, 5)


def _trade(ticker, actor="A", direction="buy", value=1000.0, source="insider"):
    return {
        "ticker": ticker, "actor": actor, "direction": direction, "value": value,
        "trade_date": "2026-11-05", "filing_date": "2026-11-06",
        "source": source, "sector": "Technology",
    }


@pytest.fixture
def session_factory():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _add_event(
    factory, ticker="ACME", actor="Doe", created_at=None, source=SourceType.INSIDER,
    reported_price=None, market_price=None,
):
    session = factory()
    event = SmartMoneyEvent(
            ticker=ticker, actor=actor, direction=Direction.BUY, size_lower=5000.0,
            size_upper=5000.0, trade_date=datetime(2026, 11, 5),
            disclosure_date=datetime(2026, 11, 6), source_type=source,
            raw_payload=json.dumps({"secret": "raw", "price_per_share": reported_price}),
            created_at=created_at or datetime.utcnow(),
    )
    session.add(event)
    session.flush()
    if market_price:
        session.add(Enrichment(event_id=event.id, price_at_signal=market_price))
    session.commit()
    session.close()


class FakePublisher:
    def __init__(self, latest=None, ok=True):
        self.latest = latest
        self.ok = ok
        self.published = []

    def read_latest(self):
        return self.latest

    def publish(self, digest):
        self.published.append(digest)
        self.latest = digest
        return self.ok


class FakeNotifier:
    def __init__(self, ok=True):
        self.ok = ok
        self.sent = []

    async def send_digest(self, digest, url):
        self.sent.append((digest, url))
        return self.ok


SETTINGS = SimpleNamespace(digest_url="https://example.test/finalyse", digest_include_congressional=False)


def test_clusters_rank_by_distinct_insiders_before_large_buys():
    trades = [
        _trade("BIG", value=9_000_000),
        _trade("TWO", "A", value=100), _trade("TWO", "B", value=200),
        _trade("THREE", "A", value=100), _trade("THREE", "B", value=200), _trade("THREE", "C", value=300),
    ]

    highlights = select_highlights(trades)

    assert [h["ticker"] for h in highlights] == ["THREE", "TWO", "BIG"]
    assert highlights[0]["headline"].startswith("3 insiders bought")
    assert highlights[2]["headline"] == "A bought $9.0M"


def test_same_insider_repeating_is_not_a_cluster():
    trades = [_trade("ONE", "A", value=100), _trade("ONE", "A", value=200)]

    assert [h["kind"] for h in select_highlights(trades)] == ["large_buy"]


def test_joint_filings_of_one_purchase_are_not_a_cluster():
    trades = [_trade("FUND", "Entity A", value=5000), _trade("FUND", "Entity B", value=5000)]

    assert [h["kind"] for h in select_highlights(trades)] == ["large_buy"]


@pytest.mark.parametrize(
    "reported, market, expected",
    [(3149.47, 4.41, None), (1.0, 30.0, None), (10.0, 12.0, 5000.0), (10.0, None, 5000.0)],
)
def test_values_with_implausible_prices_are_dropped(session_factory, reported, market, expected):
    start = datetime(2026, 11, 8)
    _add_event(
        session_factory, created_at=start + timedelta(hours=1),
        reported_price=reported, market_price=market,
    )

    assert build_digest(session_factory, start, "2026-11-09")["trades"][0]["value"] == expected


def test_sells_and_13f_are_never_highlighted():
    trades = [
        _trade("SELL", direction="sell", value=9e9),
        _trade("FUND", source="institutional", value=9e9),
        _trade("REAL", value=10),
    ]

    assert [h["ticker"] for h in select_highlights(trades)] == ["REAL"]


def test_highlights_are_capped_and_one_per_ticker():
    trades = [_trade(f"T{i}", value=i + 1) for i in range(9)] + [_trade("T8", "B", value=1)]

    highlights = select_highlights(trades)

    assert len(highlights) == 5
    assert len({h["ticker"] for h in highlights}) == 5


def test_congressional_trades_get_slots_when_included():
    trades = [_trade(f"I{i}", value=1000 + i) for i in range(5)] + [
        _trade("POL", "Rep. X", direction="sell", value=50, source="congressional")
    ]

    highlights = select_highlights(trades)

    assert "POL" in [h["ticker"] for h in highlights]
    assert [h for h in highlights if h["ticker"] == "POL"][0]["headline"] == "Rep. X sold $50"


def test_window_excludes_older_events_and_raw_payload_never_appears(session_factory):
    start = datetime(2026, 11, 8)
    _add_event(session_factory, "NEW", created_at=start + timedelta(hours=1))
    _add_event(session_factory, "OLD", created_at=start - timedelta(hours=1))

    digest = build_digest(session_factory, start, "2026-11-09")

    assert [t["ticker"] for t in digest["trades"]] == ["NEW"]
    assert set(digest["trades"][0]) == {
        "ticker", "actor", "direction", "value", "trade_date",
        "filing_date", "source", "sector",
    }
    assert "secret" not in json.dumps(digest)


def test_congressional_rows_are_excluded_by_default(session_factory):
    start = datetime(2026, 11, 8)
    _add_event(session_factory, "POL", created_at=start + timedelta(hours=1), source=SourceType.CONGRESSIONAL)

    assert build_digest(session_factory, start, "2026-11-09")["trades"] == []
    included = build_digest(session_factory, start, "2026-11-09", include_congressional=True)
    assert included["counts"]["by_source"] == {"congressional": 1}


def _run(factory, notifier, publisher, now, tmp_path):
    return asyncio.run(
        publish_digest(factory, notifier, SETTINGS, publisher, now, tmp_path / "state.json")
    )


def test_message_sends_once_per_day(session_factory, tmp_path):
    _add_event(session_factory)
    notifier, publisher = FakeNotifier(), FakePublisher()

    _run(session_factory, notifier, publisher, MONDAY_MORNING, tmp_path)
    _run(session_factory, notifier, publisher, MONDAY_MORNING, tmp_path)

    assert len(notifier.sent) == 1
    assert notifier.sent[0][1] == SETTINGS.digest_url


def test_failed_send_is_retried_by_next_morning_run(session_factory, tmp_path):
    _add_event(session_factory)
    publisher = FakePublisher()
    failing, working = FakeNotifier(ok=False), FakeNotifier()

    _run(session_factory, failing, publisher, MONDAY_MORNING, tmp_path)
    _run(session_factory, working, publisher, MONDAY_MORNING, tmp_path)

    assert len(working.sent) == 1


@pytest.mark.parametrize("now", [MONDAY_EVENING, SATURDAY_MORNING])
def test_evening_and_weekend_runs_publish_without_messaging(session_factory, tmp_path, now):
    _add_event(session_factory)
    notifier, publisher = FakeNotifier(), FakePublisher()

    _run(session_factory, notifier, publisher, now, tmp_path)

    assert notifier.sent == []
    assert len(publisher.published) == 1


def test_no_message_when_there_are_no_trades(session_factory, tmp_path):
    notifier, publisher = FakeNotifier(), FakePublisher()

    _run(session_factory, notifier, publisher, MONDAY_MORNING, tmp_path)

    assert notifier.sent == []


def test_message_still_sends_when_publishing_fails(session_factory, tmp_path):
    _add_event(session_factory)
    notifier = FakeNotifier()

    _run(session_factory, notifier, FakePublisher(ok=False), MONDAY_MORNING, tmp_path)

    assert len(notifier.sent) == 1


def test_same_day_rerun_keeps_window_and_next_day_starts_after_it(session_factory, tmp_path):
    latest = {
        "date": "2026-11-09", "window_start": "2026-11-08T12:00:00Z",
        "generated_at": "2026-11-09T12:00:00Z",
    }
    _add_event(session_factory, "EARLY", created_at=datetime(2026, 11, 9, 1))
    _add_event(session_factory, "LATER", created_at=datetime(2026, 11, 9, 20))

    same_day = _run(session_factory, FakeNotifier(), FakePublisher(latest), MONDAY_EVENING, tmp_path)
    next_day = _run(
        session_factory, FakeNotifier(), FakePublisher(latest),
        datetime(2026, 11, 10, 7, 5), tmp_path,
    )

    assert {t["ticker"] for t in same_day["trades"]} == {"EARLY", "LATER"}
    assert {t["ticker"] for t in next_day["trades"]} == {"LATER"}


def test_slack_digest_links_to_full_list():
    posted = []

    def handler(request):
        posted.append(json.loads(request.content))
        return httpx.Response(200, text="ok")

    notifier = SlackNotifier("https://hooks.test/x")
    notifier._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    digest = {
        "date": "2026-11-09",
        "counts": {"trades": 42, "tickers": 30, "buys": 10, "sells": 32},
        "highlights": [{"ticker": "ACME", "headline": "3 insiders bought $1.0M"}],
    }

    assert asyncio.run(notifier.send_digest(digest, "https://example.test/finalyse")) is True
    text = json.dumps(posted[0])
    assert "<https://example.test/finalyse|See all 42 trades>" in text
    assert "ACME" in text
    assert "not financial advice" in text


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _digest(date):
    return {
        "date": date, "window_start": "2026-11-08T00:00:00Z",
        "generated_at": f"{date}T12:00:00Z", "counts": {}, "highlights": [], "trades": [],
    }


def test_publisher_creates_branch_then_appends(tmp_path):
    remote = tmp_path / "remote.git"
    remote.mkdir()
    _git(remote, "init", "--bare", "--quiet")
    publisher = DigestPublisher(tmp_path / "clone", str(remote))

    assert publisher.read_latest() is None
    assert publisher.publish(_digest("2026-11-09")) is True
    assert publisher.publish(_digest("2026-11-10")) is True

    fresh = DigestPublisher(tmp_path / "clone2", str(remote))
    assert fresh.read_latest()["date"] == "2026-11-10"
    assert json.loads((tmp_path / "clone2" / "index.json").read_text()) == {
        "dates": ["2026-11-10", "2026-11-09"]
    }
    assert (tmp_path / "clone2" / "2026-11-09.json").exists()


def test_publisher_reports_failure_instead_of_raising(tmp_path):
    publisher = DigestPublisher(tmp_path / "clone", str(tmp_path / "does-not-exist"))

    assert publisher.publish(_digest("2026-11-09")) is False
