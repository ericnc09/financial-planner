"""Event persistence must survive duplicates without losing returned rows."""

from datetime import datetime

import pytest

from config.settings import Settings
from src.models.database import SmartMoneyEvent, get_session_factory, init_db
from src.models.schemas import Direction, SmartMoneyEventSchema, SourceType
from src.pipeline.orchestrator import Orchestrator


def signal(ticker):
    return SmartMoneyEventSchema(
        ticker=ticker,
        actor="Example disclosure actor",
        direction=Direction.BUY,
        trade_date=datetime(2026, 6, 1),
        disclosure_date=datetime(2026, 6, 5),
        source_type=SourceType.INSIDER,
    )


@pytest.fixture
def orchestrator(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'events.db'}")
    settings = Settings(_env_file=None)
    pipeline = Orchestrator(settings)
    # Do not initialize clients or run the pipeline: persistence is local only.
    pipeline.engine = init_db(settings.database_url)
    pipeline.session_factory = get_session_factory(pipeline.engine)
    yield pipeline
    pipeline.engine.dispose()


@pytest.mark.parametrize("batch", [
    ["AAPL", "EXIST", "MSFT"],  # Duplicate after a new insert.
    ["EXIST", "AAPL", "MSFT", "EXIST"],  # Duplicates at both ends.
    ["AAPL", "AAPL", "MSFT"],  # Duplicate of another new event.
])
def test_duplicate_only_rolls_back_its_own_insert(orchestrator, batch):
    original_id, _ = orchestrator._persist_events([signal("EXIST")])[0]

    inserted = orchestrator._persist_events([signal(ticker) for ticker in batch])

    assert [schema.ticker for _, schema in inserted] == ["AAPL", "MSFT"]
    assert len({event_id for event_id, _ in inserted}) == 2
    with orchestrator.session_factory() as session:
        assert session.query(SmartMoneyEvent).count() == 3
        assert session.get(SmartMoneyEvent, original_id).ticker == "EXIST"
        # Downstream enrichment must be able to use each returned ID to find
        # precisely the event it was paired with, not a reused or missing ID.
        for event_id, schema in inserted:
            persisted = session.get(SmartMoneyEvent, event_id)
            assert persisted is not None
            assert persisted.ticker == schema.ticker
            assert persisted.disclosure_date == schema.disclosure_date

    assert orchestrator._persist_events([signal(ticker) for ticker in batch]) == []
    with orchestrator.session_factory() as session:
        assert session.query(SmartMoneyEvent).count() == 3
