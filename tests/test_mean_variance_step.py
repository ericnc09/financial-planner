"""Mean-variance step must drop tickers with non-finite returns, not crash."""

import asyncio
import json

import numpy as np
import pytest

from config.settings import Settings
from src.analysis.mean_variance import MeanVarianceOptimizer
from src.models.database import MeanVarianceResult, get_session_factory, init_db
from src.pipeline.orchestrator import Orchestrator


class FakeYahoo:
    def __init__(self, returns_by_ticker):
        self.returns_by_ticker = returns_by_ticker

    async def get_price_history(self, ticker, days=252):
        return {"returns": self.returns_by_ticker[ticker]}


@pytest.fixture
def orchestrator(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'mv.db'}")
    settings = Settings(_env_file=None)
    pipeline = Orchestrator(settings)
    pipeline.engine = init_db(settings.database_url)
    pipeline.session_factory = get_session_factory(pipeline.engine)
    pipeline.mean_variance = MeanVarianceOptimizer()
    yield pipeline
    pipeline.engine.dispose()


def test_nonfinite_returns_are_skipped(orchestrator):
    rng = np.random.default_rng(0)
    bad = rng.normal(0, 0.01, 120)
    bad[10] = np.nan
    bad[20] = -np.inf  # log(0) close
    orchestrator.yahoo = FakeYahoo({
        "AAA": rng.normal(0.001, 0.01, 120),
        "BBB": rng.normal(0.0005, 0.02, 120),
        "BAD": bad,
    })

    asyncio.run(orchestrator._run_mean_variance(["AAA", "BAD", "BBB"]))

    with orchestrator.session_factory() as session:
        rows = session.query(MeanVarianceResult).all()
    assert len(rows) == 1
    assert json.loads(rows[0].tickers) == ["AAA", "BBB"]
