"""Security regression tests for the public API surface.

These encode the hardening guarantees: input validation on tickers and
backtest dates, admin gating, and CSV formula-injection neutralization.
They run against an isolated temporary database, with admin actions disabled.
"""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from config.settings import Settings
from src.api import api
from src.models.database import MeanVarianceResult


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    database_url = f"sqlite:///{tmp_path_factory.mktemp('api') / 'api.db'}"
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("DATABASE_URL", database_url)
        patch.setenv("ADMIN_API_TOKEN", "")
        patch.setenv("SENTRY_DSN", "")
        patch.setattr(api, "settings", Settings(_env_file=None))
        with TestClient(api.app) as c:
            yield c
        api.app.state.session_factory.kw["bind"].dispose()


def test_mean_variance_route_returns_latest_portfolio(client):
    with api.app.state.session_factory() as session:
        session.add_all([
            MeanVarianceResult(
                run_date=datetime(2026, 6, day),
                n_assets=2,
                tickers='["AAPL", "MSFT"]',
                ms_weights='{"AAPL": 0.6, "MSFT": 0.4}',
                ms_return=0.12,
                ms_volatility=0.20,
                ms_sharpe=0.4,
                mv_weights='{"AAPL": 0.5, "MSFT": 0.5}',
            )
            for day in (1, 2)
        ])
        session.commit()

    response = client.get("/api/analysis/mean-variance")

    assert response.status_code == 200
    portfolio = response.json()
    assert portfolio["run_date"] == "2026-06-02T00:00:00"
    assert portfolio["tickers"] == ["AAPL", "MSFT"]
    assert portfolio["n_assets"] == 2
    assert portfolio["max_sharpe"]["weights"] == {"AAPL": 0.6, "MSFT": 0.4}
    # The literal portfolio route must not change ordinary ticker routing.
    ticker_response = client.get("/api/analysis/aapl")
    assert ticker_response.status_code == 200
    assert ticker_response.json()["ticker"] == "AAPL"


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200


@pytest.mark.parametrize("bad", ["%3Dcmd", "AAPL%2F..", "A" * 11, "$(ls)"])
def test_invalid_tickers_rejected(client, bad):
    # 422 = validator caught it; 404 = encoded slash split the path and
    # missed the route entirely. Both mean the value never reached a query.
    for path in ("prices", "signals", "analysis"):
        assert client.get(f"/api/{path}/{bad}").status_code in (404, 422)


def test_backtest_rejects_bad_dates(client):
    cases = [
        {"start_date": "1900-01-01", "end_date": "2030-01-01"},  # out of bounds
        {"start_date": "bad", "end_date": "2026-01-01"},          # not a date
        {"start_date": "2026-02-01", "end_date": "2026-01-01"},   # reversed
        {"start_date": "2020-01-01", "end_date": "2026-01-01"},   # > 3 years
        {"start_date": "2026-01-01", "end_date": "2026-02-01",
         "conviction_threshold": 5.0},                             # bad threshold
    ]
    for body in cases:
        assert client.post("/api/backtest", json=body).status_code == 422, body


def test_admin_endpoints_locked_without_token(client):
    # No ADMIN_API_TOKEN configured -> disabled entirely (503), never open
    for path in ["/api/pipeline/run", "/api/performance/update", "/api/ml/xgboost/train"]:
        assert client.post(path).status_code in (401, 503)


def test_csv_export_neutralizes_formulas(client):
    r = client.get("/api/export/signals?days=30")
    assert r.status_code == 200
    for line in r.text.splitlines()[1:]:
        cells = line.split(",")
        if len(cells) > 2:
            assert not cells[2].startswith(("=", "+", "@")), line
