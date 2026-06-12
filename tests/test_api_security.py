"""Security regression tests for the public API surface.

These encode the hardening guarantees: input validation on tickers and
backtest dates, admin gating, and CSV formula-injection neutralization.
They run against whatever DB settings point at (empty is fine).
"""

import pytest
from fastapi.testclient import TestClient

from src.api.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


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
