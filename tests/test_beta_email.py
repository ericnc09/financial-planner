import asyncio
import json
from datetime import datetime
from types import SimpleNamespace

import httpx

from src.alerts.beta_email import build_email, send_top_10
from src.alerts.digest import TOP_LIST_SIZE, select_highlights

MONDAY = datetime(2026, 11, 9, 7, 30)
SATURDAY = datetime(2026, 11, 7, 7, 30)
SETTINGS = SimpleNamespace(
    buttondown_api_key="secret-key",
    beta_approved_tag="beta-approved",
    digest_url="https://example.test/finalyse",
)
TAGS = [{"name": "other", "id": "tag_other"}, {"name": "beta-approved", "id": "tag_ok"}]


def _digest(date="2026-11-09", items=2):
    return {
        "date": date,
        "counts": {"trades": 1974, "tickers": 815},
        "top_10": [
            {"ticker": f"T{i}", "headline": f"{i + 2} insiders bought $1.0M"}
            for i in range(items)
        ],
    }


def _client(tags=None, email_status=201):
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("/tags"):
            return httpx.Response(200, json={"results": TAGS if tags is None else tags, "next": None})
        return httpx.Response(email_status, json={"id": "em_1"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), requests


def _send(digest, tmp_path, now=MONDAY, settings=SETTINGS, **client_kwargs):
    client, requests = _client(**client_kwargs)
    sent = asyncio.run(send_top_10(digest, settings, client, now, tmp_path / "state.json"))
    return sent, requests


def _email_requests(requests):
    return [r for r in requests if r.url.path.endswith("/emails")]


def test_email_goes_only_to_the_approved_tag(tmp_path):
    sent, requests = _send(_digest(), tmp_path)

    assert sent is True
    email = _email_requests(requests)[0]
    assert email.headers["authorization"] == "Token secret-key"
    payload = json.loads(email.content)
    assert payload["filters"] == {
        "filters": [{"field": "subscriber.tags", "operator": "contains", "value": "tag_ok"}],
        "groups": [],
        "predicate": "and",
    }
    assert payload["status"] == "about_to_send"
    assert payload["archival_mode"] == "disabled"


def test_nothing_sends_if_the_approved_tag_does_not_exist(tmp_path):
    sent, requests = _send(_digest(), tmp_path, tags=[TAGS[0]])

    assert sent is False
    assert _email_requests(requests) == []


def test_sends_once_per_day(tmp_path):
    assert _send(_digest(), tmp_path)[0] is True

    sent, requests = _send(_digest(), tmp_path)

    assert sent is False
    assert requests == []


def test_failed_send_can_be_retried(tmp_path):
    assert _send(_digest(), tmp_path, email_status=500)[0] is False
    assert _send(_digest(), tmp_path)[0] is True


def test_stale_digest_is_never_sent_as_today(tmp_path):
    sent, requests = _send(_digest(date="2026-11-06"), tmp_path)

    assert sent is False
    assert requests == []


def test_missing_digest_does_not_send(tmp_path):
    assert _send(None, tmp_path)[0] is False


def test_weekend_empty_and_unconfigured_do_not_send(tmp_path):
    no_key = SimpleNamespace(**{**vars(SETTINGS), "buttondown_api_key": None})

    assert _send(_digest("2026-11-07"), tmp_path, now=SATURDAY)[0] is False
    assert _send(_digest(items=0), tmp_path)[0] is False
    assert _send(_digest(), tmp_path, settings=no_key)[0] is False


def test_body_has_numbered_list_link_and_disclaimer():
    subject, body = build_email(_digest(items=10), SETTINGS.digest_url)

    assert subject == "Insider trades: top 10 for 2026-11-09"
    assert "10. **T9**: 11 insiders bought $1.0M" in body
    assert "[See all 1,974 trades](https://example.test/finalyse)" in body
    assert "Not investment advice" in body


def test_names_from_filings_are_escaped():
    digest = _digest(items=1)
    digest["top_10"][0]["headline"] = "<img src=x> [click](http://bad) bought"

    _, body = build_email(digest, SETTINGS.digest_url)

    assert "<img" not in body
    assert "[click]" not in body


def test_top_list_holds_ten_tickers():
    trades = [
        {
            "ticker": f"T{i}", "actor": "A", "direction": "buy", "value": 1000.0 + i,
            "trade_date": "2026-11-05", "filing_date": "2026-11-06",
            "source": "insider", "sector": None,
        }
        for i in range(15)
    ]

    top = select_highlights(trades, TOP_LIST_SIZE)

    assert len(top) == 10
    assert top[0]["ticker"] == "T14"
