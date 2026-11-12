import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

import httpx

from src.alerts.notifier import SlackNotifier, WhatsAppNotifier, create_notifier

SETUP_DOC = Path(__file__).parent.parent / "docs" / "WHATSAPP_SETUP.md"


def _notifier(status_code=200, body=None):
    """Build a notifier that records requests instead of calling Meta."""
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status_code, json=body or {"messages": [{"id": "wamid.1"}]})

    notifier = WhatsAppNotifier("secret-token", "12345", "14165551234")
    headers = notifier._client.headers
    notifier._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), headers=headers)
    return notifier, requests


def _variables(request):
    template = json.loads(request.content)["template"]
    return [p["text"] for p in template["components"][0]["parameters"]]


def _documented_variable_count(template):
    """Count {{n}} variables under a template's heading in the setup doc."""
    section = SETUP_DOC.read_text().split(f"### `{template}`")[1].split("###")[0]
    return len(set(re.findall(r"\{\{(\d+)\}\}", section)))


def test_conviction_alert_request_shape():
    notifier, requests = _notifier()

    sent = asyncio.run(
        notifier.send_conviction_alert("RY", "buy", "Jane Doe", 0.725, 0.81, 0.64, "Banks", 120.5)
    )

    assert sent is True
    request = requests[0]
    assert str(request.url) == "https://graph.facebook.com/v24.0/12345/messages"
    assert request.headers["authorization"] == "Bearer secret-token"
    payload = json.loads(request.content)
    assert payload["to"] == "14165551234"
    assert payload["template"]["name"] == "conviction_alert"
    assert _variables(request) == [
        "RY", "BUY", "Jane Doe", "72.5%", "0.81", "0.64", "Banks", "$120.50",
    ]


def test_missing_values_become_na():
    notifier, requests = _notifier()

    asyncio.run(notifier.send_conviction_alert("RY", "buy", "Jane", 0.7, 0.8, 0.6))

    assert _variables(requests[0])[-2:] == ["N/A", "N/A"]


def test_ensemble_components_are_flattened_to_one_line():
    notifier, requests = _notifier()

    asyncio.run(
        notifier.send_ensemble_alert(
            "RY", "buy", 82.34, 0.9, "strong\nbuy", {"hmm": 70.0, "garch": None, "mc": 65.5}
        )
    )

    variables = _variables(requests[0])
    assert variables[4] == "strong buy"
    assert variables[5] == "hmm: 70.0, mc: 65.5"


def test_top_pick_handles_sparse_pick():
    notifier, requests = _notifier()

    asyncio.run(notifier.send_top_pick({"ticker": "RY", "score": 1.5}))

    variables = _variables(requests[0])
    assert variables[0:2] == ["RY", "1.50"]
    assert variables[3:6] == ["N/A", "N/A", "N/A"]
    assert variables[8] == "N/A"
    assert all(variables)


def test_analysis_summary_orders_and_caps_tickers():
    notifier, requests = _notifier()
    results = [
        {"ticker": "LOW", "verdict": "BUY", "buy_score": 60, "sell_score": 10},
        {"ticker": "HIGH", "verdict": "BUY", "buy_score": 90, "sell_score": 5},
        {"ticker": "DROP", "verdict": "SELL", "buy_score": 5, "sell_score": 75},
    ] + [
        {"ticker": f"H{i}", "verdict": "HOLD", "buy_score": 50, "sell_score": 50}
        for i in range(25)
    ]

    asyncio.run(notifier.send_analysis_summary("Run", results, ["BAD"]))

    variables = _variables(requests[0])
    assert variables[1] == "HIGH (90), LOW (60)"
    assert variables[2].endswith("H19 +5 more")
    assert variables[3] == "DROP (75)"
    assert variables[4] == "BAD"


def test_api_error_returns_false_and_does_not_raise():
    notifier, _ = _notifier(400, {"error": {"message": "Template name does not exist"}})

    assert asyncio.run(notifier.send_top_pick({"ticker": "RY", "score": 1.0})) is False


def test_unconfigured_notifier_is_a_no_op():
    notifier = WhatsAppNotifier("token", None, "14165551234")

    assert notifier.configured is False
    assert asyncio.run(notifier.send_top_pick({"ticker": "RY", "score": 1.0})) is False
    asyncio.run(notifier.close())


def test_sent_variable_counts_match_setup_doc():
    notifier, requests = _notifier()
    pick = {"ticker": "RY", "score": 1.0}

    async def send_all():
        await notifier.send_conviction_alert("RY", "buy", "Jane", 0.7, 0.8, 0.6)
        await notifier.send_ensemble_alert("RY", "buy", 80.0, 0.9, "buy", {})
        await notifier.send_top_pick(pick)
        await notifier.send_analysis_summary("Run", [])
        await notifier.send_digest(
            {
                "date": "2026-11-09",
                "counts": {"trades": 1, "tickers": 1, "buys": 1, "sells": 0},
                "highlights": [],
            },
            "https://example.test",
        )

    asyncio.run(send_all())

    for request in requests:
        name = json.loads(request.content)["template"]["name"]
        assert len(_variables(request)) == _documented_variable_count(name), name


def test_create_notifier_selects_channel():
    settings = SimpleNamespace(
        alert_channel="whatsapp",
        slack_webhook_url="https://hooks.slack.test/x",
        whatsapp_token="token",
        whatsapp_phone_number_id="12345",
        whatsapp_to="14165551234",
    )
    assert isinstance(create_notifier(settings), WhatsAppNotifier)

    settings.alert_channel = "slack"
    assert isinstance(create_notifier(settings), SlackNotifier)
