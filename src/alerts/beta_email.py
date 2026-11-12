"""
The 7:30 AM beta email: today's top 10 trades, with a link to the rest.

Sent through Buttondown, which holds the subscriber list and handles confirmation
and unsubscribes. Only subscribers carrying the approved tag receive it.
"""

import html
import json
from datetime import datetime
from pathlib import Path

import httpx
import structlog

logger = structlog.get_logger()

API_BASE = "https://api.buttondown.com/v1"
API_VERSION = "2026-04-01"
STATE_PATH = Path("data/beta_email_state.json")


def build_email(digest: dict, url: str) -> tuple[str, str]:
    """Return the subject and Markdown body. Names come from filings, so they're
    escaped before going into an email."""
    subject = f"Insider trades: top {len(digest['top_10'])} for {digest['date']}"
    lines = [
        f"{i}. **{_safe(h['ticker'])}**: {_safe(h['headline'])}"
        for i, h in enumerate(digest["top_10"], start=1)
    ]
    counts = digest["counts"]
    body = "\n\n".join(
        [
            f"{counts['trades']:,} new disclosures across {counts['tickers']:,} tickers. "
            "These stood out, by how many insiders bought and by size:",
            "\n".join(lines),
            f"[See all {counts['trades']:,} trades]({url})",
            "*Research and education only. Not investment advice or a recommendation "
            "to buy or sell any security.*",
        ]
    )
    return subject, body


async def send_top_10(
    digest: dict | None,
    settings,
    client: httpx.AsyncClient,
    now: datetime | None = None,
    state_path: Path = STATE_PATH,
) -> bool:
    """Send today's email if it's due. Returns True only if one was sent."""
    local_now = now or datetime.now()
    date = local_now.strftime("%Y-%m-%d")

    if not settings.buttondown_api_key:
        logger.info("beta_email.not_configured")
        return False
    if local_now.weekday() >= 5:
        return False
    if not digest or digest["date"] != date:
        # An old list must never go out as today's
        logger.warning("beta_email.no_digest_for_today", date=date)
        return False
    if not digest["top_10"]:
        logger.info("beta_email.nothing_to_send", date=date)
        return False
    if _last_sent_date(state_path) == date:
        logger.info("beta_email.already_sent", date=date)
        return False

    headers = {
        "Authorization": f"Token {settings.buttondown_api_key}",
        "X-API-Version": API_VERSION,
    }
    try:
        tag_id = await _find_tag_id(client, headers, settings.beta_approved_tag)
        if not tag_id:
            # Without the tag filter this would go to every subscriber, approved or not
            logger.error("beta_email.approved_tag_missing", tag=settings.beta_approved_tag)
            return False

        subject, body = build_email(digest, settings.digest_url)
        resp = await client.post(
            f"{API_BASE}/emails",
            headers=headers,
            json={
                "subject": subject,
                "body": body,
                "status": "about_to_send",
                "archival_mode": "disabled",
                "filters": {
                    "filters": [
                        {"field": "subscriber.tags", "operator": "contains", "value": tag_id}
                    ],
                    "groups": [],
                    "predicate": "and",
                },
            },
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.error(
            "beta_email.send_failed",
            status=e.response.status_code,
            error=e.response.text[:500],
        )
        return False
    except httpx.HTTPError as e:
        logger.error("beta_email.send_failed", error=str(e))
        return False

    _save_last_sent_date(date, state_path)
    logger.info("beta_email.sent", date=date, items=len(digest["top_10"]))
    return True


async def _find_tag_id(client: httpx.AsyncClient, headers: dict, name: str) -> str | None:
    url: str | None = f"{API_BASE}/tags?page_size=100"
    while url:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        page = resp.json()
        for tag in page["results"]:
            if tag["name"] == name:
                return tag["id"]
        url = page.get("next")
    return None


def _safe(text: str) -> str:
    escaped = html.escape(str(text))
    for character in "*_[]`":
        escaped = escaped.replace(character, f"\\{character}")
    return escaped


def _last_sent_date(state_path: Path) -> str | None:
    try:
        return json.loads(state_path.read_text()).get("last_sent_date")
    except (OSError, ValueError):
        return None


def _save_last_sent_date(date: str, state_path: Path) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"last_sent_date": date}))
