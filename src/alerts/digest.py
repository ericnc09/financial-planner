"""
Daily digest: one compiled message with a few highlighted trades, and a public
JSON file behind ericcosta.ca/finalyse that lists the rest.

Highlights are rule-based on purpose. The model review found conviction scores
don't predict returns while the breadth of insider buying did, so the digest
surfaces clusters and size rather than model rankings.
"""

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import structlog

from src.alerts.digest_publisher import DigestPublisher
from src.models.database import Enrichment, SmartMoneyEvent, SourceType

logger = structlog.get_logger()

MAX_HIGHLIGHTS = 5
TOP_LIST_SIZE = 10
MESSAGE_CUTOFF_HOUR = 12
# A reported price this far from the market price is a filer error or another
# currency, and the dollar value computed from it would be nonsense
MAX_PRICE_RATIO = 3.0
STATE_PATH = Path("data/digest_state.json")


def build_digest(
    session_factory,
    window_start: datetime,
    date: str,
    include_congressional: bool = False,
    now: datetime | None = None,
) -> dict:
    """Compile events ingested since window_start. Only allowlisted fields leave
    the database, so nothing else can reach the public file."""
    generated_at = now or datetime.utcnow()
    session = session_factory()
    try:
        rows = (
            session.query(SmartMoneyEvent, Enrichment.sector, Enrichment.price_at_signal)
            .outerjoin(Enrichment, SmartMoneyEvent.id == Enrichment.event_id)
            .filter(SmartMoneyEvent.created_at > window_start)
            .all()
        )
        trades = [_trade_row(event, sector, price) for event, sector, price in rows]
    finally:
        session.close()

    if not include_congressional:
        trades = [t for t in trades if t["source"] != SourceType.CONGRESSIONAL.value]
    trades.sort(key=lambda t: (t["filing_date"] or t["trade_date"], t["ticker"]), reverse=True)

    return {
        "date": date,
        "window_start": _iso(window_start),
        "generated_at": _iso(generated_at),
        "counts": _counts(trades),
        "highlights": select_highlights(trades),
        "top_10": select_highlights(trades, TOP_LIST_SIZE),
        "trades": trades,
    }


def select_highlights(trades: list[dict], limit: int = MAX_HIGHLIGHTS) -> list[dict]:
    """Pick up to limit trades, one per ticker. 13F filings are left out
    because their quarterly bulk positions would crowd out everything else."""
    insider_buys = [
        t for t in trades
        if t["source"] == SourceType.INSIDER.value and t["direction"] == "buy"
    ]
    clusters = _insider_clusters(insider_buys)[: limit * 3 // 5]
    large_buys = [
        {
            "kind": "large_buy",
            "ticker": t["ticker"],
            "sector": t["sector"],
            "value": t["value"],
            "headline": f"{t['actor']} bought {_money(t['value'])}",
        }
        for t in sorted(
            (t for t in insider_buys if t["value"]),
            key=lambda t: t["value"],
            reverse=True,
        )
    ]
    congressional = [
        {
            "kind": "congressional",
            "ticker": t["ticker"],
            "sector": t["sector"],
            "value": t["value"],
            "headline": (
                f"{t['actor']} {'bought' if t['direction'] == 'buy' else 'sold'} "
                f"{_money(t['value'])}"
            ),
        }
        for t in sorted(
            (t for t in trades if t["source"] == SourceType.CONGRESSIONAL.value),
            key=lambda t: t["value"] or 0,
            reverse=True,
        )
    ]

    # Hold back some large buys so congressional trades get slots when present
    held = limit * 2 // 5 if congressional else len(large_buys)
    ordered = clusters + large_buys[:held] + congressional + large_buys[held:]

    highlights, seen = [], set()
    for item in ordered:
        if item["ticker"] not in seen:
            seen.add(item["ticker"])
            highlights.append(item)
    return highlights[:limit]


async def publish_digest(
    session_factory,
    notifier,
    settings,
    publisher: DigestPublisher | None = None,
    now: datetime | None = None,
    state_path: Path = STATE_PATH,
) -> dict:
    """Publish the day's data on every run, and send the one compiled message on
    the first weekday-morning run. Publishing and sending fail independently."""
    local_now = now or datetime.now()
    date = local_now.strftime("%Y-%m-%d")
    publisher = publisher or DigestPublisher()

    digest = build_digest(
        session_factory,
        _window_start(publisher.read_latest(), date),
        date,
        settings.digest_include_congressional,
    )
    published = publisher.publish(digest)
    logger.info(
        "digest.built",
        date=date,
        trades=len(digest["trades"]),
        highlights=len(digest["highlights"]),
        published=published,
    )

    if _should_send(local_now, date, digest, state_path):
        if await notifier.send_digest(digest, settings.digest_url):
            _save_last_message_date(date, state_path)
    return digest


def _window_start(latest: dict | None, date: str) -> datetime:
    """Continue today's window on a re-run, or start where the last digest ended,
    so no trade lands in two daily files."""
    if latest is None:
        return datetime.utcnow() - timedelta(hours=24)
    boundary = latest["window_start"] if latest["date"] == date else latest["generated_at"]
    return datetime.strptime(boundary, "%Y-%m-%dT%H:%M:%SZ")


def _should_send(now: datetime, date: str, digest: dict, state_path: Path) -> bool:
    if now.weekday() >= 5 or now.hour >= MESSAGE_CUTOFF_HOUR or not digest["trades"]:
        return False
    return _last_message_date(state_path) != date


def _last_message_date(state_path: Path) -> str | None:
    try:
        return json.loads(state_path.read_text()).get("last_message_date")
    except (OSError, ValueError):
        return None


def _save_last_message_date(date: str, state_path: Path) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"last_message_date": date}))


def _reliable_value(event: SmartMoneyEvent, market_price: float | None) -> float | None:
    """The trade's dollar value, or None if its reported price isn't credible."""
    try:
        reported = json.loads(event.raw_payload or "{}").get("price_per_share")
    except ValueError:
        reported = None
    if reported and market_price and market_price > 0:
        ratio = reported / market_price
        if ratio > MAX_PRICE_RATIO or ratio < 1 / MAX_PRICE_RATIO:
            return None
    return event.size_lower


def _trade_row(
    event: SmartMoneyEvent, sector: str | None, market_price: float | None
) -> dict:
    return {
        "ticker": event.ticker,
        "actor": event.actor,
        "direction": event.direction.value,
        "value": _reliable_value(event, market_price),
        "trade_date": event.trade_date.strftime("%Y-%m-%d"),
        "filing_date": (
            event.disclosure_date.strftime("%Y-%m-%d") if event.disclosure_date else None
        ),
        "source": event.source_type.value,
        "sector": sector or None,
    }


def _counts(trades: list[dict]) -> dict:
    by_source: dict[str, int] = defaultdict(int)
    for t in trades:
        by_source[t["source"]] += 1
    return {
        "trades": len(trades),
        "tickers": len({t["ticker"] for t in trades}),
        "actors": len({t["actor"] for t in trades}),
        "buys": sum(t["direction"] == "buy" for t in trades),
        "sells": sum(t["direction"] == "sell" for t in trades),
        "by_source": dict(by_source),
    }


def _insider_clusters(insider_buys: list[dict]) -> list[dict]:
    """Tickers bought by two or more distinct insiders, broadest first. Related
    entities often report one purchase jointly, so identical value and date on
    the same ticker counts once."""
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for t in insider_buys:
        by_ticker[t["ticker"]].append(t)

    clusters = []
    for ticker, all_buys in by_ticker.items():
        unique: dict[tuple, dict] = {}
        for t in all_buys:
            key = (t["value"], t["trade_date"]) if t["value"] else (t["actor"],)
            unique.setdefault(key, t)
        buys = list(unique.values())
        insiders = {t["actor"] for t in buys}
        if len(insiders) < 2:
            continue
        total = sum(t["value"] or 0 for t in buys)
        clusters.append(
            {
                "kind": "cluster",
                "ticker": ticker,
                "sector": buys[0]["sector"],
                "value": total or None,
                "n_insiders": len(insiders),
                "headline": f"{len(insiders)} insiders bought {_money(total)}",
            }
        )
    clusters.sort(key=lambda c: (-c["n_insiders"], -(c["value"] or 0)))
    return clusters


def _money(value: float | None) -> str:
    if not value:
        return "an undisclosed amount"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:.0f}"


def _iso(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")
