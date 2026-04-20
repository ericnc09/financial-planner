"""
Catalyst calendar — near- and medium-term events that could move the stock.

Aggregates three catalyst sources into a single dated timeline:
  1. Earnings announcements (via yfinance earnings_dates)
  2. Recent corporate filings (8-K via SEC EDGAR submissions API)
  3. Macro releases (FOMC meetings + CPI / NFP / GDP dates from a static schedule)

Output is split into near-term (0-30 days) and medium-term (30-90 days)
buckets for the equity report.

Notes on design choices:
  - 8-K calendar is *past* filings from the last 30 days (EDGAR does not
    publish a forward calendar). Used as a signal of recent corporate
    activity, not a future-event calendar.
  - Macro dates are hard-coded for the current year. FRED release calendar
    is not exposed in a stable JSON feed, and scraping is fragile — a
    curated list is both simpler and more reliable within a small horizon.
  - Timezone-naive dates throughout. Consistent with the rest of the repo.
"""

from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta

import httpx
import structlog

logger = structlog.get_logger()


# Curated 2026 macro calendar — FOMC meetings + major data releases.
# Update annually; next year's dates are published by the Fed and BLS.
MACRO_CALENDAR_2026: list[dict] = [
    {"date": "2026-01-27", "event": "FOMC Meeting", "importance": "high"},
    {"date": "2026-01-28", "event": "FOMC Rate Decision", "importance": "high"},
    {"date": "2026-02-11", "event": "CPI Release (Jan)", "importance": "high"},
    {"date": "2026-02-06", "event": "Nonfarm Payrolls (Jan)", "importance": "high"},
    {"date": "2026-02-27", "event": "PCE Release (Jan)", "importance": "medium"},
    {"date": "2026-03-11", "event": "CPI Release (Feb)", "importance": "high"},
    {"date": "2026-03-06", "event": "Nonfarm Payrolls (Feb)", "importance": "high"},
    {"date": "2026-03-17", "event": "FOMC Meeting", "importance": "high"},
    {"date": "2026-03-18", "event": "FOMC Rate Decision + SEP", "importance": "high"},
    {"date": "2026-03-27", "event": "PCE Release (Feb)", "importance": "medium"},
    {"date": "2026-03-26", "event": "GDP Q4 Final", "importance": "medium"},
    {"date": "2026-04-10", "event": "CPI Release (Mar)", "importance": "high"},
    {"date": "2026-04-03", "event": "Nonfarm Payrolls (Mar)", "importance": "high"},
    {"date": "2026-04-28", "event": "FOMC Meeting", "importance": "high"},
    {"date": "2026-04-29", "event": "FOMC Rate Decision", "importance": "high"},
    {"date": "2026-04-30", "event": "GDP Q1 Advance", "importance": "high"},
    {"date": "2026-05-01", "event": "Nonfarm Payrolls (Apr)", "importance": "high"},
    {"date": "2026-05-13", "event": "CPI Release (Apr)", "importance": "high"},
    {"date": "2026-05-30", "event": "PCE Release (Apr)", "importance": "medium"},
    {"date": "2026-06-05", "event": "Nonfarm Payrolls (May)", "importance": "high"},
    {"date": "2026-06-11", "event": "CPI Release (May)", "importance": "high"},
    {"date": "2026-06-16", "event": "FOMC Meeting", "importance": "high"},
    {"date": "2026-06-17", "event": "FOMC Rate Decision + SEP", "importance": "high"},
    {"date": "2026-06-26", "event": "PCE Release (May)", "importance": "medium"},
    {"date": "2026-07-03", "event": "Nonfarm Payrolls (Jun)", "importance": "high"},
    {"date": "2026-07-15", "event": "CPI Release (Jun)", "importance": "high"},
    {"date": "2026-07-28", "event": "FOMC Meeting", "importance": "high"},
    {"date": "2026-07-29", "event": "FOMC Rate Decision", "importance": "high"},
    {"date": "2026-07-30", "event": "GDP Q2 Advance", "importance": "high"},
    {"date": "2026-07-31", "event": "PCE Release (Jun)", "importance": "medium"},
    {"date": "2026-08-07", "event": "Nonfarm Payrolls (Jul)", "importance": "high"},
    {"date": "2026-08-12", "event": "CPI Release (Jul)", "importance": "high"},
    {"date": "2026-08-28", "event": "PCE Release (Jul)", "importance": "medium"},
    {"date": "2026-09-04", "event": "Nonfarm Payrolls (Aug)", "importance": "high"},
    {"date": "2026-09-11", "event": "CPI Release (Aug)", "importance": "high"},
    {"date": "2026-09-15", "event": "FOMC Meeting", "importance": "high"},
    {"date": "2026-09-16", "event": "FOMC Rate Decision + SEP", "importance": "high"},
    {"date": "2026-09-25", "event": "PCE Release (Aug)", "importance": "medium"},
    {"date": "2026-10-02", "event": "Nonfarm Payrolls (Sep)", "importance": "high"},
    {"date": "2026-10-14", "event": "CPI Release (Sep)", "importance": "high"},
    {"date": "2026-10-27", "event": "FOMC Meeting", "importance": "high"},
    {"date": "2026-10-28", "event": "FOMC Rate Decision", "importance": "high"},
    {"date": "2026-10-29", "event": "GDP Q3 Advance", "importance": "high"},
    {"date": "2026-10-30", "event": "PCE Release (Sep)", "importance": "medium"},
    {"date": "2026-11-06", "event": "Nonfarm Payrolls (Oct)", "importance": "high"},
    {"date": "2026-11-12", "event": "CPI Release (Oct)", "importance": "high"},
    {"date": "2026-11-25", "event": "PCE Release (Oct)", "importance": "medium"},
    {"date": "2026-12-04", "event": "Nonfarm Payrolls (Nov)", "importance": "high"},
    {"date": "2026-12-10", "event": "CPI Release (Nov)", "importance": "high"},
    {"date": "2026-12-15", "event": "FOMC Meeting", "importance": "high"},
    {"date": "2026-12-16", "event": "FOMC Rate Decision + SEP", "importance": "high"},
    {"date": "2026-12-23", "event": "PCE Release (Nov)", "importance": "medium"},
]

USER_AGENT = "SmartMoneyFollows research@smartmoneyfollows.dev"
EDGAR_CIK_LOOKUP = "https://www.sec.gov/cgi-bin/browse-edgar"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"


async def get_catalysts(
    ticker: str,
    horizon_days: int = 90,
    include_8k: bool = True,
) -> dict:
    """
    Build a catalyst calendar for a ticker.

    Args:
        ticker: Uppercase ticker symbol.
        horizon_days: Forward window (default 90 days).
        include_8k: If True, also pull recent 8-K filings from EDGAR.

    Returns:
        Dict with near_term (0-30d), medium_term (30-90d), recent_8k
        (past 30d, if requested), and a combined timeline list.
    """
    today = datetime.utcnow().date()
    horizon_end = today + timedelta(days=horizon_days)

    # Parallelise the three source pulls
    tasks = [
        _get_earnings_catalysts(ticker, today, horizon_end),
        _get_macro_catalysts(today, horizon_end),
    ]
    if include_8k:
        tasks.append(_get_recent_8k_filings(ticker, today))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    earnings_events = results[0] if not isinstance(results[0], Exception) else []
    macro_events = results[1] if not isinstance(results[1], Exception) else []
    recent_8k = []
    if include_8k:
        if not isinstance(results[2], Exception):
            recent_8k = results[2]

    timeline = sorted(
        earnings_events + macro_events,
        key=lambda e: e["date"],
    )

    near_term = [e for e in timeline if _days_between(today, e["date"]) <= 30]
    medium_term = [
        e for e in timeline if 30 < _days_between(today, e["date"]) <= horizon_days
    ]

    return {
        "ticker": ticker,
        "as_of": today.isoformat(),
        "horizon_days": horizon_days,
        "near_term": near_term,
        "medium_term": medium_term,
        "recent_8k": recent_8k,
        "timeline": timeline,
    }


def _days_between(d0: date, iso_str: str) -> int:
    return (datetime.strptime(iso_str, "%Y-%m-%d").date() - d0).days


async def _get_earnings_catalysts(
    ticker: str, today: date, horizon_end: date
) -> list[dict]:
    """Pull the next earnings date (if any) within the horizon via yfinance."""
    return await asyncio.to_thread(_earnings_sync, ticker, today, horizon_end)


def _earnings_sync(ticker: str, today: date, horizon_end: date) -> list[dict]:
    try:
        import yfinance as yf
    except ImportError:
        return []
    try:
        stock = yf.Ticker(ticker)
        earnings_dates = stock.earnings_dates
    except Exception as e:
        logger.debug("catalyst.earnings_fetch_failed", ticker=ticker, error=str(e))
        return []

    if earnings_dates is None or earnings_dates.empty:
        return []

    events = []
    for dt in earnings_dates.index:
        dt_naive = dt.to_pydatetime().replace(tzinfo=None).date()
        if today <= dt_naive <= horizon_end:
            events.append(
                {
                    "date": dt_naive.isoformat(),
                    "event": f"{ticker} Earnings Release",
                    "type": "earnings",
                    "importance": "high",
                    "ticker": ticker,
                }
            )

    # Use only the nearest upcoming earnings — avoids duplicate guesses from yfinance
    if events:
        events.sort(key=lambda e: e["date"])
        events = events[:1]

    return events


async def _get_macro_catalysts(today: date, horizon_end: date) -> list[dict]:
    """Filter the static macro calendar to the current horizon window."""
    events = []
    for m in MACRO_CALENDAR_2026:
        try:
            event_date = datetime.strptime(m["date"], "%Y-%m-%d").date()
        except ValueError:
            continue
        if today <= event_date <= horizon_end:
            events.append(
                {
                    "date": m["date"],
                    "event": m["event"],
                    "type": "macro",
                    "importance": m["importance"],
                    "ticker": None,
                }
            )
    return events


async def _get_recent_8k_filings(ticker: str, today: date) -> list[dict]:
    """Pull the 8-K filings for a ticker from SEC EDGAR's submissions API (past 30d)."""
    try:
        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT}, timeout=20.0
        ) as client:
            cik = await _resolve_cik(client, ticker)
            if not cik:
                return []

            resp = await client.get(EDGAR_SUBMISSIONS.format(cik=cik))
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.debug("catalyst.8k_fetch_failed", ticker=ticker, error=str(e))
        return []

    recent = data.get("filings", {}).get("recent", {}) or {}
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accession_nums = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    cutoff = today - timedelta(days=30)
    filings = []
    for i, form in enumerate(forms):
        if form != "8-K":
            continue
        try:
            filing_date = datetime.strptime(dates[i], "%Y-%m-%d").date()
        except (IndexError, ValueError):
            continue
        if filing_date < cutoff:
            continue

        accession = accession_nums[i] if i < len(accession_nums) else ""
        primary = primary_docs[i] if i < len(primary_docs) else ""
        url = ""
        if accession and primary:
            no_dashes = accession.replace("-", "")
            url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{no_dashes}/{primary}"
            )

        filings.append(
            {
                "date": filing_date.isoformat(),
                "event": f"{ticker} Form 8-K filed",
                "type": "8-K",
                "importance": "medium",
                "ticker": ticker,
                "url": url,
            }
        )

    filings.sort(key=lambda f: f["date"], reverse=True)
    return filings[:10]


async def _resolve_cik(client: httpx.AsyncClient, ticker: str) -> int | None:
    """Resolve a ticker to its SEC CIK via the browse-edgar landing page."""
    try:
        resp = await client.get(
            EDGAR_CIK_LOOKUP,
            params={
                "action": "getcompany",
                "CIK": ticker,
                "type": "8-K",
                "dateb": "",
                "owner": "include",
                "count": "1",
                "output": "atom",
            },
        )
        resp.raise_for_status()
    except Exception:
        return None

    # CIK appears as a 10-digit number in the atom feed
    match = re.search(r"CIK=(\d{1,10})", resp.text)
    if match:
        return int(match.group(1))
    # Fallback: parse <id> elements which embed the CIK
    try:
        root = ET.fromstring(resp.text)
        for elem in root.iter():
            if elem.text and "CIK" in (elem.text or ""):
                m = re.search(r"CIK=(\d+)", elem.text)
                if m:
                    return int(m.group(1))
    except ET.ParseError:
        pass
    return None


def format_catalyst_calendar(catalysts: dict) -> str:
    """Render the catalyst calendar as a markdown block."""
    if not catalysts:
        return "_No catalyst data available._"

    today_iso = catalysts.get("as_of", "")
    near = catalysts.get("near_term") or []
    medium = catalysts.get("medium_term") or []
    recent_8k = catalysts.get("recent_8k") or []

    def _fmt_row(e: dict) -> str:
        imp = e.get("importance", "medium").upper()
        tag = e.get("type", "?")
        return f"- **{e['date']}** · {e['event']}  _[{tag} · {imp}]_"

    lines = [f"_As of {today_iso}._", ""]

    lines.append("**Near-term (next 30 days)**")
    if near:
        lines.extend(_fmt_row(e) for e in near)
    else:
        lines.append("- No scheduled catalysts in the near term.")

    lines.append("")
    lines.append("**Medium-term (30-90 days)**")
    if medium:
        lines.extend(_fmt_row(e) for e in medium)
    else:
        lines.append("- No scheduled catalysts in the medium term.")

    if recent_8k:
        lines.append("")
        lines.append("**Recent 8-K filings (past 30 days)**")
        for f in recent_8k[:5]:
            url = f.get("url", "")
            label = f"[{f['date']} 8-K]({url})" if url else f"{f['date']} 8-K"
            lines.append(f"- {label}")

    return "\n".join(lines)
