"""Tests for the catalyst calendar (earnings + macro + 8-K merge)."""

from datetime import date, datetime, timedelta

import pytest

from src.analysis.catalyst_calendar import (
    MACRO_CALENDAR_2026,
    _days_between,
    _get_macro_catalysts,
    format_catalyst_calendar,
)


def test_macro_calendar_has_fomc_and_cpi():
    events = [e["event"] for e in MACRO_CALENDAR_2026]
    assert any("FOMC" in e for e in events)
    assert any("CPI" in e for e in events)
    assert any("Nonfarm Payrolls" in e for e in events)


def test_days_between_basic():
    d = date(2026, 1, 1)
    assert _days_between(d, "2026-01-11") == 10


@pytest.mark.asyncio
async def test_get_macro_catalysts_filters_horizon():
    today = date(2026, 3, 1)
    horizon_end = today + timedelta(days=30)
    events = await _get_macro_catalysts(today, horizon_end)
    for e in events:
        ev_date = datetime.strptime(e["date"], "%Y-%m-%d").date()
        assert today <= ev_date <= horizon_end


@pytest.mark.asyncio
async def test_get_macro_catalysts_returns_list_structure():
    today = date(2026, 4, 1)
    horizon_end = today + timedelta(days=60)
    events = await _get_macro_catalysts(today, horizon_end)
    assert isinstance(events, list)
    for e in events:
        assert "date" in e and "event" in e and "importance" in e
        assert e["type"] == "macro"


def test_format_catalyst_calendar_with_no_data():
    assert "No catalyst data available" in format_catalyst_calendar({})


def test_format_catalyst_calendar_with_items():
    catalysts = {
        "as_of": "2026-04-20",
        "near_term": [
            {"date": "2026-05-01", "event": "Earnings Release",
             "type": "earnings", "importance": "high"},
        ],
        "medium_term": [
            {"date": "2026-06-16", "event": "FOMC Meeting",
             "type": "macro", "importance": "high"},
        ],
        "recent_8k": [
            {"date": "2026-04-10", "event": "Form 8-K filed",
             "type": "8-K", "importance": "medium",
             "url": "https://sec.gov/placeholder"},
        ],
    }
    out = format_catalyst_calendar(catalysts)
    assert "Near-term" in out
    assert "Medium-term" in out
    assert "2026-05-01" in out
    assert "FOMC" in out
    assert "8-K" in out.upper() or "8-K" in out


def test_format_catalyst_calendar_handles_empty_buckets():
    catalysts = {
        "as_of": "2026-04-20",
        "near_term": [],
        "medium_term": [],
        "recent_8k": [],
    }
    out = format_catalyst_calendar(catalysts)
    assert "No scheduled catalysts" in out
