import asyncio
from datetime import datetime, timedelta

import httpx

from src.clients.congress import CongressClient
from src.clients.edgar import EdgarClient
from src.models.schemas import Direction

FILED = datetime(2026, 11, 6)


def _form4(*transactions, ticker="ACME"):
    rows = "".join(
        f"""<nonDerivativeTransaction>
            <transactionDate><value>{day}</value></transactionDate>
            <transactionCoding><transactionCode>{code}</transactionCode></transactionCoding>
            <transactionAmounts>
              <transactionShares><value>100</value></transactionShares>
              <transactionPricePerShare><value>10</value></transactionPricePerShare>
            </transactionAmounts>
        </nonDerivativeTransaction>"""
        for code, day in transactions
    )
    return f"""<SEC-DOCUMENT>header
<XML>
<?xml version="1.0"?>
<ownershipDocument>
  <periodOfReport>2026-11-03</periodOfReport>
  <issuer><issuerTradingSymbol>{ticker}</issuerTradingSymbol></issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>Doe Jane</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><officerTitle>CEO</officerTitle></reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>{rows}</nonDerivativeTable>
</ownershipDocument>
</XML>
</SEC-DOCUMENT>"""


def test_daily_index_keeps_only_form4_and_dedupes():
    text = """Description: Daily Index
Form Type   Company Name   CIK   Date Filed  File Name
---------------------------------------------------------
4           ACME CORP      11    20261106    edgar/data/11/a.txt
4           Doe Jane       22    20261106    edgar/data/11/a.txt
4           OTHER, L.P.    33    20261106    edgar/data/33/b.txt
4/A         ACME CORP      11    20261106    edgar/data/11/c.txt
424B2       BANK           44    20261106    edgar/data/44/d.txt
"""
    assert EdgarClient._parse_daily_index(text) == {
        "edgar/data/11/a.txt": FILED,
        "edgar/data/33/b.txt": FILED,
    }


def test_only_open_market_codes_are_kept():
    filing = _form4(
        ("P", "2026-11-03"), ("S", "2026-11-04"), ("A", "2026-11-03"),
        ("M", "2026-11-03"), ("F", "2026-11-03"), ("G", "2026-11-03"),
    )

    events = EdgarClient()._parse_form4_xml(filing, FILED)

    assert [e.direction for e in events] == [Direction.BUY, Direction.SELL]


def test_disclosure_date_is_the_filing_date_not_period_of_report():
    events = EdgarClient()._parse_form4_xml(_form4(("P", "2026-11-03")), FILED)

    assert events[0].trade_date == datetime(2026, 11, 3)
    assert events[0].disclosure_date == FILED


def test_placeholder_tickers_are_skipped():
    filing = _form4(("P", "2026-11-03"), ticker="N/A")

    assert EdgarClient()._parse_form4_xml(filing, FILED) == []


def test_non_ownership_content_yields_no_events():
    assert EdgarClient()._parse_form4_xml("<html>not a filing</html>", FILED) == []


def test_bulk_ingestion_reads_every_filing_and_counts_failures():
    index = """4  ACME   11  20261106  edgar/data/11/a.txt
4  BETA   22  20261106  edgar/data/22/b.txt
4  GONE   33  20261106  edgar/data/33/missing.txt
"""
    client = EdgarClient()

    async def fake_get(url, params=None):
        if url.endswith(".idx"):
            return httpx.Response(200, text=index)
        if "missing" in url:
            raise RuntimeError("not found")
        ticker = "ACME" if "/a.txt" in url else "BETA"
        return httpx.Response(200, text=_form4(("P", "2020-01-02"), ticker=ticker))

    client._get = fake_get

    events = asyncio.run(client.get_bulk_insider_trades(since_days=7))

    # An old trade in a recent filing is kept: the window is on filing date
    assert sorted(e.ticker for e in events) == ["ACME", "BETA"]


def test_missing_daily_index_does_not_stop_ingestion():
    client = EdgarClient()

    async def fake_get(url, params=None):
        raise RuntimeError("403")

    client._get = fake_get

    assert asyncio.run(client.get_bulk_insider_trades(since_days=1)) == []


def _trade(traded_days_ago, published_days_ago, ticker="AMAT:US"):
    now = datetime.utcnow()
    return {
        "issuer": {"issuerTicker": ticker},
        "txType": "buy",
        "txDate": (now - timedelta(days=traded_days_ago)).strftime("%Y-%m-%d"),
        "pubDate": (now - timedelta(days=published_days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "chamber": "house",
        "politician": {"firstName": "Pat", "lastName": f"Lee{traded_days_ago}"},
        "value": 1000,
    }


def _congress_client(pages):
    client = CongressClient()
    fetched = []

    async def fake_fetch(page=1):
        fetched.append(page)
        return pages.get(page, [])

    client._fetch_page = fake_fetch
    return client, fetched


def test_congress_window_uses_publication_date(monkeypatch):
    monkeypatch.setattr("src.clients.congress.PAGE_DELAY_SECONDS", 0)
    client, _ = _congress_client({1: [_trade(40, 2), _trade(40, 20)], 2: [_trade(50, 30)]})

    events = asyncio.run(client.get_all_congressional_trades(since_days=14))

    assert len(events) == 1
    assert events[0].ticker == "AMAT"


def test_congress_paging_survives_pages_with_no_usable_tickers(monkeypatch):
    monkeypatch.setattr("src.clients.congress.PAGE_DELAY_SECONDS", 0)
    client, fetched = _congress_client(
        {
            1: [_trade(30, 1, ticker=None)],
            2: [_trade(31, 2)],
            3: [_trade(60, 30)],
            4: [_trade(32, 2)],
        }
    )

    events = asyncio.run(client.get_all_congressional_trades(since_days=14))

    assert len(events) == 1
    assert fetched == [1, 2, 3]  # stops at the first page that is entirely old
