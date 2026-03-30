"""
SEC EDGAR client for insider trading data (Form 3/4/5).

Uses the EDGAR RSS feed (browse-edgar?output=atom) to discover recent Form 4 filings,
then parses the raw XML for transaction details.

Free, no API key. Requires User-Agent header and respects 10 req/s rate limit.
"""

import asyncio
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models.schemas import Direction, SmartMoneyEventSchema, SourceType

logger = structlog.get_logger()

USER_AGENT = "SmartMoneyFollows research@smartmoneyfollows.dev"

BUY_CODES = {"P", "M", "A", "J", "K", "I"}
SELL_CODES = {"S", "D", "F", "G"}

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


class EdgarClient:
    """Fetches insider trades from SEC EDGAR Form 4 filings via RSS + XML parsing."""

    RSS_URL = "https://www.sec.gov/cgi-bin/browse-edgar"

    def __init__(self):
        self._client = httpx.AsyncClient(
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/xml, application/json, text/html",
            },
            timeout=30.0,
            follow_redirects=True,
        )
        self._rate_limiter = asyncio.Semaphore(8)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
    async def _get(self, url: str, params: dict | None = None) -> httpx.Response:
        async with self._rate_limiter:
            await asyncio.sleep(0.12)  # ~8 req/s max
            resp = await self._client.get(url, params=params)
            if resp.status_code == 429:
                await asyncio.sleep(5)
                raise Exception("Rate limited by SEC")
            resp.raise_for_status()
            return resp

    async def get_recent_form4_rss(self, count: int = 100) -> list[dict]:
        """
        Fetch recent Form 4 filings from EDGAR RSS (Atom) feed.
        Returns list of {title, index_url, updated} dicts.
        """
        resp = await self._get(
            self.RSS_URL,
            params={
                "action": "getcurrent",
                "type": "4",
                "owner": "include",
                "count": str(count),
                "output": "atom",
            },
        )
        root = ET.fromstring(resp.text)
        entries = root.findall("atom:entry", ATOM_NS)

        filings = []
        for entry in entries:
            category = entry.find("atom:category", ATOM_NS)
            if category is not None and category.get("term") != "4":
                continue

            title = entry.find("atom:title", ATOM_NS)
            link = entry.find("atom:link", ATOM_NS)
            updated = entry.find("atom:updated", ATOM_NS)

            if link is not None:
                filings.append(
                    {
                        "title": title.text if title is not None else "",
                        "index_url": link.get("href", ""),
                        "updated": updated.text if updated is not None else "",
                    }
                )

        logger.info("edgar.rss_form4", count=len(filings))
        return filings

    async def _get_form4_xml_url(self, index_url: str) -> str | None:
        """Parse a filing index page to find the raw Form 4 XML document URL."""
        try:
            resp = await self._get(index_url)
            # Find XML links that are NOT the xsl-rendered version
            xml_links = re.findall(
                r'href="(/Archives/edgar/data/[^"]+\.xml)"', resp.text
            )
            for link in xml_links:
                if "xsl" not in link.lower():
                    return f"https://www.sec.gov{link}"
            # Fallback: take the last XML link (usually the raw one)
            if xml_links:
                return f"https://www.sec.gov{xml_links[-1]}"
        except Exception as e:
            logger.debug("edgar.index_parse_failed", url=index_url, error=str(e))
        return None

    async def _parse_form4_xml(self, xml_url: str) -> list[SmartMoneyEventSchema]:
        """Parse a raw Form 4 XML document for transaction details."""
        try:
            resp = await self._get(xml_url)
            content = resp.text
            if "<ownershipDocument>" not in content:
                return []

            root = ET.fromstring(content)
            events = []

            # Issuer info
            issuer = root.find(".//issuer")
            ticker = ""
            if issuer is not None:
                ticker_elem = issuer.find("issuerTradingSymbol")
                if ticker_elem is not None and ticker_elem.text:
                    ticker = ticker_elem.text.strip().upper()
            if not ticker or len(ticker) > 10:
                return []

            # Reporting owner
            owner_name = "Unknown"
            owner_title = ""
            owner_elem = root.find(".//reportingOwner")
            if owner_elem is not None:
                name_elem = owner_elem.find(".//rptOwnerName")
                if name_elem is not None and name_elem.text:
                    owner_name = name_elem.text.strip()
                title_elem = owner_elem.find(".//officerTitle")
                if title_elem is not None and title_elem.text:
                    owner_title = title_elem.text.strip()

            actor = f"{owner_name} ({owner_title})" if owner_title else owner_name

            # Period of report (filing date)
            period_elem = root.find("periodOfReport")
            filing_date = datetime.utcnow()
            if period_elem is not None and period_elem.text:
                try:
                    filing_date = datetime.strptime(
                        period_elem.text.strip(), "%Y-%m-%d"
                    )
                except ValueError:
                    pass

            # Parse non-derivative transactions
            for tx in root.findall(".//nonDerivativeTransaction"):
                event = self._parse_transaction(tx, ticker, actor, filing_date)
                if event:
                    events.append(event)

            # Parse derivative transactions
            for tx in root.findall(".//derivativeTransaction"):
                event = self._parse_transaction(tx, ticker, actor, filing_date)
                if event:
                    events.append(event)

            return events

        except ET.ParseError:
            return []
        except Exception as e:
            logger.debug("edgar.form4_parse_failed", url=xml_url, error=str(e))
            return []

    def _parse_transaction(
        self,
        tx_elem,
        ticker: str,
        actor: str,
        filing_date: datetime,
    ) -> SmartMoneyEventSchema | None:
        try:
            code_elem = tx_elem.find(".//transactionCode")
            if code_elem is None or not code_elem.text:
                return None
            code = code_elem.text.strip().upper()

            if code in BUY_CODES:
                direction = Direction.BUY
            elif code in SELL_CODES:
                direction = Direction.SELL
            else:
                return None

            # Transaction date
            date_elem = tx_elem.find(".//transactionDate/value")
            trade_date = filing_date
            if date_elem is not None and date_elem.text:
                try:
                    trade_date = datetime.strptime(
                        date_elem.text.strip(), "%Y-%m-%d"
                    )
                except ValueError:
                    pass

            # Shares
            shares_elem = tx_elem.find(".//transactionShares/value")
            shares = 0.0
            if shares_elem is not None and shares_elem.text:
                try:
                    shares = float(shares_elem.text)
                except ValueError:
                    pass

            # Price
            price_elem = tx_elem.find(".//transactionPricePerShare/value")
            price = 0.0
            if price_elem is not None and price_elem.text:
                try:
                    price = float(price_elem.text)
                except ValueError:
                    pass

            dollar_value = shares * price if shares and price else None

            return SmartMoneyEventSchema(
                ticker=ticker,
                actor=actor,
                direction=direction,
                size_lower=dollar_value,
                size_upper=dollar_value,
                trade_date=trade_date,
                disclosure_date=filing_date,
                source_type=SourceType.INSIDER,
                raw_payload=json.dumps(
                    {
                        "transaction_code": code,
                        "shares": shares,
                        "price_per_share": price,
                        "dollar_value": dollar_value,
                        "source": "sec_edgar_form4",
                    }
                ),
            )
        except Exception:
            return None

    async def get_bulk_insider_trades(
        self, since_days: int = 14
    ) -> list[SmartMoneyEventSchema]:
        """
        Get recent insider trades by:
        1. Fetching Form 4 RSS feed for latest filings
        2. For each filing, finding the raw XML URL
        3. Parsing each XML for transaction details
        """
        cutoff = datetime.utcnow() - timedelta(days=since_days)
        filings = await self.get_recent_form4_rss(count=100)

        all_events = []
        for filing in filings:
            index_url = filing.get("index_url")
            if not index_url:
                continue

            xml_url = await self._get_form4_xml_url(index_url)
            if not xml_url:
                continue

            events = await self._parse_form4_xml(xml_url)
            for e in events:
                if e.trade_date >= cutoff:
                    all_events.append(e)

        # Deduplicate
        seen = set()
        deduped = []
        for e in all_events:
            key = (e.ticker, e.actor, e.trade_date.date(), e.direction)
            if key not in seen:
                seen.add(key)
                deduped.append(e)

        deduped.sort(key=lambda e: e.trade_date, reverse=True)
        logger.info("edgar.bulk_insider_trades", total=len(deduped))
        return deduped

    async def close(self):
        await self._client.aclose()
